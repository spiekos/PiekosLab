"""Nested cross-validation for the Aim 3A base models.

Structure (per F.3)
-------------------
    outer loop  -> honest performance estimate; scores only on folds the
                   tuning never saw
    inner loop  -> hyperparameter tuning, including the elastic-net mixing
                   parameter `l1_ratio`

Both loops group by `SubjectID` (`StratifiedGroupKFold`), so a participant's
samples never straddle a fold boundary. Within one gestational window each
participant contributes a single row, but grouping is kept explicit so the code
stays correct if windows are ever pooled.

Everything that learns from data sits inside the Pipeline - half-minimum
imputation, scaling and feature selection - so all three refit on each training
fold. Fitting any of them once on the whole development set lets them see every
inner validation fold before those folds are used to tune.

Elastic net must be requested in a version-appropriate way: sklearn < 1.8
needs `penalty="elasticnet"` or it silently falls back to L2 and zeroes no
coefficients (selecting every feature); sklearn >= 1.8 deprecates `penalty`
and takes `l1_ratio` alone. `enet_kwargs()` picks the right form, and
`assert_elastic_net()` warns if a fit ever produces no sparsity at all.

Models
------
    LogisticRegression (elastic-net)  performs its own
                                      dimensionality reduction while fitting
    RandomForest, XGBoost             comparators, each preceded by an explicit
                                      elastic-net selector step

(SVM was dropped from the model list on 2026-09-30.)

Class imbalance is handled by `class_weight="balanced"` for the logistic
regression and random forest, and by `scale_pos_weight` for XGBoost, which
has no `class_weight` parameter.

Reported
--------
PR-AUC (primary, per F.3), ROC-AUC, accuracy, F1, precision, recall,
specificity and Brier score. PR-AUC and ROC-AUC carry bootstrap confidence
intervals computed over pooled out-of-fold predictions; the remaining metrics
are means across outer folds.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

L1_RATIO_GRID = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
RANDOM_STATE = 42
N_BOOT = 1000

# Display name for the elastic-net-penalised logistic regression. Spelled out
# because "ElasticNet" alone reads as sklearn's linear regressor for continuous
# outcomes, which would be the wrong model for a binary endpoint.
EN_LOGREG = "LogisticRegression (elastic-net)"

MODEL_NAMES = [EN_LOGREG, "RandomForest", "XGBoost"]

# ---------------------------------------------------------------------------
# sklearn deprecated `penalty` in 1.8 (removal in 1.10): elastic net is now
# parameterised by `l1_ratio` alone. But on sklearn < 1.8 the default penalty
# is "l2" and `l1_ratio` is silently IGNORED - which is precisely the bug that
# made the old selector return every feature. So the argument has to be chosen
# by version rather than dropped.
#
#   sklearn >= 1.8 : pass l1_ratio only        (penalty= emits FutureWarning)
#   sklearn <  1.8 : pass penalty="elasticnet" (without it, no sparsity at all)
# ---------------------------------------------------------------------------
_SKL_GE_18 = tuple(int(x) for x in sklearn_version.split(".")[:2]) >= (1, 8)


# ---------------------------------------------------------------------------
# RandomForest runs single-threaded (n_jobs=1).
#
# sklearn >= 1.8 wraps every parallel task in warnings.catch_warnings() +
# warnings.resetwarnings() to propagate warning filters to workers. RF's
# n_jobs=-1 uses joblib's *threading* backend, and catch_warnings is not
# thread-safe before Python 3.14: concurrent workers can leave the process-wide
# warnings.filters list empty. Once empty it stays empty, and every later task
# emits "`sklearn.utils.parallel.delayed` should be used with
# `sklearn.utils.parallel.Parallel`..." - thousands per run.
#
# Reproduced on sklearn 1.8.0 / Python 3.12: filters went 11 -> 0 at fit #9,
# then 390 warnings from 120 fits. With n_jobs=1: 0 warnings, filters intact.
#
# Suppressing the message does not work: the ignore-filter lives in the same
# list the race erases. Removing the threads removes the cause.
#
# Cost: none measurable at our data size (84 x 1399: 0.20 s per fit either
# way). Predictions are bit-identical to n_jobs=-1 at a fixed random_state.
# XGBoost keeps n_jobs=-1: it threads natively in C++, not through joblib.
# ---------------------------------------------------------------------------
RF_N_JOBS = 1


def enet_kwargs(**kw) -> dict:
    """Return kwargs that give a genuine elastic net on this sklearn version."""
    if not _SKL_GE_18:
        kw["penalty"] = "elasticnet"
    return kw


def assert_elastic_net(estimator, n_features: int) -> None:
    """Fail loudly if a fitted linear model produced no sparsity at all.

    Version-independent guard against the historical failure mode where the
    penalty silently fell back to L2 and every feature survived selection.
    Only meaningful when l1_ratio > 0 and the problem is wide.
    """
    coef = getattr(estimator, "coef_", None)
    if coef is None or n_features < 50:
        return
    if int((coef.ravel() == 0).sum()) == 0:
        logger.warning(
            "Elastic-net fit produced ZERO exactly-zero coefficients across %d "
            "features. On this sklearn (%s) that usually means the penalty fell "
            "back to L2 and no selection is happening.", n_features, sklearn_version,
        )




def scale_pos_weight(y) -> float:
    """XGBoost's imbalance lever; it has no `class_weight`."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return float(n_neg / n_pos) if n_pos else 1.0


class HalfMinimumImputer(BaseEstimator, TransformerMixin):
    """Half-minimum imputation, fitted per training fold.

    Fills missing values with (column minimum - 1) on the log2 scale, which is
    halving in linear space: the value sits just below the lowest detected
    concentration. This matches the assumption the SOP pipeline makes at
    Step 10 - in mass spectrometry a missing value usually means below the
    detection limit, not missing at random.

    Previously this stage used `SimpleImputer(strategy="median")`, which
    assumes missing-at-random and substitutes a *typical* value where the
    evidence points to a *low* one. That inconsistency was material: 4.5% of
    LIPD_plasma cells reach the model still missing, spread across every
    feature.

    Fitted inside the CV pipeline, so each fold's floors come from its own
    training rows. A column entirely missing in training falls back to the
    global training minimum, and to 0.0 if the whole fold is empty.
    """

    def fit(self, X, y=None):
        A = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column
            mins = np.nanmin(A, axis=0)
        allnan = np.all(np.isnan(A), axis=0)
        if allnan.any():
            finite = mins[~allnan]
            mins = np.where(allnan, float(np.min(finite)) if finite.size else 1.0, mins)
        self.floors_ = mins - 1.0
        self.n_features_in_ = A.shape[1]
        return self

    def transform(self, X):
        A = np.asarray(X, dtype=float).copy()
        idx = np.where(np.isnan(A))
        if idx[0].size:
            A[idx] = np.take(self.floors_, idx[1])
        return A


def _prep_steps() -> list:
    """Imputation + scaling, refit per fold.

    Both belong here rather than upfront: fitting either on the whole
    development set before cross-validation lets each outer fold's parameters
    absorb its own test fold.
    """
    return [
        ("impute", HalfMinimumImputer()),
        ("scale", RobustScaler()),
    ]


def _selector(random_state: int = RANDOM_STATE) -> SelectFromModel:
    """Elastic-net selector placed ahead of the non-linear comparators.

    Uses a fixed C / l1_ratio. A CV-tuned selector here would nest a third
    cross-validation inside the inner loop inside the outer loop, which is
    prohibitively slow on a ~1400-feature matrix. The elastic-net logistic
    regression tunes `l1_ratio` properly, as F.3 requires.

    Elastic net is requested via `enet_kwargs()` - see module docstring.
    """
    return SelectFromModel(
        LogisticRegression(
            **enet_kwargs(C=0.1, l1_ratio=0.5),
            solver="saga", class_weight="balanced",
            random_state=random_state, max_iter=2000, tol=1e-3,
        ),
        threshold=1e-10,
    )


def build_model(name: str, params: dict, y_train, random_state: int = RANDOM_STATE) -> Pipeline:
    """Instantiate a full pipeline for `name` from a parameter dict."""
    p = dict(params or {})

    if name == EN_LOGREG:
        return Pipeline(_prep_steps() + [
            ("clf", LogisticRegression(
                **enet_kwargs(C=p.get("C", 1.0), l1_ratio=p.get("l1_ratio", 0.5)),
                solver="saga", class_weight="balanced",
                random_state=random_state, max_iter=2000, tol=1e-3,
            )),
        ])

    if name == "RandomForest":
        return Pipeline(_prep_steps() + [
            ("select", _selector(random_state)),
            ("clf", RandomForestClassifier(
                n_estimators=p.get("n_estimators", 300),
                max_depth=p.get("max_depth", None),
                min_samples_split=p.get("min_samples_split", 2),
                min_samples_leaf=p.get("min_samples_leaf", 1),
                max_features=p.get("max_features", "sqrt"),
                class_weight="balanced", random_state=random_state,
                # Single-threaded on purpose - see RF_N_JOBS.
                n_jobs=RF_N_JOBS,
            )),
        ])

    if name == "XGBoost":
        from xgboost import XGBClassifier
        return Pipeline(_prep_steps() + [
            ("select", _selector(random_state)),
            ("clf", XGBClassifier(
                n_estimators=p.get("n_estimators", 300),
                max_depth=p.get("max_depth", 4),
                learning_rate=p.get("learning_rate", 0.1),
                subsample=p.get("subsample", 1.0),
                colsample_bytree=p.get("colsample_bytree", 1.0),
                min_child_weight=p.get("min_child_weight", 1),
                gamma=p.get("gamma", 0.0),
                reg_alpha=p.get("reg_alpha", 1e-5),
                reg_lambda=p.get("reg_lambda", 1.0),
                scale_pos_weight=scale_pos_weight(y_train),
                eval_metric="logloss", random_state=random_state,
                tree_method="hist", n_jobs=-1, verbosity=0,
            )),
        ])

    raise ValueError(f"Unknown model: {name}")


def _bootstrap_ci(y_true, y_prob, metric_fn, n_boot: int = N_BOOT,
                  random_state: int = RANDOM_STATE) -> tuple[float, float]:
    """Percentile bootstrap CI, matching the previous pipeline's approach.

    Resamples pooled out-of-fold predictions, so the interval describes
    uncertainty in the estimate rather than dispersion across folds.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], y_prob[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _mean(values) -> float:
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(a.mean()) if a.size else float("nan")


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = RANDOM_STATE,
    n_trials: int = 40,
    models: list[str] | None = None,
) -> dict:
    """Nested CV with Optuna TPE tuning in the inner loop.

    Returns {model: {metrics, per-fold params, out-of-fold predictions}}.
    """
    from optuna_tuning import tune_on_fold

    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    outer_splits = max(2, min(outer_splits, n_pos, n_neg))
    inner_splits = max(2, min(inner_splits, outer_splits))
    logger.info(
        "Nested CV: n=%d (%d pos / %d neg), %d participants, outer=%d inner=%d, trials=%d",
        len(y), n_pos, n_neg, groups.nunique(), outer_splits, inner_splits, n_trials,
    )

    outer = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    folds = list(outer.split(X, y, groups))

    results = {}
    for name in (models or MODEL_NAMES):
        try:
            build_model(name, {}, y, random_state)
        except ImportError:
            logger.warning("%s unavailable - skipping.", name)
            continue

        oof_prob = np.full(len(y), np.nan)
        oof_fold = np.full(len(y), -1)
        per_fold, best_params = [], []

        for fold, (tr, te) in enumerate(folds, start=1):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y.iloc[tr], y.iloc[te]

            model, chosen, _ = tune_on_fold(
                X_tr, y_tr, groups.iloc[tr], name,
                n_trials=n_trials, inner_splits=inner_splits, random_state=random_state,
            )
            prob = model.predict_proba(X_te)[:, 1]
            oof_prob[te] = prob
            oof_fold[te] = fold
            best_params.append(chosen)

            pred = (prob >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
            n_sel = None
            if "select" in model.named_steps:
                n_sel = int(model.named_steps["select"].get_support().sum())
            elif hasattr(model.named_steps["clf"], "coef_"):
                n_sel = int((model.named_steps["clf"].coef_.ravel() != 0).sum())

            per_fold.append({
                "fold": fold,
                "pr_auc": average_precision_score(y_te, prob),
                "roc_auc": roc_auc_score(y_te, prob) if y_te.nunique() > 1 else np.nan,
                "accuracy": accuracy_score(y_te, pred),
                "f1": f1_score(y_te, pred, zero_division=0),
                "precision": precision_score(y_te, pred, zero_division=0),
                "recall": recall_score(y_te, pred, zero_division=0),
                "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
                "brier": brier_score_loss(y_te, prob),
                "n_features_selected": n_sel,
            })
            logger.info("  %-32s fold %d/%d: PR-AUC=%.3f feats=%s",
                        name, fold, len(folds), per_fold[-1]["pr_auc"], n_sel)

        fd = pd.DataFrame(per_fold)
        pr_lo, pr_hi = _bootstrap_ci(y, oof_prob, average_precision_score, random_state=random_state)
        roc_lo, roc_hi = _bootstrap_ci(y, oof_prob, roc_auc_score, random_state=random_state)

        results[name] = {
            "pr_auc_mean": _mean(fd.pr_auc),
            "pr_auc_oof": float(average_precision_score(y, oof_prob)),
            "pr_auc_ci95": [pr_lo, pr_hi],
            "roc_auc_mean": _mean(fd.roc_auc),
            "roc_auc_oof": float(roc_auc_score(y, oof_prob)),
            "roc_auc_ci95": [roc_lo, roc_hi],
            "accuracy_mean": _mean(fd.accuracy),
            "f1_mean": _mean(fd.f1),
            "precision_mean": _mean(fd.precision),
            "recall_mean": _mean(fd.recall),
            "specificity_mean": _mean(fd.specificity),
            "brier_mean": _mean(fd.brier),
            "n_features_mean": _mean(fd.n_features_selected),
            "per_fold": per_fold,
            "best_params_per_fold": best_params,
            "oof_prob": oof_prob.tolist(),
            "oof_fold": oof_fold.tolist(),
            "prevalence": n_pos / (n_pos + n_neg),
        }
        logger.info(
            "  %-32s PR-AUC=%.3f (OOF %.3f [%.3f-%.3f])  ROC=%.3f  Brier=%.3f  baseline=%.3f",
            name, results[name]["pr_auc_mean"], results[name]["pr_auc_oof"],
            pr_lo, pr_hi, results[name]["roc_auc_mean"],
            results[name]["brier_mean"], results[name]["prevalence"],
        )
    return results
