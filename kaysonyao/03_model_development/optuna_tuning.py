"""Optuna (TPE) hyperparameter search for the inner loop of the nested CV.

Search spaces mirror the previous pipeline's `_build_model_from_trial` so
results stay comparable; the only deliberate addition is `l1_ratio` for the
logistic regression, which the old code pinned to 0 (i.e. ridge, not elastic
net).

Placement is the point of the structure: the study runs **inside** each outer
fold and is scored only on inner validation folds, so the outer fold is never
seen during tuning.

No pruner is used, matching the previous pipeline - every trial runs to
completion, so the TPE sampler sees an unbiased set of results.
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from nested_cv import EN_LOGREG, RANDOM_STATE, build_model

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_params(trial, model: str) -> dict:
    """Search space per model. Ranges match the previous pipeline."""
    if model == EN_LOGREG:
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            # New: the old pipeline hard-coded l1_ratio=0, which is ridge.
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }

    if model == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }

    if model == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
        }

    if model == "SVM":
        return {
            "C": trial.suggest_float("C", 1e-2, 100.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }

    raise ValueError(f"Unknown model: {model}")


def tune_on_fold(
    X, y, groups,
    model: str,
    n_trials: int = 40,
    inner_splits: int = 3,
    random_state: int = RANDOM_STATE,
):
    """Run a TPE study on one outer-training fold.

    Returns (pipeline refit on the whole fold, best params, best inner score).
    `X`, `y`, `groups` must be the *training* half of an outer fold only.
    """
    inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    splits = list(inner.split(X, y, groups))

    def objective(trial):
        params = suggest_params(trial, model)
        scores = []
        for tr, va in splits:
            pipe = build_model(model, params, y.iloc[tr], random_state)
            try:
                pipe.fit(X.iloc[tr], y.iloc[tr])
                prob = pipe.predict_proba(X.iloc[va])[:, 1]
            except Exception:
                return float("-inf")
            scores.append(average_precision_score(y.iloc[va], prob))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    pipe = build_model(model, best.params, y, random_state)
    pipe.fit(X, y)
    return pipe, dict(best.params), float(best.value)
