"""
Shared utilities for the DP3 proteomics modeling pipeline.

Provides data loading, feature selection (LASSO), train/val/test splitting,
cross-validation helpers, and evaluation metrics used by:
 - binary_classifier.py   (per-outcome binary models)
 - multilabel_classifier.py  (joint multi-label models)

Run all scripts from the project root so that os.getcwd() resolves correctly.
"""

# Standard library
import logging
import os

# Third-party
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV, MultiTaskElasticNetCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirror 02_exploratory_analysis/utilities.py)
# ---------------------------------------------------------------------------

_METADATA_COLS = [
    "SubjectID", "Group", "Subgroup", "Batch", "GestAgeDelivery", "SampleGestAge"
]
_GROUP_LABEL_MAP = {"sptb": "sPTB"}

OUTCOMES   = ["HDP", "FGR", "sPTB"]
TIMEPOINTS = ["A", "B", "C", "D", "E"]

N_CV_FOLDS = 10
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load cleaned wide-format CSV (index=SampleID, columns=metadata+analytes)."""
    return pd.read_csv(path, index_col=0)

def load_significant_analytes(
    diff_results_csv: str,
    q_threshold: float = 0.05,
) -> list | None:
    """
    Load significant analyte IDs from a differential-analysis results CSV.

    Reads the full ``_differential_results.csv`` (all tested analytes with
    q-values) and returns those whose ``q_value`` is below *q_threshold*.
    The fold-change criterion used when generating ``_significant_analytes.csv``
    is intentionally omitted here; the downstream elastic-net regularisation
    handles effect-size filtering.

    Parameters
   ----------
    diff_results_csv : path to ``<comparison>_differential_results.csv``
    q_threshold      : FDR q-value cut-off (default 0.05)

    Returns
    -------
    list of str  - analyte names, or None if the file is missing / no analytes
                   survive the threshold.
    """
    if not os.path.exists(diff_results_csv):
        return None
    df = pd.read_csv(diff_results_csv, index_col=0)
    if df.empty or "q_value" not in df.columns:
        return None
    tested = df[df.get("excluded", pd.Series(False, index=df.index)) == False]
    sig = tested[tested["q_value"] < q_threshold]
    analytes = sig.index.dropna().tolist()
    return analytes if analytes else None

def get_analyte_columns(df: pd.DataFrame) -> list:
    """Return analyte (feature) column names by excluding metadata columns."""
    return [c for c in df.columns if c not in _METADATA_COLS]

def normalise_group_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise Group column capitalisation (e.g. 'sptb' -> 'sPTB')."""
    df = df.copy()
    if "Group" in df.columns:
        df["Group"] = df["Group"].replace(_GROUP_LABEL_MAP)
    return df

# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def split_70_15_15(
    X: pd.DataFrame,
    y: pd.Series,
    stratify: bool = True,
    random_state: int = RANDOM_STATE,
):
    """
    Stratified 70 / 15 / 15 split.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split

    strat = y if stratify else None

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=strat
    )
    # 15/30 = 0.50 of the remaining 30 % -> each 15 %
    strat_tmp = y_tmp if stratify else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=random_state, stratify=strat_tmp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def plot_correlation_matrix(
    X: pd.DataFrame,
    output_path: str,
    title: str = "Feature correlation matrix",
    max_features: int = 100,
) -> None:
    """
    Compute and save a Pearson correlation heatmap of the feature matrix.

    If X has more than *max_features* columns only the first *max_features*
    columns are plotted (post-LASSO selection this is rarely an issue).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X_plot = X.iloc[:, :max_features]
    corr = X_plot.corr(method="pearson")

    fig_size = max(10, len(corr) * 0.18)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=True,
        yticklabels=True,
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=12, pad=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Correlation matrix saved -> %s", output_path)

# ---------------------------------------------------------------------------
# LASSO feature selection
# ---------------------------------------------------------------------------

def lasso_feature_selection_binary(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = N_CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> list:
    """
    Elastic-net logistic regression CV (l1_ratios 0.1..1.0) for binary feature selection.
    Returns column names with non-zero coefficients at the CV-chosen (ratio, C).
    """
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)

    vc = pd.Series(y_train).value_counts()
    if len(vc) < 2:
        logger.warning(
            "ElasticNet binary: training set has only one class (%s) - returning no features.",
            vc.index[0],
        )
        return []

    # Cap folds to smallest class size - avoids ValueError when n < n_splits
    min_class = int(vc.min())
    cv_actual = max(2, min(cv, min_class))
    if cv_actual < cv:
        logger.warning(
            "ElasticNet binary: capping CV folds %d -> %d (small class).", cv, cv_actual
        )

    enet = LogisticRegressionCV(
        l1_ratios=(0.1, 0.5, 0.7, 0.9, 1.0),
        solver="saga",
        cv=cv_actual,
        class_weight="balanced",
        random_state=random_state,
        max_iter=20000,
        tol=1e-3,
        n_jobs=-1,
    )
    enet.fit(X_scaled, y_train)

    chosen_ratio = enet.l1_ratio_[0] if hasattr(enet.l1_ratio_, "__len__") else enet.l1_ratio_
    coef = enet.coef_.ravel()
    selected = [col for col, c in zip(X_train.columns, coef) if c != 0.0]
    logger.info(
        "ElasticNet binary: %d / %d features selected (best l1_ratio=%.2f).",
        len(selected), X_train.shape[1], chosen_ratio,
    )
    return selected

def lasso_feature_selection_multilabel(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    cv: int = N_CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> list:
    """
    Multi-task elastic-net CV to select a shared feature set for all outcomes jointly.

    MultiTaskElasticNetCV searches over an l1_ratio grid [0.1, 0.5, 0.7, 0.9, 1.0],
    mixing L1 (group sparsity across outcomes) with L2 (retains correlated features).
    A feature is kept if its coefficient is non-zero for ANY outcome.

    Returns
    -------
    selected_features : list of str
    """
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)

    enet = MultiTaskElasticNetCV(
        l1_ratio=(0.1, 0.5, 0.7, 0.9, 1.0),
        cv=cv,
        random_state=random_state,
        max_iter=10000,
        n_jobs=-1,
    )
    enet.fit(X_scaled, Y_train.values)

    coef = enet.coef_   # (n_outcomes, n_features)
    mask = np.any(coef != 0.0, axis=0)
    selected = [col for col, m in zip(X_train.columns, mask) if m]
    logger.info(
        "ElasticNet multi-task: %d / %d features selected (best l1_ratio=%.2f).",
        len(selected), X_train.shape[1], enet.l1_ratio_,
    )
    return selected

# ---------------------------------------------------------------------------
# Base models
# ---------------------------------------------------------------------------

def get_base_models_binary(random_state: int = RANDOM_STATE) -> dict:
    """
    Return a dict of {name: estimator} for binary classification.

    All sklearn models use class_weight='balanced'.
    XGBoost scale_pos_weight is set dynamically per dataset (see binary_classifier.py).
    """
    return {
        "LogisticRegression": LogisticRegression(
            l1_ratio=0,            # pure L2; replaces deprecated penalty='l2'
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        ),
        "SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        ),
    }

def get_base_models_multilabel(random_state: int = RANDOM_STATE) -> dict:
    """
    Return base estimators for multi-label classification (wrapped in MultiOutputClassifier).
    """
    return {
        "LogisticRegression": LogisticRegression(
            l1_ratio=0,            # pure L2; replaces deprecated penalty='l2'
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        ),
        "SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        ),
    }

# ---------------------------------------------------------------------------
# Optuna TPE hyperparameter tuning
# ---------------------------------------------------------------------------

def _build_model_from_trial_binary(trial, model_name: str, y_train, random_state: int):
    """Instantiate a model from an Optuna trial for binary classification."""
    if model_name == "LogisticRegression":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 100.0, log=True),
            l1_ratio=0, solver="lbfgs", class_weight="balanced",
            max_iter=5000, random_state=random_state,
        )
    elif model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
    elif model_name == "XGBoost":
        spw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
            scale_pos_weight=spw,
            eval_metric="logloss", random_state=random_state, n_jobs=-1, verbosity=0,
        )
    elif model_name == "SVM":
        return SVC(
            C=trial.suggest_float("C", 1e-2, 100.0, log=True),
            gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
            kernel="rbf", class_weight="balanced",
            probability=True, random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def _build_model_from_trial_multilabel(trial, model_name: str, random_state: int):
    """Instantiate a base model from an Optuna trial for multi-label classification.

    No scale_pos_weight for XGBoost - each output head is fitted independently
    by MultiOutputClassifier so class imbalance is handled per-outcome at fit time.
    """
    if model_name == "LogisticRegression":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 100.0, log=True),
            l1_ratio=0, solver="lbfgs", class_weight="balanced",
            max_iter=5000, random_state=random_state,
        )
    elif model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
    elif model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
            eval_metric="logloss", random_state=random_state, n_jobs=-1, verbosity=0,
        )
    elif model_name == "SVM":
        return SVC(
            C=trial.suggest_float("C", 1e-2, 100.0, log=True),
            gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
            kernel="rbf", class_weight="balanced",
            probability=True, random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def tune_hyperparams_binary(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Optuna TPE search for binary classification.

    Trains on X_train, scores PR-AUC on X_val for each trial.
    Scaling is applied inside the objective so the scaler is fit only on X_train.

    Returns
    -------
    best_params : dict
    best_val_pr_auc : float
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna is required for hyperparameter tuning: pip install optuna")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    scaler = RobustScaler()
    X_tr_s  = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    def objective(trial):
        model = _build_model_from_trial_binary(trial, model_name, y_train, random_state)
        model.fit(X_tr_s, y_train)
        y_prob = model.predict_proba(X_val_s)[:, 1]
        return average_precision_score(y_val, y_prob)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "  [Optuna/%s] best val PR-AUC=%.4f  params=%s",
        model_name, study.best_value, study.best_params,
    )
    return study.best_params, float(study.best_value)

def tune_hyperparams_multilabel(
    model_name: str,
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    Y_val: pd.DataFrame,
    n_trials: int = 50,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Optuna TPE search for multi-label classification.

    Objective = macro-average PR-AUC across all outcomes on X_val.

    Returns
    -------
    best_params : dict
    best_val_macro_pr_auc : float
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna is required for hyperparameter tuning: pip install optuna")

    from sklearn.multioutput import MultiOutputClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    scaler = RobustScaler()
    X_tr_s  = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    outcome_names = list(Y_train.columns)

    def objective(trial):
        base = _build_model_from_trial_multilabel(trial, model_name, random_state)
        model = MultiOutputClassifier(base, n_jobs=-1)
        model.fit(X_tr_s, Y_train)
        Y_prob = model.predict_proba(X_val_s)
        scores = []
        for i, _ in enumerate(outcome_names):
            prob = Y_prob[i][:, 1]
            try:
                scores.append(average_precision_score(Y_val.iloc[:, i], prob))
            except ValueError:
                pass   # only one class in val - skip this outcome
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "  [Optuna/%s] best val macro PR-AUC=%.4f  params=%s",
        model_name, study.best_value, study.best_params,
    )
    return study.best_params, float(study.best_value)

def build_tuned_model_binary(
    model_name: str,
    params: dict,
    y_train=None,
    random_state: int = RANDOM_STATE,
):
    """Instantiate a fresh binary model from tuned hyperparameters. y_train sets XGBoost scale_pos_weight."""
    if model_name == "LogisticRegression":
        return LogisticRegression(
            C=params["C"], l1_ratio=0, solver="lbfgs",
            class_weight="balanced", max_iter=5000, random_state=random_state,
        )
    elif model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
    elif model_name == "XGBoost":
        spw = (
            float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
            if y_train is not None else 1.0
        )
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            gamma=params["gamma"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            scale_pos_weight=spw,
            eval_metric="logloss", random_state=random_state, n_jobs=-1, verbosity=0,
        )
    elif model_name == "SVM":
        return SVC(
            C=params["C"], gamma=params["gamma"], kernel="rbf",
            class_weight="balanced", probability=True, random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def build_tuned_model_multilabel(
    model_name: str,
    params: dict,
    random_state: int = RANDOM_STATE,
):
    """
    Instantiate a fresh unfitted MultiOutputClassifier from tuned hyperparameters.
    """
    from sklearn.multioutput import MultiOutputClassifier
    base = build_tuned_model_binary(model_name, params, y_train=None, random_state=random_state)
    return MultiOutputClassifier(base, n_jobs=-1)

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cv_binary(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Stratified k-fold CV for a binary model.

    Returns dict with mean +/- std for PR-AUC, ROC-AUC, F1, Accuracy.
    """
    # Cap folds to smallest class size - avoids ValueError when n < n_splits
    min_class = int(y.value_counts().min())
    n_splits_actual = max(2, min(n_splits, min_class))
    if n_splits_actual < n_splits:
        logger.warning("CV binary: capping folds %d -> %d (small class).", n_splits, n_splits_actual)

    cv = StratifiedKFold(n_splits=n_splits_actual, shuffle=True, random_state=random_state)
    scaler = RobustScaler()

    pr_aucs, roc_aucs, f1s, accs = [], [], [], []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model.fit(X_tr_s, y_tr)
        y_prob = model.predict_proba(X_val_s)[:, 1]
        y_pred = model.predict(X_val_s)

        pr_aucs.append(average_precision_score(y_val, y_prob))
        roc_aucs.append(roc_auc_score(y_val, y_prob))
        f1s.append(f1_score(y_val, y_pred, zero_division=0))
        accs.append(accuracy_score(y_val, y_pred))

    return {
        "pr_auc_mean":  float(np.mean(pr_aucs)),
        "pr_auc_std":   float(np.std(pr_aucs)),
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std":  float(np.std(roc_aucs)),
        "f1_mean":      float(np.mean(f1s)),
        "f1_std":       float(np.std(f1s)),
        "acc_mean":     float(np.mean(accs)),
        "acc_std":      float(np.std(accs)),
    }

def run_cv_multilabel(
    model,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    n_splits: int = N_CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    K-fold CV for a multi-label model.

    Returns dict with mean +/- std PR-AUC and ROC-AUC averaged across outcomes,
    plus per-outcome breakdown.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scaler = RobustScaler()

    outcome_names = list(Y.columns)
    fold_pr  = {o: [] for o in outcome_names}
    fold_roc = {o: [] for o in outcome_names}

    for train_idx, val_idx in cv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        Y_tr, Y_val = Y.iloc[train_idx], Y.iloc[val_idx]

        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model.fit(X_tr_s, Y_tr)
        Y_prob = model.predict_proba(X_val_s)

        for i, outcome in enumerate(outcome_names):
            prob = Y_prob[i][:, 1] if hasattr(Y_prob[i], "shape") else Y_prob[:, i]
            try:
                fold_pr[outcome].append(average_precision_score(Y_val.iloc[:, i], prob))
                fold_roc[outcome].append(roc_auc_score(Y_val.iloc[:, i], prob))
            except ValueError:
                # only one class in fold
                fold_pr[outcome].append(np.nan)
                fold_roc[outcome].append(np.nan)

    results = {}
    all_pr, all_roc = [], []
    for outcome in outcome_names:
        pr_vals  = [v for v in fold_pr[outcome]  if not np.isnan(v)]
        roc_vals = [v for v in fold_roc[outcome] if not np.isnan(v)]
        results[outcome] = {
            "pr_auc_mean":  float(np.mean(pr_vals))  if pr_vals  else np.nan,
            "pr_auc_std":   float(np.std(pr_vals))   if pr_vals  else np.nan,
            "roc_auc_mean": float(np.mean(roc_vals)) if roc_vals else np.nan,
            "roc_auc_std":  float(np.std(roc_vals))  if roc_vals else np.nan,
        }
        all_pr.extend(pr_vals)
        all_roc.extend(roc_vals)

    results["macro_avg"] = {
        "pr_auc_mean":  float(np.nanmean(all_pr)),
        "roc_auc_mean": float(np.nanmean(all_roc)),
    }
    return results

# ---------------------------------------------------------------------------
# Final evaluation on held-out test set
# ---------------------------------------------------------------------------

def evaluate_binary(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Fit on training data and evaluate on test set. Returns metrics dict."""
    scaler = RobustScaler()
    X_tr_s  = scaler.fit_transform(X_train)
    X_te_s  = scaler.transform(X_test)

    model.fit(X_tr_s, y_train)
    y_prob = model.predict_proba(X_te_s)[:, 1]
    y_pred = model.predict(X_te_s)

    return {
        "pr_auc":    float(average_precision_score(y_test, y_prob)),
        "roc_auc":   float(roc_auc_score(y_test, y_prob)),
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
    }, scaler

def evaluate_multilabel(
    model,
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
) -> dict:
    """Fit on training data and evaluate multi-label model on test set."""
    scaler = RobustScaler()
    X_tr_s  = scaler.fit_transform(X_train)
    X_te_s  = scaler.transform(X_test)

    model.fit(X_tr_s, Y_train)
    Y_prob = model.predict_proba(X_te_s)
    Y_pred = model.predict(X_te_s)

    outcome_names = list(Y_test.columns)
    results = {}
    for i, outcome in enumerate(outcome_names):
        prob = Y_prob[i][:, 1] if hasattr(Y_prob[i], "shape") else Y_prob[:, i]
        pred = Y_pred[:, i] if hasattr(Y_pred, "shape") else Y_pred[i]
        results[outcome] = {
            "pr_auc":    float(average_precision_score(Y_test.iloc[:, i], prob)),
            "roc_auc":   float(roc_auc_score(Y_test.iloc[:, i], prob)),
            "accuracy":  float(accuracy_score(Y_test.iloc[:, i], pred)),
            "precision": float(precision_score(Y_test.iloc[:, i], pred, zero_division=0)),
            "recall":    float(recall_score(Y_test.iloc[:, i], pred, zero_division=0)),
            "f1":        float(f1_score(Y_test.iloc[:, i], pred, zero_division=0)),
        }
    return results, scaler

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str,
    output_path: str,
) -> None:
    """Save a precision-recall curve as PNG."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=1.5, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str,
    output_path: str,
) -> None:
    """Save a ROC curve as PNG."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=1.5, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    output_path: str,
    top_n: int = 30,
    title: str = "Top features",
) -> None:
    """Bar plot of top-N feature importances / coefficients."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    idx = np.argsort(np.abs(importances))[::-1][:top_n]
    top_names  = [feature_names[i] for i in idx]
    top_values = importances[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.28)))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top_values]
    ax.barh(range(len(top_names)), top_values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Importance / Coefficient")
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

