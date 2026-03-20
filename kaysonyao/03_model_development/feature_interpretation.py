"""
Feature interpretation (Gini / SHAP / LIME) for the best binary model per timepoint.

Features used: union of LASSO-selected features across ALL timepoints + placenta
(superset), so each timepoint's model is re-fitted on the full shared feature set.
This lets you compare the importance trajectory of any analyte across timepoints,
even when it was only selected by LASSO at one of them.

Each output CSV is annotated with log2_fold_change and direction (up/down in
complications vs controls) pulled from the differential analysis results.

Artifacts needed per timepoint (produced by binary_classifier.py):
    summary.json                 -> data_path, complications_pooled
    sample_splits.csv            -> exact train/val/test SampleIDs
    tuned_hyperparams.json       -> best model hyperparameters
    lasso_selected_features.csv  -> contributes to the cross-timepoint superset

Usage:
    python 03_model_development/feature_interpretation.py [--binary-results-dir DIR]
        [--timepoints A B C] [--sig-analytes-dir DIR]
        [--shap-bg-start 100] [--shap-bg-max 1000]
"""

import json
import os
import sys
import logging
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import RobustScaler
import shap
from lime.lime_tabular import LimeTabularExplainer

sys.path.insert(0, os.path.dirname(__file__))
from utilities import (
    TIMEPOINTS,
    RANDOM_STATE,
    load_data,
    get_analyte_columns,
    normalise_group_labels,
    build_tuned_model_binary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Superset helpers
# ---------------------------------------------------------------------------

def collect_superset_features(base_dir: str, timepoints: list) -> list:
    """Union of LASSO-selected features across all plasma timepoints and placenta."""
    union = set()
    for tp in timepoints:
        p = os.path.join(base_dir, "plasma", tp, "lasso_selected_features.csv")
        if os.path.exists(p):
            union.update(pd.read_csv(p)["feature"].tolist())
    p = os.path.join(base_dir, "placenta", "all", "lasso_selected_features.csv")
    if os.path.exists(p):
        union.update(pd.read_csv(p)["feature"].tolist())
    return sorted(union)


def _build_superset_data(results_dir: str, data_csv: str,
                         superset_features: list, complications: list):
    """
    Reload raw data, intersect with superset features, and reconstruct
    the exact train+val / test split recorded in sample_splits.csv.

    Returns (X_trainval, X_test, y_trainval, y_test, feat_cols).
    """
    df = load_data(data_csv)
    df = normalise_group_labels(df)

    keep = set(["Control"] + complications)
    mask = df["Group"].isin(keep)
    sub  = df.loc[mask, "Group"]
    y_all = sub.isin(complications).astype(int)

    available  = set(get_analyte_columns(df))
    feat_cols  = [f for f in superset_features if f in available]
    n_missing  = len(superset_features) - len(feat_cols)
    if n_missing:
        logger.warning("  %d / %d superset features absent from this dataset - using %d",
                       n_missing, len(superset_features), len(feat_cols))

    X_all = df.loc[y_all.index, feat_cols]

    splits  = pd.read_csv(os.path.join(results_dir, "sample_splits.csv"))
    tv_ids  = splits.loc[splits["split"].isin(["train", "val"]), "SampleID"].tolist()
    te_ids  = splits.loc[splits["split"] == "test",              "SampleID"].tolist()

    # Guard against samples that were in the split but might not be in this data slice
    tv_ids = [i for i in tv_ids if i in X_all.index]
    te_ids = [i for i in te_ids if i in X_all.index]

    return (X_all.loc[tv_ids], X_all.loc[te_ids],
            y_all.loc[tv_ids], y_all.loc[te_ids],
            feat_cols)


# ---------------------------------------------------------------------------
# Fold-change annotation
# ---------------------------------------------------------------------------

_FC_COLS = ("log2_fold_change", "logFC", "log2FC", "log2FoldChange",
            "log_fold_change", "fold_change")

def load_fold_changes(diff_results_csv: str) -> dict:
    """Return {analyte: log2_fc} from a differential-results CSV. Returns {} on failure."""
    if not diff_results_csv or not os.path.exists(diff_results_csv):
        return {}
    df = pd.read_csv(diff_results_csv, index_col=0)
    for col in _FC_COLS:
        if col in df.columns:
            return df[col].dropna().to_dict()
    logger.warning("No fold-change column found in %s (tried: %s)",
                   diff_results_csv, _FC_COLS)
    return {}


def annotate_with_fc(series: pd.Series, fold_changes: dict) -> pd.DataFrame:
    """
    Attach log2_fold_change and direction (up / down / unknown) to an
    importance Series. 'up' means higher in complications than controls.
    """
    df = series.rename("importance").to_frame()
    df["log2_fold_change"] = df.index.map(fold_changes)
    df["direction"] = df["log2_fold_change"].apply(
        lambda v: "up" if (pd.notna(v) and v > 0)
                  else "down" if (pd.notna(v) and v < 0)
                  else "unknown"
    )
    return df


# ---------------------------------------------------------------------------
# 1. Gini / coefficient importance
# ---------------------------------------------------------------------------

def compute_gini_importance(model, feature_names: list) -> pd.Series:
    if hasattr(model, "coef_"):
        vals = model.coef_.ravel()
    elif hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    else:
        return pd.Series(dtype=float)
    return pd.Series(vals, index=feature_names, name="importance")


def plot_gini(importance: pd.Series, title: str, output_path: str, top_n: int = 30):
    idx  = importance.abs().nlargest(top_n).index
    vals = importance[idx][::-1]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
    fig, ax = plt.subplots(figsize=(8, max(4, len(vals) * 0.30)))
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(vals.index, fontsize=8)
    ax.set_xlabel("Importance / Coefficient")
    ax.set_title(title, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. SHAP
# ---------------------------------------------------------------------------

def _extract_shap_matrix(shap_result, n_features: int) -> np.ndarray:
    """Normalise any SHAP output format to (n_samples, n_features)."""
    vals = shap_result.values if hasattr(shap_result, "values") else shap_result
    if isinstance(vals, list):          # KernelExplainer -> [class0, class1]
        vals = vals[1] if len(vals) == 2 else vals[0]
    vals = np.array(vals)
    if vals.ndim == 3 and vals.shape[2] == 2:   # TreeExplainer binary (n, f, 2)
        vals = vals[:, :, 1]
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    assert vals.shape[1] == n_features
    return vals


def run_shap(
    model,
    model_name: str,
    X_trainval_sc: np.ndarray,
    X_test_sc: np.ndarray,
    feature_names: list,
    output_dir: str,
    fold_changes: dict,
    bg_start: int = 100,
    bg_max: int = 1000,
) -> pd.Series | None:
    os.makedirs(output_dir, exist_ok=True)
    n_features = len(feature_names)

    def _background(n: int) -> np.ndarray:
        n   = min(n, len(X_trainval_sc))
        idx = np.random.default_rng(RANDOM_STATE).choice(len(X_trainval_sc), n, replace=False)
        return X_trainval_sc[idx]

    def _build_explainer(bg: np.ndarray):
        if model_name in ("RandomForest", "XGBoost"):
            return shap.TreeExplainer(model), "TreeExplainer"
        elif model_name == "LogisticRegression":
            return shap.LinearExplainer(model, shap.maskers.Independent(bg)), "LinearExplainer"
        else:
            return shap.KernelExplainer(model.predict_proba, bg), "KernelExplainer"

    def _compute(n_bg: int) -> np.ndarray:
        bg = _background(n_bg)
        explainer, etype = _build_explainer(bg)
        logger.info("  SHAP: %s, background n=%d", etype, n_bg)
        X_df = pd.DataFrame(X_test_sc, columns=feature_names)
        if etype == "KernelExplainer":
            result = explainer.shap_values(X_test_sc, nsamples=n_bg,
                                           l1_reg="num_features(10)")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = explainer(X_df)
        return _extract_shap_matrix(result, n_features)

    # Try bg_max first; fall back to bg_start if it fails (single computation)
    shap_matrix, final_bg = None, bg_start
    for n_bg in sorted({bg_start, bg_max}, reverse=True):
        try:
            shap_matrix = _compute(n_bg)
            final_bg    = n_bg
            logger.info("  SHAP: success at n=%d", n_bg)
            break
        except Exception as exc:
            logger.warning("  SHAP: failed at n=%d - %s", n_bg, exc)

    if shap_matrix is None:
        logger.error("  SHAP: all attempts failed - skipping.")
        return None

    mean_abs = pd.Series(np.abs(shap_matrix).mean(axis=0),
                         index=feature_names, name="mean_abs_shap"
                         ).sort_values(ascending=False)

    pd.DataFrame(shap_matrix, columns=feature_names).to_csv(
        os.path.join(output_dir, "shap_values.csv"), index=False)
    annotate_with_fc(mean_abs, fold_changes).to_csv(
        os.path.join(output_dir, "shap_mean_abs.csv"), header=True)

    # Bar plot
    top = mean_abs.head(30)
    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.30)))
    ax.barh(range(len(top)), top.values[::-1], color="#ff7f0e")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1], fontsize=8)
    ax.set_xlabel(f"Mean |SHAP|  (background n={final_bg})")
    ax.set_title("SHAP Feature Importance", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Beeswarm
    shap.summary_plot(shap_matrix,
                      pd.DataFrame(X_test_sc, columns=feature_names),
                      max_display=30, show=False, plot_type="dot")
    plt.gcf().savefig(os.path.join(output_dir, "shap_beeswarm.png"),
                      dpi=150, bbox_inches="tight")
    plt.close()

    return mean_abs


# ---------------------------------------------------------------------------
# 3. LIME - num_samples stability tuning
# ---------------------------------------------------------------------------

def _lime_explain(lime_exp, predict_fn, inst: np.ndarray,
                  feat_idx: dict, n_features: int, ns: int) -> np.ndarray:
    """Explain one instance. feat_idx and n_features are precomputed by the caller."""
    exp = lime_exp.explain_instance(inst, predict_fn,
                                    num_features=n_features, num_samples=ns, labels=(1,))
    vec = np.zeros(n_features)
    for fname, w in exp.as_list(label=1):
        if fname in feat_idx:
            vec[feat_idx[fname]] = w
    return vec


def _lime_tune(lime_exp, predict_fn, X_test_sc: np.ndarray,
               feat_idx: dict, n_features: int,
               candidates: list) -> tuple:
    """Tune num_samples via Spearman rho stability. Returns (chosen_ns, probe_idx, refs, rhos)."""
    probe_idx = np.linspace(0, len(X_test_sc) - 1,
                            min(3, len(X_test_sc)), dtype=int)
    ns_max = max(candidates)
    refs   = [_lime_explain(lime_exp, predict_fn, X_test_sc[i],
                            feat_idx, n_features, ns_max) for i in probe_idx]
    rhos   = []
    for ns in sorted(candidates):
        corrs = []
        for i, ref in zip(probe_idx, refs):
            w = _lime_explain(lime_exp, predict_fn, X_test_sc[i], feat_idx, n_features, ns)
            r, _ = spearmanr(np.abs(w), np.abs(ref))
            if not np.isnan(r):
                corrs.append(r)
        mean_r = np.mean(corrs) if corrs else 0.0
        rhos.append(mean_r)
        logger.info("  LIME stability: num_samples=%5d  rho=%.3f", ns, mean_r)
        if mean_r >= 0.95:
            logger.info("  LIME: chosen num_samples=%d", ns)
            return ns, probe_idx, refs, rhos

    logger.warning("  LIME: no candidate reached rho>=0.95 - using max=%d", ns_max)
    return ns_max, probe_idx, refs, rhos


def _plot_stability_curve(candidates, rhos, chosen_ns, output_path):
    """Plot LIME stability curve from precomputed rhos (no LIME re-runs)."""
    # rhos may be shorter than candidates when _lime_tune exits early on success
    evaluated = sorted(candidates)[:len(rhos)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(evaluated, rhos, marker="o", color="#2ca02c")
    ax.axhline(0.95, color="gray", linestyle="--", lw=0.8, label="rho = 0.95")
    ax.axvline(chosen_ns, color="red", linestyle="--", lw=0.8,
               label=f"chosen = {chosen_ns}")
    ax.set_xlabel("num_samples")
    ax.set_ylabel("Mean Spearman rho")
    ax.set_title("LIME stability across num_samples", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_lime(
    model,
    X_trainval_sc: np.ndarray,
    X_test_sc: np.ndarray,
    feature_names: list,
    output_dir: str,
    fold_changes: dict,
    candidates: list | None = None,
) -> pd.Series | None:
    if candidates is None:
        candidates = [100, 500, 1000, 2000, 5000]

    os.makedirs(output_dir, exist_ok=True)
    n        = len(feature_names)
    feat_idx = {f: i for i, f in enumerate(feature_names)}   # precomputed once

    lime_exp   = LimeTabularExplainer(
        training_data=X_trainval_sc,
        feature_names=feature_names,
        class_names=["Control", "Complication"],
        mode="classification",
        random_state=RANDOM_STATE,
        discretize_continuous=False,
    )
    predict_fn = model.predict_proba

    tuned_ns, _, _, rhos = _lime_tune(
        lime_exp, predict_fn, X_test_sc, feat_idx, n, candidates
    )

    logger.info("  LIME: explaining %d test samples (num_samples=%d) ...",
                len(X_test_sc), tuned_ns)
    weight_matrix = np.zeros((len(X_test_sc), n))
    for i, inst in enumerate(X_test_sc):
        try:
            weight_matrix[i] = _lime_explain(lime_exp, predict_fn,
                                             inst, feat_idx, n, tuned_ns)
        except Exception as exc:
            logger.warning("  LIME: sample %d failed - %s", i, exc)

    mean_abs = pd.Series(np.abs(weight_matrix).mean(axis=0),
                         index=feature_names, name="mean_abs_lime"
                         ).sort_values(ascending=False)

    pd.DataFrame(weight_matrix, columns=feature_names).to_csv(
        os.path.join(output_dir, "lime_weights.csv"), index=False)
    annotate_with_fc(mean_abs, fold_changes).to_csv(
        os.path.join(output_dir, "lime_mean_abs.csv"), header=True)

    top = mean_abs.head(30)
    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.30)))
    ax.barh(range(len(top)), top.values[::-1], color="#2ca02c")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1], fontsize=8)
    ax.set_xlabel(f"Mean |LIME weight|  (num_samples={tuned_ns})")
    ax.set_title("LIME Feature Importance", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "lime_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    _plot_stability_curve(candidates, rhos, tuned_ns,
                          output_path=os.path.join(output_dir, "lime_stability.png"))
    return mean_abs


# ---------------------------------------------------------------------------
# Combined panel plot
# ---------------------------------------------------------------------------

def plot_combined(gini, shap_vals, lime_vals, title, output_path, top_n=20):
    panels  = [(s, l, c) for s, l, c in [
        (gini,      "Gini/Coef", "#1f77b4"),
        (shap_vals, "SHAP",      "#ff7f0e"),
        (lime_vals, "LIME",      "#2ca02c"),
    ] if s is not None and not s.empty]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), max(5, top_n * 0.30)))
    if len(panels) == 1:
        axes = [axes]
    for ax, (series, lbl, col) in zip(axes, panels):
        s = series.abs().nlargest(top_n).sort_values()
        ax.barh(range(len(s)), s.values, color=col)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s.index, fontsize=8)
        ax.set_xlabel("Score")
        ax.set_title(lbl, fontsize=10)
    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-(tissue, timepoint) driver
# ---------------------------------------------------------------------------

def interpret_one(tissue, timepoint, results_dir, output_dir,
                  best_model_name, bg_start, bg_max,
                  superset_features, data_csv, complications, fold_changes):
    tag = f"[{tissue.upper()} {timepoint}]"
    logger.info("%s Best model: %s | superset features: %d",
                tag, best_model_name, len(superset_features))

    if not data_csv or not os.path.exists(data_csv):
        logger.error("%s data_csv not found (%s) - re-run binary_classifier.py.", tag, data_csv)
        return

    # Reconstruct superset data using the original split indices
    try:
        X_tv_raw, X_te_raw, y_trainval, y_test, feat_cols = _build_superset_data(
            results_dir, data_csv, superset_features, complications
        )
    except Exception as exc:
        logger.error("%s Failed to build superset data - %s", tag, exc)
        return

    logger.info("%s  trainval=%d  test=%d  features=%d",
                tag, len(X_tv_raw), len(X_te_raw), len(feat_cols))

    # Scale on superset trainval
    scaler        = RobustScaler()
    X_trainval_sc = scaler.fit_transform(X_tv_raw)
    X_test_sc     = scaler.transform(X_te_raw)

    # Rebuild and retrain best model on superset with its original tuned params
    hp_path = os.path.join(results_dir, "tuned_hyperparams.json")
    if not os.path.exists(hp_path):
        logger.error("%s tuned_hyperparams.json missing - skipping.", tag)
        return
    with open(hp_path) as f:
        hp = json.load(f)
    params = hp["model_params"][best_model_name]
    model  = build_tuned_model_binary(best_model_name, params, y_trainval, RANDOM_STATE)
    model.fit(X_trainval_sc, y_trainval)
    logger.info("%s  Model retrained on superset.", tag)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Gini
    logger.info("%s  Gini ...", tag)
    gini = compute_gini_importance(model, feat_cols)
    if not gini.empty:
        annotate_with_fc(
            gini.sort_values(key=lambda s: s.abs(), ascending=False),
            fold_changes,
        ).to_csv(os.path.join(output_dir, "gini_importance.csv"), header=True)
        plot_gini(gini,
                  title=f"Gini/Coef - {best_model_name} | {tissue} {timepoint}",
                  output_path=os.path.join(output_dir, "gini_importance.png"))
    else:
        logger.warning("%s  Model has no Gini/coef_ attribute.", tag)

    # 2. SHAP
    logger.info("%s  SHAP (bg_start=%d, bg_max=%d) ...", tag, bg_start, bg_max)
    shap_result = run_shap(model, best_model_name,
                           X_trainval_sc, X_test_sc, feat_cols,
                           output_dir, fold_changes,
                           bg_start=bg_start, bg_max=bg_max)

    # 3. LIME
    logger.info("%s  LIME (stability tuning) ...", tag)
    lime_result = run_lime(model, X_trainval_sc, X_test_sc,
                           feat_cols, output_dir, fold_changes)

    # Combined panel
    plot_combined(gini if not gini.empty else None, shap_result, lime_result,
                  title=f"{best_model_name} | {tissue} {timepoint}",
                  output_path=os.path.join(output_dir, "combined_importance.png"))

    logger.info("%s  Done -> %s", tag, output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    wkdir = os.getcwd()
    p = argparse.ArgumentParser(
        description="Feature interpretation (Gini/SHAP/LIME) for best binary model."
    )
    p.add_argument(
        "--binary-results-dir",
        default=os.path.join(wkdir, "04_results_and_figures", "models", "binary"),
    )
    p.add_argument("--timepoints",          nargs="+", default=TIMEPOINTS)
    p.add_argument(
        "--superset-timepoints",
        nargs="+",
        default=["A", "B", "C", "D"],
        help=(
            "Plasma timepoints whose LASSO-selected features contribute to the "
            "cross-timepoint superset.  Placenta is always included.  Defaults to "
            "A B C D (E excluded because its LASSO regularisation was too weak, "
            "selecting ~1600 features)."
        ),
    )
    p.add_argument("--shap-bg-start",  type=int, default=100)
    p.add_argument("--shap-bg-max",    type=int, default=1000)
    p.add_argument("--skip-plasma",    action="store_true")
    p.add_argument("--skip-placenta",  action="store_true")
    p.add_argument(
        "--sig-analytes-dir",
        default=os.path.join(wkdir, "04_results_and_figures", "differential_analysis"),
        help=(
            "Root directory of differential analysis outputs used to pull "
            "fold-change values for output annotation. Pass '' to skip."
        ),
    )
    args = p.parse_args()

    base     = args.binary_results_dir
    sig_dir  = args.sig_analytes_dir if args.sig_analytes_dir else None

    summary_csv = os.path.join(base, "all_results_summary.csv")
    if not os.path.exists(summary_csv):
        logger.error("all_results_summary.csv not found at %s", summary_csv)
        sys.exit(1)

    summary = pd.read_csv(summary_csv)
    best = (summary.loc[summary.groupby(["tissue", "timepoint"])["pr_auc"].idxmax()]
            .set_index(["tissue", "timepoint"])["model"].to_dict())

    logger.info("Best models by test PR-AUC:")
    for (t, tp), m in best.items():
        pr = summary.loc[(summary.tissue == t) & (summary.timepoint == tp)
                         & (summary.model == m), "pr_auc"].values[0]
        logger.info("  %-10s %-5s -> %-20s (%.4f)", t, tp, m, pr)

    # Build superset from selected timepoints only (plasma + placenta)
    superset = collect_superset_features(base, args.superset_timepoints)
    logger.info("Cross-timepoint superset (%s + placenta): %d features",
                " ".join(args.superset_timepoints), len(superset))
    if not superset:
        logger.error("No lasso_selected_features.csv files found - run binary_classifier.py first.")
        sys.exit(1)

    def _summary_json(tissue, tp):
        path = os.path.join(base, tissue, tp, "summary.json")
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    def _fold_changes(tissue, tp):
        if not sig_dir:
            return {}
        if tissue == "plasma":
            diff_csv = os.path.join(sig_dir, "plasma", "cross_sectional", tp,
                                    "Control_vs_Complication_differential_results.csv")
        else:
            diff_csv = os.path.join(sig_dir, "placenta", "cross_sectional",
                                    "Control_vs_Complication_differential_results.csv")
        fc = load_fold_changes(diff_csv)
        if fc:
            logger.info("  Fold changes loaded: %d analytes from %s", len(fc), diff_csv)
        else:
            logger.warning("  No fold changes available for %s %s", tissue, tp)
        return fc

    if not args.skip_plasma:
        logger.info("=== PLASMA ===")
        for tp in args.timepoints:
            bm = best.get(("plasma", tp))
            if bm is None:
                logger.warning("Plasma %s not in summary - skipping.", tp)
                continue
            summ         = _summary_json("plasma", tp)
            data_csv     = summ.get("data_path", "")
            complications = summ.get("complications_pooled", ["HDP", "FGR", "sPTB"])
            fold_changes  = _fold_changes("plasma", tp)
            interpret_one(
                tissue="plasma", timepoint=tp,
                results_dir=os.path.join(base, "plasma", tp),
                output_dir=os.path.join(base, "plasma", tp, "interpretation"),
                best_model_name=bm,
                bg_start=args.shap_bg_start,
                bg_max=args.shap_bg_max,
                superset_features=superset,
                data_csv=data_csv,
                complications=complications,
                fold_changes=fold_changes,
            )

    if not args.skip_placenta:
        logger.info("=== PLACENTA ===")
        bm = best.get(("placenta", "all"))
        if bm is None:
            logger.warning("Placenta not in summary - skipping.")
        else:
            summ         = _summary_json("placenta", "all")
            data_csv     = summ.get("data_path", "")
            complications = summ.get("complications_pooled", ["HDP", "FGR", "sPTB"])
            fold_changes  = _fold_changes("placenta", "all")
            interpret_one(
                tissue="placenta", timepoint="all",
                results_dir=os.path.join(base, "placenta", "all"),
                output_dir=os.path.join(base, "placenta", "all", "interpretation"),
                best_model_name=bm,
                bg_start=args.shap_bg_start,
                bg_max=args.shap_bg_max,
                superset_features=superset,
                data_csv=data_csv,
                complications=complications,
                fold_changes=fold_changes,
            )


if __name__ == "__main__":
    main()
