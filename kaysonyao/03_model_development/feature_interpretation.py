"""
Feature interpretation for the best binary classifier at each (tissue, timepoint).

Three methods applied to the best model selected by test PR-AUC:

  1. Gini / coefficient importance   – model-native weights (coef_ or
                                       feature_importances_), ranked by |value|.
  2. SHAP (Shapley Additive Explanations)
       - TreeExplainer  for RF / XGBoost (exact, no background required)
       - LinearExplainer for LogisticRegression
       - KernelExplainer for SVM
       Background size: 100 samples first; upgraded to 1000 if no error.
  3. LIME (Local Interpretable Model-agnostic Explanations)
       - num_samples tuned via Spearman stability across [100, 500, 1000, 2000, 5000].
       - Picks smallest num_samples with mean ρ ≥ 0.95 vs the highest setting.

Loads pre-saved artifacts from binary_classifier.py — no model reconstruction:
    {ModelName}.joblib       → fitted model
    scaler.joblib            → fitted RobustScaler (train+val)
    X_trainval_scaled.csv    → scaled training+validation features
    X_test_scaled.csv        → scaled test features
    y_test.csv               → test labels
    lasso_selected_features.csv → feature names
    all_results_summary.csv  → best model per (tissue, timepoint) by test PR-AUC

Usage
-----
    python 03_model_development/feature_interpretation.py

    # Optional flags:
    python 03_model_development/feature_interpretation.py \\
        --binary-results-dir 04_results_and_figures/models/binary \\
        --timepoints A B C     \\
        --shap-bg-start 100    \\
        --shap-bg-max   1000
"""

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
import shap
from lime.lime_tabular import LimeTabularExplainer

sys.path.insert(0, os.path.dirname(__file__))
from utilities import TIMEPOINTS, RANDOM_STATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


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
    if isinstance(vals, list):          # KernelExplainer → [class0, class1]
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

    # Try bg_start; upgrade to bg_max on success
    shap_matrix, final_bg = None, bg_start
    for n_bg in sorted({bg_start, bg_max}):
        try:
            shap_matrix = _compute(n_bg)
            final_bg    = n_bg
            logger.info("  SHAP: success at n=%d", n_bg)
        except Exception as exc:
            logger.warning("  SHAP: failed at n=%d — %s", n_bg, exc)
            break   # don't try larger if smaller already failed

    if shap_matrix is None:
        logger.error("  SHAP: all attempts failed — skipping.")
        return None

    mean_abs = pd.Series(np.abs(shap_matrix).mean(axis=0),
                         index=feature_names, name="mean_abs_shap"
                         ).sort_values(ascending=False)

    pd.DataFrame(shap_matrix, columns=feature_names).to_csv(
        os.path.join(output_dir, "shap_values.csv"), index=False)
    mean_abs.to_csv(os.path.join(output_dir, "shap_mean_abs.csv"), header=True)

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
# 3. LIME — num_samples stability tuning
# ---------------------------------------------------------------------------

def _lime_explain(lime_exp, predict_fn, inst: np.ndarray,
                  feature_names: list, ns: int) -> np.ndarray:
    n = len(feature_names)
    feat_idx = {f: i for i, f in enumerate(feature_names)}
    exp = lime_exp.explain_instance(inst, predict_fn,
                                    num_features=n, num_samples=ns, labels=(1,))
    vec = np.zeros(n)
    for fname, w in exp.as_list(label=1):
        if fname in feat_idx:
            vec[feat_idx[fname]] = w
    return vec


def _lime_tune(lime_exp, predict_fn, X_test_sc: np.ndarray,
               feature_names: list, candidates: list) -> int:
    probe_idx = np.linspace(0, len(X_test_sc) - 1,
                            min(3, len(X_test_sc)), dtype=int)
    ns_max  = max(candidates)
    refs    = [_lime_explain(lime_exp, predict_fn, X_test_sc[i],
                             feature_names, ns_max) for i in probe_idx]

    for ns in sorted(candidates):
        corrs = []
        for i, ref in zip(probe_idx, refs):
            w = _lime_explain(lime_exp, predict_fn, X_test_sc[i], feature_names, ns)
            r, _ = spearmanr(np.abs(w), np.abs(ref))
            if not np.isnan(r):
                corrs.append(r)
        mean_r = np.mean(corrs) if corrs else 0.0
        logger.info("  LIME stability: num_samples=%5d  ρ=%.3f", ns, mean_r)
        if mean_r >= 0.95:
            logger.info("  LIME: chosen num_samples=%d", ns)
            return ns

    logger.warning("  LIME: no candidate reached ρ≥0.95 — using max=%d", ns_max)
    return ns_max


def _plot_stability_curve(lime_exp, predict_fn, X_test_sc, feature_names,
                          candidates, chosen_ns, output_path):
    probe_idx = np.linspace(0, len(X_test_sc) - 1,
                            min(3, len(X_test_sc)), dtype=int)
    ns_max = max(candidates)
    refs   = [_lime_explain(lime_exp, predict_fn, X_test_sc[i],
                            feature_names, ns_max) for i in probe_idx]
    rhos   = []
    for ns in sorted(candidates):
        c = []
        for i, ref in zip(probe_idx, refs):
            w = _lime_explain(lime_exp, predict_fn, X_test_sc[i], feature_names, ns)
            r, _ = spearmanr(np.abs(w), np.abs(ref))
            if not np.isnan(r):
                c.append(r)
        rhos.append(np.mean(c) if c else np.nan)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sorted(candidates), rhos, marker="o", color="#2ca02c")
    ax.axhline(0.95, color="gray", linestyle="--", lw=0.8, label="ρ = 0.95")
    ax.axvline(chosen_ns, color="red", linestyle="--", lw=0.8,
               label=f"chosen = {chosen_ns}")
    ax.set_xlabel("num_samples")
    ax.set_ylabel("Mean Spearman ρ")
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
    candidates: list | None = None,
) -> pd.Series | None:
    if candidates is None:
        candidates = [100, 500, 1000, 2000, 5000]

    os.makedirs(output_dir, exist_ok=True)
    n = len(feature_names)

    lime_exp   = LimeTabularExplainer(
        training_data=X_trainval_sc,
        feature_names=feature_names,
        class_names=["Control", "Complication"],
        mode="classification",
        random_state=RANDOM_STATE,
        discretize_continuous=False,
    )
    predict_fn = model.predict_proba

    tuned_ns = _lime_tune(lime_exp, predict_fn, X_test_sc, feature_names, candidates)

    logger.info("  LIME: explaining %d test samples (num_samples=%d) …",
                len(X_test_sc), tuned_ns)
    weight_matrix = np.zeros((len(X_test_sc), n))
    for i, inst in enumerate(X_test_sc):
        try:
            weight_matrix[i] = _lime_explain(lime_exp, predict_fn,
                                             inst, feature_names, tuned_ns)
        except Exception as exc:
            logger.warning("  LIME: sample %d failed — %s", i, exc)

    mean_abs = pd.Series(np.abs(weight_matrix).mean(axis=0),
                         index=feature_names, name="mean_abs_lime"
                         ).sort_values(ascending=False)

    pd.DataFrame(weight_matrix, columns=feature_names).to_csv(
        os.path.join(output_dir, "lime_weights.csv"), index=False)
    mean_abs.to_csv(os.path.join(output_dir, "lime_mean_abs.csv"), header=True)

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

    _plot_stability_curve(lime_exp, predict_fn, X_test_sc, feature_names,
                          candidates, tuned_ns,
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
                  best_model_name, bg_start, bg_max):
    tag = f"[{tissue.upper()} {timepoint}]"
    logger.info("%s Best model: %s", tag, best_model_name)

    # ── load artifacts ─────────────────────────────────────────────────────
    def _load(fname):
        p = os.path.join(results_dir, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return p

    try:
        model         = joblib.load(_load(f"{best_model_name}.joblib"))
        feature_names = pd.read_csv(_load("lasso_selected_features.csv"))["feature"].tolist()
        X_trainval_sc = pd.read_csv(_load("X_trainval_scaled.csv"), index_col=0).values
        X_test_sc     = pd.read_csv(_load("X_test_scaled.csv"),     index_col=0).values
        y_test        = pd.read_csv(_load("y_test.csv"), index_col=0).squeeze()
    except FileNotFoundError as exc:
        logger.warning("%s Missing artifact %s — skipping. Re-run binary_classifier.py first.", tag, exc)
        return

    logger.info("%s  trainval=%d  test=%d  features=%d",
                tag, len(X_trainval_sc), len(X_test_sc), len(feature_names))

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Gini ────────────────────────────────────────────────────────────
    logger.info("%s  Gini …", tag)
    gini = compute_gini_importance(model, feature_names)
    if not gini.empty:
        gini.sort_values(key=lambda s: s.abs(), ascending=False).to_csv(
            os.path.join(output_dir, "gini_importance.csv"), header=True)
        plot_gini(gini,
                  title=f"Gini/Coef — {best_model_name} | {tissue} {timepoint}",
                  output_path=os.path.join(output_dir, "gini_importance.png"))
    else:
        logger.warning("%s  Model has no Gini/coef_ attribute.", tag)

    # ── 2. SHAP ────────────────────────────────────────────────────────────
    logger.info("%s  SHAP (bg_start=%d, bg_max=%d) …", tag, bg_start, bg_max)
    shap_result = run_shap(model, best_model_name,
                           X_trainval_sc, X_test_sc, feature_names,
                           output_dir, bg_start=bg_start, bg_max=bg_max)

    # ── 3. LIME ────────────────────────────────────────────────────────────
    logger.info("%s  LIME (stability tuning) …", tag)
    lime_result = run_lime(model, X_trainval_sc, X_test_sc,
                           feature_names, output_dir)

    # ── Combined panel ─────────────────────────────────────────────────────
    plot_combined(gini if not gini.empty else None, shap_result, lime_result,
                  title=f"{best_model_name} | {tissue} {timepoint}",
                  output_path=os.path.join(output_dir, "combined_importance.png"))

    logger.info("%s  Done → %s", tag, output_dir)


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
    p.add_argument("--timepoints",   nargs="+", default=TIMEPOINTS)
    p.add_argument("--shap-bg-start", type=int, default=100)
    p.add_argument("--shap-bg-max",   type=int, default=1000)
    p.add_argument("--skip-plasma",   action="store_true")
    p.add_argument("--skip-placenta", action="store_true")
    args = p.parse_args()

    base = args.binary_results_dir

    # Best model per (tissue, timepoint) by test PR-AUC
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
        logger.info("  %-10s %-5s → %-20s (%.4f)", t, tp, m, pr)

    if not args.skip_plasma:
        logger.info("=== PLASMA ===")
        for tp in args.timepoints:
            bm = best.get(("plasma", tp))
            if bm is None:
                logger.warning("Plasma %s not in summary — skipping.", tp)
                continue
            interpret_one(
                tissue="plasma", timepoint=tp,
                results_dir=os.path.join(base, "plasma", tp),
                output_dir=os.path.join(base, "plasma", tp, "interpretation"),
                best_model_name=bm,
                bg_start=args.shap_bg_start,
                bg_max=args.shap_bg_max,
            )

    if not args.skip_placenta:
        logger.info("=== PLACENTA ===")
        bm = best.get(("placenta", "all"))
        if bm is None:
            logger.warning("Placenta not in summary — skipping.")
        else:
            interpret_one(
                tissue="placenta", timepoint="all",
                results_dir=os.path.join(base, "placenta", "all"),
                output_dir=os.path.join(base, "placenta", "all", "interpretation"),
                best_model_name=bm,
                bg_start=args.shap_bg_start,
                bg_max=args.shap_bg_max,
            )


if __name__ == "__main__":
    main()
