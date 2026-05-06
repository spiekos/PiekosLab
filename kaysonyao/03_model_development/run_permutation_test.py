import argparse
import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import RobustScaler

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from utilities import build_tuned_model_binary, RANDOM_STATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_condition(result_dir: str):
    needed = [
        "X_trainval_scaled.csv", "X_test_scaled.csv",
        "sample_splits.csv", "y_test.csv",
        "tuned_hyperparams.json", "summary.json",
    ]
    for f in needed:
        if not os.path.exists(os.path.join(result_dir, f)):
            logger.warning("Missing %s in %s — skipping.", f, result_dir)
            return None

    X_tv  = pd.read_csv(os.path.join(result_dir, "X_trainval_scaled.csv"), index_col=0)
    X_te  = pd.read_csv(os.path.join(result_dir, "X_test_scaled.csv"),     index_col=0)
    y_te  = pd.read_csv(os.path.join(result_dir, "y_test.csv"),             index_col=0).squeeze()

    splits = pd.read_csv(os.path.join(result_dir, "sample_splits.csv"))
    tv_mask = splits["split"].isin(["train", "val"])
    tv_rows = splits.loc[tv_mask].set_index("SampleID")
    y_tv = tv_rows.loc[X_tv.index, "label"].astype(int)

    with open(os.path.join(result_dir, "tuned_hyperparams.json")) as fh:
        hp = json.load(fh)

    with open(os.path.join(result_dir, "summary.json")) as fh:
        summary = json.load(fh)

    best_name   = summary["best_model_val"]
    best_params = hp["model_params"][best_name]
    actual_pr   = summary["test_metrics"][best_name]["pr_auc"]

    return X_tv, y_tv, X_te, y_te, best_name, best_params, actual_pr


def run_permutation_test(
    result_dir: str,
    label: str,
    n_perm: int = 1000,
    random_state: int = RANDOM_STATE,
) -> dict | None:
    loaded = _load_condition(result_dir)
    if loaded is None:
        return None

    X_tv, y_tv, X_te, y_te, best_name, best_params, actual_pr = loaded

    n_ctrl  = int((y_tv == 0).sum())
    n_compl = int((y_tv == 1).sum())
    n_test  = len(y_te)

    logger.info(
        "%s | model=%s | trainval(ctrl=%d, compl=%d) | test=%d | actual PR-AUC=%.4f",
        label, best_name, n_ctrl, n_compl, n_test, actual_pr,
    )

    rng = np.random.default_rng(random_state)
    null_prs = []

    X_tv_np = X_tv.values
    X_te_np = X_te.values
    y_tv_np = y_tv.values.copy()
    y_te_np = y_te.values

    # Warn if test set is tiny
    if n_test < 10:
        logger.warning(
            "%s: test set has only %d samples — p-value will be unreliable.", label, n_test
        )

    for i in range(n_perm):
        y_perm = rng.permutation(y_tv_np)
        # Skip if only one class after permutation (shouldn't happen but guard)
        if len(np.unique(y_perm)) < 2:
            continue
        model = build_tuned_model_binary(best_name, best_params, pd.Series(y_perm))
        model.fit(X_tv_np, y_perm)
        if len(np.unique(y_te_np)) < 2:
            continue
        y_prob = model.predict_proba(X_te_np)[:, 1]
        null_prs.append(float(average_precision_score(y_te_np, y_prob)))

        if (i + 1) % 100 == 0:
            logger.info("  %s: %d / %d permutations done.", label, i + 1, n_perm)

    null_prs = np.array(null_prs)
    p_value  = float((null_prs >= actual_pr).sum() / len(null_prs))
    null_mean = float(null_prs.mean())
    null_std  = float(null_prs.std())
    null_max  = float(null_prs.max())

    logger.info(
        "%s | null: mean=%.3f  std=%.3f  max=%.3f | p=%.4f %s",
        label, null_mean, null_std, null_max, p_value,
        "✓ significant" if p_value < 0.05 else "✗ NOT significant",
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_prs, bins=40, color="#94a3b8", edgecolor="white", linewidth=0.4, label="Null distribution")
    ax.axvline(actual_pr, color="#ef4444", linewidth=2.0, label=f"Observed PR-AUC = {actual_pr:.3f}")
    ax.axvline(null_mean, color="#64748b", linewidth=1.0, linestyle="--", label=f"Null mean = {null_mean:.3f}")
    sig_str = f"p = {p_value:.4f}" + (" *" if p_value < 0.05 else " (n.s.)")
    ax.set_title(f"Permutation Test — {label}\n{best_name}  |  {sig_str}", fontsize=11, fontweight="bold")
    ax.set_xlabel("PR-AUC (permuted labels)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out_png = os.path.join(result_dir, "permutation_test.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved plot: %s", out_png)

    return {
        "label":       label,
        "tissue":      label.split("/")[0],
        "timepoint":   label.split("/")[1] if "/" in label else "all",
        "model":       best_name,
        "actual_pr":   actual_pr,
        "null_mean":   null_mean,
        "null_std":    null_std,
        "null_max":    null_max,
        "n_perm":      len(null_prs),
        "p_value":     p_value,
        "significant": p_value < 0.05,
        "n_test":      n_test,
        "plot_path":   out_png,
    }


def main():
    parser = argparse.ArgumentParser(description="Permutation test on saved MTBL_sop model results.")
    parser.add_argument("--dataset", default="MTBL_sop", choices=["MTBL_sop", "LIPD_sop"],
                        help="Which dataset's saved results to test.")
    parser.add_argument("--n-perm", type=int, default=1000,
                        help="Number of permutations per condition (default: 1000).")
    args = parser.parse_args()

    wkdir      = os.getcwd()
    result_root = os.path.join(wkdir, "04_results_and_figures", "models", "binary", args.dataset)

    conditions = []
    for tp in ["A", "B", "C", "D", "E"]:
        d = os.path.join(result_root, "plasma", tp)
        if os.path.isdir(d):
            conditions.append((d, f"plasma/{tp}"))
    d = os.path.join(result_root, "placenta", "all")
    if os.path.isdir(d):
        conditions.append((d, "placenta/all"))

    if not conditions:
        logger.error("No result directories found under %s", result_root)
        sys.exit(1)

    all_results = []
    for result_dir, label in conditions:
        res = run_permutation_test(result_dir, label, n_perm=args.n_perm)
        if res:
            all_results.append(res)

    if not all_results:
        logger.error("No results produced.")
        sys.exit(1)

    summary_path = os.path.join(result_root, "permutation_test_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    logger.info("Summary saved: %s", summary_path)

    print("\n" + "=" * 80)
    print(f"Permutation Test Summary — {args.dataset}  (n_perm={args.n_perm})")
    print("=" * 80)
    print(f"{'Condition':<20} {'Model':<22} {'Actual PR':<12} {'Null mean':<12} {'Null max':<12} {'p-value':<10} {'Sig?'}")
    print("-" * 80)
    for r in all_results:
        sig = "✓ p<0.05" if r["significant"] else "✗ n.s."
        if r["n_test"] < 10:
            sig += " ⚠ tiny n"
        print(
            f"{r['label']:<20} {r['model']:<22} {r['actual_pr']:<12.4f} "
            f"{r['null_mean']:<12.4f} {r['null_max']:<12.4f} {r['p_value']:<10.4f} {sig}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
