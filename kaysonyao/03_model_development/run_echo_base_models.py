"""Aim 3A base models on the ECHO gestational windows.

Per the current spec:
  * full analyte matrix as input - no differential pre-filter. Selecting
    features from differential results computed on the whole cohort lets the
    test set help choose which features exist, which inflates test performance.
    Elastic net does the dimensionality reduction instead, during fitting.
  * primary target "any complication" (HDP | FGR | sPTB vs Control)
  * development set only; the locked 30% test set is not touched here
  * nested CV, PR-AUC with 95% CIs

Metabolomics and lipidomics only. Proteomics keeps its existing runner.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "01_data_cleaning"))

from nested_cv import run_nested_cv  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

COMPLICATIONS = ["HDP", "FGR", "sPTB"]
DATASETS = ["MTBL_plasma", "LIPD_plasma"]
WINDOWS = ["W1_early", "W2_third"]

METADATA_COLS = {
    "SampleID", "SubjectID", "Batch", "Group", "Subgroup",
    "GestAgeDelivery", "SampleGestAge", "Timepoint", "Tissue",
}


def analyte_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in METADATA_COLS]


def load_window(root: str, dataset: str, window: str, locked: pd.DataFrame):
    path = os.path.join(root, "data", "cleaned", "echo_windows", dataset,
                        f"{dataset}_echo_{window}.csv")
    if not os.path.exists(path):
        logger.warning("%s / %s: not found (%s)", dataset, window, path)
        return None

    df = pd.read_csv(path, low_memory=False)
    df = df[df["Group"].isin(["Control"] + COMPLICATIONS)].copy()

    dev_ids = set(locked.loc[locked["split"] == "dev", "SubjectID"])
    before = df["SubjectID"].nunique()
    df = df[df["SubjectID"].isin(dev_ids)]
    logger.info(
        "%s / %s: %d participants -> %d in development set (test set untouched)",
        dataset, window, before, df["SubjectID"].nunique(),
    )
    if df.empty:
        return None

    cols = analyte_columns(df)
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    n_dropped = len(cols) - X.shape[1]
    if n_dropped:
        logger.info("  dropped %d all-NaN analyte column(s)", n_dropped)
    # Median imputation is fit on the development set only; the locked test set
    # is never read here, so no test information enters.
    X = X.fillna(X.median())

    y = df["Group"].isin(COMPLICATIONS).astype(int)
    groups = df["SubjectID"]
    logger.info("  X=%s | any-complication=%d control=%d", X.shape, int(y.sum()), int((y == 0).sum()))
    return X.reset_index(drop=True), y.reset_index(drop=True), groups.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="ECHO-window base models, full analyte matrix.")
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()

    root = os.path.abspath(args.repo_root)
    locked_path = os.path.join(root, "data", "cleaned", "locked_split.csv")
    if not os.path.exists(locked_path):
        raise SystemExit("Locked split not found. Run 01_data_cleaning/make_locked_split.py first.")
    locked = pd.read_csv(locked_path)

    out_root = os.path.join(root, "04_results_and_figures", "models_v2", "echo_base")
    os.makedirs(out_root, exist_ok=True)

    summaries = []
    for dataset in args.datasets:
        for window in WINDOWS:
            logger.info("=== %s / %s ===", dataset, window)
            loaded = load_window(root, dataset, window, locked)
            if loaded is None:
                continue
            X, y, groups = loaded
            if y.nunique() < 2 or min(int(y.sum()), int((y == 0).sum())) < 5:
                logger.warning("  too few in one class - skipping")
                continue

            res = run_nested_cv(
                X, y, groups,
                outer_splits=args.outer_splits,
                inner_splits=args.inner_splits,
            )
            out_dir = os.path.join(out_root, dataset, window)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "nested_cv_results.json"), "w") as f:
                json.dump(res, f, indent=2, default=float)

            for model, m in res.items():
                summaries.append({
                    "dataset": dataset, "window": window, "model": model,
                    "n_samples": int(len(y)), "n_participants": int(groups.nunique()),
                    "n_features_in": int(X.shape[1]),
                    "n_features_selected_mean": m["n_features_mean"],
                    "pr_auc": m["pr_auc_mean"], "pr_auc_lo": m["pr_auc_ci95"][0],
                    "pr_auc_hi": m["pr_auc_ci95"][1],
                    "roc_auc": m["roc_auc_mean"], "brier": m["brier_mean"],
                    "prevalence_baseline": m["prevalence"],
                })

    if summaries:
        s = pd.DataFrame(summaries)
        p = os.path.join(out_root, "summary.csv")
        s.to_csv(p, index=False)
        logger.info("Summary -> %s", p)
        logger.info("\n%s", s.to_string(index=False))
    logger.info("ECHO base models complete.")


if __name__ == "__main__":
    main()
