"""
Binary classification pipeline for SOP v4 pipeline outputs (MTBL_sop / LIPD_sop).

Mirrors binary_classifier.py but points to:
  - data/cleaned/sop_omics_pipeline_v2/{tissue}/
  - 04_results_and_figures/differential_analysis/{dataset}/
  - 04_results_and_figures/models/binary/{dataset}/

Usage (from kaysonyao folder):
    python 03_model_development/run_sop_models.py --dataset MTBL_sop
    python 03_model_development/run_sop_models.py --dataset LIPD_sop
    python 03_model_development/run_sop_models.py              # both
    python 03_model_development/run_sop_models.py --dataset MTBL_sop --n-trials 30
    python 03_model_development/run_sop_models.py --dataset MTBL_sop --skip-placenta
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from binary_classifier import run_binary_pipeline
from utilities import (
    load_data,
    load_significant_analytes,
    normalise_group_labels,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

COMPLICATIONS = ["HDP", "FGR", "sPTB"]
TIMEPOINTS    = ["A", "B", "C", "D", "E"]


def run_dataset(
    dataset: str,
    wkdir: str,
    n_trials: int,
    skip_plasma: bool,
    skip_placenta: bool,
) -> None:
    tissue, _ = dataset.split("_", 1)          # "MTBL" or "LIPD"
    has_placenta = (tissue == "MTBL")

    plasma_dir   = os.path.join(wkdir, "data", "cleaned", "sop_omics_pipeline_v2", f"{tissue}_plasma")
    placenta_dir = os.path.join(wkdir, "data", "cleaned", "sop_omics_pipeline_v2", f"{tissue}_placenta")
    diff_root    = os.path.join(wkdir, "04_results_and_figures", "differential_analysis", dataset)
    output_root  = os.path.join(wkdir, "04_results_and_figures", "models", "binary", dataset)

    logger.info("=== %s | n_trials=%d ===", dataset, n_trials)
    all_summaries = []

    # ── Plasma ────────────────────────────────────────────────────────────────
    if not skip_plasma:
        logger.info("--- PLASMA ---")
        for tp in TIMEPOINTS:
            csv_path = os.path.join(plasma_dir, f"{tissue}_plasma_suffix_{tp}.csv")
            if not os.path.exists(csv_path):
                logger.warning("Plasma TP %s not found, skipping: %s", tp, csv_path)
                continue

            df = normalise_group_labels(load_data(csv_path))
            logger.info("Plasma TP %s: %d samples x %d cols", tp, *df.shape)

            # Load significant analytes from differential analysis (q < 0.05 filter)
            sig_csv = os.path.join(
                diff_root, "plasma", "cross_sectional", tp,
                "Control_vs_Complication_differential_results.csv",
            )
            sig_analytes = load_significant_analytes(sig_csv)
            if sig_analytes:
                logger.info("Plasma TP %s: %d significant analytes loaded (q<0.05).", tp, len(sig_analytes))
            else:
                logger.warning("Plasma TP %s: no diff results found — using all features.", tp)

            out_dir = os.path.join(output_root, "plasma", tp)
            result = run_binary_pipeline(
                df, "plasma", tp, COMPLICATIONS, out_dir, n_trials, sig_analytes,
                data_path=os.path.abspath(csv_path),
            )
            if result:
                all_summaries.append(result)

    # ── Placenta ──────────────────────────────────────────────────────────────
    if not skip_placenta and has_placenta:
        logger.info("--- PLACENTA ---")
        placenta_csv = os.path.join(placenta_dir, f"{tissue}_placenta_cleaned_with_metadata.csv")
        if not os.path.exists(placenta_csv):
            logger.warning("Placenta CSV not found: %s", placenta_csv)
        else:
            df = normalise_group_labels(load_data(placenta_csv))
            logger.info("Placenta: %d samples x %d cols", *df.shape)

            sig_csv = os.path.join(
                diff_root, "placenta", "cross_sectional",
                "Control_vs_Complication_differential_results.csv",
            )
            sig_analytes = load_significant_analytes(sig_csv)
            if sig_analytes:
                logger.info("Placenta: %d significant analytes loaded (q<0.05).", len(sig_analytes))
            else:
                logger.warning("Placenta: no diff results found — using all features.")

            out_dir = os.path.join(output_root, "placenta", "all")
            result = run_binary_pipeline(
                df, "placenta", "all", COMPLICATIONS, out_dir, n_trials, sig_analytes,
                data_path=os.path.abspath(placenta_csv),
            )
            if result:
                all_summaries.append(result)

    # ── Save combined summary ─────────────────────────────────────────────────
    if all_summaries:
        summary_path = os.path.join(output_root, "all_summaries.json")
        os.makedirs(output_root, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        logger.info("All summaries saved to %s", summary_path)

        logger.info("\n%-25s %-10s %-10s %-10s", "condition", "n_feat_pre", "n_feat_post", "test_PR-AUC")
        for s in all_summaries:
            cond = f"{s.get('tissue','?')}/{s.get('timepoint','?')}"
            pr   = s.get("test_metrics", {}).get("pr_auc", float("nan"))
            logger.info(
                "%-25s %-10d %-10d %.4f",
                cond,
                s.get("n_features_pretlasso", 0),
                s.get("n_features_postlasso", 0),
                pr,
            )

    logger.info("=== %s complete ===", dataset)


def main():
    parser = argparse.ArgumentParser(
        description="Binary classification on SOP v4 pipeline outputs."
    )
    parser.add_argument(
        "--dataset",
        choices=["MTBL_sop", "LIPD_sop"],
        default=None,
        help="Dataset to run. If omitted, runs both.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Optuna hyperparameter tuning trials per model (default: 50).",
    )
    parser.add_argument("--skip-plasma",   action="store_true")
    parser.add_argument("--skip-placenta", action="store_true")
    args = parser.parse_args()

    wkdir    = os.getcwd()
    datasets = [args.dataset] if args.dataset else ["MTBL_sop", "LIPD_sop"]

    for ds in datasets:
        run_dataset(ds, wkdir, args.n_trials, args.skip_plasma, args.skip_placenta)


if __name__ == "__main__":
    main()
