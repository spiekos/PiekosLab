import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from binary_classifier import run_binary_pipeline
from utilities import load_data, normalise_group_labels, get_analyte_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

COMPLICATIONS = ["HDP", "FGR", "sPTB"]
TIMEPOINTS    = ["A", "B", "C", "D", "E"]
DATASET       = "MTBL_sop"


def main():
    parser = argparse.ArgumentParser(
        description="MTBL_sop binary classifier WITHOUT diff-analysis pre-filter."
    )
    parser.add_argument("--n-trials",     type=int, default=50)
    parser.add_argument("--skip-plasma",  action="store_true")
    parser.add_argument("--skip-placenta",action="store_true")
    args = parser.parse_args()

    wkdir       = os.getcwd()
    tissue      = "MTBL"
    plasma_dir  = os.path.join(wkdir, "data", "cleaned", "sop_omics_pipeline_v2", "MTBL_plasma")
    placenta_dir= os.path.join(wkdir, "data", "cleaned", "sop_omics_pipeline_v2", "MTBL_placenta")
    output_root = os.path.join(wkdir, "04_results_and_figures", "models", "binary", "MTBL_sop_nodiff")

    logger.info("=== %s_nodiff | n_trials=%d | NO differential filter ===", DATASET, args.n_trials)
    all_summaries = []

    # ── Plasma ────────────────────────────────────────────────────────────────
    if not args.skip_plasma:
        for tp in TIMEPOINTS:
            csv_path = os.path.join(plasma_dir, f"{tissue}_plasma_suffix_{tp}.csv")
            if not os.path.exists(csv_path):
                logger.warning("Plasma TP %s not found: %s — skipping.", tp, csv_path)
                continue

            df = normalise_group_labels(load_data(csv_path))
            logger.info(
                "Plasma TP %s: %d samples × %d analyte cols (no diff filter)",
                tp, len(df), len(get_analyte_columns(df)),
            )

            out_dir = os.path.join(output_root, "plasma", tp)
            result = run_binary_pipeline(
                df, "plasma", tp, COMPLICATIONS,
                out_dir, args.n_trials,
                sig_analytes=None,          # ← key: skip differential pre-filter
                data_path=os.path.abspath(csv_path),
            )
            if result:
                all_summaries.append(result)

    # ── Placenta ──────────────────────────────────────────────────────────────
    if not args.skip_placenta:
        placenta_csv = os.path.join(placenta_dir, f"{tissue}_placenta_cleaned_with_metadata.csv")
        if not os.path.exists(placenta_csv):
            logger.warning("Placenta CSV not found: %s — skipping.", placenta_csv)
        else:
            df = normalise_group_labels(load_data(placenta_csv))
            logger.info("Placenta: %d samples (no diff filter)", len(df))

            out_dir = os.path.join(output_root, "placenta", "all")
            result = run_binary_pipeline(
                df, "placenta", "all", COMPLICATIONS,
                out_dir, args.n_trials,
                sig_analytes=None,
                data_path=os.path.abspath(placenta_csv),
            )
            if result:
                all_summaries.append(result)

    # ── Save combined summary ─────────────────────────────────────────────────
    if all_summaries:
        os.makedirs(output_root, exist_ok=True)
        summary_path = os.path.join(output_root, "all_summaries.json")
        with open(summary_path, "w") as fh:
            json.dump(all_summaries, fh, indent=2)
        logger.info("All summaries saved to %s", summary_path)

        logger.info("\n%-22s %-12s %-12s %-10s", "condition", "n_pre_lasso", "n_post_lasso", "best_PR-AUC")
        logger.info("-" * 60)
        for s in all_summaries:
            cond      = f"{s['tissue']}/{s['timepoint']}"
            best_name = s["best_model_val"]
            best_pr   = s["test_metrics"][best_name]["pr_auc"]
            logger.info(
                "%-22s %-12d %-12d %.4f  (%s)",
                cond,
                s["n_features_pretlasso"],
                s["n_features_postlasso"],
                best_pr,
                best_name,
            )

    logger.info("=== MTBL_sop_nodiff complete ===")


if __name__ == "__main__":
    main()
