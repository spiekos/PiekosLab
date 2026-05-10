"""
Differential analysis for sop_omics_pipeline_v2 outputs (MTBL_sop and LIPD_sop).

Runs the same cross-sectional + longitudinal pipeline as identify_differential_analytes.py
but pointed at the new SOP v4 output paths.

Usage (from the kaysonyao folder):
    python 02_exploratory_analysis/run_sop_differential.py --dataset MTBL_sop
    python 02_exploratory_analysis/run_sop_differential.py --dataset LIPD_sop
    python 02_exploratory_analysis/run_sop_differential.py  # runs both
"""

import os
import sys
import logging
import argparse
import pandas as pd

# ── Import shared utilities from sibling script ────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from utilities import (
    load_data,
    normalise_group_labels,
    get_analyte_columns,
    run_cross_sectional,
    run_longitudinal,
    write_sample_count_report,
    _start_analysis_log,
    plot_cross_sectional_boxplots,
    plot_longitudinal_boxplots,
    write_metaboanalyst_export,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _merge_to_complication(df: pd.DataFrame, group_col: str = "Group") -> pd.DataFrame:
    df = df.copy()
    df[group_col] = df[group_col].apply(lambda g: g if g == "Control" else "Complication")
    return df


def get_numeric_analyte_columns(df: pd.DataFrame) -> list:
    """
    Like get_analyte_columns() but additionally restricts to numeric dtype.
    Handles extra string columns (e.g. MetadataCanonicalID) that the SOP pipeline
    writes alongside the metadata but that the base utility doesn't know to exclude.
    """
    base = get_analyte_columns(df)
    return [c for c in base if pd.api.types.is_numeric_dtype(df[c])]


def run_dataset(dataset: str, wkdir: str) -> None:
    """
    dataset : "MTBL_sop" or "LIPD_sop"
    """
    tissue, _ = dataset.split("_", 1)     # "MTBL" or "LIPD"
    has_placenta = (tissue == "MTBL")     # LIPD has no placenta in current run

    cleaned_root = os.path.join(wkdir, "data", "cleaned", "sop_omics_pipeline_v2")

    plasma_dir   = os.path.join(cleaned_root, f"{tissue}_plasma")
    placenta_dir = os.path.join(cleaned_root, f"{tissue}_placenta")

    output_root = os.path.join(
        wkdir, "04_results_and_figures", "differential_analysis", dataset
    )
    os.makedirs(output_root, exist_ok=True)
    _start_analysis_log(output_root)
    logger.info("=== Differential analysis: %s ===", dataset)

    prefix = f"{tissue}_plasma"  # e.g. MTBL_plasma

    # ── Sample count report ────────────────────────────────────────────────
    placenta_csv = os.path.join(placenta_dir, f"{tissue}_placenta_cleaned_with_metadata.csv")
    write_sample_count_report(
        output_root,
        plasma_dir,
        placenta_csv if has_placenta and os.path.exists(placenta_csv) else None,
        file_prefix=prefix,
    )

    # ── Load feature metadata for MetaboAnalyst m/z + RT lookup ──────────────
    plasma_fm_path = os.path.join(plasma_dir, f"{tissue}_plasma_feature_metadata.csv")
    plasma_feature_meta = (
        pd.read_csv(plasma_fm_path, index_col=0, low_memory=False)
        if os.path.exists(plasma_fm_path) else pd.DataFrame()
    )
    placenta_fm_path = os.path.join(placenta_dir, f"{tissue}_placenta_feature_metadata.csv")
    placenta_feature_meta = (
        pd.read_csv(placenta_fm_path, index_col=0, low_memory=False)
        if has_placenta and os.path.exists(placenta_fm_path) else pd.DataFrame()
    )

    # ── Cross-sectional: placenta ──────────────────────────────────────────
    if has_placenta and os.path.exists(placenta_csv):
        df = normalise_group_labels(load_data(placenta_csv))
        df = _merge_to_complication(df)
        analyte_cols = get_numeric_analyte_columns(df)
        logger.info(
            "Cross-sectional [placenta]: %d samples x %d analytes", df.shape[0], len(analyte_cols)
        )
        placenta_cs_dir = os.path.join(output_root, "placenta", "cross_sectional")
        run_cross_sectional(
            df,
            analyte_cols,
            group_col="Group",
            output_dir=placenta_cs_dir,
        )
        write_metaboanalyst_export(
            placenta_cs_dir, "Control_vs_Complication", placenta_feature_meta,
            metaboanalyst_dir=os.path.join(output_root, "placenta", "metaboanalyst"),
        )
    elif has_placenta:
        logger.warning("Placenta file not found, skipping: %s", placenta_csv)

    # ── Cross-sectional: plasma per timepoint ──────────────────────────────
    available_tps = []
    for tp in ["A", "B", "C", "D", "E"]:
        tp_csv = os.path.join(plasma_dir, f"{prefix}_suffix_{tp}.csv")
        if not os.path.exists(tp_csv):
            logger.warning("Plasma timepoint %s not found, skipping: %s", tp, tp_csv)
            continue
        available_tps.append(tp)
        df_tp = normalise_group_labels(load_data(tp_csv))
        df_tp = _merge_to_complication(df_tp)
        analyte_cols = get_numeric_analyte_columns(df_tp)
        logger.info(
            "Cross-sectional [plasma %s]: %d samples x %d analytes", tp, df_tp.shape[0], len(analyte_cols)
        )
        tp_cs_dir = os.path.join(output_root, "plasma", "cross_sectional", tp)
        run_cross_sectional(
            df_tp,
            analyte_cols,
            group_col="Group",
            output_dir=tp_cs_dir,
        )
        write_metaboanalyst_export(
            tp_cs_dir, "Control_vs_Complication", plasma_feature_meta,
            metaboanalyst_dir=os.path.join(
                output_root, "plasma", "metaboanalyst", "cross_sectional", tp
            ),
        )

    # ── Cross-sectional boxplots: plasma ───────────────────────────────────
    tp_dfs_cs = {}
    for tp in available_tps:
        tp_csv = os.path.join(plasma_dir, f"{prefix}_suffix_{tp}.csv")
        tp_dfs_cs[tp] = normalise_group_labels(
            _merge_to_complication(load_data(tp_csv))
        )

    if tp_dfs_cs:
        plot_cross_sectional_boxplots(
            tp_dfs             = tp_dfs_cs,
            cross_results_root = os.path.join(output_root, "plasma", "cross_sectional"),
            output_dir         = os.path.join(output_root, "plasma", "cross_sectional_boxplots"),
            group_col          = "Group",
            ctrl_group         = "Control",
            compl_sources      = ["Complication"],
            top_n              = 50,
        )

    # ── Longitudinal: plasma ───────────────────────────────────────────────
    tp_dfs = {}
    for tp in available_tps:
        tp_csv = os.path.join(plasma_dir, f"{prefix}_suffix_{tp}.csv")
        tp_dfs[tp] = normalise_group_labels(load_data(tp_csv))

    if not tp_dfs:
        logger.warning("No plasma timepoint files found; skipping longitudinal.")
        return

    analyte_cols = get_numeric_analyte_columns(next(iter(tp_dfs.values())))
    long_dir     = os.path.join(output_root, "plasma", "longitudinal")

    for group in ["Control", "FGR", "HDP", "sPTB"]:
        logger.info("Longitudinal [%s]", group)
        run_longitudinal(
            tp_dfs,
            analyte_cols,
            group=group,
            group_col="Group",
            subject_col="SubjectID",
            output_dir=long_dir,
        )

    # pooled Complication
    tp_dfs_merged = {k: _merge_to_complication(v) for k, v in tp_dfs.items()}
    logger.info("Longitudinal [Complication pooled]")
    run_longitudinal(
        tp_dfs_merged,
        analyte_cols,
        group="Complication",
        group_col="Group",
        subject_col="SubjectID",
        output_dir=long_dir,
    )

    # ── MetaboAnalyst exports: longitudinal ───────────────────────────────
    import glob as _glob
    for long_csv in sorted(_glob.glob(os.path.join(long_dir, "*_longitudinal_results.csv"))):
        stem = os.path.basename(long_csv).replace("_longitudinal_results.csv", "")
        write_metaboanalyst_export(
            long_dir, stem, plasma_feature_meta,
            results_suffix="_longitudinal_results.csv",
            metaboanalyst_dir=os.path.join(
                output_root, "plasma", "metaboanalyst", "longitudinal"
            ),
        )

    # ── Longitudinal boxplots ──────────────────────────────────────────────
    plot_longitudinal_boxplots(
        tp_dfs           = tp_dfs_merged,
        long_results_dir = long_dir,
        output_dir       = os.path.join(output_root, "plasma", "longitudinal_boxplots"),
        group_col        = "Group",
        subject_col      = "SubjectID",
        ctrl_group       = "Control",
        compl_sources    = ["Complication"],
        top_n            = 50,
    )

    logger.info("=== %s complete. Results in: %s ===", dataset, output_root)


def main():
    parser = argparse.ArgumentParser(
        description="Run differential analysis on SOP v4 pipeline outputs."
    )
    parser.add_argument(
        "--dataset",
        choices=["MTBL_sop", "LIPD_sop"],
        default=None,
        help="Which dataset to run. If omitted, runs both MTBL_sop and LIPD_sop.",
    )
    args = parser.parse_args()

    wkdir = os.getcwd()
    datasets = [args.dataset] if args.dataset else ["MTBL_sop", "LIPD_sop"]

    for ds in datasets:
        run_dataset(ds, wkdir)

    logger.info("All done.")


if __name__ == "__main__":
    main()
