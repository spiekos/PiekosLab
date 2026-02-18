"""
Title: clean_proteomics_data.py
Author: Kayson Yao
Date: 02/08/2026
Description:
DP3 proteomics preprocessing for Olink Explore long-format exports.

Workflow:
1) Load each Olink file in long format.
2) Standardize missing NPX values:
   - Convert blank/"NA"/"N/A"/0 in the NPX column to NaN.
3) QC masking:
   - If QC_Warning != PASS or Assay_Warning != PASS, set NPX to NaN.
4) Load metadata and extract batch (omics set#), group, subgroup, gest age del and sample gest age (only for plasma):
   - Match on "sample Id" column
5) Panel normalization (long format, uses Olink internal control samples):
   - Identify internal controls by SampleID prefix "CONTROL_SAMPLE".
   - For each (Panel, Assay), compute the median NPX among control samples.
   - For each Assay, compute a global reference as the median of these panel medians across panels.
   - Adjustment = global_median - panel_median; add adjustment to NPX for all samples in that Panel+Assay.
6) Remove Olink internal control samples from downstream analysis matrices.
7) Reshape to a wide matrix (rows = SampleID, columns = Assay, values = NPX; aggregate duplicates by median).
8) ComBat normalization (wide matrix):
   - Apply ComBat batch correction using metadata Batch labels.
   - Preserve original missingness pattern after correction.
9) Missingness filter on the ComBat-normalized wide matrix (pre-imputation):
   - Keep assays with missing fraction < 25%.
   - For assays failing the cutoff, compute group-wise missingness (Control vs Complication),
     test imbalance using Fisher's exact test, and apply Benjamini-Hochberg correction.
   - Save a dropped-assay missingness report CSV.
10) Impute remaining missing values (last):
    - Per assay, fill NaN with (minimum observed NPX - 1), i.e. half the linear minimum.
11) Output scaling:
    - Convert NPX (log2 scale) to linear positive scale via 2**NPX.
12) Merge metadata (Batch, Group, Subgroup, GestAgeDelivery) into final matrix
13) Save final cleaned matrix (wide, SampleID x [metadata + Assays]) to CSV.

"""

import logging
import os
import sys
import argparse

import numpy as np
import pandas as pd

from utilities import (
    CUTOFF_PERCENT_MISSING,
    collect_olink_files,
    load_metadata_with_batch,
    process_single_file,
    combine_batches,
    apply_panel_normalization_long,
    is_olink_control_sample,
    is_olink_control_assay,
    missingness_filter_and_group_check,
    combat_normalize_wide,
    half_min_impute_wide,
)

logger = logging.getLogger(__name__)


# -----------------------------
# Main wrapper
# -----------------------------
def process_all_files(
    file_paths: list[str],
    output_csv: str,
    metadata_path: str,
    meta_type: str = "proteomics",
):
    """
    Complete proteomics data processing pipeline.

    Args:
        file_paths: List of paths to Olink CSV files
        output_csv: Path to save final cleaned matrix
        metadata_path: Path to metadata Excel file
        meta_type: 'proteomics' or 'placenta' - which sheet to use
    """
    run_label = "plasma" if meta_type == "proteomics" else "placenta"
    logger.info("%s: start", run_label)

    # 1) Load metadata
    metadata = load_metadata_with_batch(metadata_path, meta_type=meta_type)

    # 2) Load/process all Olink files
    batches_long = []
    for fp in file_paths:
        df_single = process_single_file(fp)
        df_single["SourceFile"] = os.path.basename(fp)
        batches_long.append(df_single)
    df_long = combine_batches(batches_long)

    # 3) Panel normalization
    df_long = apply_panel_normalization_long(df_long)

    # 4) Remove Olink controls
    ctrl_mask = is_olink_control_sample(df_long)
    df_long_bio = df_long.loc[~ctrl_mask].copy()

    assay_ctrl_mask = is_olink_control_assay(df_long_bio)
    n_assay_ctrl_rows = int(assay_ctrl_mask.sum())
    if n_assay_ctrl_rows > 0:
        n_assay_ctrl_names = int(df_long_bio.loc[assay_ctrl_mask, "Assay"].nunique())
        df_long_bio = df_long_bio.loc[~assay_ctrl_mask].copy()
        logger.info(
            "[%s] removed %d rows from %d control assays (assay name contains 'control').",
            run_label, n_assay_ctrl_rows, n_assay_ctrl_names,
        )

    # 5) Build wide matrix
    X = df_long_bio.pivot_table(
        index="SampleID",
        columns="Assay",
        values="NPX",
        aggfunc="median",
    )

    # Metadata matching diagnostics
    matched = X.index.isin(metadata.index)
    n_matched = matched.sum()
    if n_matched < len(X):
        unmatched = X.index[~matched]
        logger.warning(
            "%d samples missing metadata. examples=%s",
            len(unmatched), list(unmatched[:5]),
        )
        data_no_space = X.index.str.replace(" ", "", regex=False)
        meta_no_space = metadata.index.str.replace(" ", "", regex=False)
        matched_no_space = data_no_space.isin(meta_no_space)
        if matched_no_space.sum() > matched.sum():
            logger.debug(
                "Stripping internal spaces would improve matching (%d vs %d).",
                matched_no_space.sum(), matched.sum(),
            )

    if n_matched == 0:
        logger.warning("No metadata matches found; downstream results may be unreliable.")

    # 6) Group labels for missingness check
    groups = metadata["Group"].reindex(X.index)
    has_metadata = groups.notna()
    n_no_metadata = (~has_metadata).sum()
    if n_no_metadata > 0:
        logger.warning("Filtering %d samples without metadata group.", n_no_metadata)
        X = X.loc[has_metadata].copy()
        groups = groups.loc[has_metadata]

    groups_binary = pd.Series(
        np.where(
            groups.astype(str).str.strip().str.upper() == "CONTROL",
            "Control",
            "Complication",
        ),
        index=groups.index,
        name="GroupBinary",
    )

    metadata = metadata.reindex(X.index)

    # 7) ComBat normalization (before missingness filter)
    batch_labels = metadata["Batch"].reindex(X.index)
    has_batch = batch_labels.notna()
    if not has_batch.all():
        n_no_batch = int((~has_batch).sum())
        logger.warning("Filtering %d samples without batch labels.", n_no_batch)
        X = X.loc[has_batch].copy()
        groups_binary = groups_binary.loc[has_batch].copy()
        metadata = metadata.reindex(X.index)
        batch_labels = batch_labels.loc[has_batch]

    X_norm = combat_normalize_wide(X, batch_labels)

    # 8) Missingness filter + report (after ComBat)
    X_kept, dropped_report = missingness_filter_and_group_check(
        X_norm, groups_binary, cutoff=CUTOFF_PERCENT_MISSING, alpha_bh=0.05
    )
    if not dropped_report.empty:
        rep_path = os.path.splitext(output_csv)[0] + "_dropped_missingness_report.csv"
        dropped_report.to_csv(rep_path, index=False)

    # 9) Imputation
    X_final = half_min_impute_wide(X_kept)

    # 10) Convert to linear scale
    X_final_linear = np.power(2.0, X_final)

    # 11) Merge metadata
    metadata_aligned = metadata.reindex(X_final_linear.index)

    if meta_type == "proteomics":
        subject_ids = X_final_linear.index.to_series().str.replace(
            r"\s*[A-Z]+$", "", regex=True
        )
        metadata_aligned.insert(0, "SubjectID", subject_ids.values)

    final_output = pd.concat([metadata_aligned, X_final_linear], axis=1)

    # 12) Save output
    final_output.to_csv(output_csv, index=True)

    n_dropped = len(dropped_report)
    logger.info(
        "%s: done | samples=%d | assays=%d | dropped_assays=%d "
        "| metadata_matched=%d/%d | output=%s",
        run_label, final_output.shape[0], X_final_linear.shape[1],
        n_dropped, n_matched, len(matched), output_csv,
    )
    if n_dropped > 0:
        logger.info(
            "Missingness report: %s_dropped_missingness_report.csv",
            os.path.splitext(output_csv)[0],
        )

    n_missing_meta = metadata_aligned.isna().all(axis=1).sum()
    if n_missing_meta > 0:
        logger.warning("%d samples have no metadata.", n_missing_meta)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    def _run_default_mode() -> None:
        """Backward-compatible run mode (current behavior)."""
        wkdir = os.getcwd()
        data_dir = os.path.join(wkdir, "data", "proteomics")
        output_dir = os.path.join(wkdir, "data", "cleaned", "proteomics", "normalized_full_results")
        os.makedirs(output_dir, exist_ok=True)
        metadata_path = os.path.join(wkdir, "data", "dp3 master table v2.xlsx")

        plasma_files, placenta_files = collect_olink_files(data_dir)
        logger.info(
            "files | plasma=%d | placenta=%d", len(plasma_files), len(placenta_files)
        )

        if plasma_files:
            out_csv_plasma = os.path.join(
                output_dir, "proteomics_plasma_cleaned_with_metadata.csv"
            )
            process_all_files(plasma_files, out_csv_plasma, metadata_path, meta_type="proteomics")

        if placenta_files:
            out_csv_placenta = os.path.join(
                output_dir, "proteomics_placenta_cleaned_with_metadata.csv"
            )
            process_all_files(
                placenta_files, out_csv_placenta, metadata_path, meta_type="placenta"
            )

        logger.info("All processing complete.")

    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Clean Olink proteomics data (core pipeline; diagnostics run separately)."
        )
        parser.add_argument(
            "--mode",
            choices=["auto", "single"],
            default="auto",
            help="auto: discover plasma/placenta CSVs in --data-dir (default). "
            "single: run one dataset with explicit --files/--meta-type/--output-csv.",
        )
        parser.add_argument(
            "--data-dir",
            default=None,
            help="Directory containing raw Olink CSV files (used in auto mode).",
        )
        parser.add_argument(
            "--metadata-path",
            default=None,
            help="Path to metadata Excel file.",
        )
        parser.add_argument(
            "--output-dir",
            default=None,
            help="Directory for cleaned output CSVs (used in auto mode).",
        )
        parser.add_argument(
            "--files",
            nargs="+",
            default=None,
            help="Input CSV files (required in single mode).",
        )
        parser.add_argument(
            "--meta-type",
            choices=["proteomics", "placenta"],
            default=None,
            help="Metadata sheet type (required in single mode).",
        )
        parser.add_argument(
            "--output-csv",
            default=None,
            help="Output CSV path (required in single mode).",
        )
        return parser

    def main() -> None:
        if len(sys.argv) == 1:
            _run_default_mode()
            return

        parser = _build_parser()
        args = parser.parse_args()

        wkdir = os.getcwd()
        metadata_path = args.metadata_path or os.path.join(
            wkdir, "data", "dp3 master table v2.xlsx"
        )

        if args.mode == "auto":
            data_dir = args.data_dir or os.path.join(wkdir, "data", "proteomics")
            output_dir = args.output_dir or os.path.join(
                wkdir, "data", "cleaned", "proteomics", "normalized_full_results"
            )
            os.makedirs(output_dir, exist_ok=True)

            plasma_files, placenta_files = collect_olink_files(data_dir)
            logger.info(
                "files | plasma=%d | placenta=%d",
                len(plasma_files), len(placenta_files),
            )

            if plasma_files:
                out_csv_plasma = os.path.join(
                    output_dir, "proteomics_plasma_cleaned_with_metadata.csv"
                )
                process_all_files(
                    plasma_files, out_csv_plasma, metadata_path, meta_type="proteomics"
                )

            if placenta_files:
                out_csv_placenta = os.path.join(
                    output_dir, "proteomics_placenta_cleaned_with_metadata.csv"
                )
                process_all_files(
                    placenta_files, out_csv_placenta, metadata_path, meta_type="placenta"
                )

            logger.info("All processing complete.")
            return

        # mode == "single"
        missing = []
        if not args.files:
            missing.append("--files")
        if args.meta_type is None:
            missing.append("--meta-type")
        if args.output_csv is None:
            missing.append("--output-csv")
        if missing:
            parser.error(f"single mode requires: {', '.join(missing)}")

        process_all_files(args.files, args.output_csv, metadata_path, meta_type=args.meta_type)

    main()
