"""
clean_metabolomics_data.py
Author: Kayson Yao

Metabolomics preprocessing pipeline for DP3 wide-format LC-MS data.

The data has already been log2-transformed, imputed, and batch-normalized
by the collaborating lab. This script:

1. Loads Samples_<batch>_<timepoint>.csv (plasma) and Samples_<batch>.csv
   (placenta) — Pooled QC files are ignored.
2. Normalises SampleID formatting (removes internal spaces).
3. Drops duplicate SampleIDs, keeping the first occurrence.
4. Merges authoritative metadata (Group, Subgroup, Batch, GestAgeDelivery,
   SampleGestAge) from the master Excel table.
5. Applies a missingness filter (<20% missing per analyte).
6. Re-imputes any NaN introduced post-filter using half-minimum imputation.
7. Saves final wide matrices to normalized_full_results/.

After this, run format_proteomics.py to slice the plasma output by timepoint.

Usage:
    python 01_data_cleaning/clean_metabolomics_data.py
"""

import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utilities import (
    CUTOFF_PERCENT_MISSING,
    missingness_filter_and_group_check,
    half_min_impute_wide,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Columns embedded in the data files that are replaced by master-table metadata
# The two gestational age columns are clinical covariates, not metabolites
_DATA_META_COLS = {
    "patient_ID", "group", "subgroup",
    "gestational_age", "gestational_age_at_collection",
}
_GROUP_LABEL_MAP = {"sptb": "sPTB"}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_samples(data_dir: str) -> pd.DataFrame:
    """
    Load and combine all Samples_*.csv files in data_dir.
    Pooled_*.csv files are skipped.

    For plasma files (Samples_<batch>_<timepoint>.csv), the SampleID is
    reconstructed as patient_ID + timepoint to correctly handle subjects
    whose SubjectID ends in 'E' (e.g. DP3-0165E at timepoint A → DP3-0165EA).

    Duplicate SampleIDs across batches are resolved by averaging numeric
    analyte columns; non-numeric metadata columns use the first occurrence.
    """
    frames = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.startswith("Samples_") or not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(data_dir, fname), index_col=0)
        df.index = df.index.astype(str).str.replace(r"\s+", "", regex=True)

        parts = fname.replace(".csv", "").split("_")  # ["Samples", batch, timepoint?]
        if len(parts) == 3 and "patient_ID" in df.columns:
            # Plasma: reconstruct SampleID = patient_ID + timepoint from filename
            # This fixes E-subjects where the file index drops the trailing 'E'
            timepoint = parts[2]
            df.index = (df["patient_ID"].astype(str) + timepoint).values

        frames.append(df)

    combined = pd.concat(frames)
    n_dups = combined.index.duplicated().sum()

    if n_dups:
        # Separate non-numeric metadata columns from analyte columns so we can
        # average the numeric values while taking first-occurrence for strings.
        meta_cols_present = [c for c in combined.columns if c in _DATA_META_COLS]
        analyte_cols = [c for c in combined.columns if c not in _DATA_META_COLS]

        averaged = combined[analyte_cols].groupby(level=0).mean()

        if meta_cols_present:
            meta_first = combined[meta_cols_present].groupby(level=0).first()
            result = averaged.join(meta_first)
        else:
            result = averaged

        logger.info(
            "Averaged %d duplicate SampleID row(s) → %d unique SampleIDs.",
            n_dups, len(result),
        )
        return result

    return combined


def _load_metadata(
    meta_path: str,
    sheet: str,
    sample_col: str,
    include_sample_gest_age: bool = False,
) -> pd.DataFrame:
    """
    Load and standardise metadata from the master Excel table.
    Returns a DataFrame indexed by normalised SampleID.
    """
    cols = [sample_col, "omics set#", "group", "subgroup", "gest age del"]
    if include_sample_gest_age:
        cols.append("sample gest Age")

    meta = pd.read_excel(meta_path, sheet_name=sheet)[cols].dropna(subset=[sample_col])
    meta[sample_col] = meta[sample_col].astype(str).str.strip().str.replace(r"\s+", "", regex=True)

    rename = {
        sample_col:     "SampleID",
        "omics set#":   "Batch",
        "group":        "Group",
        "subgroup":     "Subgroup",
        "gest age del": "GestAgeDelivery",
    }
    if include_sample_gest_age:
        rename["sample gest Age"] = "SampleGestAge"

    meta = meta.rename(columns=rename).set_index("SampleID")
    meta["Group"] = meta["Group"].replace(_GROUP_LABEL_MAP)
    return meta[~meta.index.duplicated()]


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def _process(
    data: pd.DataFrame,
    meta: pd.DataFrame,
    output_csv: str,
    label: str,
) -> None:
    """
    Merge metadata, run missingness filter, re-impute, and save.

    SubjectID is taken from the embedded patient_ID column (plasma) or
    directly from the SampleID index (placenta, where index = SubjectID).
    """
    # Capture SubjectID before dropping data metadata columns
    subject_ids = (
        data["patient_ID"]
        if "patient_ID" in data.columns
        else pd.Series(data.index, index=data.index)
    )

    analyte_cols = [c for c in data.columns if c not in _DATA_META_COLS]

    # Merge authoritative metadata
    data = data[analyte_cols].join(meta, how="left")

    n_unmatched = data["Group"].isna().sum()
    if n_unmatched:
        logger.warning("[%s] %d sample(s) have no metadata match — dropping.", label, n_unmatched)
    data = data[data["Group"].notna()].copy()
    subject_ids = subject_ids.reindex(data.index)

    groups_binary = data["Group"].map(
        lambda g: "Control" if g == "Control" else "Complication"
    )

    # Missingness filter
    X_kept, dropped = missingness_filter_and_group_check(
        data[analyte_cols], groups_binary, cutoff=CUTOFF_PERCENT_MISSING
    )
    if not dropped.empty:
        rep_path = output_csv.replace(".csv", "_dropped_missingness_report.csv")
        dropped.to_csv(rep_path, index=False)
        logger.info(
            "[%s] Dropped %d analyte(s) exceeding %.0f%% missingness threshold.",
            label, len(dropped), CUTOFF_PERCENT_MISSING * 100,
        )

    X_final = half_min_impute_wide(X_kept)

    # Assemble: SubjectID | metadata | analytes
    meta_order = [
        c for c in ["Group", "Subgroup", "Batch", "GestAgeDelivery", "SampleGestAge"]
        if c in data.columns
    ]
    output = pd.concat(
        [subject_ids.rename("SubjectID"), data[meta_order], X_final], axis=1
    )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    output.index.name = "SampleID"
    output.to_csv(output_csv)
    logger.info(
        "[%s] Done — %d samples × %d analytes → %s",
        label, len(X_final), len(X_final.columns), output_csv,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    wkdir    = os.getcwd()
    meta_path = os.path.join(wkdir, "data", "dp3 master table v2.xlsx")
    output_dir = os.path.join(
        wkdir, "data", "cleaned", "metabolomics", "normalized_full_results"
    )

    # Plasma
    plasma_data = _load_samples(
        os.path.join(wkdir, "data", "cleaned", "metabolomics", "plasma")
    )
    plasma_meta = _load_metadata(
        meta_path, "n=133 metabolomics", "Sample ID", include_sample_gest_age=True
    )
    _process(
        plasma_data, plasma_meta,
        os.path.join(output_dir, "metabolomics_plasma_cleaned_with_metadata.csv"),
        label="plasma",
    )

    # Placenta
    placenta_data = _load_samples(
        os.path.join(wkdir, "data", "cleaned", "metabolomics", "placenta")
    )
    placenta_meta = _load_metadata(meta_path, "n=133 placenta", "ID")
    _process(
        placenta_data, placenta_meta,
        os.path.join(output_dir, "metabolomics_placenta_cleaned_with_metadata.csv"),
        label="placenta",
    )

    logger.info("All metabolomics processing complete.")
