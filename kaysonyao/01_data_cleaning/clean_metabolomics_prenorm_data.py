"""
clean_metabolomics_prenorm_data.py
==================================
Processes pre-normalized plasma metabolomics data through log2 transformation,
ComBat batch correction, missingness filtering, and imputation — identical
pipeline to proteomics and lipids.

Replaces the colleague's normalization step with ComBat-based batch correction.
Input files contain raw peak areas that have been cleaned (QC, peak picking,
feature alignment) but NOT normalized.

Input
-----
  data/metabolomics_prenormalized/Samples_{batch}_{timepoint}_preNorm.csv
    Batches   : 51223, 110123, 112524  (instrument run dates)
    Timepoints: A, B, C, D, E
    Columns   : analyte columns (p*_POS / n*_NEG) + metadata columns
                (patient_ID, group, subgroup, gestational_age,
                 gestational_age_at_collection)
    Values    : raw peak areas, linear scale, no zeros, NaN = missing

Output
------
  data/cleaned/metabolomics_combat/normalized_full_results/
    metabolomics_plasma_cleaned_with_metadata.csv
    metabolomics_plasma_dropped_missingness_report.csv
    metabolomics_pca_pre_post_combat_batch.png
    metabolomics_pca_pre_post_combat_group.png
  data/cleaned/metabolomics_combat/normalized_sliced_by_suffix/
    metabolomics_plasma_formatted_suffix_{A-E}.csv

Output CSV format (identical to proteomics / lipids pipeline)
-------------------------------------------------------------
  Index   : SampleID  (DP3-XXXXZ)
  Columns : SubjectID, Batch, Group, Subgroup, GestAgeDelivery,
            SampleGestAge, <analyte_1>, <analyte_2>, …
  Values  : log2(peak_area + 1), ComBat-corrected, imputed

Pipeline
--------
  1.  Load all Samples_*_preNorm.csv files; extract instrument batch and
      timepoint from filename.
  2.  Construct canonical SampleID:
        - patient_ID ending in E (postpartum-only subject) → SampleID = patient_ID
        - otherwise → SampleID = patient_ID + filename_timepoint
  3.  Combine across files; average analyte values for duplicate SampleIDs
      (cross-batch technical replicates); keep first occurrence for metadata.
  4.  Log2(x + 1) transform all analyte columns.
  5.  Build per-sample batch labels (instrument batch date string).
  6.  Filter samples missing group or batch labels.
  7.  Build binary group labels (Control vs Complication).
  8.  ComBat batch correction on the wide log2 matrix.
  9.  Pre/post ComBat PCA plots (colored by batch, colored by group).
  10. Missingness filter: drop analytes with ≥ 20% missing; Fisher's exact +
      Benjamini-Hochberg group-imbalance test; save dropped-analyte report.
  11. Half-minimum imputation per analyte in log2 space (fill = min_log2 − 1).
  12. Attach metadata columns (SubjectID, Batch, Group, Subgroup,
      GestAgeDelivery, SampleGestAge).
  13. Save full matrix CSV.
  14. Slice by timepoint suffix (A–E) and save per-timepoint CSVs.

Usage
-----
  python 01_data_cleaning/clean_metabolomics_prenorm_data.py
"""

import argparse
import logging
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utilities import (
    CUTOFF_PERCENT_MISSING,
    combat_normalize_wide,
    half_min_impute_wide,
    missingness_filter_and_group_check,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns present in the pre-normalised files that are metadata, not analytes
_META_COLS = {
    "patient_ID",
    "group",
    "subgroup",
    "gestational_age",
    "gestational_age_at_collection",
}

VALID_TIMEPOINTS = set("ABCDE")


# ---------------------------------------------------------------------------
# SampleID construction
# ---------------------------------------------------------------------------

def _canonical_sid(patient_id: str, file_timepoint: str) -> str | None:
    """
    Build the canonical DP3-XXXXZ SampleID from a patient_ID and the
    timepoint letter extracted from the filename.

    Rules
    -----
    • Strip any whitespace from patient_id.
    • Postpartum-only subjects have patient_ID ending in a timepoint letter
      (e.g. "DP3-0130E").  Their biological timepoint is always E, so the
      canonical SampleID is just the patient_ID with no extra suffix.
    • All other subjects get the filename timepoint appended:
      patient_ID="DP3-0071", file_timepoint="A"  →  "DP3-0071A"
    • Returns None if the result cannot be resolved to a valid DP3-XXXXZ.
    """
    pid = re.sub(r"\s+", "", str(patient_id).strip())

    # Postpartum subject: patient_ID already ends in a timepoint letter
    m_post = re.match(r"^(DP3-\d{4})([A-Ea-e])$", pid)
    if m_post:
        return f"{m_post.group(1)}{m_post.group(2).upper()}"

    # Regular subject: append filename timepoint
    m_reg = re.match(r"^(DP3-\d{4})$", pid)
    if m_reg and file_timepoint.upper() in VALID_TIMEPOINTS:
        return f"{m_reg.group(1)}{file_timepoint.upper()}"

    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_prenorm_samples(data_dir: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and combine all Samples_*_preNorm.csv files.

    Returns
    -------
    combined_analytes : DataFrame  (SampleID × analyte)   — raw peak areas
    batch_series      : Series     (SampleID → batch date string)
    """
    analyte_frames: list[pd.DataFrame] = []
    meta_frames:    list[pd.DataFrame] = []
    batch_records:  list[tuple[str, str]] = []   # (SampleID, batch_date)

    for fname in sorted(os.listdir(data_dir)):
        if not fname.startswith("Samples_") or not fname.endswith("_preNorm.csv"):
            continue

        # Filename pattern: Samples_{batch}_{timepoint}_preNorm.csv
        parts = fname.replace("_preNorm.csv", "").split("_")
        # parts = ['Samples', batch_date, timepoint]
        if len(parts) != 3:
            logger.warning("Unexpected filename format, skipping: %s", fname)
            continue
        batch_date   = parts[1]   # e.g. "51223"
        file_tp      = parts[2]   # e.g. "A"

        df = pd.read_csv(os.path.join(data_dir, fname), index_col=0)

        # Require patient_ID column
        if "patient_ID" not in df.columns:
            logger.warning("No patient_ID column in %s — skipping.", fname)
            continue

        # Build canonical SampleID
        sids = [_canonical_sid(pid, file_tp) for pid in df["patient_ID"]]
        n_unresolved = sum(1 for s in sids if s is None)
        if n_unresolved:
            logger.warning(
                "[%s] %d row(s) with unresolvable SampleID — dropped.",
                fname, n_unresolved,
            )

        df.index = sids
        df = df[df.index.notna()]        # drop unresolvable rows

        # Split analyte vs metadata columns
        analyte_cols = [c for c in df.columns if c not in _META_COLS]
        meta_cols    = [c for c in df.columns if c in _META_COLS]

        analyte_frames.append(df[analyte_cols])
        if meta_cols:
            meta_frames.append(df[meta_cols])

        for sid in df.index:
            batch_records.append((sid, batch_date))

        logger.info(
            "[%s] batch=%s tp=%s → %d samples, %d analytes",
            fname, batch_date, file_tp, len(df), len(analyte_cols),
        )

    if not analyte_frames:
        raise RuntimeError(f"No Samples_*_preNorm.csv files found in {data_dir}")

    # --- Combine and resolve duplicates ---
    combined_analytes = pd.concat(analyte_frames)
    combined_meta     = pd.concat(meta_frames) if meta_frames else pd.DataFrame()

    # Batch label: for duplicated SampleIDs, keep the batch of the first occurrence
    batch_series = (
        pd.DataFrame(batch_records, columns=["SampleID", "Batch"])
        .drop_duplicates(subset="SampleID", keep="first")
        .set_index("SampleID")["Batch"]
    )

    n_dup_rows = combined_analytes.index.duplicated().sum()
    if n_dup_rows:
        logger.info(
            "%d duplicate SampleID row(s) found — averaging analyte values.",
            n_dup_rows,
        )
        combined_analytes = combined_analytes.groupby(level=0).mean()
        if not combined_meta.empty:
            combined_meta = combined_meta.groupby(level=0).first()
    else:
        combined_analytes = combined_analytes[~combined_analytes.index.duplicated(keep="first")]

    logger.info(
        "Combined matrix: %d samples × %d analytes",
        combined_analytes.shape[0], combined_analytes.shape[1],
    )
    return combined_analytes, batch_series, combined_meta


# ---------------------------------------------------------------------------
# PCA helpers
# ---------------------------------------------------------------------------

def _generate_pca_plots(
    X_pre: pd.DataFrame,
    X_post: pd.DataFrame,
    batch_labels: pd.Series,
    group_labels: pd.Series,
    out_dir: str,
) -> None:
    """Save side-by-side pre/post ComBat PCA plots colored by batch and group."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA as _PCA
    except ImportError:
        logger.warning("matplotlib or scikit-learn not installed — PCA plots skipped.")
        return

    os.makedirs(out_dir, exist_ok=True)

    def _prep(X: pd.DataFrame) -> pd.DataFrame:
        Xp = X.copy().apply(lambda col: col.fillna(col.median(skipna=True)), axis=0)
        return Xp.dropna(axis=1, how="all")

    def _pca2d(X: pd.DataFrame):
        Xp = _prep(X)
        if Xp.shape[0] < 3 or Xp.shape[1] < 2:
            return None, None
        pca = _PCA(n_components=2)
        coords = pca.fit_transform(Xp.values)
        return (
            pd.DataFrame(coords, index=Xp.index, columns=["PC1", "PC2"]),
            pca.explained_variance_ratio_,
        )

    def _scatter(ax, df, labs, title, var):
        for u in sorted(labs.unique()):
            m = labs == u
            ax.scatter(df.loc[m, "PC1"], df.loc[m, "PC2"], label=str(u), alpha=0.7, s=50)
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")

    shared = X_pre.index.intersection(X_post.index)
    if len(shared) < 3:
        logger.warning("PCA skipped — too few shared samples (%d).", len(shared))
        return

    for color_by, labels, fname_suffix in [
        ("batch", batch_labels, "batch"),
        ("group", group_labels, "group"),
    ]:
        labs = labels.reindex(shared).astype(str)
        pre_coords,  pre_var  = _pca2d(X_pre.loc[shared])
        post_coords, post_var = _pca2d(X_post.loc[shared])
        if pre_coords is None or post_coords is None:
            logger.warning("PCA (%s) skipped — too few features.", color_by)
            continue

        x_all = np.concatenate([pre_coords["PC1"].values, post_coords["PC1"].values])
        y_all = np.concatenate([pre_coords["PC2"].values, post_coords["PC2"].values])
        xpad  = 0.05 * (x_all.max() - x_all.min() + 1e-9)
        ypad  = 0.05 * (y_all.max() - y_all.min() + 1e-9)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True, sharey=True)
        _scatter(axes[0], pre_coords,  labs,
                 f"PCA pre-ComBat (colored by {color_by})",  pre_var)
        _scatter(axes[1], post_coords, labs,
                 f"PCA post-ComBat (colored by {color_by})", post_var)
        for ax in axes:
            ax.set_xlim(x_all.min() - xpad, x_all.max() + xpad)
            ax.set_ylim(y_all.min() - ypad, y_all.max() + ypad)

        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper right", fontsize=9)
        fig.suptitle(
            f"Metabolomics Plasma — Pre/Post ComBat PCA (colored by {color_by})",
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 0.97, 0.95])

        out_path = os.path.join(out_dir, f"metabolomics_pca_pre_post_combat_{fname_suffix}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info("PCA plot saved → %s", out_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_cleaned_matrix(
    data_dir: str,
    output_full_dir: str,
    output_sliced_dir: str,
) -> None:

    os.makedirs(output_full_dir,   exist_ok=True)
    os.makedirs(output_sliced_dir, exist_ok=True)

    # --- Step 1-3: Load pre-normalized files, build SampleIDs, average duplicates ---
    logger.info("Loading pre-normalised sample files from %s …", data_dir)
    raw_analytes, batch_series, embedded_meta = _load_prenorm_samples(data_dir)

    # --- Step 4: Log2(x + 1) transform ---
    logger.info("Applying log2(x + 1) transformation …")
    df_log2 = pd.DataFrame(
        np.log2(raw_analytes.values.astype(float) + 1.0),
        index=raw_analytes.index,
        columns=raw_analytes.columns,
    )

    # --- Step 5: Build per-sample batch and group labels ---
    # Group comes from embedded metadata (already in the pre-norm files)
    if "group" in embedded_meta.columns:
        groups = embedded_meta["group"].reindex(df_log2.index)
        groups = groups.str.replace("sptb", "sPTB", regex=False)
    else:
        logger.warning("No 'group' column found in pre-norm files — group will be NaN.")
        groups = pd.Series(np.nan, index=df_log2.index, name="group")

    batch_labels = batch_series.reindex(df_log2.index)

    # Metadata matching diagnostics
    n_matched = groups.notna().sum()
    logger.info("Group labels matched: %d / %d samples", n_matched, len(groups))

    # --- Step 6: Filter samples missing group or batch ---
    has_group = groups.notna()
    if not has_group.all():
        logger.warning("Filtering %d samples without group label.", int((~has_group).sum()))
        df_log2      = df_log2.loc[has_group].copy()
        groups       = groups.loc[has_group]
        batch_labels = batch_labels.loc[has_group]
        embedded_meta = embedded_meta.reindex(df_log2.index)

    has_batch = batch_labels.notna()
    if not has_batch.all():
        logger.warning("Filtering %d samples without batch label.", int((~has_batch).sum()))
        df_log2      = df_log2.loc[has_batch].copy()
        groups       = groups.loc[has_batch]
        batch_labels = batch_labels.loc[has_batch]
        embedded_meta = embedded_meta.reindex(df_log2.index)

    # --- Step 7: Binary group labels for missingness Fisher test ---
    groups_binary = pd.Series(
        np.where(
            groups.astype(str).str.strip().str.upper() == "CONTROL",
            "Control",
            "Complication",
        ),
        index=groups.index,
        name="GroupBinary",
    )

    # --- Step 8: ComBat batch correction ---
    logger.info(
        "Applying ComBat batch correction (batches: %s) …",
        sorted(batch_labels.unique()),
    )
    X_norm = combat_normalize_wide(df_log2, batch_labels)

    # --- Step 9: Pre/post ComBat PCA plots ---
    logger.info("Generating pre/post ComBat PCA plots …")
    _generate_pca_plots(
        X_pre=df_log2,
        X_post=X_norm,
        batch_labels=batch_labels,
        group_labels=groups,
        out_dir=output_full_dir,
    )

    # --- Step 10: Missingness filter (≥20%) + Fisher/BH report ---
    logger.info(
        "Applying missingness filter (cutoff = %.0f%%) …",
        CUTOFF_PERCENT_MISSING * 100,
    )
    X_kept, dropped_report = missingness_filter_and_group_check(
        X_norm, groups_binary, cutoff=CUTOFF_PERCENT_MISSING, alpha_bh=0.05,
    )
    n_dropped = len(dropped_report)
    if n_dropped > 0:
        rep_path = os.path.join(
            output_full_dir, "metabolomics_plasma_dropped_missingness_report.csv"
        )
        dropped_report.to_csv(rep_path, index=False)
        logger.info("Dropped %d analytes (≥20%% missing) — report → %s", n_dropped, rep_path)
    logger.info(
        "Analytes after missingness filter: %d (dropped %d)",
        X_kept.shape[1], n_dropped,
    )

    # --- Step 11: Half-minimum imputation ---
    n_missing = int(X_kept.isna().sum().sum())
    logger.info("Missing values before imputation: %d", n_missing)
    X_final = half_min_impute_wide(X_kept)
    logger.info("Imputation complete.")

    # --- Step 12: Attach metadata columns ---
    logger.info("Attaching metadata columns …")

    # SubjectID: strip the timepoint suffix (last character of SampleID)
    subject_ids = X_final.index.to_series().str.replace(r"[A-E]$", "", regex=True)

    meta_out = pd.DataFrame(index=X_final.index)
    meta_out["SubjectID"]    = subject_ids.values
    meta_out["Batch"]        = batch_labels.reindex(X_final.index).values
    meta_out["Group"]        = groups.reindex(X_final.index).values
    if "subgroup" in embedded_meta.columns:
        meta_out["Subgroup"] = embedded_meta["subgroup"].reindex(X_final.index).values
    if "gestational_age" in embedded_meta.columns:
        meta_out["GestAgeDelivery"] = embedded_meta["gestational_age"].reindex(X_final.index).values
    if "gestational_age_at_collection" in embedded_meta.columns:
        meta_out["SampleGestAge"] = (
            embedded_meta["gestational_age_at_collection"].reindex(X_final.index).values
        )

    df_final = pd.concat([meta_out, X_final], axis=1)

    # --- Step 13: Save full matrix ---
    full_csv = os.path.join(output_full_dir, "metabolomics_plasma_cleaned_with_metadata.csv")
    df_final.to_csv(full_csv)
    logger.info(
        "Saved full matrix → %s  (%d rows × %d cols)", full_csv, *df_final.shape
    )

    # --- Step 14: Slice by timepoint and save ---
    logger.info("Slicing by timepoint suffix …")
    for tp in "ABCDE":
        mask = df_final.index.str.endswith(tp)
        df_tp = df_final[mask].copy()
        df_tp.index = df_tp.index.str[:-1]   # "DP3-0071A" → "DP3-0071"
        df_tp.index.name = "SampleID"
        out_path = os.path.join(
            output_sliced_dir, f"metabolomics_plasma_formatted_suffix_{tp}.csv"
        )
        df_tp.to_csv(out_path)
        logger.info("  Timepoint %s: %d samples → %s", tp, len(df_tp), out_path)

    logger.info(
        "Done. Final matrix: %d samples × %d analytes | dropped: %d | batches: %s",
        X_final.shape[0], X_final.shape[1], n_dropped, sorted(batch_labels.unique()),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Log2 + ComBat pipeline for pre-normalized metabolomics data."
    )
    p.add_argument(
        "--data-dir",
        default=os.path.join("data", "metabolomics_prenormalized"),
        help="Directory containing Samples_*_preNorm.csv files.",
    )
    p.add_argument(
        "--output-full-dir",
        default=os.path.join(
            "data", "cleaned", "metabolomics_combat", "normalized_full_results"
        ),
        help="Directory for the full merged matrix CSV and PCA plots.",
    )
    p.add_argument(
        "--output-sliced-dir",
        default=os.path.join(
            "data", "cleaned", "metabolomics_combat", "normalized_sliced_by_suffix"
        ),
        help="Directory for per-timepoint CSV slices.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    build_cleaned_matrix(
        data_dir=args.data_dir,
        output_full_dir=args.output_full_dir,
        output_sliced_dir=args.output_sliced_dir,
    )
