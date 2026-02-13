"""
Title: clean_proteomics_data_corrected_v2.py
Author: Samantha Piekos, Kayson Yao
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
8) Missingness filter on the combined wide matrix (pre-imputation):
   - Keep assays with missing fraction < 25%.
   - For assays failing the cutoff, compute group-wise missingness (Control vs Complication),
     test imbalance using Fisher's exact test, and apply Benjamini-Hochberg correction.
   - Save a dropped-assay missingness report CSV.
9) Quantile normalization (wide matrix) to adjust batch-related distributional differences:
    - Quantile-normalize per-sample NPX distributions using a NaN-aware implementation.
10) Impute remaining missing values (last):
    - Per assay, fill NaN with 0.5 x minimum observed value in that assay.
11) Output scaling:
    - Convert NPX (log2 scale) to linear positive scale via 2**NPX.
12) Merge metadata (Batch, Group, Subgroup, GestAgeDelivery) into final matrix
13) Save final cleaned matrix (wide, SampleID x [metadata + Assays]) to CSV.

"""
# Move missingness checking after batch normalization (9 -> 8 -> break by timepoint)

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, false_discovery_control

try:
    from statsmodels.stats.multitest import multipletests  # Optional dependency
except Exception:
    multipletests = None



CUTOFF_PERCENT_MISSING = 0.25
CONTROL_SAMPLE_PREFIXES = ("CONTROL", "NEG", "PLATE")


# -----------------------------
# Metadata Loading
# -----------------------------
def load_metadata_with_batch(
    metadata_path: str,
    meta_type: str = "proteomics",
    sample_id_col: str = None,
    batch_col: str = "omics set#"
) -> pd.DataFrame:
    """
    Load metadata from Excel file and extract:
    - Batch (from 'omics set#' column)
    - Group
    - Subgroup  
    - Gestational age at delivery (gest age del)
    - Sample gestational age (sample gest Age) - for proteomics only
    
    IMPORTANT: The sample_id_col contains the FULL sample identifier including
    letter suffixes (e.g., DP3-0005A, DP3-0005B) which represent different
    timepoints/samples from the same subject.
    
    Returns a DataFrame with SampleID (with letters) as index and metadata columns.
    
    Args:
        metadata_path: path to the metadata Excel file
        meta_type: "proteomics" or "placenta" - which sheet to load
        sample_id_col: column name containing unique sample identifiers 
                      (default None = auto-detect based on meta_type)
        batch_col: column name containing batch/omics set info (default "omics set#")
    
    Returns:
        DataFrame with SampleID as index and columns: Batch, Group, Subgroup, GestAgeDelivery, SampleGestAge (proteomics only)
    """
    # Load appropriate sheet and determine sample ID column
    if meta_type == "proteomics":
        sheet_name = "n=133 proteomics"
        if sample_id_col is None:
            sample_id_col = "sample Id"
    elif meta_type == "placenta":
        sheet_name = "n=133 placenta"
        if sample_id_col is None:
            sample_id_col = "ID"  # Use ID column for placenta - matches Olink sample IDs
    else:
        raise ValueError(f"meta_type must be 'proteomics' or 'placenta', got '{meta_type}'")
    
    meta = pd.read_excel(metadata_path, sheet_name=sheet_name)
    
    # Build list of required columns based on meta_type
    required_cols = [sample_id_col, batch_col, "group", "subgroup", "gest age del"]
    
    # Add sample gest Age only for proteomics (not available in placenta sheet)
    if meta_type == "proteomics":
        if "sample gest Age" in meta.columns:
            required_cols.append("sample gest Age")
        else:
            print("   WARNING: 'sample gest Age' column not found in proteomics sheet")
    
    # Check required columns exist
    missing_cols = [col for col in required_cols if col not in meta.columns]
    if missing_cols:
        raise ValueError(f"Metadata missing required columns: {missing_cols}. "
                        f"Available columns: {meta.columns.tolist()}")
    
    # Extract relevant columns
    meta_subset = meta[required_cols].copy()
    
    # Filter out rows where sample ID is missing/blank (including whitespace-only)
    sid_clean = meta_subset[sample_id_col].astype(str).str.strip()
    meta_subset = meta_subset[meta_subset[sample_id_col].notna() & (sid_clean != "")].copy()
    
    # Rename columns for consistency
    rename_dict = {
        sample_id_col: "SampleID",
        batch_col: "Batch",
        "group": "Group",
        "subgroup": "Subgroup",
        "gest age del": "GestAgeDelivery"
    }
    
    if meta_type == "proteomics" and "sample gest Age" in meta_subset.columns:
        rename_dict["sample gest Age"] = "SampleGestAge"
    
    meta_subset = meta_subset.rename(columns=rename_dict)
    
    # Clean up SampleID
    meta_subset["SampleID"] = (
        meta_subset["SampleID"]
        .astype(str)
        .str.strip()  # Remove leading/trailing spaces
        .str.replace(r'\s+', ' ', regex=True)  # Normalize multiple spaces to single space
    )
    
    # Set SampleID as index
    meta_subset = meta_subset.set_index("SampleID")
    
    # Remove duplicates if any (keep first occurrence)
    meta_subset = meta_subset[~meta_subset.index.duplicated(keep='first')]
    
    return meta_subset


# -----------------------------
# Standardize missing + QC mask
# -----------------------------
def standardize_missing_npx(df: pd.DataFrame) -> pd.DataFrame:
    """Convert various missing value representations to NaN"""
    df = df.copy()
    if "NPX" not in df.columns:
        return df

    s = df["NPX"]
    if s.dtype == object:
        s = s.replace(["", " ", "NA", "N/A", "na", "n/a", "NaN", "nan", None], np.nan)
    s = pd.to_numeric(s, errors="coerce")
    s = s.mask(s == 0, np.nan)
    df["NPX"] = s
    return df


def qc_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Set NPX to NaN for samples that fail QC warnings"""
    df = df.copy()
    if "NPX" not in df.columns:
        return df

    if "QC_Warning" in df.columns:
        bad = df["QC_Warning"].astype(str).str.upper() != "PASS"
        df.loc[bad, "NPX"] = np.nan

    if "Assay_Warning" in df.columns:
        bad = df["Assay_Warning"].astype(str).str.upper() != "PASS"
        df.loc[bad, "NPX"] = np.nan

    return df


def process_single_file(file_input: str) -> pd.DataFrame:
    """
    Process a single Olink CSV file:
    1. Load data
    2. Standardize missing values
    3. Apply QC masking
    4. Keep SampleID as-is (including letter suffixes for longitudinal samples)
    """
    df = pd.read_csv(file_input, sep=";")
    df = standardize_missing_npx(df)
    df = qc_mask(df)
    
    # Clean up SampleID
    if "SampleID" in df.columns:
        df["SampleID"] = (
            df["SampleID"]
            .astype(str)
            .str.strip()  # Remove leading/trailing spaces
            .str.replace(r'\s+', ' ', regex=True)  # Normalize multiple spaces to single space
        )
    
    return df


def combine_batches(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate multiple dataframes"""
    return pd.concat(df_list, axis=0, ignore_index=True)


def is_olink_control_sample(df_long: pd.DataFrame) -> pd.Series:
    """Identify Olink technical control samples from SampleID prefixes."""
    sid = df_long["SampleID"].astype(str).str.strip().str.upper()
    return sid.str.startswith(CONTROL_SAMPLE_PREFIXES)


def is_olink_control_assay(df_long: pd.DataFrame) -> pd.Series:
    """Identify technical control assays (e.g., incubation/amplification controls)."""
    assay = df_long["Assay"].astype(str).str.strip().str.lower()
    return assay.str.contains(r"\bcontrol\b", regex=True)


# -----------------------------
# Panel normalization (long)
# -----------------------------
def apply_panel_normalization_long(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Panel normalization using Olink internal control samples.
    This normalizes across panels to adjust for systematic differences.
    """
    df = df_long.copy()
    required = {"SampleID", "Panel", "Assay", "NPX"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Panel normalization requires columns: {sorted(required)}; missing: {sorted(missing)}")

    ctrl = is_olink_control_sample(df)

    panel_medians = (
        df.loc[ctrl]
        .groupby(["Panel", "Assay"])["NPX"]
        .median()
        .rename("panel_median")
        .reset_index()
    )
    if panel_medians.empty:
        raise ValueError(
            "No Olink control samples found "
            f"(SampleID startswith one of {CONTROL_SAMPLE_PREFIXES})."
        )

    global_ref = (
        panel_medians.groupby("Assay")["panel_median"]
        .median()
        .rename("global_median")
        .reset_index()
    )

    adj = panel_medians.merge(global_ref, on="Assay", how="left")
    adj["adjustment"] = adj["global_median"] - adj["panel_median"]

    df = df.merge(adj[["Panel", "Assay", "adjustment"]], on=["Panel", "Assay"], how="left")
    df["adjustment"] = df["adjustment"].fillna(0.0)
    df["NPX"] = df["NPX"] + df["adjustment"]
    df.drop(columns=["adjustment"], inplace=True)

    return df


# -----------------------------
# Missingness + group check
# -----------------------------
def benjamini_hochberg_rejections(pvals: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Apply Benjamini-Hochberg FDR correction using package implementations."""
    pvals = pvals.dropna().astype(float).clip(0, 1)
    if pvals.empty:
        return pd.Series(dtype=bool)

    if multipletests is not None:
        rejected, _, _, _ = multipletests(pvals.values, alpha=alpha, method="fdr_bh")
        return pd.Series(rejected, index=pvals.index, dtype=bool)

    pvals_adj = false_discovery_control(pvals.values, method="bh")
    return pd.Series(pvals_adj <= alpha, index=pvals.index, dtype=bool)


def missingness_filter_and_group_check(
    X: pd.DataFrame,
    groups: pd.Series | None,
    cutoff: float = CUTOFF_PERCENT_MISSING,
    alpha_bh: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter out assays with high missingness and test for group imbalance.
    """
    miss_frac = X.isna().mean(axis=0)
    keep = miss_frac[miss_frac < cutoff].index
    drop = miss_frac[miss_frac >= cutoff].index

    X_kept = X.loc[:, keep].copy()

    if len(drop) == 0:
        return X_kept, pd.DataFrame(columns=["Assay", "missing_frac", "p_value", "bh_reject"])

    report_rows = []
    pvals = {}

    if groups is None:
        for assay in drop:
            report_rows.append({"Assay": assay, "missing_frac": float(miss_frac[assay]),
                                "p_value": np.nan, "bh_reject": False})
        return X_kept, pd.DataFrame(report_rows).sort_values("missing_frac", ascending=False)

    g = groups.reindex(X.index)
    valid = g.notna()
    g = g[valid]
    Xg = X.loc[valid, drop]

    uniq = pd.unique(g)
    if len(uniq) != 2:
        print(f"   WARNING: Expected 2 groups for Fisher's test, but got {len(uniq)}: {uniq}")
        print(f"   This may be because samples didn't match metadata properly.")
        print(f"   Skipping group-wise missingness testing.")
        for assay in drop:
            report_rows.append({
                "Assay": assay,
                "missing_frac": float(miss_frac[assay]),
                "p_value": np.nan,
                "bh_reject": False
            })
        report = pd.DataFrame(report_rows).sort_values("missing_frac", ascending=False)
        return X_kept, report

    g0, g1 = uniq[0], uniq[1]

    for assay in drop:
        miss = Xg[assay].isna()
        a = int(((g == g0) & miss).sum())
        b = int(((g == g0) & (~miss)).sum())
        c = int(((g == g1) & miss).sum())
        d = int(((g == g1) & (~miss)).sum())

        if (a + b == 0) or (c + d == 0):
            p = 1.0
        else:
            _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")

        pvals[assay] = p
        report_rows.append({
            "Assay": assay,
            "missing_frac": float(miss_frac[assay]),
            f"{g0}_missing": a, f"{g0}_observed": b,
            f"{g1}_missing": c, f"{g1}_observed": d,
            "p_value": p
        })

    report = pd.DataFrame(report_rows).set_index("Assay")
    reject = benjamini_hochberg_rejections(pd.Series(pvals), alpha=alpha_bh)
    report["bh_reject"] = reject.reindex(report.index).fillna(False).astype(bool)
    report = report.sort_values(["bh_reject", "p_value"], ascending=[False, True]).reset_index()

    return X_kept, report


# -----------------------------
# Quantile normalization (wide)
# -----------------------------
def quantile_normalize_wide(X: pd.DataFrame) -> pd.DataFrame:
    """
    Quantile normalize samples (rows) to have the same distribution.
    Handles missing values by only normalizing observed values.
    """
    A = X.to_numpy(dtype=float)
    n, p = A.shape
    out = np.full_like(A, np.nan)

    sorted_vals, sorted_cols, counts = [], [], []
    for i in range(n):
        row = A[i, :]
        mask = np.isfinite(row)
        vals = row[mask]
        cols = np.where(mask)[0]
        if vals.size == 0:
            sorted_vals.append(np.array([], dtype=float))
            sorted_cols.append(np.array([], dtype=int))
            counts.append(0)
            continue
        order = np.argsort(vals)
        sorted_vals.append(vals[order])
        sorted_cols.append(cols[order])
        counts.append(vals.size)

    max_k = max(counts) if counts else 0
    if max_k == 0:
        return X.copy()

    rank_means = np.zeros(max_k, dtype=float)
    for k in range(max_k):
        v = [sorted_vals[i][k] for i in range(n) if counts[i] > k]
        rank_means[k] = float(np.mean(v))

    for i in range(n):
        if counts[i] == 0:
            continue
        out[i, sorted_cols[i]] = rank_means[: counts[i]]

    return pd.DataFrame(out, index=X.index, columns=X.columns)


# -----------------------------
# Imputation
# -----------------------------
def half_min_impute_wide(X: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with 0.5 * minimum observed value per assay"""
    Xi = X.copy()
    for col in Xi.columns:
        s = Xi[col]
        if s.notna().any():
            Xi[col] = s.fillna(0.5 * s.min(skipna=True))
    return Xi


# -----------------------------
# Main wrapper
# -----------------------------
def process_all_files(
    file_paths: list[str],
    output_csv: str,
    metadata_path: str,
    meta_type: str = "proteomics"
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
    print(f"{run_label}: start")

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
    n_controls = ctrl_mask.sum()
    df_long_bio = df_long.loc[~ctrl_mask].copy()

    # Remove assay-level technical controls from downstream feature matrix.
    assay_ctrl_mask = is_olink_control_assay(df_long_bio)
    n_assay_ctrl_rows = int(assay_ctrl_mask.sum())
    if n_assay_ctrl_rows > 0:
        n_assay_ctrl_names = int(df_long_bio.loc[assay_ctrl_mask, "Assay"].nunique())
        df_long_bio = df_long_bio.loc[~assay_ctrl_mask].copy()
        print(
            f"[{run_label}] removed {n_assay_ctrl_rows} rows from {n_assay_ctrl_names} control assays "
            "(assay name contains 'control')."
        )

    # 5) Build wide matrix
    X = df_long_bio.pivot_table(
        index="SampleID",
        columns="Assay",
        values="NPX",
        aggfunc="median"
    )

    # Metadata matching diagnostics (warnings only)
    matched = X.index.isin(metadata.index)
    n_matched = matched.sum()
    if n_matched < len(X):
        unmatched = X.index[~matched]
        print(f"warning: {len(unmatched)} samples missing metadata. examples={list(unmatched[:5])}")

        # Check if removing spaces improves matching (compact diagnostic hint)
        data_no_space = X.index.str.replace(' ', '', regex=False)
        meta_no_space = metadata.index.str.replace(' ', '', regex=False)
        matched_no_space = data_no_space.isin(meta_no_space)
        if matched_no_space.sum() > matched.sum():
            print(f"hint: stripping internal spaces would improve matching ({matched_no_space.sum()} vs {matched.sum()}).")

    if n_matched == 0:
        print(f"warning: no metadata matches found; downstream results may be unreliable.")

    # 6) Group labels for missingness check
    groups = metadata["Group"].reindex(X.index)
    has_metadata = groups.notna()
    n_no_metadata = (~has_metadata).sum()
    if n_no_metadata > 0:
        print(f"warning: filtering {n_no_metadata} samples without metadata group.")
        X = X.loc[has_metadata].copy()
        groups = groups.loc[has_metadata]

    # Binary labels: Control vs everything else
    groups_binary = pd.Series(
        np.where(groups.astype(str).str.strip().str.upper() == "CONTROL", "Control", "Complication"),
        index=groups.index,
        name="GroupBinary"
    )

    # Align metadata to remaining samples
    metadata = metadata.reindex(X.index)

    # 7) Missingness filter + report
    X_kept, dropped_report = missingness_filter_and_group_check(
        X, groups_binary, cutoff=CUTOFF_PERCENT_MISSING, alpha_bh=0.05
    )
    if not dropped_report.empty:
        rep_path = os.path.splitext(output_csv)[0] + "_dropped_missingness_report.csv"
        dropped_report.to_csv(rep_path, index=False)

    # 8) Quantile normalization
    X_qn = quantile_normalize_wide(X_kept)

    # 9) Imputation
    X_final = half_min_impute_wide(X_qn)

    # 10) Convert to linear scale
    X_final_linear = np.power(2.0, X_final)

    # 11) Merge metadata
    metadata_aligned = metadata.reindex(X_final_linear.index)

    if meta_type == "proteomics":
        subject_ids = X_final_linear.index.to_series().str.replace(r'\s*[A-Z]+$', '', regex=True)
        metadata_aligned.insert(0, 'SubjectID', subject_ids.values)
        final_output = pd.concat([metadata_aligned, X_final_linear], axis=1)
    else:
        final_output = pd.concat([metadata_aligned, X_final_linear], axis=1)

    # 12) Save output
    final_output.to_csv(output_csv, index=True)

    n_dropped = len(dropped_report)
    msg = (
        f"{run_label}: done | samples={final_output.shape[0]} "
        f"| assays={X_final_linear.shape[1]} | dropped_assays={n_dropped} "
        f"| metadata_matched={n_matched}/{len(matched)} | output={output_csv}"
    )
    print(msg)
    if n_dropped > 0:
        print(f"missingness report: {os.path.splitext(output_csv)[0]}_dropped_missingness_report.csv")

    n_missing_meta = metadata_aligned.isna().all(axis=1).sum()
    if n_missing_meta > 0:
        print(f"warning: {n_missing_meta} samples have no metadata.")


if __name__ == "__main__":
    def _collect_files_by_type(data_dir: str) -> tuple[list[str], list[str]]:
        plasma_files = []
        placenta_files = []
        for fn in os.listdir(data_dir):
            if not fn.endswith(".csv"):
                continue
            full_path = os.path.join(data_dir, fn)
            fn_lower = fn.lower()
            if "plasma" in fn_lower:
                plasma_files.append(full_path)
            elif "placenta" in fn_lower or "tissue" in fn_lower or "lysate" in fn_lower:
                placenta_files.append(full_path)
            else:
                print(f"WARNING: Cannot determine type for file: {fn}")
        return sorted(plasma_files), sorted(placenta_files)

    def _run_default_mode() -> None:
        """Backward-compatible run mode (current behavior)."""
        wkdir = os.getcwd()
        data_dir = os.path.join(wkdir, "data", "proteomics")
        output_dir = os.path.join(wkdir, "data", "cleaned", "proteomics")
        os.makedirs(output_dir, exist_ok=True)
        metadata_path = os.path.join(wkdir, "data", "dp3 master table v2.xlsx")

        plasma_files, placenta_files = _collect_files_by_type(data_dir)
        print(f" files | plasma={len(plasma_files)} | placenta={len(placenta_files)}")

        if plasma_files:
            out_csv_plasma = os.path.join(output_dir, "proteomics_plasma_cleaned_with_metadata.csv")
            process_all_files(plasma_files, out_csv_plasma, metadata_path, meta_type="proteomics")

        if placenta_files:
            out_csv_placenta = os.path.join(output_dir, "proteomics_placenta_cleaned_with_metadata.csv")
            process_all_files(placenta_files, out_csv_placenta, metadata_path, meta_type="placenta")

        print("all processing complete")

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
        # Keep current behavior if no CLI args are provided.
        if len(sys.argv) == 1:
            _run_default_mode()
            return

        parser = _build_parser()
        args = parser.parse_args()

        wkdir = os.getcwd()
        metadata_path = args.metadata_path or os.path.join(wkdir, "data", "dp3 master table v2.xlsx")

        if args.mode == "auto":
            data_dir = args.data_dir or os.path.join(wkdir, "data", "proteomics")
            output_dir = args.output_dir or os.path.join(wkdir, "data", "cleaned", "proteomics")
            os.makedirs(output_dir, exist_ok=True)

            plasma_files, placenta_files = _collect_files_by_type(data_dir)
            print(f"files | plasma={len(plasma_files)} | placenta={len(placenta_files)}")

            if plasma_files:
                out_csv_plasma = os.path.join(output_dir, "proteomics_plasma_cleaned_with_metadata.csv")
                process_all_files(plasma_files, out_csv_plasma, metadata_path, meta_type="proteomics")

            if placenta_files:
                out_csv_placenta = os.path.join(output_dir, "proteomics_placenta_cleaned_with_metadata.csv")
                process_all_files(placenta_files, out_csv_placenta, metadata_path, meta_type="placenta")

            print("all processing complete")
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
