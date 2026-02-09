"""
Title: clean_proteomics_data_corrected_v2.py
Author: Samantha Piekos, Kayson Yao
Date: 02/08/2026
Description:
DP3 proteomics preprocessing for Olink Explore long-format exports.

KEY CORRECTIONS (v2):
1. Batch is now defined by "omics set#" column in metadata (not filename)
2. Metadata columns (group, subgroup, gest age del, omics set#) are merged into final output
3. Sample IDs with letters (e.g., DP3-0163E, DP3-0005A) represent DIFFERENT TIMEPOINTS/SAMPLES
   - These are LONGITUDINAL samples from the same subject ID
   - Each letter suffix is a distinct sample and should be kept separate
   - DO NOT remove letters or aggregate them
4. CONTROL_SAMPLE samples are excluded from final output (used only for normalization)
5. The metadata Excel has "ID" (subject) and "sample Id" (unique sample identifier with letters)
   - We match on "sample Id" which includes the letter suffixes

Workflow:
1) Load each Olink file in long format.
2) Standardize missing NPX values:
   - Convert blank/"NA"/"N/A"/0 in the NPX column to NaN.
3) QC masking:
   - If QC_Warning != PASS or Assay_Warning != PASS, set NPX to NaN.
4) Load metadata and extract batch (omics set#), group, subgroup, gest age del
   - Match on "sample Id" column (which includes letter suffixes like DP3-0005A)
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
9) PCA diagnostics (wide matrix):
   - PCA before quantile normalization (color points by Batch).
10) Quantile normalization (wide matrix) to adjust batch-related distributional differences:
    - Quantile-normalize per-sample NPX distributions using a NaN-aware implementation.
11) PCA diagnostics after quantile normalization (color points by Batch).
12) Impute remaining missing values (last):
    - Per assay, fill NaN with 0.5 x minimum observed value in that assay.
13) Output scaling:
    - Convert NPX (log2 scale) to linear positive scale via 2**NPX.
14) Merge metadata (Batch, Group, Subgroup, GestAgeDelivery) into final matrix
15) Save final cleaned matrix (wide, SampleID x [metadata + Assays]) to CSV.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


CUTOFF_PERCENT_MISSING = 0.25


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
        DataFrame with SampleID as index and columns: Batch, Group, Subgroup, GestAgeDelivery
    """
    # Load appropriate sheet
    if meta_type == "proteomics":
        sheet_name = "n=133 proteomics"
        # Auto-detect sample ID column if not specified
        if sample_id_col is None:
            sample_id_col = "sample Id"
    elif meta_type == "placenta":
        sheet_name = "n=133 placenta"
        # For placenta, we need to check which TP# column to use
        # We'll determine this based on the filename later
        if sample_id_col is None:
            sample_id_col = "TP# tissue"  # Default for placenta
    else:
        raise ValueError(f"meta_type must be 'proteomics' or 'placenta', got '{meta_type}'")
    
    meta = pd.read_excel(metadata_path, sheet_name=sheet_name)
    
    print(f"   Sheet '{sheet_name}' columns: {meta.columns.tolist()}")
    
    # For placenta sheet, need to handle potentially having both TP# tissue and TP# rnalater
    # We'll create a combined SampleID from both columns if they exist
    if meta_type == "placenta":
        if "TP# tissue" in meta.columns and "TP# rnalater" in meta.columns:
            # Combine both tissue and rnalater samples
            # First, create separate dataframes for each
            tissue_df = meta[meta["TP# tissue"].notna()].copy()
            tissue_df["SampleID"] = tissue_df["TP# tissue"]
            
            rnalater_df = meta[meta["TP# rnalater"].notna()].copy()
            rnalater_df["SampleID"] = rnalater_df["TP# rnalater"]
            
            # Combine them
            meta_subset = pd.concat([tissue_df, rnalater_df], axis=0, ignore_index=True)
            
            # Extract relevant columns
            meta_subset = meta_subset[["SampleID", batch_col, "group", "subgroup", "gest age del"]].copy()
        else:
            # Fallback if only one column exists
            required_cols = [sample_id_col, batch_col, "group", "subgroup", "gest age del"]
            missing_cols = [col for col in required_cols if col not in meta.columns]
            if missing_cols:
                raise ValueError(f"Metadata missing required columns: {missing_cols}. "
                                f"Available columns: {meta.columns.tolist()}")
            meta_subset = meta[[sample_id_col, batch_col, "group", "subgroup", "gest age del"]].copy()
            meta_subset = meta_subset.rename(columns={sample_id_col: "SampleID"})
    else:
        # Proteomics sheet - standard processing
        required_cols = [sample_id_col, batch_col, "group", "subgroup", "gest age del"]
        missing_cols = [col for col in required_cols if col not in meta.columns]
        if missing_cols:
            raise ValueError(f"Metadata missing required columns: {missing_cols}. "
                            f"Available columns: {meta.columns.tolist()}")
        meta_subset = meta[[sample_id_col, batch_col, "group", "subgroup", "gest age del"]].copy()
        meta_subset = meta_subset.rename(columns={sample_id_col: "SampleID"})
    
    # Rename columns for consistency
    meta_subset = meta_subset.rename(columns={
        batch_col: "Batch",
        "group": "Group",
        "subgroup": "Subgroup",
        "gest age del": "GestAgeDelivery"
    })
    
    # Clean up SampleID (strip whitespace, but keep letter suffixes)
    meta_subset["SampleID"] = meta_subset["SampleID"].astype(str).str.strip()
    
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
    
    # Clean up SampleID (strip whitespace but keep letter suffixes)
    if "SampleID" in df.columns:
        df["SampleID"] = df["SampleID"].astype(str).str.strip()
    
    return df


def combine_batches(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate multiple dataframes"""
    return pd.concat(df_list, axis=0, ignore_index=True)


def is_olink_control_sample(df_long: pd.DataFrame) -> pd.Series:
    """Identify Olink internal control samples"""
    return df_long["SampleID"].astype(str).str.strip().str.upper().str.startswith("CONTROL_SAMPLE")


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

    ctrl = df["SampleID"].astype(str).str.strip().str.upper().str.startswith("CONTROL_SAMPLE")

    panel_medians = (
        df.loc[ctrl]
        .groupby(["Panel", "Assay"])["NPX"]
        .median()
        .rename("panel_median")
        .reset_index()
    )
    if panel_medians.empty:
        raise ValueError("No Olink control samples found (SampleID startswith CONTROL_SAMPLE).")

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
    """Apply Benjamini-Hochberg FDR correction"""
    pvals = pvals.dropna().astype(float).clip(0, 1)
    if pvals.empty:
        return pd.Series(dtype=bool)

    p_sorted = pvals.sort_values()
    n = len(p_sorted)
    thresh = (np.arange(1, n + 1) / n) * alpha
    passed = p_sorted.values <= thresh

    out = pd.Series(False, index=pvals.index)
    if np.any(passed):
        k_star = np.max(np.where(passed)[0])
        out.loc[p_sorted.index[: k_star + 1]] = True
    return out


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
# PCA plotting
# -----------------------------
def plot_pca(X: pd.DataFrame, labels: pd.Series, out_png: str, title: str):
    """Generate PCA plot colored by labels (typically Batch)"""
    if X.shape[0] < 3 or X.shape[1] < 2:
        print("[WARN] PCA skipped (too few samples/features).")
        return

    Xp = X.copy()
    Xp = Xp.apply(lambda col: col.fillna(col.median(skipna=True)), axis=0)
    Xp = Xp.dropna(axis=1, how="all")
    if Xp.shape[0] < 3 or Xp.shape[1] < 2:
        print("[WARN] PCA skipped (not enough data after dropping all-NaN columns).")
        return

    labs = labels.reindex(Xp.index).astype(str)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(Xp.values)
    dfc = pd.DataFrame(coords, index=Xp.index, columns=["PC1", "PC2"])

    plt.figure(figsize=(10, 8))
    for u in labs.unique():
        m = labs == u
        plt.scatter(dfc.loc[m, "PC1"], dfc.loc[m, "PC2"], label=u, alpha=0.7, s=50)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


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
    meta_type: str = "proteomics",
    pca_dir: str | None = None
):
    """
    Complete proteomics data processing pipeline.
    
    Args:
        file_paths: List of paths to Olink CSV files
        output_csv: Path to save final cleaned matrix
        metadata_path: Path to metadata Excel file
        meta_type: 'proteomics' or 'placenta' - which sheet to use
        pca_dir: Directory to save PCA plots (optional)
    """
    print("=" * 80)
    print("PROTEOMICS DATA CLEANING PIPELINE - LONGITUDINAL DESIGN")
    print("=" * 80)
    print("\nNOTE: Letter suffixes (A, B, C, E, etc.) represent different timepoints/samples")
    print("      from the same subject and will be treated as separate samples.\n")
    
    # Step 1: Load metadata
    print("[1/15] Loading metadata...")
    metadata = load_metadata_with_batch(metadata_path, meta_type=meta_type)
    print(f"   Loaded metadata for {len(metadata)} samples")
    print(f"   Metadata columns: {metadata.columns.tolist()}")
    print(f"   Unique batches (Omics Sets): {metadata['Batch'].unique().tolist()}")
    print(f"   Sample examples: {list(metadata.index[:5])}")
    
    # Step 2: Load and process all Olink files
    print("\n[2/15] Loading and processing Olink files...")
    batches_long = []
    for fp in file_paths:
        print(f"   Processing: {os.path.basename(fp)}")
        batches_long.append(process_single_file(fp))
    df_long = combine_batches(batches_long)
    print(f"   Combined data: {len(df_long)} rows, {df_long['SampleID'].nunique()} unique samples")
    
    # Show sample ID examples from data
    olink_samples = df_long[~is_olink_control_sample(df_long)]['SampleID'].unique()[:10]
    print(f"   Sample examples from data: {list(olink_samples)}")
    
    # Step 3: Apply panel normalization using Olink controls
    print("\n[3/15] Applying panel normalization...")
    df_long = apply_panel_normalization_long(df_long)
    print("   Panel normalization complete")
    
    # Step 4: Exclude Olink internal controls
    print("\n[4/15] Removing Olink internal control samples...")
    ctrl_mask = is_olink_control_sample(df_long)
    n_controls = ctrl_mask.sum()
    df_long_bio = df_long.loc[~ctrl_mask].copy()
    print(f"   Removed {n_controls} control sample rows")
    print(f"   Remaining: {len(df_long_bio)} rows, {df_long_bio['SampleID'].nunique()} unique samples")
    
    # Step 5: Create wide matrix
    print("\n[5/15] Creating wide matrix (SampleID x Assay)...")
    X = df_long_bio.pivot_table(
        index="SampleID",
        columns="Assay",
        values="NPX",
        aggfunc="median"
    )
    print(f"   Wide matrix: {X.shape[0]} samples × {X.shape[1]} assays")
    
    # Check metadata matching
    matched = X.index.isin(metadata.index)
    print(f"   Samples in data matching metadata: {matched.sum()} / {len(X)}")
    if matched.sum() < len(X):
        unmatched = X.index[~matched]
        print(f"   WARNING: {len(unmatched)} samples in data not found in metadata:")
        print(f"            {list(unmatched[:5])}...")
        print(f"\n   Sample ID format in data: {list(X.index[:5])}")
        print(f"   Sample ID format in metadata: {list(metadata.index[:5])}")
        
        # For placenta data, suggest checking the ID mapping
        if meta_type == "placenta":
            print("\n   NOTE: For placenta samples, there may be an ID mapping issue.")
            print("         The 'ID' column in the metadata might map to the Olink SampleIDs.")
            print("         Attempting to use 'ID' column from metadata instead...")
            
            # Reload metadata using the ID column
            meta_raw = pd.read_excel(metadata_path, sheet_name="n=133 placenta")
            if "ID" in meta_raw.columns:
                # Try to match on ID column - filter out rows with no ID
                meta_with_id = meta_raw[meta_raw["ID"].notna()].copy()
                id_mapping = meta_with_id.set_index("ID")[["omics set#", "group", "subgroup", "gest age del"]].copy()
                id_mapping = id_mapping.rename(columns={
                    "omics set#": "Batch",
                    "group": "Group",
                    "subgroup": "Subgroup",
                    "gest age del": "GestAgeDelivery"
                })
                # Clean up index - remove spaces and standardize
                id_mapping.index = id_mapping.index.astype(str).str.strip()
                id_mapping = id_mapping[~id_mapping.index.duplicated(keep='first')]
                
                # Check if this matches better
                matched_id = X.index.isin(id_mapping.index)
                print(f"   Samples matching using 'ID' column: {matched_id.sum()} / {len(X)}")
                
                if matched_id.sum() > matched.sum():
                    print(f"   ✓ Better match found! Using 'ID' column for placenta metadata.")
                    metadata = id_mapping
                else:
                    print(f"   ✗ 'ID' column didn't improve matching.")
                    print(f"   Sample IDs in data: {list(X.index[:5])}")
                    print(f"   Sample IDs in 'ID' column: {list(id_mapping.index[:5])}")
    
    # Re-check matching after potential fix
    matched = X.index.isin(metadata.index)
    n_matched = matched.sum()
    print(f"\n   Final metadata matching: {n_matched} / {len(X)} samples matched")
    if n_matched == 0:
        print("\n   ⚠️  ERROR: No samples matched between data and metadata!")
        print("   The pipeline will continue, but results may not be reliable.")
        print("   Please verify the sample ID mapping between Olink files and metadata Excel.")
    elif n_matched < len(X):
        print(f"\n   ⚠️  WARNING: {len(X) - n_matched} samples have no metadata and will be treated as 'Complication' group.")
    
    # Step 6: Create binary group labels for missingness check
    print("\n[6/15] Creating binary group labels...")
    # Get group labels from metadata
    groups = metadata["Group"].reindex(X.index)
    
    # Filter out samples with NaN/blank group labels (no metadata match)
    has_metadata = groups.notna()
    n_no_metadata = (~has_metadata).sum()
    
    if n_no_metadata > 0:
        print(f"   ⚠️  Filtering out {n_no_metadata} samples with no metadata (NaN group values)")
        print(f"   These samples will be excluded from analysis until you clarify with your professor.")
        # Remove samples without metadata from the matrix
        X = X.loc[has_metadata].copy()
        groups = groups.loc[has_metadata]
        print(f"   Remaining samples after filtering: {len(X)}")
    
    # Make binary: Control vs everything else
    # Only applies to samples that have valid metadata
    groups_binary = pd.Series(
        np.where(groups.astype(str).str.strip().str.upper() == "CONTROL", "Control", "Complication"),
        index=groups.index,
        name="GroupBinary"
    )
    print(f"   Group distribution: {groups_binary.value_counts().to_dict()}")
    
    # Step 7: Missingness filter
    print("\n[7/15] Applying missingness filter...")
    X_kept, dropped_report = missingness_filter_and_group_check(
        X, groups_binary, cutoff=CUTOFF_PERCENT_MISSING, alpha_bh=0.05
    )
    print(f"   Kept {X_kept.shape[1]} assays, dropped {len(dropped_report)} assays")
    
    if not dropped_report.empty:
        rep_path = os.path.splitext(output_csv)[0] + "_dropped_missingness_report.csv"
        dropped_report.to_csv(rep_path, index=False)
        print(f"   Saved missingness report to: {rep_path}")
    
    # Step 8: Get batch labels from metadata for PCA
    print("\n[8/15] Extracting batch labels for PCA...")
    # Only get batch for samples that remain after filtering
    sample_to_batch = metadata["Batch"].reindex(X_kept.index)
    print(f"   Batch distribution: {sample_to_batch.value_counts().to_dict()}")
    
    # Step 9: PCA before quantile normalization
    if pca_dir:
        print("\n[9/15] Generating PCA (pre-quantile normalization)...")
        os.makedirs(pca_dir, exist_ok=True)
        plot_pca(
            X_kept,
            sample_to_batch,
            os.path.join(pca_dir, "pca_pre_quantile_norm.png"),
            "PCA (panel-normalized, pre-quantile normalization)"
        )
        print(f"   Saved to: {os.path.join(pca_dir, 'pca_pre_quantile_norm.png')}")
    else:
        print("\n[9/15] Skipping PCA (pre-quantile normalization) - no output directory specified")
    
    # Step 10: Quantile normalization
    print("\n[10/15] Applying quantile normalization...")
    X_qn = quantile_normalize_wide(X_kept)
    print("   Quantile normalization complete")
    
    # Step 11: PCA after quantile normalization
    if pca_dir:
        print("\n[11/15] Generating PCA (post-quantile normalization)...")
        plot_pca(
            X_qn,
            sample_to_batch,
            os.path.join(pca_dir, "pca_post_quantile_norm.png"),
            "PCA (post quantile normalization)"
        )
        print(f"   Saved to: {os.path.join(pca_dir, 'pca_post_quantile_norm.png')}")
    else:
        print("\n[11/15] Skipping PCA (post-quantile normalization) - no output directory specified")
    
    # Step 12: Imputation
    print("\n[12/15] Imputing missing values (0.5 × minimum per assay)...")
    X_final = half_min_impute_wide(X_qn)
    print("   Imputation complete")
    
    # Step 13: Convert to linear scale
    print("\n[13/15] Converting NPX (log2) to linear scale (2^NPX)...")
    X_final_linear = np.power(2.0, X_final)
    print("   Conversion complete")
    
    # Step 14: Merge metadata
    print("\n[14/15] Merging metadata into final matrix...")
    # Reindex metadata to match final matrix samples
    metadata_aligned = metadata.reindex(X_final_linear.index)
    
    # Combine metadata with protein data
    final_output = pd.concat([metadata_aligned, X_final_linear], axis=1)
    print(f"   Final matrix: {final_output.shape[0]} samples × {final_output.shape[1]} columns")
    print(f"   Metadata columns: {metadata_aligned.columns.tolist()}")
    print(f"   Protein assays: {X_final_linear.shape[1]}")
    
    # Step 15: Save output
    print("\n[15/15] Saving final cleaned matrix...")
    final_output.to_csv(output_csv, index=True)
    print(f"   Saved to: {output_csv}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nFinal output summary:")
    print(f"  - Samples: {final_output.shape[0]}")
    print(f"  - Total columns: {final_output.shape[1]}")
    print(f"  - Metadata columns: {len(metadata_aligned.columns)}")
    print(f"  - Protein assays: {X_final_linear.shape[1]}")
    print(f"  - Output file: {output_csv}")
    if pca_dir:
        print(f"  - PCA plots: {pca_dir}")
    
    # Show metadata coverage
    n_missing_meta = metadata_aligned.isna().all(axis=1).sum()
    if n_missing_meta > 0:
        print(f"\n  WARNING: {n_missing_meta} samples have no metadata (missing from Excel)")


if __name__ == "__main__":
    # Example usage
    wkdir = os.getcwd()
    data_dir = os.path.join(wkdir, "data", "proteomics")
    output_dir = os.path.join(wkdir, "data", "cleaned", "proteomics")
    os.makedirs(output_dir, exist_ok=True)

    # Metadata file path
    metadata_path = os.path.join(wkdir, 'data', "dp3 master table v2.xlsx")

    # Collect all CSV files and separate by type
    plasma_files = []
    placenta_files = []
    
    for fn in os.listdir(data_dir):
        if fn.endswith(".csv"):
            full_path = os.path.join(data_dir, fn)
            # Determine file type based on filename
            fn_lower = fn.lower()
            if "plasma" in fn_lower:
                plasma_files.append(full_path)
            elif "placenta" in fn_lower or "tissue" in fn_lower or "lysate" in fn_lower:
                placenta_files.append(full_path)
            else:
                print(f"WARNING: Cannot determine type for file: {fn}")
                print(f"         Please add it to plasma_files or placenta_files manually")

    # Sort for deterministic processing
    plasma_files = sorted(plasma_files)
    placenta_files = sorted(placenta_files)

    print("=" * 80)
    print("FILE CLASSIFICATION")
    print("=" * 80)
    print(f"\nPlasma files (will use 'n=133 proteomics' sheet):")
    for f in plasma_files:
        print(f"  - {os.path.basename(f)}")
    
    print(f"\nPlacenta/Tissue files (will use 'n=133 placenta' sheet):")
    for f in placenta_files:
        print(f"  - {os.path.basename(f)}")
    print("\n" + "=" * 80 + "\n")

    # Process plasma files
    if plasma_files:
        print("\n" + "=" * 80)
        print("PROCESSING PLASMA FILES")
        print("=" * 80 + "\n")
        out_csv_plasma = os.path.join(output_dir, "proteomics_plasma_cleaned_with_metadata.csv")
        pca_dir_plasma = os.path.join(output_dir, "pca_plasma")
        process_all_files(plasma_files, out_csv_plasma, metadata_path, 
                         meta_type="proteomics", pca_dir=pca_dir_plasma)

    # Process placenta files
    if placenta_files:
        print("\n" + "=" * 80)
        print("PROCESSING PLACENTA/TISSUE FILES")
        print("=" * 80 + "\n")
        out_csv_placenta = os.path.join(output_dir, "proteomics_placenta_cleaned_with_metadata.csv")
        pca_dir_placenta = os.path.join(output_dir, "pca_placenta")
        process_all_files(placenta_files, out_csv_placenta, metadata_path, 
                         meta_type="placenta", pca_dir=pca_dir_placenta)

    # Summary
    print("\n" + "=" * 80)
    print("ALL PROCESSING COMPLETE")
    print("=" * 80)
    if plasma_files:
        print(f"\n✓ Plasma output: {os.path.join(output_dir, 'proteomics_plasma_cleaned_with_metadata.csv')}")
    if placenta_files:
        print(f"✓ Placenta output: {os.path.join(output_dir, 'proteomics_placenta_cleaned_with_metadata.csv')}")
    print()