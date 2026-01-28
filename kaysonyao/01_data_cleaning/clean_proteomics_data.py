"""
Title: clean_proteomics_data.py (DP3 updated)
Author: Samantha Piekos, Kayson Yao
Date: 01/27/2026 (updated per PI notes)
Description:
DP3 proteomics preprocessing for Olink Explore long-format exports.

Workflow:
1) Standardize missing values for analyte measurements (blank/"NA"/0 -> NaN on NPX only)
2) QC mask: if QC_Warning != PASS or Assay_Warning != PASS -> NPX = NaN
3) Combine batches (each file = batch)
4) Filter proteins with <20% missingness (on combined data, before imputation)
   - For proteins failing cutoff, compute missingness proportion by Group and test imbalance (Fisher) + BH
5) Panel normalization (using CONTROL_SAMPLE medians within each Panel)
6) PCA before batch correction (color = batch)
7) Batch correction using replicates across batches (if duplicates exist)
8) PCA after batch correction
9) 1/2-minimum imputation (last)
10) Save cleaned matrix
"""

import os
import math
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

CUTOFF_PERCENT_MISSING = 0.25 # Move threshold to 25%

# -----------------------------
# Standardize missing + QC mask + batch tagging (long)
# -----------------------------
def standardize_missing_npx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert blanks/'NA'/'N/A'/0 in NPX to NaN.
    Applies ONLY to the NPX column.
    """
    df = df.copy()
    if "NPX" not in df.columns:
        return df

    # Convert to string for pattern replacement, then coerce to numeric
    s = df["NPX"]

    # Handle string NA-like values
    if s.dtype == object:
        s = s.replace(
            to_replace=["", " ", "NA", "N/A", "na", "n/a", "NaN", "nan", None],
            value=np.nan,
        )

    # Coerce numeric
    s = pd.to_numeric(s, errors="coerce")

    # Treat 0 as missing
    s = s.mask(s == 0, np.nan)

    df["NPX"] = s
    return df


def qc_mask(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mask NPX values (set to NaN) when QC_Warning or Assay_Warning are not PASS.
    Applies ONLY to NPX.
    """
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


def infer_batch_id_from_filename(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


# -----------------------------
# Missingness + group check (binary)
# -----------------------------
def benjamini_hochberg_rejections(pvals: pd.Series, alpha: float = 0.05) -> pd.Series:
    """
    BH procedure returning a boolean Series indicating rejected hypotheses.
    pvals: Series indexed by feature name
    """
    pvals = pvals.dropna().astype(float).clip(0, 1)
    if pvals.empty:
        return pd.Series(dtype=bool)

    p_sorted = pvals.sort_values()
    n = len(p_sorted)
    thresh = (np.arange(1, n + 1) / n) * alpha
    passed = p_sorted.values <= thresh

    if not np.any(passed):
        return pd.Series(False, index=pvals.index)

    k_star = np.max(np.where(passed)[0])  # 0-based
    reject_features = p_sorted.index[: k_star + 1]
    out = pd.Series(False, index=pvals.index)
    out.loc[reject_features] = True
    return out


def missingness_filter_and_group_check(
    X: pd.DataFrame,
    groups: pd.Series | None,
    cutoff: float = CUTOFF_PERCENT_MISSING,
    alpha_bh: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    X: wide matrix (rows=SampleID, cols=Assay), values=NPX
    groups: Series indexed by SampleID, values in {0,1} or {control, complication}
            Used ONLY to compute missingness proportions and Fisher/BH on discarded assays.

    Returns:
      X_kept: filtered matrix with assays missing < cutoff
      dropped_report: DataFrame with missingness stats and p/q for dropped assays
    """
    miss_frac = X.isna().mean(axis=0)
    keep = miss_frac[miss_frac < cutoff].index
    drop = miss_frac[miss_frac >= cutoff].index

    X_kept = X.loc[:, keep].copy()

    # Build report for dropped assays
    report_rows = []
    pvals = {}

    if len(drop) == 0:
        return X_kept, pd.DataFrame(columns=["Assay", "missing_frac", "p_value", "bh_reject"])

    if groups is None:
        # No group labels: still report missingness fraction
        for assay in drop:
            report_rows.append(
                {"Assay": assay, "missing_frac": float(miss_frac[assay]), "p_value": np.nan, "bh_reject": False}
            )
        return X_kept, pd.DataFrame(report_rows).sort_values("missing_frac", ascending=False)

    # Align groups
    g = groups.reindex(X.index)
    valid = g.notna()
    g = g[valid]
    Xg = X.loc[valid, drop]
    
    # Ensure binary groups
    uniq = pd.unique(g)
    if len(uniq) != 2:
        raise ValueError(f"Expected exactly 2 groups for missingness check; got {uniq}")

    g0, g1 = uniq[0], uniq[1]

    for assay in drop:
        miss = Xg[assay].isna()
        a = int(((g == g0) & miss).sum())        # group0 missing
        b = int(((g == g0) & (~miss)).sum())     # group0 observed
        c = int(((g == g1) & miss).sum())        # group1 missing
        d = int(((g == g1) & (~miss)).sum())     # group1 observed

        # Fisher exact on [[a,b],[c,d]]
        # If a group has 0 samples, set p=1
        if (a + b == 0) or (c + d == 0):
            p = 1.0
        else:
            _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")

        pvals[assay] = p

        report_rows.append(
            {
                "Assay": assay,
                "missing_frac": float(miss_frac[assay]),
                f"{g0}_missing": a,
                f"{g0}_observed": b,
                f"{g1}_missing": c,
                f"{g1}_observed": d,
                "p_value": p,
            }
        )

    report = pd.DataFrame(report_rows).set_index("Assay")
    pser = pd.Series(pvals, name="p_value")
    reject = benjamini_hochberg_rejections(pser, alpha=alpha_bh)
    report["bh_reject"] = reject.reindex(report.index).fillna(False).astype(bool)
    report = report.sort_values(["bh_reject", "p_value"], ascending=[False, True]).reset_index()

    return X_kept, report

# -----------------------------
#  Panel normalization (long)
# -----------------------------

def apply_panel_normalization_long(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Panel normalization using Olink internal control samples encoded in SampleID:

      For each Assay:
        panel_median = median NPX across CONTROL_SAMPLEs within each Panel
        global_median = median of panel_medians across all Panels
        adjustment(panel) = global_median - panel_median
        NPX_adj = NPX + adjustment(panel)

    Requires columns: SampleID, Panel, Assay, NPX
    """
    df = df_long.copy()

    required = {"SampleID", "Panel", "Assay", "NPX"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Panel normalization requires columns: {sorted(required)}; missing: {sorted(missing)}")

    # Identify Olink control samples from SampleID pattern
    ctrl = df["SampleID"].astype(str).str.strip().str.upper().str.startswith("CONTROL_SAMPLE")

    # Compute panel-specific medians using control samples only
    panel_medians = (
        df.loc[ctrl]
        .groupby(["Panel", "Assay"])["NPX"]
        .median()
        .rename("panel_median")
        .reset_index()
    )

    if panel_medians.empty:
        raise ValueError(
            "No control samples found using SampleID prefix 'CONTROL_SAMPLE'. "
            "Check the exact SampleID pattern in your file."
        )

    # Global reference = median of panel medians across panels (per Assay)
    global_ref = (
        panel_medians.groupby("Assay")["panel_median"]
        .median()
        .rename("global_median")
        .reset_index()
    )

    # Adjustment = Global - Panel median
    adj = panel_medians.merge(global_ref, on="Assay", how="left")
    adj["adjustment"] = adj["global_median"] - adj["panel_median"]

    # Apply adjustment to every sample based on its Panel+Assay
    df = df.merge(adj[["Panel", "Assay", "adjustment"]], on=["Panel", "Assay"], how="left")

    # If some Panel+Assay combinations lacked controls, adjustment will be NaN.
    # Leave those NPX unchanged (adjustment=0).
    df["adjustment"] = df["adjustment"].fillna(0.0)

    df["NPX"] = df["NPX"] + df["adjustment"]
    df.drop(columns=["adjustment"], inplace=True)

    return df



# -----------------------------
#       PCA plots
# -----------------------------
def plot_pca(X: pd.DataFrame, labels: pd.Series, out_png: str, title: str):
    """
    PCA scatter plot (PC1 vs PC2). Colors by labels.
    """

    # Drop samples with any NaN for PCA
    Xp = X.dropna(axis=0, how="any")
    if Xp.shape[0] < 3 or Xp.shape[1] < 2:
        print("[WARN] PCA skipped (not enough complete cases).")
        return

    labs = labels.reindex(Xp.index).astype(str)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(Xp.values)

    dfc = pd.DataFrame(coords, index=Xp.index, columns=["PC1", "PC2"])
    uniq = labs.unique()

    plt.figure()
    for u in uniq:
        m = labs == u
        plt.scatter(dfc.loc[m, "PC1"], dfc.loc[m, "PC2"], label=u, alpha=0.7)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------
# Replicate-based batch correction (wide)
# -----------------------------
def find_replicates_across_batches(batch_to_samples: dict[str, set[str]]) -> dict[tuple[str, str], set[str]]:
    """
    Given batch_to_samples: {batch: set(SampleID)}, returns intersections for batch pairs.
    """
    batches = list(batch_to_samples.keys())
    reps = {}
    for i in range(len(batches)):
        for j in range(i + 1, len(batches)):
            b1, b2 = batches[i], batches[j]
            reps[(b1, b2)] = batch_to_samples[b1].intersection(batch_to_samples[b2])
    return reps


def compute_correction_factors_difference(
    X_ref: pd.DataFrame,
    X_tgt: pd.DataFrame,
    replicate_ids: list[str],
    outlier_iqr_k: float = 3.0,
) -> pd.Series:
    """
    Compute per-assay additive correction for NPX (log2 scale):
      delta = median( X_ref - X_tgt ) across replicate pairs, excluding outliers.
      corrected_tgt = X_tgt + delta
    """
    if len(replicate_ids) == 0:
        return pd.Series(index=X_tgt.columns, data=0.0)

    common_assays = X_ref.columns.intersection(X_tgt.columns)
    ref = X_ref.loc[replicate_ids, common_assays]
    tgt = X_tgt.loc[replicate_ids, common_assays]
    diff = ref - tgt  # additive in NPX

    deltas = {}
    for assay in common_assays:
        v = diff[assay].dropna()
        if v.empty:
            deltas[assay] = 0.0
            continue
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - outlier_iqr_k * iqr
        hi = q3 + outlier_iqr_k * iqr
        v2 = v[(v >= lo) & (v <= hi)]
        deltas[assay] = float(v2.median()) if not v2.empty else float(v.median())

    return pd.Series(deltas)

def compute_correction_factors_ratio_linear(
    X_ref: pd.DataFrame,
    X_tgt: pd.DataFrame,
    replicate_ids: list[str],
    outlier_iqr_k: float = 3.0,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Ratio correction in LINEAR space:
      For each assay:
        ratio = (2**NPX_tgt) / (2**NPX_ref) across replicate pairs
        exclude outliers
        median_ratio = median(ratio)
        correction_factor = 1 / median_ratio
      Apply to target in linear space, then convert back to NPX.

    Returns:
      factors: Series indexed by assay giving multiplicative correction factor (linear space).
    """
    if len(replicate_ids) == 0:
        return pd.Series(index=X_tgt.columns, data=1.0)

    common_assays = X_ref.columns.intersection(X_tgt.columns)
    ref = X_ref.loc[replicate_ids, common_assays]
    tgt = X_tgt.loc[replicate_ids, common_assays]

    # Convert NPX (log2) -> linear
    ref_lin = np.power(2.0, ref)
    tgt_lin = np.power(2.0, tgt)

    # ratio = tgt/ref
    ratio = tgt_lin / (ref_lin + eps)

    factors = {}
    for assay in common_assays:
        v = ratio[assay].replace([np.inf, -np.inf], np.nan).dropna()
        if v.empty:
            factors[assay] = 1.0
            continue

        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - outlier_iqr_k * iqr
        hi = q3 + outlier_iqr_k * iqr
        v2 = v[(v >= lo) & (v <= hi)]
        med = float(v2.median()) if not v2.empty else float(v.median())
        factors[assay] = 1.0 / med if med > 0 else 1.0

    return pd.Series(factors)


def apply_correction_factors_ratio_linear(
    X_tgt: pd.DataFrame,
    factors: pd.Series,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Apply multiplicative correction in linear space to target batch, return NPX (log2).
    """
    common_assays = X_tgt.columns.intersection(factors.index)
    # NPX -> linear
    tgt_lin = np.power(2.0, X_tgt[common_assays])
    # apply factors (broadcast by columns)
    corrected_lin = tgt_lin.mul(factors[common_assays], axis=1)
    # linear -> NPX
    corrected_npx = np.log2(corrected_lin + eps)

    out = X_tgt.copy()
    out[common_assays] = corrected_npx
    return out

def apply_batch_correction_using_replicates(
    X: pd.DataFrame,
    sample_to_batch: pd.Series,
    reference_batch: str,
    mode: str = "difference",
    outlier_iqr_k: float = 3.0,
) -> pd.DataFrame:
    """
    Adjust non-reference batches as a function of reference_batch using duplicate samples.

    Parameters
    ----------
    mode:
      - "difference": additive correction on NPX (log2). delta = median(ref - tgt)
      - "ratio_linear": multiplicative correction in linear space. factor = 1/median((tgt/ref)_linear)
    """
    if mode not in {"difference", "ratio_linear"}:
        raise ValueError("mode must be one of {'difference', 'ratio_linear'}")

    Xc = X.copy()

    batch_to_samples = {
        b: set(sample_to_batch[sample_to_batch == b].index)
        for b in sample_to_batch.dropna().unique()
    }

    reps = find_replicates_across_batches(batch_to_samples)

    total_reps = sum(len(v) for v in reps.values())
    if total_reps == 0:
        print("[INFO] No replicate SampleIDs found across batches; skipping replicate-based batch correction.")
        return Xc

    print(f"[INFO] Replicate counts by batch pair: " +
          ", ".join([f"{a}∩{b}={len(s)}" for (a, b), s in reps.items()]))

    for b in batch_to_samples.keys():
        if b == reference_batch:
            continue

        key = (reference_batch, b) if (reference_batch, b) in reps else (b, reference_batch)
        rep_ids = sorted(list(reps.get(key, set())))
        if len(rep_ids) == 0:
            print(f"[INFO] No replicates between {reference_batch} and {b}; skipping adjustment for {b}.")
            continue

        X_ref = Xc.loc[sample_to_batch == reference_batch]
        X_tgt = Xc.loc[sample_to_batch == b]

        rep_ids = [sid for sid in rep_ids if (sid in X_ref.index) and (sid in X_tgt.index)]
        if len(rep_ids) == 0:
            print(f"[INFO] Replicate IDs not present in both matrices for {b}; skipping.")
            continue

        if mode == "difference":
            deltas = compute_correction_factors_difference(
                X_ref, X_tgt, rep_ids, outlier_iqr_k=outlier_iqr_k
            )
            common_assays = deltas.index.intersection(X_tgt.columns)
            Xc.loc[sample_to_batch == b, common_assays] = (
                Xc.loc[sample_to_batch == b, common_assays].add(deltas[common_assays], axis=1)
            )
            print(f"[INFO] Applied DIFFERENCE correction to batch {b} vs {reference_batch} using {len(rep_ids)} replicates.")

        else:  # mode == "ratio_linear"
            factors = compute_correction_factors_ratio_linear(
                X_ref, X_tgt, rep_ids, outlier_iqr_k=outlier_iqr_k
            )
            corrected = apply_correction_factors_ratio_linear(X_tgt, factors)
            common_assays = corrected.columns.intersection(Xc.columns)
            Xc.loc[sample_to_batch == b, common_assays] = corrected[common_assays].values
            print(f"[INFO] Applied RATIO(LINEAR) correction to batch {b} vs {reference_batch} using {len(rep_ids)} replicates.")

    return Xc



# -----------------------------
#  Imputation (wide)
# -----------------------------
def half_min_impute_wide(X: pd.DataFrame) -> pd.DataFrame:
    """
    1/2-minimum imputation per assay column. Impute at the very end.
    X: wide matrix (SampleID x Assay)
    returns imputed DataFrame.
    """
    Xi = X.copy()
    for col in Xi.columns:
        s = Xi[col]
        if s.notna().any():
            half_min = 0.5 * s.min(skipna=True)
            Xi[col] = s.fillna(half_min)
    return Xi


# -----------------------------
#     Wrapper functions
# -----------------------------
def process_single_file(file_input: str, create_file_output: bool = False, file_output: str | None = None) -> pd.DataFrame:
    """
    Reads ONE Olink batch file (long format), standardizes missing, QC masks, and tags batch.
    Returns long-format cleaned (pre-filter, pre-panelnorm) dataframe with a Batch column.

    Use for testing or intermediate steps.
    """
    df = pd.read_csv(file_input, sep=";")
    df = standardize_missing_npx(df)
    df = qc_mask(df)

    df["Batch"] = infer_batch_id_from_filename(file_input)

    if create_file_output and file_output is not None:
        df.to_csv(file_output, sep=";", index=False)

    return df


def combine_batches(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    df_combined = pd.concat(df_list, axis=0, ignore_index=True)
    return df_combined

'''
def merge_with_metadata(df: pd.DataFrame, 
                        metadata_path: str, 
                        omic_sample_id_col: str = "SampleID",
                        meta_sample_id_col: str = 'sample Id') -> pd.DataFrame:
    """
    Loads group labels from metadata.
    """
    meta = pd.read_excel(metadata_path, sheet_name='n=133 proteomics')

    if omic_sample_id_col not in df.columns or \
        meta_sample_id_col not in meta.columns:
        raise ValueError('ID column not found')
    
    if 'group' not in meta.columns:
        raise ValueError('Group column not found')

    g = df.merge(meta[[meta_sample_id_col, 'group']], 
                left_on=omic_sample_id_col, 
                right_on=meta_sample_id_col,
                how='left')
    
    return g
'''

def load_groups_from_metadata(
    metadata_path: str,
    sample_id_col: str = "SampleID",
    group_col: str = "Group",
    make_binary: bool = False,
    control_label: str = "Control",
) -> pd.Series:
    """
    Loads group labels from metadata. Optionally collapses to binary:
      Control vs Complication (everything else).
    """
    meta = pd.read_excel(metadata_path, sheet_name='n=133 proteomics')
    if sample_id_col not in meta.columns or group_col not in meta.columns:
        raise ValueError(
            f"Metadata must contain columns '{sample_id_col}' and '{group_col}'. "
            f"Found: {meta.columns.tolist()}"
        )

    g = meta.set_index(sample_id_col)[group_col].astype(str)

    if make_binary:
        g_bin = np.where(g == control_label, "Control", "Complication")
        g = pd.Series(g_bin, index=g.index, name="GroupBinary")

    return g



def process_all_files(
    file_paths: list[str],
    output_csv: str,
    metadata_path: str,
    pca_dir: str | None = None,
):
    """
    DP3 full pipeline for one tissue x timepoint x modality set of Olink batch files.
    """
    # Load and QC-mask each batch (long format)
    batches_long = [process_single_file(fp) for fp in file_paths]
    df_long = combine_batches(batches_long)

    # Pivot to wide for missingness filter (still pre-panelnorm)
    X = df_long.pivot_table(index="SampleID", columns="Assay", values="NPX", 
                            aggfunc='median')

    # Group labels for missingness diagnostic (only for dropped assays)
    groups = load_groups_from_metadata(metadata_path, sample_id_col="sample Id", group_col="group", make_binary=True)

    # Missingness filter + group-wise check on dropped assays
    X_kept, dropped_report = missingness_filter_and_group_check(X, groups, cutoff=CUTOFF_PERCENT_MISSING, alpha_bh=0.05)

    # Save dropped report alongside output
    if dropped_report is not None and not dropped_report.empty:
        rep_path = os.path.splitext(output_csv)[0] + "_dropped_missingness_report.csv"
        dropped_report.to_csv(rep_path, index=False)

    keep_assays = set(X_kept.columns)

    # Panel normalization (long format) on kept assays only
    df_long_kept = df_long[df_long["Assay"].isin(keep_assays)].copy()
    df_long_panelnorm = apply_panel_normalization_long(df_long_kept)

    # PCA before replicate batch correction (wide)
    X_panel = df_long_panelnorm.pivot_table(index="SampleID", columns="Assay", values="NPX", aggfunc='median')

    sample_to_batch = (
        df_long_panelnorm[["SampleID", "Batch"]]
        .drop_duplicates()
        .groupby("SampleID")["Batch"]
        .first()
    ) # only takes the first batch per sampleID for PCA


    if pca_dir:
        os.makedirs(pca_dir, exist_ok=True)
        plot_pca(X_panel, sample_to_batch, os.path.join(pca_dir, "pca_pre_batch_correction.png"),
                 "PCA (panel-normalized, pre-batch correction)")

    # Replicate-based batch correction (if duplicates exist)
    # Choose reference batch = first file's inferred batch id
    ref_batch = infer_batch_id_from_filename(file_paths[0])
    X_batchcorr = apply_batch_correction_using_replicates(X_panel, sample_to_batch, reference_batch=ref_batch, mode='ratio_linear')

    # Debug line - delete after use
    diff = (X_batchcorr - X_panel).abs()
    print("[DEBUG] max |post-pre|:", np.nanmax(diff.to_numpy()))
    print("[DEBUG] mean |post-pre|:", np.nanmean(diff.to_numpy()))


    if pca_dir:
        plot_pca(X_batchcorr, sample_to_batch, os.path.join(pca_dir, "pca_post_batch_correction.png"),
                 "PCA (post replicate-based batch correction)")

    # 1/2-min imputation LAST
    X_final = half_min_impute_wide(X_batchcorr)

    # Save final
    # Save as wide matrix (SampleID index)
    X_final.to_csv(output_csv, index=True)

if __name__ == "__main__":
    # Test all files
    wkdir = os.getcwd()
    data_dir = os.path.join(wkdir, "data", "proteomics")
    output_dir = os.path.join(wkdir, "data", "cleaned", "proteomics")
    os.makedirs(output_dir, exist_ok=True)

    # Metadata file must have SampleID and Group columns
    metadata_path = os.path.join(wkdir, 'data', "dp3 master table v2.xlsx")

    # Collect batch files (exclude metadata)
    file_paths = []
    for fn in os.listdir(data_dir):
        if fn.endswith(".csv") and fn != "dp3 master table v2.xlsx":
            file_paths.append(os.path.join(data_dir, fn))

    # Deterministic order (so reference batch is stable)
    file_paths = sorted(file_paths)

    out_csv = os.path.join(output_dir, "proteomics_cleaned_panelnorm_batchcorr_imputed.csv")
    pca_dir = os.path.join(output_dir, "pca")

    process_all_files(file_paths, out_csv, metadata_path, pca_dir=pca_dir)