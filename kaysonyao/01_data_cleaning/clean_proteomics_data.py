"""
Title: clean_proteomics_data.py
Author: Samantha Piekos, Kayson Yao
Date: 02/01/2026
Description:
DP3 proteomics preprocessing for Olink Explore long-format exports.

Workflow:
1) Load each Olink file in long format.
2) Standardize missing NPX values:
   - Convert blank/"NA"/"N/A"/0 in the NPX column to NaN.
3) QC masking:
   - If QC_Warning != PASS or Assay_Warning != PASS, set NPX to NaN.
4) Tag batch identity from filename and concatenate all batches into one long table.
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
14) Save final cleaned matrix (wide, SampleID x Assay) to CSV.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

CUTOFF_PERCENT_MISSING = 0.25


# -----------------------------
# Standardize missing + QC mask + batch tagging
# -----------------------------
def standardize_missing_npx(df: pd.DataFrame) -> pd.DataFrame:
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


def process_single_file(file_input: str) -> pd.DataFrame:
    df = pd.read_csv(file_input, sep=";")
    df = standardize_missing_npx(df)
    df = qc_mask(df)
    df["Batch"] = infer_batch_id_from_filename(file_input)
    return df


def combine_batches(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(df_list, axis=0, ignore_index=True)


def is_olink_control_sample(df_long: pd.DataFrame) -> pd.Series:
    return df_long["SampleID"].astype(str).str.strip().str.upper().str.startswith("CONTROL_SAMPLE")


# -----------------------------
# Missingness + group check
# -----------------------------
def benjamini_hochberg_rejections(pvals: pd.Series, alpha: float = 0.05) -> pd.Series:
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
        raise ValueError(f"Expected exactly 2 groups for missingness check; got {uniq}")

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


def load_groups_from_metadata(
    metadata_path: str,
    sample_id_col: str = "sample Id",
    group_col: str = "group",
    make_binary: bool = True,
    control_label: str = "Control",
) -> pd.Series:
    meta = pd.read_excel(metadata_path, sheet_name="n=133 proteomics")
    if sample_id_col not in meta.columns or group_col not in meta.columns:
        raise ValueError(f"Metadata must contain columns '{sample_id_col}' and '{group_col}'. "
                         f"Found: {meta.columns.tolist()}")

    g = meta.set_index(sample_id_col)[group_col].astype(str)
    if make_binary:
        g = pd.Series(np.where(g == control_label, "Control", "Complication"),
                      index=g.index, name="GroupBinary")
    return g


# -----------------------------
# Panel normalization (long)
# -----------------------------
def apply_panel_normalization_long(df_long: pd.DataFrame) -> pd.DataFrame:
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
# Quantile normalization (wide)
# -----------------------------
def quantile_normalize_wide(X: pd.DataFrame) -> pd.DataFrame:
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

    plt.figure()
    for u in labs.unique():
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
# Imputation
# -----------------------------
def half_min_impute_wide(X: pd.DataFrame) -> pd.DataFrame:
    Xi = X.copy()
    for col in Xi.columns:
        s = Xi[col]
        if s.notna().any():
            Xi[col] = s.fillna(0.5 * s.min(skipna=True))
    return Xi


# -----------------------------
# Main wrapper
# -----------------------------
def process_all_files(file_paths: list[str], output_csv: str, metadata_path: str, pca_dir: str | None = None):
    # QC masking + combine batches
    batches_long = [process_single_file(fp) for fp in file_paths]
    df_long = combine_batches(batches_long)

    # Apply panel normalization using Olink controls (keeps controls in long df)
    # but downstream wide matrices should exclude controls.
    df_long = apply_panel_normalization_long(df_long)

    # Exclude Olink internal controls for analysis matrices
    ctrl_mask = is_olink_control_sample(df_long)
    df_long_bio = df_long.loc[~ctrl_mask].copy()

    # Wide matrix for missingness filter
    X = df_long_bio.pivot_table(
        index="SampleID",
        columns="Assay",
        values="NPX",
        aggfunc="median"
        )

    groups = load_groups_from_metadata(metadata_path, make_binary=True)

    X_kept, dropped_report = missingness_filter_and_group_check(X,
    groups, cutoff=CUTOFF_PERCENT_MISSING, alpha_bh=0.05)

    if dropped_report is not None and not dropped_report.empty:
        rep_path = os.path.splitext(output_csv)[0] + "_dropped_missingness_report.csv"
        dropped_report.to_csv(rep_path, index=False)

    # Batch labels for PCA
    sample_to_batch = (
        df_long_bio[["SampleID", "Batch"]]
        .drop_duplicates()
        .groupby("SampleID")["Batch"]
        .first()
    )

    #
    if pca_dir:
        os.makedirs(pca_dir, exist_ok=True)
        plot_pca(
            X_kept,
            sample_to_batch,
            os.path.join(pca_dir, "pca_pre_quantile_norm.png"),
            "PCA (panel-normalized, pre-quantile normalization)"
                )

    # Quantile normalization
    X_qn = quantile_normalize_wide(X_kept)

    if pca_dir:
        plot_pca(
            X_qn,
            sample_to_batch,
            os.path.join(pca_dir, "pca_post_quantile_norm.png"),
            "PCA (post quantile normalization)"
                )

    # Impute LAST
    X_final = half_min_impute_wide(X_qn)
    x_final_linear = np.power(2.0, X_final)
    x_final_linear.to_csv(output_csv, index=True)


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

    out_csv = os.path.join(output_dir, "proteomics_cleaned_panelnorm_quantilenorm_imputed.csv")
    pca_dir = os.path.join(output_dir, "pca")

    process_all_files(file_paths, out_csv, metadata_path, pca_dir=pca_dir)