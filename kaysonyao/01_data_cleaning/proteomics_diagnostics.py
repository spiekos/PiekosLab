"""
Optional diagnostics for the proteomics cleaning pipeline.

This module intentionally separates slower QC/diagnostic routines from
`clean_proteomics_data.py` so core cleaning can run faster.

Diagnostics included:
1) Duplicate SampleID×Assay report (before median collapse in wide pivot)
2) PCA pre-quantile normalization
3) PCA post-quantile normalization
4) Shared-axis pre/post PCA comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from clean_proteomics_data import (
    CUTOFF_PERCENT_MISSING,
    load_metadata_with_batch,
    process_single_file,
    combine_batches,
    apply_panel_normalization_long,
    is_olink_control_sample,
    missingness_filter_and_group_check,
    quantile_normalize_wide,
)


def duplicate_sample_assay_report(df_long_bio: pd.DataFrame) -> pd.DataFrame:
    """Build diagnostics for duplicated SampleID×Assay combinations."""
    required = {"SampleID", "Assay", "NPX"}
    missing = required - set(df_long_bio.columns)
    if missing:
        raise ValueError(
            f"Duplicate diagnostics requires columns: {sorted(required)}; missing: {sorted(missing)}"
        )

    group_cols = ["SampleID", "Assay"]
    base_agg = (
        df_long_bio.groupby(group_cols, dropna=False)
        .agg(
            n_records=("NPX", "size"),
            n_non_missing=("NPX", lambda s: int(s.notna().sum())),
            n_missing=("NPX", lambda s: int(s.isna().sum())),
            npx_median=("NPX", "median"),
            npx_std=("NPX", "std"),
            npx_min=("NPX", "min"),
            npx_max=("NPX", "max"),
        )
        .reset_index()
    )

    if "Panel" in df_long_bio.columns:
        panel_counts = (
            df_long_bio.groupby(group_cols, dropna=False)["Panel"]
            .nunique(dropna=True)
            .rename("n_panels")
            .reset_index()
        )
        panels_list = (
            df_long_bio.groupby(group_cols, dropna=False)["Panel"]
            .apply(lambda s: ";".join(sorted(set(s.dropna().astype(str)))))
            .rename("panels")
            .reset_index()
        )
        base_agg = base_agg.merge(panel_counts, on=group_cols, how="left")
        base_agg = base_agg.merge(panels_list, on=group_cols, how="left")

    if "PlateID" in df_long_bio.columns:
        plate_counts = (
            df_long_bio.groupby(group_cols, dropna=False)["PlateID"]
            .nunique(dropna=True)
            .rename("n_plate_ids")
            .reset_index()
        )
        base_agg = base_agg.merge(plate_counts, on=group_cols, how="left")

    report = base_agg[base_agg["n_records"] > 1].copy()
    report = report.sort_values(["n_records", "SampleID", "Assay"], ascending=[False, True, True])
    return report


def _prepare_pca_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Prepare matrix for PCA with per-assay median imputation."""
    if X.shape[0] < 3 or X.shape[1] < 2:
        return pd.DataFrame(index=X.index)
    Xp = X.copy()
    Xp = Xp.apply(lambda col: col.fillna(col.median(skipna=True)), axis=0)
    Xp = Xp.dropna(axis=1, how="all")
    return Xp


def _compute_pca_coords(X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray] | tuple[None, None]:
    """Compute 2D PCA coordinates and explained variance ratio."""
    Xp = _prepare_pca_matrix(X)
    if Xp.shape[0] < 3 or Xp.shape[1] < 2:
        return None, None
    pca = PCA(n_components=2)
    coords = pca.fit_transform(Xp.values)
    return pd.DataFrame(coords, index=Xp.index, columns=["PC1", "PC2"]), pca.explained_variance_ratio_


def _plot_pca_on_axis(ax, dfc: pd.DataFrame, labs: pd.Series, title: str, var_ratio: np.ndarray):
    for u in labs.unique():
        m = labs == u
        ax.scatter(dfc.loc[m, "PC1"], dfc.loc[m, "PC2"], label=u, alpha=0.7, s=50)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}%)")


def plot_pca(X: pd.DataFrame, labels: pd.Series, out_png: str, title: str):
    dfc, var_ratio = _compute_pca_coords(X)
    if dfc is None:
        print("[WARN] PCA skipped (too few samples/features).")
        return
    labs = labels.reindex(dfc.index).astype(str)
    plt.figure(figsize=(10, 8))
    _plot_pca_on_axis(plt.gca(), dfc, labs, title, var_ratio)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_pca_pre_post_comparison(X_pre: pd.DataFrame, X_post: pd.DataFrame, labels: pd.Series, out_png: str):
    df_pre, var_pre = _compute_pca_coords(X_pre)
    df_post, var_post = _compute_pca_coords(X_post)
    if df_pre is None or df_post is None:
        print("[WARN] PCA comparison skipped (too few samples/features).")
        return

    shared = df_pre.index.intersection(df_post.index)
    if len(shared) < 3:
        print("[WARN] PCA comparison skipped (too few shared samples).")
        return

    df_pre = df_pre.loc[shared]
    df_post = df_post.loc[shared]
    labs = labels.reindex(shared).astype(str)

    x_all = np.concatenate([df_pre["PC1"].values, df_post["PC1"].values])
    y_all = np.concatenate([df_pre["PC2"].values, df_post["PC2"].values])
    xpad = 0.05 * (x_all.max() - x_all.min() + 1e-9)
    ypad = 0.05 * (y_all.max() - y_all.min() + 1e-9)
    xlim = (x_all.min() - xpad, x_all.max() + xpad)
    ylim = (y_all.min() - ypad, y_all.max() + ypad)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True, sharey=True)
    _plot_pca_on_axis(axes[0], df_pre, labs, "PCA (panel-normalized, pre-quantile normalization)", var_pre)
    _plot_pca_on_axis(axes[1], df_post, labs, "PCA (post quantile normalization)", var_post)
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", fontsize=9)
    fig.suptitle("Pre/Post PCA Comparison (Shared Axis Limits)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.97, 0.95])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def run_diagnostics(
    file_paths: list[str],
    output_csv: str,
    metadata_path: str,
    meta_type: str = "proteomics",
    pca_dir: str | None = None,
):
    """Run diagnostics separately from core cleaning."""
    print("=" * 80)
    print("PROTEOMICS DIAGNOSTICS")
    print("=" * 80)

    print("\n[1/5] Loading metadata...")
    metadata = load_metadata_with_batch(metadata_path, meta_type=meta_type)

    print("\n[2/5] Loading and preprocessing Olink files...")
    batches_long = []
    for fp in file_paths:
        print(f"   Processing: {os.path.basename(fp)}")
        df_single = process_single_file(fp)
        df_single["SourceFile"] = os.path.basename(fp)
        batches_long.append(df_single)
    df_long = combine_batches(batches_long)

    print("\n[3/5] Panel normalization + control removal...")
    df_long = apply_panel_normalization_long(df_long)
    df_long_bio = df_long.loc[~is_olink_control_sample(df_long)].copy()

    dup_report = duplicate_sample_assay_report(df_long_bio)
    dup_path = os.path.splitext(output_csv)[0] + "_duplicate_sample_assay_report.csv"
    dup_report.to_csv(dup_path, index=False)
    print(f"   Duplicate combos: {len(dup_report)}")
    print(f"   Saved: {dup_path}")

    print("\n[4/5] Building assay matrix for PCA diagnostics...")
    X = df_long_bio.pivot_table(index="SampleID", columns="Assay", values="NPX", aggfunc="median")
    groups = metadata["Group"].reindex(X.index)
    has_metadata = groups.notna()
    X = X.loc[has_metadata].copy()
    groups = groups.loc[has_metadata]
    groups_binary = pd.Series(
        np.where(groups.astype(str).str.strip().str.upper() == "CONTROL", "Control", "Complication"),
        index=groups.index,
        name="GroupBinary",
    )

    X_kept, _ = missingness_filter_and_group_check(X, groups_binary, cutoff=CUTOFF_PERCENT_MISSING, alpha_bh=0.05)
    X_qn = quantile_normalize_wide(X_kept)
    sample_to_batch = metadata["Batch"].reindex(X_kept.index)

    print("\n[5/5] Saving PCA diagnostics...")
    if pca_dir is None:
        pca_dir = os.path.join(os.path.dirname(output_csv), f"pca_{meta_type}")
    os.makedirs(pca_dir, exist_ok=True)

    pre_png = os.path.join(pca_dir, "pca_pre_quantile_norm.png")
    post_png = os.path.join(pca_dir, "pca_post_quantile_norm.png")
    cmp_png = os.path.join(pca_dir, "pca_pre_post_shared_axes.png")
    plot_pca(X_kept, sample_to_batch, pre_png, "PCA (panel-normalized, pre-quantile normalization)")
    plot_pca(X_qn, sample_to_batch, post_png, "PCA (post quantile normalization)")
    plot_pca_pre_post_comparison(X_kept, X_qn, sample_to_batch, cmp_png)
    print(f"   Saved: {pre_png}")
    print(f"   Saved: {post_png}")
    print(f"   Saved: {cmp_png}")


def _collect_files(data_dir: str):
    plasma_files, placenta_files = [], []
    for fn in os.listdir(data_dir):
        if not fn.endswith(".csv"):
            continue
        full_path = os.path.join(data_dir, fn)
        fn_lower = fn.lower()
        if "plasma" in fn_lower:
            plasma_files.append(full_path)
        elif "placenta" in fn_lower or "tissue" in fn_lower or "lysate" in fn_lower:
            placenta_files.append(full_path)
    return sorted(plasma_files), sorted(placenta_files)


if __name__ == "__main__":
    wkdir = os.getcwd()
    data_dir = os.path.join(wkdir, "data", "proteomics")
    output_dir = os.path.join(wkdir, "data", "cleaned", "proteomics")
    metadata_path = os.path.join(wkdir, "data", "dp3 master table v2.xlsx")

    plasma_files, placenta_files = _collect_files(data_dir)

    if plasma_files:
        out_csv_plasma = os.path.join(output_dir, "proteomics_plasma_cleaned_with_metadata.csv")
        pca_dir_plasma = os.path.join(output_dir, "pca_plasma")
        run_diagnostics(plasma_files, out_csv_plasma, metadata_path, meta_type="proteomics", pca_dir=pca_dir_plasma)

    if placenta_files:
        out_csv_placenta = os.path.join(output_dir, "proteomics_placenta_cleaned_with_metadata.csv")
        pca_dir_placenta = os.path.join(output_dir, "pca_placenta")
        run_diagnostics(placenta_files, out_csv_placenta, metadata_path, meta_type="placenta", pca_dir=pca_dir_placenta)
