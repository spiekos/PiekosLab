"""
Optional diagnostics for the proteomics cleaning pipeline.

This module intentionally separates slower QC/diagnostic routines from
`clean_proteomics_data.py` so core cleaning can run faster.

Diagnostics included:
1) Duplicate SampleID×Assay report (before median collapse in wide pivot)
2) PCA pre-ComBat normalization
3) PCA post-ComBat normalization
4) Shared-axis pre/post PCA comparison
5) Normalization quality plots (sample distributions, boxplots, density overlay,
   quantile heatmap, and optional batch comparison)
"""

import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

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
)

logger = logging.getLogger(__name__)

# Metadata columns present in the final cleaned CSV (used by plot_normalization_quality).
_METADATA_COLS = [
    "SampleID", "SubjectID", "Group", "Subgroup",
    "Batch", "GestAgeDelivery", "SampleGestAge",
]

# Seed for reproducible random sampling in plots.
_RNG_SEED = 42


def _extract_protein_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Drop known metadata columns and return log2-transformed protein data."""
    meta_cols = [c for c in _METADATA_COLS if c in df.columns]
    return np.log2(df.drop(columns=meta_cols))


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
        logger.warning("PCA skipped (too few samples/features).")
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
        logger.warning("PCA comparison skipped (too few samples/features).")
        return

    shared = df_pre.index.intersection(df_post.index)
    if len(shared) < 3:
        logger.warning("PCA comparison skipped (too few shared samples).")
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
    _plot_pca_on_axis(axes[0], df_pre, labs, "PCA (panel-normalized, pre-ComBat)", var_pre)
    _plot_pca_on_axis(axes[1], df_post, labs, "PCA (post ComBat)", var_post)
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


def plot_combat_assessment(
    X_pre: pd.DataFrame,
    X_post: pd.DataFrame,
    batch_labels: pd.Series,
    out_dir: str,
) -> None:
    """
    Visualise the effect of ComBat batch correction.

    Generates two plots saved under out_dir/:
      1. combat_batch_medians.png  — side-by-side boxplots of per-batch
         sample medians before and after ComBat.
      2. combat_assay_batch_cv.png — overlapping histograms of the
         coefficient of variation (CV) of per-batch assay means before
         and after ComBat; a left-shift of the post-ComBat curve indicates
         effective batch-effect removal.

    Args:
        X_pre:         Wide assay matrix before ComBat (samples × assays, may contain NaN).
        X_post:        Wide assay matrix after ComBat (same shape as X_pre, may contain NaN).
        batch_labels:  Per-sample batch identifiers, index aligned to X_pre / X_post.
        out_dir:       Directory where plots are saved (created if absent).
    """
    shared = X_pre.index.intersection(X_post.index)
    if len(shared) < 3:
        logger.warning("ComBat assessment skipped (too few shared samples: %d).", len(shared))
        return

    batches = batch_labels.reindex(shared).astype(str)
    unique_batches = sorted(batches.unique())
    if len(unique_batches) < 2:
        logger.warning("ComBat assessment skipped (fewer than 2 batches).")
        return

    os.makedirs(out_dir, exist_ok=True)
    X_pre  = X_pre.loc[shared]
    X_post = X_post.loc[shared]

    # ------------------------------------------------------------------
    # 1. Per-batch sample median distributions: pre vs post
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, X, title in [
        (axes[0], X_pre,  "Pre-ComBat"),
        (axes[1], X_post, "Post-ComBat"),
    ]:
        data   = [X.loc[batches == b].median(axis=1, skipna=True).dropna().values
                  for b in unique_batches]
        labels = [f"Batch {b}" for b in unique_batches]
        ax.boxplot(data, labels=labels)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Per-sample Median NPX (log2)")
        ax.grid(alpha=0.3, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.suptitle("ComBat Batch Correction: Per-batch Sample Medians", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "combat_batch_medians.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("   Saved: %s", out_path)

    # ------------------------------------------------------------------
    # 2. Per-assay batch CV: distribution before vs after ComBat
    # ------------------------------------------------------------------
    # For each assay, compute the CV of per-batch means.  A successful
    # ComBat run should reduce the long right-tail of high-CV assays.
    pre_batch_means  = pd.DataFrame(
        {b: X_pre.loc[batches == b].mean(axis=0, skipna=True)  for b in unique_batches}
    )
    post_batch_means = pd.DataFrame(
        {b: X_post.loc[batches == b].mean(axis=0, skipna=True) for b in unique_batches}
    )

    def _batch_cv(df: pd.DataFrame) -> pd.Series:
        row_mean = df.mean(axis=1).abs().clip(lower=1e-9)
        return df.std(axis=1) / row_mean

    pre_cv  = _batch_cv(pre_batch_means).dropna()
    post_cv = _batch_cv(post_batch_means).dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(pre_cv,  bins=50, alpha=0.6, label="Pre-ComBat",  color="salmon")
    ax.hist(post_cv, bins=50, alpha=0.6, label="Post-ComBat", color="steelblue")
    ax.set_xlabel("CV of per-batch assay means")
    ax.set_ylabel("Number of assays")
    ax.set_title("ComBat Effect: Distribution of Per-assay Batch CVs\n"
                 "(left-shift = better batch alignment)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "combat_assay_batch_cv.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("   Saved: %s", out_path)


def run_diagnostics(
    file_paths: list[str],
    output_csv: str,
    metadata_path: str,
    meta_type: str = "proteomics",
    pca_dir: str | None = None,
    combat_dir: str | None = None,
    reports_dir: str | None = None,
):
    """Run diagnostics separately from core cleaning."""
    logger.info("=" * 60)
    logger.info("PROTEOMICS DIAGNOSTICS")
    logger.info("=" * 60)

    logger.info("[1/6] Loading metadata...")
    metadata = load_metadata_with_batch(metadata_path, meta_type=meta_type)

    logger.info("[2/6] Loading and preprocessing Olink files...")
    batches_long = []
    for fp in file_paths:
        logger.info("   Processing: %s", os.path.basename(fp))
        df_single = process_single_file(fp)
        df_single["SourceFile"] = os.path.basename(fp)
        batches_long.append(df_single)
    df_long = combine_batches(batches_long)

    logger.info("[3/6] Panel normalization + control removal...")
    df_long = apply_panel_normalization_long(df_long)
    df_long_bio = df_long.loc[~is_olink_control_sample(df_long)].copy()
    df_long_bio = df_long_bio.loc[~is_olink_control_assay(df_long_bio)].copy()

    dup_report = duplicate_sample_assay_report(df_long_bio)
    dup_path = os.path.splitext(output_csv)[0] + "_duplicate_sample_assay_report.csv"
    dup_report.to_csv(dup_path, index=False)
    logger.info("   Duplicate combos: %d", len(dup_report))
    logger.info("   Saved: %s", dup_path)

    logger.info("[4/6] Building assay matrix for ComBat + PCA diagnostics...")
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

    sample_to_batch = metadata["Batch"].reindex(X.index)
    has_batch = sample_to_batch.notna()
    if not has_batch.all():
        X = X.loc[has_batch].copy()
        groups_binary = groups_binary.loc[has_batch].copy()
        sample_to_batch = sample_to_batch.loc[has_batch]

    X_combat = combat_normalize_wide(X, sample_to_batch)
    X_kept, _ = missingness_filter_and_group_check(
        X_combat, groups_binary, cutoff=CUTOFF_PERCENT_MISSING, alpha_bh=0.05
    )
    sample_to_batch = sample_to_batch.reindex(X_kept.index)

    logger.info("[5/6] Saving ComBat assessment...")
    if combat_dir is None:
        combat_dir = os.path.join(os.path.dirname(output_csv), f"combat_{meta_type}")
    plot_combat_assessment(X, X_combat, sample_to_batch, combat_dir)

    logger.info("[6/6] Saving PCA diagnostics...")
    if pca_dir is None:
        pca_dir = os.path.join(os.path.dirname(output_csv), f"pca_{meta_type}")
    # Each PCA variant gets its own subfolder.
    _pca_dirs = {
        "pre_combat":  os.path.join(pca_dir, "pre_combat"),
        "post_combat": os.path.join(pca_dir, "post_combat"),
        "comparison":  os.path.join(pca_dir, "comparison"),
    }
    for d in _pca_dirs.values():
        os.makedirs(d, exist_ok=True)

    pre_png  = os.path.join(_pca_dirs["pre_combat"],  "pca_pre_combat.png")
    post_png = os.path.join(_pca_dirs["post_combat"], "pca_post_combat.png")
    cmp_png  = os.path.join(_pca_dirs["comparison"],  "pca_pre_post_shared_axes.png")
    plot_pca(X, sample_to_batch, pre_png, "PCA (panel-normalized, pre-ComBat)")
    plot_pca(X_kept, sample_to_batch, post_png, "PCA (post ComBat + post-missingness filter)")
    plot_pca_pre_post_comparison(X, X_kept, sample_to_batch, cmp_png)
    logger.info("   Saved: %s", pre_png)
    logger.info("   Saved: %s", post_png)
    logger.info("   Saved: %s", cmp_png)


def plot_normalization_quality(
    result_path: str,
    output_dir: str | None = None,
    verbose: bool = False,
) -> None:
    """
    Generate normalization quality plots from a cleaned proteomics CSV.

    Produces four diagnostic outputs:
      1. Histogram of per-sample medians, means, and standard deviations.
      2. Boxplot of protein distributions across samples.
      3. Density overlay for a random subset of samples.
      4. Per-batch median boxplot (only if a Batch column is present).

    Args:
        result_path: Path to the cleaned proteomics CSV (wide format, index = SampleID).
        output_dir:  Directory for saved plots. Defaults to the same directory as result_path.
        verbose:     Log additional per-metric statistics.
    """
    df = pd.read_csv(result_path, index_col=0)

    if output_dir is None:
        output_dir = os.path.dirname(result_path)

    # Each plot type gets its own subfolder under output_dir.
    _dirs = {
        "distributions": os.path.join(output_dir, "sample_distributions"),
        "boxplots":       os.path.join(output_dir, "sample_boxplots"),
        "density":        os.path.join(output_dir, "density_overlay"),
        "batch":          os.path.join(output_dir, "batch_comparison"),
    }
    for d in _dirs.values():
        os.makedirs(d, exist_ok=True)

    file_prefix = os.path.splitext(os.path.basename(result_path))[0]
    X_log2 = _extract_protein_matrix(df)

    logger.info("=" * 70)
    logger.info("NORMALIZATION QUALITY ASSESSMENT: %s", file_prefix)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Sample distribution alignment
    # ------------------------------------------------------------------
    logger.info("[1/4] Sample distribution alignment")

    sample_means = X_log2.mean(axis=1)
    sample_stds = X_log2.std(axis=1)
    sample_medians = X_log2.median(axis=1)

    cv_stds = sample_stds.std() / sample_stds.mean()
    cv_medians = (
        sample_medians.std() / abs(sample_medians.mean())
        if abs(sample_medians.mean()) > 0.01
        else float("inf")
    )

    if verbose:
        logger.info(
            "   Sample medians: mean=%.4f, std=%.4f, CV=%.4f",
            sample_medians.mean(), sample_medians.std(), cv_medians,
        )
        logger.info(
            "   Sample stds:    mean=%.4f, std=%.4f, CV=%.4f",
            sample_stds.mean(), sample_stds.std(), cv_stds,
        )

    if cv_stds < 0.05:
        logger.info("   ✓ EXCELLENT: Sample std devs very uniform (CV < 5%%)")
    elif cv_stds < 0.10:
        logger.info("   ✓ GOOD: Sample std devs reasonably uniform (CV < 10%%)")
    else:
        logger.info("   ✗ WARNING: Sample std devs variable (CV >= 10%%)")

    if abs(sample_medians.mean()) > 0.1 and cv_medians < 0.10:
        logger.info("   ✓ EXCELLENT: Sample medians very uniform (CV < 10%%)")
    elif abs(sample_medians.mean()) > 0.1 and cv_medians < 0.20:
        logger.info("   ✓ GOOD: Sample medians reasonably uniform (CV < 20%%)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(sample_medians, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(sample_medians.mean(), color="red", linestyle="--",
                    label=f"Mean={sample_medians.mean():.3f}")
    axes[0].axvline(sample_medians.median(), color="orange", linestyle="--",
                    label=f"Median={sample_medians.median():.3f}")
    axes[0].set_xlabel("Sample Median (log2)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Sample Medians")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(sample_means, bins=50, edgecolor="black", alpha=0.7, color="green")
    axes[1].axvline(sample_means.mean(), color="red", linestyle="--",
                    label=f"Mean={sample_means.mean():.3f}")
    axes[1].set_xlabel("Sample Mean (log2)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Sample Means")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].hist(sample_stds, bins=50, edgecolor="black", alpha=0.7, color="purple")
    axes[2].axvline(sample_stds.mean(), color="red", linestyle="--",
                    label=f"Mean={sample_stds.mean():.3f}")
    axes[2].set_xlabel("Sample Std Dev (log2)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Distribution of Sample Std Devs")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(_dirs["distributions"], f"{file_prefix}_sample_distributions.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("   Saved: %s", out_path)

    # ------------------------------------------------------------------
    # 2. Sample distribution boxplots
    # ------------------------------------------------------------------
    logger.info("[2/4] Sample distribution boxplots")

    n_samples_to_plot = min(50, X_log2.shape[0])
    fig, ax = plt.subplots(figsize=(20, 6))
    X_log2.iloc[:n_samples_to_plot].T.boxplot(ax=ax)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("log2(Protein Abundance)")
    ax.set_title(f"Sample Distribution Boxplots (first {n_samples_to_plot} samples)")
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=90)
    plt.tight_layout()
    out_path = os.path.join(_dirs["boxplots"], f"{file_prefix}_sample_boxplots.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("   Saved: %s", out_path)

    median_range = sample_medians.max() - sample_medians.min()
    if verbose:
        logger.info("   Boxplot median range: %.4f", median_range)
    if median_range < 0.5:
        logger.info("   ✓ EXCELLENT: Boxplot medians very well aligned (range < 0.5)")
    elif median_range < 1.0:
        logger.info("   ✓ GOOD: Boxplot medians reasonably aligned (range < 1.0)")
    else:
        logger.info("   ✗ WARNING: Boxplot medians show substantial variation (range >= 1.0)")

    # ------------------------------------------------------------------
    # 3. Density overlay
    # ------------------------------------------------------------------
    logger.info("[3/4] Sample density overlay")

    n_samples_density = min(20, X_log2.shape[0])
    rng = np.random.default_rng(_RNG_SEED)
    sample_indices = rng.choice(X_log2.index, n_samples_density, replace=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx in sample_indices:
        X_log2.loc[idx].plot.density(ax=ax, alpha=0.3, linewidth=1)
    ax.set_xlabel("log2(Protein Abundance)")
    ax.set_ylabel("Density")
    ax.set_title(f"Sample Density Plots (n={n_samples_density} random samples)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(_dirs["density"], f"{file_prefix}_density_overlay.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("   Saved: %s", out_path)

    # ------------------------------------------------------------------
    # 4. Batch comparison (optional)
    # ------------------------------------------------------------------
    if "Batch" in df.columns:
        logger.info("[4/4] Batch effect assessment")

        batches = df["Batch"]
        unique_batches = batches.unique()
        logger.info("   Number of batches: %d", len(unique_batches))
        if verbose:
            logger.info("   Batch distribution: %s", batches.value_counts().to_dict())

        if len(unique_batches) > 1:
            batch_means = {
                b: X_log2.loc[batches == b].median(axis=1).mean()
                for b in unique_batches
            }
            median_diff = max(batch_means.values()) - min(batch_means.values())
            logger.info("   Median difference between batches: %.4f", median_diff)
            if median_diff < 0.1:
                logger.info("   ✓ EXCELLENT: Negligible batch effect on medians (< 0.1)")
            elif median_diff < 0.3:
                logger.info("   ✓ GOOD: Small batch effect on medians (< 0.3)")
            else:
                logger.info("   ✗ WARNING: Noticeable batch effect on medians (>= 0.3)")

        data_for_boxplot = [
            X_log2.loc[batches == b].median(axis=1) for b in sorted(unique_batches)
        ]
        labels_for_boxplot = [f"Batch {b}" for b in sorted(unique_batches)]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(data_for_boxplot, labels=labels_for_boxplot)
        ax.set_ylabel("Sample Median (log2)")
        ax.set_title("Sample Medians by Batch")
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        out_path = os.path.join(_dirs["batch"], f"{file_prefix}_batch_comparison.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("   Saved: %s", out_path)
    else:
        logger.info("[4/4] Batch comparison skipped (no Batch column in output).")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("NORMALIZATION QUALITY SUMMARY")
    logger.info("=" * 70)

    issues = []
    if cv_stds >= 0.10:
        issues.append("Sample standard deviations are highly variable")
    if abs(sample_medians.mean()) > 0.1 and cv_medians >= 0.20:
        issues.append("Sample medians are highly variable")
    if median_range >= 1.0:
        issues.append("Boxplot medians show substantial variation")

    if not issues:
        logger.info("✓ NORMALIZATION QUALITY: EXCELLENT — no major issues detected.")
    else:
        logger.info("✗ NORMALIZATION QUALITY: NEEDS REVIEW — issues detected:")
        for issue in issues:
            logger.info("    - %s", issue)

    logger.info("Plots saved to: %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    wkdir = os.getcwd()
    data_dir = os.path.join(wkdir, "data", "proteomics")
    output_dir = os.path.join(wkdir, "data", "cleaned", "proteomics", "normalized_full_results")
    metadata_path = os.path.join(wkdir, "data", "dp3 master table v2.xlsx")

    # All diagnostic outputs live under a single diagnostics/ subfolder,
    # with one subdirectory per sample type (plasma / placenta), and each
    # plot type in its own subfolder within that.
    #
    # Final layout:
    #   diagnostics/
    #     plasma/
    #       pca/pre_combat/   post_combat/   comparison/
    #       sample_distributions/   sample_boxplots/
    #       density_overlay/   quantile_heatmap/   batch_comparison/
    #     placenta/
    #       (same structure)
    diag_base = os.path.join(output_dir, "diagnostics")

    plasma_files, placenta_files = collect_olink_files(data_dir)

    if plasma_files:
        out_csv_plasma = os.path.join(output_dir, "proteomics_plasma_cleaned_with_metadata.csv")
        plasma_diag = os.path.join(diag_base, "plasma")
        run_diagnostics(
            plasma_files, out_csv_plasma, metadata_path,
            meta_type="proteomics",
            pca_dir=os.path.join(plasma_diag, "pca"),
            combat_dir=os.path.join(plasma_diag, "combat_assessment"),
        )
        if os.path.exists(out_csv_plasma):
            plot_normalization_quality(
                out_csv_plasma,
                output_dir=plasma_diag,
                verbose=True,
            )

    if placenta_files:
        out_csv_placenta = os.path.join(output_dir, "proteomics_placenta_cleaned_with_metadata.csv")
        placenta_diag = os.path.join(diag_base, "placenta")
        run_diagnostics(
            placenta_files, out_csv_placenta, metadata_path,
            meta_type="placenta",
            pca_dir=os.path.join(placenta_diag, "pca"),
            combat_dir=os.path.join(placenta_diag, "combat_assessment"),
        )
        if os.path.exists(out_csv_placenta):
            plot_normalization_quality(
                out_csv_placenta,
                output_dir=placenta_diag,
                verbose=True,
            )
