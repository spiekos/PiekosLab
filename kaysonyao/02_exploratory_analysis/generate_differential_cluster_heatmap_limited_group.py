"""
Heatmap visualization for intraomics differential analysis results.

Cross-sectional heatmap:
    Rows = significant analytes (q < 0.05 in >= 1 pairwise comparison); capped at
           100, ranked by minimum q-value across comparisons.
    Columns = individual samples, clustered within each group separately then
              concatenated in alphabetical group order.
    Values = z-score of per-analyte abundance across all samples, clipped ±2.5.
    Row clustering: Ward linkage, Euclidean distance.

Longitudinal heatmap:
    Rows = analytes significant (q < 0.05) in >= 1 delta comparison; capped at
           100, ranked by number of significant comparisons.
    Columns = delta comparisons in fixed chronological order (earlier T_a first,
              T_b ascending within; not clustered).
    Values = median delta per analyte per comparison, row-wise z-scored, clipped ±2.0.
    Row clustering: Ward linkage, Euclidean distance.
    Significant cells (q < 0.05) marked with a dot; non-significant cells dimmed.

Usage:
    # Cross-sectional
    python generate_differential_cluster_heatmap_limited_group.py \
        --mode cross_sectional \
        --input cleaned.csv \
        --results-dir results/cross_sectional \
        --output-dir results/cross_sectional

    # Longitudinal
    python generate_differential_cluster_heatmap_limited_group.py \
        --mode longitudinal \
        --results-dir results/longitudinal \
        --output-dir results/longitudinal \
        --group Control
"""

import os
import sys
import glob
import argparse
import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

FDR_THRESHOLD = 0.05
MAX_ANALYTES = 100          # cap keeps row height readable; ranked by significance
MIN_SIG_ANALYTES = 5

# Per-format DPI: PNG uses 150 (smaller file, still sharp); PDF is vector so DPI is ignored.
_PNG_DPI = 150
_PDF_DPI = 300              # used only as a fallback; PDF output is vector-based

# Chronological timepoint order for this project (plasma suffixes).
# Used to sort longitudinal delta comparison columns correctly.
_TIMEPOINT_ORDER = ["A", "B", "C", "D", "E", "EA", "EB", "EC", "ED", "EE"]

_METADATA_COLS = [
    "SubjectID", "Group", "Subgroup", "Batch",
    "GestAgeDelivery", "SampleGestAge",
]


def get_analyte_columns(df: pd.DataFrame) -> list:
    """Return analyte column names, excluding known metadata columns."""
    return [c for c in df.columns if c not in _METADATA_COLS]


# ---------------------------------------------------------------------------
# Cross-sectional heatmap
# ---------------------------------------------------------------------------

def collect_significant_analytes(results_dir: str) -> set:
    """Union of significant analyte IDs across all pairwise comparison CSVs."""
    sig_files = glob.glob(os.path.join(results_dir, "*_significant_analytes.csv"))
    analytes = set()
    for f in sorted(sig_files):
        df = pd.read_csv(f, index_col=0)
        analytes.update(df.index.tolist())
    return analytes


def _rank_analytes_by_min_q(sig_analytes: list, results_dir: str) -> pd.Series:
    """Return a Series of min q-value per analyte across all result CSVs."""
    min_q = pd.Series(1.0, index=sig_analytes)
    for f in sorted(glob.glob(os.path.join(results_dir, "*_differential_results.csv"))):
        res = pd.read_csv(f, index_col=0)
        if "q_value" not in res.columns:
            continue
        for analyte in sig_analytes:
            if analyte in res.index:
                q = res.loc[analyte, "q_value"]
                if pd.notna(q) and q < min_q[analyte]:
                    min_q[analyte] = q
    return min_q


def _within_group_order(X: pd.DataFrame) -> list:
    """Return sample order after Ward/Euclidean clustering within this group."""
    if X.shape[0] <= 2:
        return list(X.index)
    X_clean = X.fillna(X.median())
    dist = pdist(X_clean.values, metric="euclidean")
    Z = linkage(dist, method="ward")
    return list(X.index[leaves_list(Z)])


def plot_cross_sectional_heatmap(
    data_path: str,
    results_dir: str,
    output_dir: str,
    group_col: str = "Group",
    label: str = "cross_sectional",
) -> None:
    """Generate and save a cross-sectional z-score heatmap.

    Saves <label>_heatmap.pdf, <label>_heatmap.png (300 dpi),
    and <label>_heatmap_data.csv to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path, index_col=0)
    analyte_cols = get_analyte_columns(df)

    # Collect significant analytes across all comparisons.
    sig_set = collect_significant_analytes(results_dir)
    sig_analytes = [a for a in analyte_cols if a in sig_set]

    if len(sig_analytes) < MIN_SIG_ANALYTES:
        print(f"Skipping cross-sectional heatmap: only {len(sig_analytes)} significant analytes "
              f"(threshold = {MIN_SIG_ANALYTES}).")
        return

    # Cap at MAX_ANALYTES, ranking by minimum q-value across comparisons.
    if len(sig_analytes) > MAX_ANALYTES:
        min_q = _rank_analytes_by_min_q(sig_analytes, results_dir)
        sig_analytes = min_q.nsmallest(MAX_ANALYTES).index.tolist()
        print(f"Capped at {MAX_ANALYTES} analytes (ranked by minimum q-value).")

    # Z-score per analyte (row) across all samples.
    X = df[sig_analytes].copy()
    col_mean = X.mean(axis=0)
    col_std = X.std(axis=0).replace(0, np.nan)
    X_z = (X - col_mean) / col_std
    X_z = X_z.clip(-2.5, 2.5).fillna(0.0)

    # Cluster samples within each group separately, concatenate groups.
    groups = df[group_col].dropna()
    group_order = sorted(groups.unique())
    ordered_samples = []
    for g in group_order:
        g_idx = groups[groups == g].index
        g_X = X_z.loc[X_z.index.isin(g_idx)]
        ordered_samples.extend(_within_group_order(g_X))

    # Build the heatmap matrix: analytes × samples.
    X_z_plot = X_z.loc[ordered_samples].T

    # Group color bar for columns.
    palette = sns.color_palette("tab10", n_colors=len(group_order))
    group_color_map = dict(zip(group_order, palette))
    col_colors = pd.Series(
        [group_color_map.get(groups.get(s, group_order[0]), "gray") for s in ordered_samples],
        index=ordered_samples,
        name=group_col,
    )

    n_samples = X_z_plot.shape[1]
    n_sig = X_z_plot.shape[0]
    # Row height: 0.20 in per analyte keeps labels readable; minimum 10 in
    figw = max(12, n_samples * 0.15)
    figh = max(10, n_sig * 0.20)
    # Font size for row labels: shrink a bit if many rows
    row_fontsize = max(5, min(9, int(180 / n_sig)))

    cg = sns.clustermap(
        X_z_plot,
        col_cluster=False,       # columns ordered by group; no cross-group clustering
        row_cluster=True,
        method="ward",
        metric="euclidean",
        cmap="RdBu_r",
        center=0,
        vmin=-2.5,
        vmax=2.5,
        col_colors=col_colors,
        figsize=(figw, figh),
        xticklabels=False,
        yticklabels=True,        # always show analyte labels
    )
    cg.ax_heatmap.set_xlabel("")
    cg.ax_heatmap.set_ylabel("")
    # Apply row label font size after draw
    plt.setp(cg.ax_heatmap.get_yticklabels(), fontsize=row_fontsize)
    # Place suptitle inside the figure boundary to avoid PDF clipping
    cg.fig.suptitle(
        f"Cross-sectional Heatmap  |  {n_sig} analytes  |  {n_samples} samples",
        y=0.99, fontsize=11,
    )

    cg.fig.savefig(
        os.path.join(output_dir, f"{label}_heatmap.pdf"),
        bbox_inches="tight", pad_inches=0.3,
    )
    cg.fig.savefig(
        os.path.join(output_dir, f"{label}_heatmap.png"),
        dpi=_PNG_DPI, bbox_inches="tight", pad_inches=0.3,
    )
    plt.close(cg.fig)

    X_z_plot.to_csv(os.path.join(output_dir, f"{label}_heatmap_data.csv"))
    print(f"Saved cross-sectional heatmap: {n_sig} analytes × {n_samples} samples → {output_dir}/")


# ---------------------------------------------------------------------------
# Longitudinal heatmap
# ---------------------------------------------------------------------------

def _sort_delta_columns(columns: list) -> list:
    """Sort delta comparison column labels chronologically.

    Ordering rule (matching SOP example): earlier T_a first (chronological
    anchor); within the same T_a, T_b ascending (shorter span before longer).

    Example for 4 timepoints: B−A, C−A, D−A, C−B, D−B, D−C
    which reads left-to-right as earliest transition → latest transition.

    Timepoint order defined by _TIMEPOINT_ORDER (prenatal A–E, then
    postnatal EA–EE where EA mirrors A, EB mirrors B, etc.).

    Column name format: "<T_b>_minus_<T_a>"
    Falls back to alphabetical for labels not in _TIMEPOINT_ORDER.
    """
    def _sort_key(col: str):
        parts = col.split("_minus_", 1)
        if len(parts) != 2:
            return (999, 999)
        t_b, t_a = parts
        b_idx = _TIMEPOINT_ORDER.index(t_b) if t_b in _TIMEPOINT_ORDER else 999
        a_idx = _TIMEPOINT_ORDER.index(t_a) if t_a in _TIMEPOINT_ORDER else 999
        return (a_idx, b_idx)   # earlier T_a first; within same T_a, T_b ascending

    return sorted(columns, key=_sort_key)


def plot_longitudinal_heatmap(
    results_dir: str,
    output_dir: str,
    group: str,
) -> None:
    """Generate and save a longitudinal median-delta heatmap.

    Saves <group>_longitudinal_heatmap.pdf, .png (300 dpi),
    and <group>_longitudinal_heatmap_data.csv to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    result_files = sorted(glob.glob(
        os.path.join(results_dir, f"{group}_*_longitudinal_results.csv")
    ))
    if not result_files:
        print(f"No longitudinal result files found for group '{group}' in {results_dir}.")
        return

    # Build matrix: rows=analytes, columns=delta comparisons.
    delta_medians = {}     # comp_label → pd.Series(analyte → median_delta)
    sig_per_comp = {}      # comp_label → set of significant analyte IDs

    for f in result_files:
        res = pd.read_csv(f, index_col=0)
        if "delta_comparison" not in res.columns or res.empty:
            continue
        comp_label = res["delta_comparison"].iloc[0]
        delta_medians[comp_label] = res["median_delta"]
        sig_per_comp[comp_label] = set(
            res.index[res["significant"] == True].tolist()
        )

    if not delta_medians:
        print("No valid longitudinal results to plot.")
        return

    # Build matrix and sort columns chronologically (earlier T_a first, T_b ascending within).
    matrix = pd.DataFrame(delta_medians)
    matrix = matrix[_sort_delta_columns(list(matrix.columns))]

    # Keep analytes significant in at least one delta comparison.
    all_sig = set().union(*sig_per_comp.values())
    sig_analytes = [a for a in matrix.index if a in all_sig]

    if len(sig_analytes) < MIN_SIG_ANALYTES:
        print(f"Skipping longitudinal heatmap: only {len(sig_analytes)} significant analytes "
              f"(threshold = {MIN_SIG_ANALYTES}).")
        return

    # Cap at MAX_ANALYTES, ranked by number of significant comparisons then min q-value.
    if len(sig_analytes) > MAX_ANALYTES:
        n_sig_comps = pd.Series(
            {a: sum(a in s for s in sig_per_comp.values()) for a in sig_analytes}
        )
        sig_analytes = n_sig_comps.nlargest(MAX_ANALYTES).index.tolist()
        print(f"Capped at {MAX_ANALYTES} analytes (ranked by number of significant comparisons).")

    matrix = matrix.loc[sig_analytes]

    # Row-wise z-score across delta comparisons.
    row_mean = matrix.mean(axis=1)
    row_std = matrix.std(axis=1).replace(0, np.nan)
    matrix_z = matrix.sub(row_mean, axis=0).div(row_std, axis=0).clip(-2.0, 2.0).fillna(0.0)

    n_sig = len(sig_analytes)
    n_comps = matrix_z.shape[1]
    # Row height: 0.20 in per analyte; column width: 1.4 in per comparison
    figw = max(10, n_comps * 1.4 + 4)
    figh = max(8, n_sig * 0.20)
    row_fontsize = max(5, min(9, int(180 / n_sig)))

    # Column labels using delta notation (− instead of _minus_).
    col_labels = [c.replace("_minus_", "\u2212") for c in matrix_z.columns]

    cg = sns.clustermap(
        matrix_z,
        col_cluster=False,       # columns in fixed chronological order
        row_cluster=True,
        method="ward",
        metric="euclidean",
        cmap="RdBu_r",
        center=0,
        vmin=-2.0,
        vmax=2.0,
        figsize=(figw, figh),
        xticklabels=col_labels,
        yticklabels=True,        # always show analyte labels
    )

    # Get the row order from the dendrogram.
    reordered_rows = [matrix_z.index[i] for i in cg.dendrogram_row.reordered_ind]
    ax = cg.ax_heatmap

    # Overlay: dim non-significant cells (alpha=0.4 white rectangle),
    # mark significant cells with a dot.
    for row_i, analyte in enumerate(reordered_rows):
        for col_j, comp in enumerate(matrix_z.columns):
            is_sig = analyte in sig_per_comp.get(comp, set())
            if not is_sig:
                ax.add_patch(plt.Rectangle(
                    (col_j, row_i), 1, 1,
                    fill=True, color="white", alpha=0.6, zorder=2,
                ))
            else:
                ax.text(
                    col_j + 0.5, row_i + 0.5, "\u2022",
                    ha="center", va="center", fontsize=8, color="black", zorder=3,
                )

    ax.set_xlabel("Delta comparison")
    ax.set_ylabel("")
    # Apply row label font size after draw
    plt.setp(cg.ax_heatmap.get_yticklabels(), fontsize=row_fontsize)
    # Place suptitle inside the figure boundary to avoid PDF clipping
    cg.fig.suptitle(
        f"Longitudinal Heatmap  |  {group}  |  {n_sig} analytes  |  {n_comps} comparisons",
        y=0.99, fontsize=11,
    )

    cg.fig.savefig(
        os.path.join(output_dir, f"{group}_longitudinal_heatmap.pdf"),
        bbox_inches="tight", pad_inches=0.3,
    )
    cg.fig.savefig(
        os.path.join(output_dir, f"{group}_longitudinal_heatmap.png"),
        dpi=_PNG_DPI, bbox_inches="tight", pad_inches=0.3,
    )
    plt.close(cg.fig)

    matrix_z.to_csv(os.path.join(output_dir, f"{group}_longitudinal_heatmap_data.csv"))
    print(f"Saved longitudinal heatmap: {n_sig} analytes × {n_comps} comparisons → {output_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Differential analysis heatmap visualization."
    )
    p.add_argument(
        "--mode",
        choices=["cross_sectional", "longitudinal", "both"],
        default="cross_sectional",
        help="Heatmap type to generate (default: cross_sectional).",
    )
    p.add_argument(
        "--input",
        help="Cleaned wide-format CSV (rows=samples). Required for cross_sectional mode.",
    )
    p.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing differential results CSVs.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save heatmap outputs.",
    )
    p.add_argument(
        "--group-col",
        default="Group",
        help="Column name for group labels in the cleaned CSV (default: Group).",
    )
    p.add_argument(
        "--group",
        help="Target group for longitudinal heatmap (e.g. Control). "
             "Required for longitudinal mode.",
    )
    p.add_argument(
        "--label",
        default="cross_sectional",
        help="Filename prefix for cross-sectional heatmap outputs (default: cross_sectional).",
    )
    return p


def main():
    if len(sys.argv) == 1:
        _run_default_mode()
        return

    parser = _build_parser()
    args = parser.parse_args()

    if args.mode in ("cross_sectional", "both"):
        if not args.input:
            parser.error("--input is required for cross_sectional mode.")
        plot_cross_sectional_heatmap(
            data_path=args.input,
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            group_col=args.group_col,
            label=args.label,
        )
    if args.mode in ("longitudinal", "both"):
        if not args.group:
            parser.error("--group is required for longitudinal mode.")
        plot_longitudinal_heatmap(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            group=args.group,
        )



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    def _run_default_mode() -> None:
        wkdir = os.getcwd()
        cleaned_dir_placenta = os.path.join(
            wkdir, "data", "cleaned", "proteomics", "normalized_full_results"
        )
        cleaned_dir_plasma = os.path.join(
            wkdir, "data", "cleaned", "proteomics", "normalized_sliced_by_suffix"
        )
        diff_dir = os.path.join(wkdir, "data", "diff_analysis", "results")

        # ── Cross-sectional heatmap: placenta ────────────────────────────
        placenta_csv  = os.path.join(cleaned_dir_placenta, "proteomics_placenta_cleaned_with_metadata.csv")
        cross_results = os.path.join(diff_dir, "placenta", "cross_sectional")
        if os.path.exists(placenta_csv) and os.path.isdir(cross_results):
            plot_cross_sectional_heatmap(
                data_path=placenta_csv,
                results_dir=cross_results,
                output_dir=cross_results,
                group_col="Group",
                label="placenta_cross_sectional",
            )
        else:
            logger.warning(
                "Skipping placenta cross-sectional heatmap: CSV or results dir not found.\n"
                "  CSV:     %s\n  Results: %s", placenta_csv, cross_results,
            )

        # ── Cross-sectional heatmaps: plasma per timepoint (A, B, C, D) ──
        cs_plasma_timepoints = ["A", "B", "C", "D"]
        for tp in cs_plasma_timepoints:
            tp_csv        = os.path.join(cleaned_dir_plasma, "proteomics_plasma_formatted_suffix_{}.csv".format(tp))
            tp_results    = os.path.join(diff_dir, "plasma", "cross_sectional", tp)
            if os.path.exists(tp_csv) and os.path.isdir(tp_results):
                plot_cross_sectional_heatmap(
                    data_path=tp_csv,
                    results_dir=tp_results,
                    output_dir=tp_results,
                    group_col="Group",
                    label="plasma_{}_cross_sectional".format(tp),
                )
            else:
                logger.warning(
                    "Skipping plasma cross-sectional heatmap [%s]: CSV or results dir not found.\n"
                    "  CSV:     %s\n  Results: %s", tp, tp_csv, tp_results,
                )

        # ── Longitudinal heatmap: plasma ─────────────────────────────────
        long_results = os.path.join(diff_dir, "plasma", "longitudinal")
        for group in ["Control"]:
            if os.path.isdir(long_results):
                plot_longitudinal_heatmap(
                    results_dir=long_results,
                    output_dir=long_results,
                    group=group,
                )
            else:
                logger.warning(
                    "Skipping longitudinal heatmap [%s]: results dir not found: %s",
                    group, long_results,
                )

        logger.info("Heatmap generation complete.")

    main()
