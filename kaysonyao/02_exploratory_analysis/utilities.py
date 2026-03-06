"""
Utilities for intraomics differential analysis and heatmap visualization.

Shared constants, data loading helpers, statistical analysis functions
(cross-sectional and longitudinal), and heatmap generation functions used by:
  - identify_differential_analytes.py
  - generate_differential_cluster_heatmap_limited_group.py
"""

# Standard library
import datetime
import glob
import logging
import os
import platform

# Third-party
import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
import statsmodels.stats.multitest as multitest
import seaborn as sns
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

FDR_THRESHOLD    = 0.05
_METADATA_COLS   = ["SubjectID", "Group", "Subgroup", "Batch", "GestAgeDelivery", "SampleGestAge"]
_GROUP_LABEL_MAP = {"sptb": "sPTB"}

# Analysis-specific constants
MIN_N             = 5
LOG2_FC_THRESHOLD = np.log2(1.5)   # ≈ 0.585 — 1.5× linear FC in log2 space

# Heatmap-specific constants
MAX_ANALYTES    = 500
MIN_SIG_ANALYTES = 5
_PNG_DPI         = 300
_TIMEPOINT_ORDER = ["A", "B", "C", "D", "E"]

# Binary column colour scheme: Control (blue) vs Complication (orange-red)
_CTRL_COLOUR  = "steelblue"
_COMPL_COLOUR = "tomato"


# ---------------------------------------------------------------------------
# Shared data helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load cleaned wide-format CSV (index=SampleID, columns=metadata+analytes)."""
    return pd.read_csv(path, index_col=0)


def normalise_group_labels(df: pd.DataFrame, group_col: str = "Group") -> pd.DataFrame:
    """Standardise group label capitalisation using _GROUP_LABEL_MAP (returns df for chaining)."""
    if group_col in df.columns:
        before = df[group_col].value_counts()
        df[group_col] = df[group_col].replace(_GROUP_LABEL_MAP)
        for old, canonical in _GROUP_LABEL_MAP.items():
            n = before.get(old, 0)
            if n > 0:
                logger.info("  Group label fix: '%s' → '%s' (%d sample(s))", old, canonical, n)
    return df


def get_analyte_columns(df: pd.DataFrame) -> list:
    """Return analyte column names by excluding known metadata columns."""
    return [c for c in df.columns if c not in _METADATA_COLS]


# ---------------------------------------------------------------------------
# Analysis log
# ---------------------------------------------------------------------------

def _start_analysis_log(output_dir: str) -> None:
    """Attach a FileHandler to the root logger and write a parameter/version header.

    Creates <output_dir>/analysis_log.txt (overwriting any previous run).
    All subsequent logger calls in this process will be mirrored to that file.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "analysis_log.txt")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("=" * 70)
    logger.info("Intraomics Differential Analysis Log")
    logger.info("Date/time    : %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Python       : %s", platform.python_version())
    logger.info("scipy        : %s", scipy.__version__)
    logger.info("statsmodels  : %s", statsmodels.__version__)
    logger.info("pandas       : %s", pd.__version__)
    logger.info("numpy        : %s", np.__version__)
    logger.info("-" * 70)
    logger.info("Parameters:")
    logger.info("  MIN_N             = %d  (min non-missing observations per group)", MIN_N)
    logger.info("  FDR_THRESHOLD     = %.2f", FDR_THRESHOLD)
    logger.info("  LOG2_FC_THRESHOLD = %.4f  (log2(1.5); both CS and longitudinal)", LOG2_FC_THRESHOLD)
    logger.info("  FDR method        = Benjamini-Hochberg")
    logger.info("  CS test           = Mann-Whitney U (two-sided)")
    logger.info("  Long test         = Wilcoxon signed-rank (two-sided, zero_method='wilcox')")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Cross-sectional analysis
# ---------------------------------------------------------------------------

def _test_one_pair(v1: pd.Series, v2: pd.Series) -> dict:
    """Run Mann-Whitney U for one analyte between two groups.

    Returns a dict of result fields. Sets excluded=True and NaN statistics
    when either group has fewer than MIN_N non-missing observations.
    """
    v1 = v1.dropna()
    v2 = v2.dropna()
    if len(v1) < MIN_N or len(v2) < MIN_N:
        return {
            "U_statistic":  np.nan,
            "p_value":       np.nan,
            "median_group1": v1.median() if len(v1) else np.nan,
            "median_group2": v2.median() if len(v2) else np.nan,
            "fold_change":   np.nan,
            "n_group1":      len(v1),
            "n_group2":      len(v2),
            "_excluded":     True,
        }
    u_stat, p_val = stats.mannwhitneyu(v1, v2, alternative="two-sided", method="auto")
    med1, med2 = v1.median(), v2.median()
    # fold_change is the log2 difference (= log2 fold change in linear space),
    # since NPX values are already on a log2 scale.
    fc = med2 - med1
    return {
        "U_statistic":  u_stat,
        "p_value":       p_val,
        "median_group1": med1,
        "median_group2": med2,
        "fold_change":   fc,
        "n_group1":      len(v1),
        "n_group2":      len(v2),
        "_excluded":     False,
    }


def run_cross_sectional(
    df: pd.DataFrame,
    analyte_cols: list,
    group_col: str = "Group",
    output_dir: str = "results/cross_sectional",
    control_group: str = "Control",
) -> None:
    """Run Mann-Whitney U tests (control vs each complication group) with BH FDR correction.

    Only comparisons of the form (control_group, X) are run — one per non-control group.

    For each comparison:
      - Requires >= MIN_N non-missing observations per group per analyte.
      - Applies BH FDR correction independently per comparison.
      - Saves <g1>_vs_<g2>_differential_results.csv (all analytes, sorted by q-value).
      - Saves <g1>_vs_<g2>_significant_analytes.csv (q < FDR_THRESHOLD only).
    """
    os.makedirs(output_dir, exist_ok=True)
    groups = sorted(df[group_col].dropna().unique())
    pairs  = [(control_group, g) for g in groups if g != control_group]
    logger.info(
        "Cross-sectional: %s vs %d complication group(s) → %d comparison(s)",
        control_group, len(pairs), len(pairs),
    )

    for g1, g2 in pairs:
        df_g1 = df.loc[df[group_col] == g1, analyte_cols]
        df_g2 = df.loc[df[group_col] == g2, analyte_cols]

        rows = []
        for analyte in analyte_cols:
            row = _test_one_pair(df_g1[analyte], df_g2[analyte])
            row["analyte_id"] = analyte
            rows.append(row)

        results_df = pd.DataFrame(rows).set_index("analyte_id")

        tested_mask = ~results_df["_excluded"]
        results_df["q_value"]    = np.nan
        results_df["significant"] = False
        if tested_mask.sum() > 0:
            _, q_vals, _, _ = multitest.multipletests(
                results_df.loc[tested_mask, "p_value"], method="fdr_bh"
            )
            results_df.loc[tested_mask, "q_value"] = q_vals
            abs_fc = results_df.loc[tested_mask, "fold_change"].abs()
            results_df.loc[tested_mask, "significant"] = (
                (q_vals < FDR_THRESHOLD) & (abs_fc >= LOG2_FC_THRESHOLD)
            )

        results_df = results_df.rename(columns={"_excluded": "excluded"})
        col_order  = [
            "U_statistic", "p_value", "q_value", "significant",
            "median_group1", "median_group2", "fold_change",
            "n_group1", "n_group2", "excluded",
        ]
        results_df = results_df[col_order].sort_values("q_value")

        label    = f"{g1}_vs_{g2}"
        out_path = os.path.join(output_dir, f"{label}_differential_results.csv")
        results_df.to_csv(out_path)

        sig_df   = results_df[results_df["significant"] == True]
        sig_path = os.path.join(output_dir, f"{label}_significant_analytes.csv")
        sig_df.to_csv(sig_path)

        n_excluded = results_df["excluded"].sum()
        logger.info(
            "  %s → %d tested, %d excluded (n < %d), %d significant (q < %.2f)",
            label, tested_mask.sum(), n_excluded, MIN_N, len(sig_df), FDR_THRESHOLD,
        )


# ---------------------------------------------------------------------------
# Longitudinal analysis
# ---------------------------------------------------------------------------

def _compute_deltas(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    analyte_cols: list,
    subject_col: str,
) -> pd.DataFrame:
    """Compute per-participant deltas (T_b - T_a) for each analyte.

    Only participants with non-missing values at both timepoints are included
    per analyte. Returns a DataFrame (participants × analytes).
    """
    def _index_by_subject(df):
        if subject_col in df.columns:
            return df.set_index(subject_col)[analyte_cols]
        if df.index.name == subject_col:
            return df[analyte_cols]
        return df[analyte_cols]

    da = _index_by_subject(df_a)
    db = _index_by_subject(df_b)
    shared = da.index.intersection(db.index)
    return db.loc[shared] - da.loc[shared]


def run_longitudinal(
    timepoint_dfs: dict,
    analyte_cols: list,
    group: str,
    group_col: str = "Group",
    subject_col: str = "SubjectID",
    output_dir: str = "results/longitudinal",
) -> None:
    """Run Wilcoxon signed-rank tests on per-participant deltas.

    For each pair of ADJACENT timepoints (T_n, T_n+1) in the order supplied:
      - Computes delta = value_T_later - value_T_earlier per participant.
      - Requires >= MIN_N paired observations per analyte.
      - Applies BH FDR correction independently per delta comparison.
      - Saves <group>_<T_later>_minus_<T_earlier>_longitudinal_results.csv.

    Only adjacent steps are tested (e.g. B-A, C-B, D-C, E-D), not all
    pairwise combinations, per PI specification.
    """
    os.makedirs(output_dir, exist_ok=True)

    filtered = {}
    for label, df in timepoint_dfs.items():
        if group_col in df.columns:
            df = df.loc[df[group_col] == group].copy()
        elif df.index.name != subject_col and group_col not in df.columns:
            logger.warning(
                "No group column '%s' in timepoint '%s'; using all rows.", group_col, label
            )
        filtered[label] = df

    labels = list(filtered.keys())
    pairs  = list(zip(range(len(labels) - 1), range(1, len(labels))))
    logger.info(
        "Longitudinal [%s]: %d timepoints → %d adjacent delta comparisons",
        group, len(labels), len(pairs),
    )

    for i, j in pairs:
        t_a, t_b = labels[i], labels[j]
        deltas   = _compute_deltas(filtered[t_a], filtered[t_b], analyte_cols, subject_col)

        if deltas.shape[0] == 0:
            logger.warning(
                "  %s−%s [%s]: 0 paired subjects — skipping (no result file written).",
                t_b, t_a, group,
            )
            continue

        rows = []
        for analyte in analyte_cols:
            if analyte not in deltas.columns:
                continue
            d = deltas[analyte].dropna()
            if len(d) < MIN_N:
                rows.append({
                    "analyte_id":      analyte,
                    "delta_comparison": f"{t_b}_minus_{t_a}",
                    "W_statistic":     np.nan,
                    "p_value":          np.nan,
                    "median_delta":    d.median() if len(d) else np.nan,
                    "n_pairs":          len(d),
                    "_excluded":        True,
                })
                continue
            w_stat, p_val = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            rows.append({
                "analyte_id":      analyte,
                "delta_comparison": f"{t_b}_minus_{t_a}",
                "W_statistic":     w_stat,
                "p_value":          p_val,
                "median_delta":    d.median(),
                "n_pairs":          len(d),
                "_excluded":        False,
            })

        results_df  = pd.DataFrame(rows).set_index("analyte_id")
        tested_mask = ~results_df["_excluded"]
        results_df["q_value"]    = np.nan
        results_df["significant"] = False
        if tested_mask.sum() > 0:
            _, q_vals, _, _ = multitest.multipletests(
                results_df.loc[tested_mask, "p_value"], method="fdr_bh"
            )
            results_df.loc[tested_mask, "q_value"] = q_vals
            abs_delta = results_df.loc[tested_mask, "median_delta"].abs()
            results_df.loc[tested_mask, "significant"] = (
                (q_vals < FDR_THRESHOLD) & (abs_delta >= LOG2_FC_THRESHOLD)
            )

        results_df = results_df.rename(columns={"_excluded": "excluded"})
        col_order  = [
            "delta_comparison", "W_statistic", "p_value", "q_value",
            "significant", "median_delta", "n_pairs", "excluded",
        ]
        results_df = results_df[col_order].sort_values("q_value")

        fname    = f"{group}_{t_b}_minus_{t_a}_longitudinal_results.csv"
        out_path = os.path.join(output_dir, fname)
        results_df.to_csv(out_path)
        n_excluded = results_df["excluded"].sum()
        logger.info(
            "  %s−%s → %d tested, %d excluded (n < %d), %d significant (q < %.2f)",
            t_b, t_a, tested_mask.sum(), n_excluded, MIN_N,
            results_df["significant"].sum(), FDR_THRESHOLD,
        )



# ---------------------------------------------------------------------------
# Sample count report
# ---------------------------------------------------------------------------

def write_sample_count_report(
    output_dir: str,
    cleaned_dir_plasma: str,
    placenta_csv: str,
) -> None:
    """Write a sample count table: n per (sample_type, Group, timepoint).

    Reads per-timepoint plasma files (A–E) and the placenta cleaned CSV.
    Saves sample_counts_per_group_timepoint.csv and logs the pivot table.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for tp in _TIMEPOINT_ORDER:
        tp_csv = os.path.join(
            cleaned_dir_plasma, f"proteomics_plasma_formatted_suffix_{tp}.csv"
        )
        if not os.path.exists(tp_csv):
            continue
        df = normalise_group_labels(load_data(tp_csv))
        if "Group" not in df.columns:
            continue
        for grp, n in df["Group"].value_counts().items():
            rows.append({"sample_type": "plasma", "timepoint": tp, "Group": grp, "n": int(n)})

    if os.path.exists(placenta_csv):
        df_p = normalise_group_labels(load_data(placenta_csv))
        if "Group" in df_p.columns:
            for grp, n in df_p["Group"].value_counts().items():
                rows.append({"sample_type": "placenta", "timepoint": "—", "Group": grp, "n": int(n)})

    if not rows:
        logger.warning("Sample count report: no data found.")
        return

    report    = pd.DataFrame(rows)
    long_path = os.path.join(output_dir, "sample_counts_per_group_timepoint.csv")
    report.to_csv(long_path, index=False)

    pivot = report.pivot_table(
        index="Group",
        columns=["sample_type", "timepoint"],
        values="n",
        aggfunc="sum",
    ).fillna(0).astype(int)
    logger.info("Sample counts per group × timepoint:\n%s", pivot.to_string())
    logger.info("Sample count report saved: %s", long_path)


# ---------------------------------------------------------------------------
# Heatmap: shared helpers
# ---------------------------------------------------------------------------

def collect_significant_analytes(results_dir: str) -> set:
    """Union of significant analyte IDs across all pairwise comparison CSVs."""
    sig_files = glob.glob(os.path.join(results_dir, "*_significant_analytes.csv"))
    analytes  = set()
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
    Z    = linkage(dist, method="ward")
    return list(X.index[leaves_list(Z)])


# ---------------------------------------------------------------------------
# Heatmap: cross-sectional
# ---------------------------------------------------------------------------

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

    df          = pd.read_csv(data_path, index_col=0)
    analyte_cols = get_analyte_columns(df)
    sig_set      = collect_significant_analytes(results_dir)

    if len(sig_set) < MIN_SIG_ANALYTES:
        print(f"Skipping cross-sectional heatmap: only {len(sig_set)} significant analytes "
              f"(threshold = {MIN_SIG_ANALYTES}).")
        return

    # Significant analytes only
    sig_list = [a for a in analyte_cols if a in sig_set]
    if len(sig_list) > MAX_ANALYTES:
        min_q             = _rank_analytes_by_min_q(sig_list, results_dir)
        selected_analytes = min_q.nsmallest(MAX_ANALYTES).index.tolist()
        print(f"Capped at {MAX_ANALYTES} significant analytes.")
    else:
        selected_analytes = sig_list

    X       = df[selected_analytes].copy()
    col_mean = X.mean(axis=0)
    col_std  = X.std(axis=0).replace(0, np.nan)
    X_z      = (X - col_mean) / col_std
    X_z      = X_z.clip(-2.5, 2.5).fillna(0.0)

    groups        = df[group_col].dropna()
    group_order   = sorted(groups.unique())
    ordered_samples = []
    for g in group_order:
        g_idx = groups[groups == g].index
        g_X   = X_z.loc[X_z.index.isin(g_idx)]
        ordered_samples.extend(_within_group_order(g_X))

    X_z_plot       = X_z.loc[ordered_samples].T
    X_z_plot.index = X_z_plot.index.tolist()  # all rows are significant

    palette        = sns.color_palette("tab10", n_colors=len(group_order))
    group_color_map = dict(zip(group_order, palette))
    col_colors      = pd.Series(
        [group_color_map.get(groups.get(s, group_order[0]), "gray") for s in ordered_samples],
        index=ordered_samples,
        name=group_col,
    )

    n_samples    = X_z_plot.shape[1]
    n_total      = X_z_plot.shape[0]
    figw         = max(20, n_samples * 0.22 + 6)
    figh         = max(10, n_total * 0.20)
    row_fontsize = max(5, min(9, int(180 / n_total)))

    cg = sns.clustermap(
        X_z_plot,
        col_cluster=False,
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
        yticklabels=True,
        dendrogram_ratio=0.25,
    )
    cg.ax_heatmap.set_xlabel("")
    cg.ax_heatmap.set_ylabel("")
    plt.setp(cg.ax_heatmap.get_yticklabels(), fontsize=row_fontsize)
    cg.figure.suptitle(
        f"Cross-sectional Heatmap  |  {n_total} significant analytes  "
        f"|  {n_samples} samples",
        y=0.99, fontsize=11,
    )

    cg.figure.savefig(
        os.path.join(output_dir, f"{label}_heatmap.pdf"),
        bbox_inches="tight", pad_inches=0.3,
    )
    cg.figure.savefig(
        os.path.join(output_dir, f"{label}_heatmap.png"),
        dpi=_PNG_DPI, bbox_inches="tight", pad_inches=0.3,
    )
    plt.close(cg.figure)

    X_z_plot.to_csv(os.path.join(output_dir, f"{label}_heatmap_data.csv"))
    print(f"Saved cross-sectional heatmap: {n_total} significant analytes "
          f"× {n_samples} samples → {output_dir}/")


def plot_pairwise_cross_sectional_heatmap(
    g1: str,
    g2: str,
    data_path: str,
    results_dir: str,
    output_dir: str,
    group_col: str = "Group",
    g2_source_groups: list = None,
) -> None:
    """Generate a pairwise cross-sectional heatmap for one group comparison.

    Shows only significant analytes for this pair (up to MAX_ANALYTES, ranked by q-value).
    Column colour bar uses a binary Control vs Complication scheme.
    Saves outputs to <output_dir>/<g1>_vs_<g2>/ subfolder.

    Parameters
    ----------
    g2_source_groups : list, optional
        When g2 is a pooled label (e.g. "Complication"), supply the original
        group names that were merged into it (e.g. ["FGR", "HDP", "sPTB"]).
        The function will filter the data to [g1] + g2_source_groups and
        relabel g2_source_groups as g2 before plotting.
    """
    pair_label = f"{g1}_vs_{g2}"
    pair_dir   = os.path.join(output_dir, pair_label)

    result_csv = os.path.join(results_dir, f"{pair_label}_differential_results.csv")
    if not os.path.exists(result_csv):
        print(f"  Skipping {pair_label}: result CSV not found ({result_csv}).")
        return

    res     = pd.read_csv(result_csv, index_col=0)
    sig_set = set(res.index[res["significant"] == True].tolist())

    if len(sig_set) < MIN_SIG_ANALYTES:
        print(f"  Skipping {pair_label}: only {len(sig_set)} significant analytes "
              f"(threshold = {MIN_SIG_ANALYTES}).")
        return

    os.makedirs(pair_dir, exist_ok=True)

    df = pd.read_csv(data_path, index_col=0)
    if g2_source_groups is not None:
        # Filter to g1 + underlying source groups, then relabel them as g2
        mask = df[group_col].isin([g1] + list(g2_source_groups))
        df   = df.loc[mask].copy()
        df.loc[df[group_col] != g1, group_col] = g2
    else:
        mask = df[group_col].isin([g1, g2])
        df   = df.loc[mask].copy()
    analyte_cols = get_analyte_columns(df)

    # Significant analytes only
    sig_list = [a for a in analyte_cols if a in sig_set]
    if len(sig_list) > MAX_ANALYTES:
        sig_list_ranked = (
            res.loc[[a for a in sig_list if a in res.index], "q_value"]
            .sort_values()
            .index.tolist()
        )
        selected_analytes = sig_list_ranked[:MAX_ANALYTES]
        print(f"  {pair_label}: capped at {MAX_ANALYTES} significant analytes.")
    else:
        selected_analytes = sig_list

    X        = df[selected_analytes].copy()
    col_mean = X.mean(axis=0)
    col_std  = X.std(axis=0).replace(0, np.nan)
    X_z      = (X - col_mean) / col_std
    X_z      = X_z.clip(-2.5, 2.5).fillna(0.0)

    groups          = df[group_col]
    ordered_samples = []
    for g in [g1, g2]:
        g_idx = groups[groups == g].index
        g_X   = X_z.loc[X_z.index.isin(g_idx)]
        ordered_samples.extend(_within_group_order(g_X))

    X_z_plot       = X_z.loc[ordered_samples].T
    X_z_plot.index = X_z_plot.index.tolist()  # all rows are significant

    # Binary colour bar: Control (steelblue) vs Complication (tomato)
    col_colors = pd.Series(
        [_CTRL_COLOUR if groups[s] == "Control" else _COMPL_COLOUR for s in ordered_samples],
        index=ordered_samples,
        name="Control vs Complication",
    )

    n_samples    = X_z_plot.shape[1]
    n_total      = X_z_plot.shape[0]
    figw         = max(20, n_samples * 0.22 + 6)
    figh         = max(10, n_total * 0.20)
    row_fontsize = max(5, min(9, int(180 / n_total)))

    cg = sns.clustermap(
        X_z_plot,
        col_cluster=False,
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
        yticklabels=True,
        dendrogram_ratio=0.25,
    )
    cg.ax_heatmap.set_xlabel("")
    cg.ax_heatmap.set_ylabel("")
    plt.setp(cg.ax_heatmap.get_yticklabels(), fontsize=row_fontsize)
    cg.figure.suptitle(
        f"Cross-sectional Heatmap  |  {g1} vs {g2}  |  "
        f"{n_total} significant analytes  |  {n_samples} samples",
        y=0.99, fontsize=11,
    )

    cg.figure.savefig(
        os.path.join(pair_dir, f"{pair_label}_heatmap.pdf"),
        bbox_inches="tight", pad_inches=0.3,
    )
    cg.figure.savefig(
        os.path.join(pair_dir, f"{pair_label}_heatmap.png"),
        dpi=_PNG_DPI, bbox_inches="tight", pad_inches=0.3,
    )
    plt.close(cg.figure)

    X_z_plot.to_csv(os.path.join(pair_dir, f"{pair_label}_heatmap_data.csv"))
    print(f"  Saved {pair_label}: {n_total} significant analytes "
          f"× {n_samples} samples → {pair_dir}/")


# ---------------------------------------------------------------------------
# Heatmap: longitudinal
# ---------------------------------------------------------------------------

def _sort_delta_columns(columns: list) -> list:
    """Sort delta comparison column labels chronologically.

    Ordering rule: earlier T_a first (chronological anchor); within the same
    T_a, T_b ascending (shorter span before longer).

    Column name format: "<T_b>_minus_<T_a>"
    """
    def _sort_key(col: str):
        parts = col.split("_minus_", 1)
        if len(parts) != 2:
            return (999, 999)
        t_b, t_a = parts
        b_idx = _TIMEPOINT_ORDER.index(t_b) if t_b in _TIMEPOINT_ORDER else 999
        a_idx = _TIMEPOINT_ORDER.index(t_a) if t_a in _TIMEPOINT_ORDER else 999
        return (a_idx, b_idx)

    return sorted(columns, key=_sort_key)


def _is_adjacent_comparison(comp_label: str) -> bool:
    """Return True if comp_label is a single-step adjacent comparison.

    Adjacent means T_b and T_a are consecutive entries in _TIMEPOINT_ORDER
    (index difference == 1). Filters out stale non-adjacent result files.
    """
    parts = comp_label.split("_minus_", 1)
    if len(parts) != 2:
        return False
    t_b, t_a = parts
    if t_b not in _TIMEPOINT_ORDER or t_a not in _TIMEPOINT_ORDER:
        return False
    return _TIMEPOINT_ORDER.index(t_b) - _TIMEPOINT_ORDER.index(t_a) == 1


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

    delta_medians = {}
    sig_per_comp  = {}

    for f in result_files:
        res = pd.read_csv(f, index_col=0)
        if "delta_comparison" not in res.columns or res.empty:
            continue
        comp_label = res["delta_comparison"].iloc[0]
        if not _is_adjacent_comparison(comp_label):
            print(f"  Skipping non-adjacent comparison: {comp_label}")
            continue
        delta_medians[comp_label] = res["median_delta"]
        sig_per_comp[comp_label]  = set(res.index[res["significant"] == True].tolist())

    if not delta_medians:
        print("No valid longitudinal results to plot.")
        return

    matrix = pd.DataFrame(delta_medians)
    matrix = matrix[_sort_delta_columns(list(matrix.columns))]

    all_sig = set().union(*sig_per_comp.values())
    if len(all_sig) < MIN_SIG_ANALYTES:
        print(f"Skipping longitudinal heatmap: only {len(all_sig)} significant analytes "
              f"(threshold = {MIN_SIG_ANALYTES}).")
        return

    # Significant analytes only
    sig_list = [a for a in matrix.index if a in all_sig]
    if len(sig_list) > MAX_ANALYTES:
        n_sig_comps  = pd.Series(
            {a: sum(a in s for s in sig_per_comp.values()) for a in sig_list}
        )
        all_analytes = n_sig_comps.nlargest(MAX_ANALYTES).index.tolist()
        print(f"Capped at {MAX_ANALYTES} significant analytes.")
    else:
        all_analytes = sig_list

    matrix   = matrix.loc[all_analytes]
    row_mean = matrix.mean(axis=1)
    row_std  = matrix.std(axis=1).replace(0, np.nan)
    matrix_z = matrix.sub(row_mean, axis=0).div(row_std, axis=0).clip(-2.0, 2.0).fillna(0.0)

    n_total      = matrix_z.shape[0]
    n_sig        = len(all_sig)
    n_comps      = matrix_z.shape[1]
    figw         = max(18, n_comps * 2.5 + 8)
    figh         = max(8, n_total * 0.20)
    row_fontsize = max(5, min(9, int(180 / n_total)))
    col_labels   = [c.replace("_minus_", "\u2212") for c in matrix_z.columns]

    cg = sns.clustermap(
        matrix_z,
        col_cluster=False,
        row_cluster=True,
        method="ward",
        metric="euclidean",
        cmap="RdBu_r",
        center=0,
        vmin=-2.0,
        vmax=2.0,
        figsize=(figw, figh),
        xticklabels=col_labels,
        yticklabels=True,
        dendrogram_ratio=0.25,
    )

    reordered_rows = [matrix_z.index[i] for i in cg.dendrogram_row.reordered_ind]
    ax = cg.ax_heatmap

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
    plt.setp(cg.ax_heatmap.get_yticklabels(), fontsize=row_fontsize)
    cg.figure.suptitle(
        f"Longitudinal Heatmap  |  {group}  |  {n_total} analytes shown  "
        f"({n_sig} significant ●)  |  {n_comps} comparisons",
        y=0.99, fontsize=11,
    )

    cg.figure.savefig(
        os.path.join(output_dir, f"{group}_longitudinal_heatmap.pdf"),
        bbox_inches="tight", pad_inches=0.3,
    )
    cg.figure.savefig(
        os.path.join(output_dir, f"{group}_longitudinal_heatmap.png"),
        dpi=_PNG_DPI, bbox_inches="tight", pad_inches=0.3,
    )
    plt.close(cg.figure)

    matrix_z.to_csv(os.path.join(output_dir, f"{group}_longitudinal_heatmap_data.csv"))
    print(f"Saved longitudinal heatmap: {n_total} analytes ({n_sig} significant) "
          f"× {n_comps} comparisons → {output_dir}/")


