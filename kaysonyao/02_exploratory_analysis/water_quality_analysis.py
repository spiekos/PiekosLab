"""water_quality_analysis.py
============================
Compare water-quality metrics (average THM concentrations and exceedance
rates) between Control and each complication group (FGR, HDP, sPTB).

Input
-----
  data/cleaned/survey/water_cleaned.csv
    Must contain columns produced by clean_survey_data.py:
      SubjectID, Group, Subgroup,
      TTHM_avg, Br.THM_avg, CHCl3_avg, CHBr3_avg, BDCM_avg, CDBM_avg,
      TTHM_exceed_rate, CHCl3_exceed_rate, CHBr3_exceed_rate,
      BDCM_exceed_rate, CDBM_exceed_rate

Statistical approach
--------------------
  • Kruskal-Wallis H  : overall test across all 4 groups.
  • Mann-Whitney U    : pairwise Control vs FGR / HDP / sPTB.
  • BH FDR            : applied across all pairwise tests within
                        each panel (avg or exceed).

Output
------
  04_results_and_figures/survey/water/
    water_avg_distribution.png       – boxplot: average concentrations
    water_exceed_distribution.png    – boxplot: exceedance rates
    water_combined_distribution.png  – two-panel figure (avg | exceed)
    water_avg_stats_results.csv
    water_avg_significant_pairs.csv
    water_exceed_stats_results.csv
    water_exceed_significant_pairs.csv
"""

import os
import warnings
import logging
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.dirname(HERE)
WATER_CSV = os.path.join(ROOT, "data", "cleaned", "survey", "water_cleaned.csv")
OUT_DIR   = os.path.join(ROOT, "04_results_and_figures", "survey", "water")
os.makedirs(OUT_DIR, exist_ok=True)

PNG_DPI   = 300
FDR_ALPHA = 0.05
MIN_N     = 5

GROUP_ORDER   = ["Control", "FGR", "HDP", "sPTB"]
GROUP_PALETTE = {
    "Control": "steelblue",
    "FGR":     "tomato",
    "HDP":     "darkorange",
    "sPTB":    "mediumpurple",
}

# Analyte panels
AVG_ANALYTES = [
    ("TTHM_avg",   "TTHM"),
    ("Br.THM_avg", "Br-THM"),
    ("CHCl3_avg",  "CHCl₃"),
    ("CHBr3_avg",  "CHBr₃"),
    ("BDCM_avg",   "BDCM"),
    ("CDBM_avg",   "CDBM"),
]

EXCEED_ANALYTES = [
    ("TTHM_exceed_rate",   "TTHM"),
    ("CHCl3_exceed_rate",  "CHCl₃"),
    ("CHBr3_exceed_rate",  "CHBr₃"),
    ("BDCM_exceed_rate",   "BDCM"),
    ("CDBM_exceed_rate",   "CDBM"),
]


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def kruskal_wallis(groups_data: dict) -> tuple:
    arrays = [v for v in groups_data.values() if len(v) >= MIN_N]
    if len(arrays) < 2:
        return np.nan, np.nan
    all_vals = np.concatenate(arrays)
    if np.all(all_vals == all_vals[0]):   # all identical — test is undefined
        return np.nan, np.nan
    try:
        h, p = stats.kruskal(*arrays)
    except ValueError:
        return np.nan, np.nan
    return float(h), float(p)


def mann_whitney_pairwise(df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    rows = []
    comparisons = [(g1, g2) for g1, g2 in itertools.combinations(GROUP_ORDER, 2)
                   if g1 == "Control"]
    for g1, g2 in comparisons:
        a = df.loc[df["Group"] == g1, col].dropna().values
        b = df.loc[df["Group"] == g2, col].dropna().values
        if len(a) < MIN_N or len(b) < MIN_N:
            rows.append(dict(analyte=label, g1=g1, g2=g2,
                             n_g1=len(a), n_g2=len(b),
                             U=np.nan, p_value=np.nan, q_value=np.nan,
                             significant=False, excluded=True))
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(dict(analyte=label, g1=g1, g2=g2,
                         n_g1=len(a), n_g2=len(b),
                         U=float(u), p_value=float(p), q_value=np.nan,
                         significant=False, excluded=False))

    result = pd.DataFrame(rows)
    tested = result[~result["excluded"]]
    if len(tested):
        reject, q_vals, _, _ = multipletests(tested["p_value"], method="fdr_bh")
        result.loc[~result["excluded"], "q_value"]    = q_vals
        result.loc[~result["excluded"], "significant"] = reject
    return result


# ---------------------------------------------------------------------------
# Plotting helper: one analyte → one axes
# ---------------------------------------------------------------------------
def _plot_one_analyte(ax, df, col, label, stats_row_df):
    """Box + strip plot for a single analyte, coloured by group."""
    present = [g for g in GROUP_ORDER if g in df["Group"].values
               and df.loc[df["Group"] == g, col].dropna().shape[0] >= 1]
    if not present:
        ax.set_visible(False)
        return

    # Box plot
    sns.boxplot(
        data=df[df["Group"].isin(present)],
        x="Group", y=col, order=present,
        palette=GROUP_PALETTE,
        width=0.55, linewidth=1.2,
        fliersize=0, ax=ax,
    )
    # Strip
    sns.stripplot(
        data=df[df["Group"].isin(present)],
        x="Group", y=col, order=present,
        palette=GROUP_PALETTE,
        size=4, alpha=0.65, jitter=True, ax=ax,
    )

    # n per group
    ylo, yhi = ax.get_ylim()
    for i, grp in enumerate(present):
        n = df.loc[df["Group"] == grp, col].dropna().shape[0]
        ax.text(i, ylo - 0.04 * (yhi - ylo), f"n={n}",
                ha="center", va="top", fontsize=7)

    # Significance brackets for this analyte
    sig = stats_row_df[
        (stats_row_df["analyte"] == label) & (stats_row_df["significant"] == True)
    ]
    y_top = df[col].dropna().max()
    y_range = df[col].dropna().max() - df[col].dropna().min()
    step = y_range * 0.10 if y_range > 0 else 0.1
    for _, row in sig.iterrows():
        if row["g1"] not in present or row["g2"] not in present:
            continue
        x1, x2 = present.index(row["g1"]), present.index(row["g2"])
        y_top += step
        ax.plot([x1, x1, x2, x2],
                [y_top - step * 0.3, y_top, y_top, y_top - step * 0.3],
                lw=1, color="black")
        q = row["q_value"]
        sig_str = "***" if q < 0.001 else ("**" if q < 0.01 else "*")
        ax.text((x1 + x2) / 2, y_top + step * 0.1, sig_str,
                ha="center", va="bottom", fontsize=8)

    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8)


# ---------------------------------------------------------------------------
# Panel-level plot: one analyte list → one figure
# ---------------------------------------------------------------------------
def plot_panel(
    df: pd.DataFrame,
    analytes: list,          # list of (col_name, display_label)
    panel_title: str,
    y_label: str,
    out_path: str,
    stats_df: pd.DataFrame,
) -> None:
    n = len(analytes)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 4.5 * nrows),
                              squeeze=False)

    for idx, (col, label) in enumerate(analytes):
        r, c = divmod(idx, ncols)
        if col not in df.columns:
            axes[r][c].set_visible(False)
            continue
        _plot_one_analyte(axes[r][c], df, col, label, stats_df)
        if c == 0:
            axes[r][c].set_ylabel(y_label, fontsize=9)

    # Hide unused axes
    for idx in range(len(analytes), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    # Shared legend
    patches = [mpatches.Patch(color=GROUP_PALETTE[g], label=g)
               for g in GROUP_ORDER]
    fig.legend(handles=patches, title="Group",
               loc="upper right", bbox_to_anchor=(0.99, 0.99),
               fontsize=10, title_fontsize=11,
               framealpha=0.9, ncol=2)

    fig.suptitle(panel_title, y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Combined two-panel figure (avg | exceed rate)
# ---------------------------------------------------------------------------
def plot_combined(
    df: pd.DataFrame,
    avg_stats: pd.DataFrame,
    exc_stats: pd.DataFrame,
    out_path: str,
) -> None:
    """
    Two-column figure:
      Left  column  = average concentrations (AVG_ANALYTES)
      Right column  = exceedance rates       (EXCEED_ANALYTES)
    Each column has its own rows of subplots (one subplot per analyte).
    """
    n_avg = len(AVG_ANALYTES)
    n_exc = len(EXCEED_ANALYTES)
    n_rows = max(n_avg, n_exc)

    fig = plt.figure(figsize=(14, 4.2 * n_rows))
    # Left block: column 1-3; right block: column 4-6 (add gap column between)
    gs = fig.add_gridspec(n_rows, 7, hspace=0.55, wspace=0.45)

    def make_axes(panel_analytes, col_offset, stats_df, y_label, header):
        n = len(panel_analytes)
        ncols_panel = min(n, 3)
        for idx, (col, label) in enumerate(panel_analytes):
            r   = idx // ncols_panel
            c   = idx %  ncols_panel
            ax  = fig.add_subplot(gs[r, col_offset + c])
            if col not in df.columns:
                ax.set_visible(False)
                continue
            _plot_one_analyte(ax, df, col, label, stats_df)
            if c == 0:
                ax.set_ylabel(y_label, fontsize=8)
        # panel header as text
        header_ax = fig.add_subplot(gs[0, col_offset : col_offset + ncols_panel])
        header_ax.set_visible(False)
        fig.text(
            (col_offset + ncols_panel / 2) / 7, 1.005,
            header, ha="center", va="bottom",
            fontsize=12, fontweight="bold",
            transform=fig.transFigure,
        )

    make_axes(AVG_ANALYTES,    0, avg_stats, "Concentration (µg/L)",
              "Average Concentrations")
    make_axes(EXCEED_ANALYTES, 4, exc_stats, "Exceedance Rate (fraction)",
              "Exceedance Rates")

    # Shared legend
    patches = [mpatches.Patch(color=GROUP_PALETTE[g], label=g)
               for g in GROUP_ORDER]
    fig.legend(handles=patches, title="Group",
               loc="upper right", bbox_to_anchor=(0.99, 1.01),
               fontsize=9, title_fontsize=10,
               framealpha=0.9, ncol=1)

    fig.suptitle("Water Quality — Control vs Complication Groups",
                 y=1.03, fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Run statistics for one set of analytes
# ---------------------------------------------------------------------------
def run_stats(df: pd.DataFrame, analytes: list) -> pd.DataFrame:
    all_rows = []
    for col, label in analytes:
        if col not in df.columns:
            logger.warning("Column %s not found — skipping.", col)
            continue
        kw_h, kw_p = kruskal_wallis(
            {g: df.loc[df["Group"] == g, col].dropna().values
             for g in GROUP_ORDER}
        )
        logger.info("  %-22s KW H=%.3f  p=%.4f",
                    label,
                    kw_h if not np.isnan(kw_h) else -1,
                    kw_p if not np.isnan(kw_p) else 1)
        pw = mann_whitney_pairwise(df, col, label)
        pw["KW_H"] = kw_h
        pw["KW_p"] = kw_p
        all_rows.append(pw)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(WATER_CSV):
        logger.error("Water CSV not found: %s", WATER_CSV)
        logger.error("Run clean_survey_data.py first.")
        return

    df = pd.read_csv(WATER_CSV)
    logger.info("Loaded water data: %d rows, %d columns.", *df.shape)
    logger.info("Groups present: %s", df["Group"].value_counts().to_dict())

    # ── Statistics ──────────────────────────────────────────────────────────
    logger.info("=== Average concentrations ===")
    avg_stats = run_stats(df, AVG_ANALYTES)

    logger.info("=== Exceedance rates ===")
    exc_stats = run_stats(df, EXCEED_ANALYTES)

    # Save stats
    for stats_df, tag in [(avg_stats, "avg"), (exc_stats, "exceed")]:
        if stats_df.empty:
            continue
        stats_df.to_csv(os.path.join(OUT_DIR, f"water_{tag}_stats_results.csv"),
                        index=False)
        sig_df = stats_df[stats_df["significant"] == True]
        sig_df.to_csv(os.path.join(OUT_DIR, f"water_{tag}_significant_pairs.csv"),
                      index=False)
        logger.info("%s: %d significant pairs (BH q < %.2f).",
                    tag.upper(), len(sig_df), FDR_ALPHA)

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_panel(
        df, AVG_ANALYTES,
        panel_title="Water Quality — Average THM Concentrations",
        y_label="Concentration (µg/L)",
        out_path=os.path.join(OUT_DIR, "water_avg_distribution.png"),
        stats_df=avg_stats,
    )

    plot_panel(
        df, EXCEED_ANALYTES,
        panel_title="Water Quality — Exceedance Rates",
        y_label="Exceedance Rate (fraction of samples above limit)",
        out_path=os.path.join(OUT_DIR, "water_exceed_distribution.png"),
        stats_df=exc_stats,
    )

    plot_combined(
        df, avg_stats, exc_stats,
        out_path=os.path.join(OUT_DIR, "water_combined_distribution.png"),
    )

    logger.info("Water quality analysis complete.")


if __name__ == "__main__":
    main()
