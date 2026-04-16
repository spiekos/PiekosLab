"""survey_distribution_analysis.py
==================================
Check score distributions across groups for each survey instrument
(EPDS, PSS, PUQE24, Diet) and test whether scores differ between
Control and each complication group at each visit.

Statistical approach
--------------------
  Survey scores are ordinal / skewed → non-parametric tests.
    • Kruskal-Wallis H  : overall test across all 4 groups.
    • Mann-Whitney U    : pairwise Control vs FGR / HDP / sPTB.
    • BH FDR            : applied across all pairwise tests within
                         each survey × visit combination.

  Same framework as the omics differential analysis so results are
  directly comparable.

Input
-----
  data/cleaned/survey/{epds,pss,puqe24,diet}_cleaned.csv

Output
------
  04_results_and_figures/survey/
    {survey}/
      {survey}_{visit}_distribution.png   – violin + strip plot
      {survey}_stats_results.csv          – all pairwise test results
      {survey}_significant_pairs.csv      – BH-significant pairs only
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
HERE       = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(HERE)
SURVEY_DIR = os.path.join(ROOT, "data", "cleaned", "survey")
OUT_ROOT   = os.path.join(ROOT, "04_results_and_figures", "survey")

PNG_DPI    = 300
FDR_ALPHA  = 0.05
MIN_N      = 5          # minimum group size to test

GROUP_ORDER    = ["Control", "FGR", "HDP", "sPTB"]
GROUP_PALETTE  = {
    "Control": "steelblue",
    "FGR":     "tomato",
    "HDP":     "darkorange",
    "sPTB":    "mediumpurple",
}

# Survey instrument → (csv filename, score column, y-axis label, title)
SURVEYS = {
    "epds":   ("epds_cleaned.csv",   "score",            "EPDS Score",             "Edinburgh Postnatal Depression Scale"),
    "pss":    ("pss_cleaned.csv",    "score",            "PSS Score",              "Perceived Stress Scale"),
    "puqe24": ("puqe24_cleaned.csv", "score",            "PUQE-24 Score",          "Pregnancy Nausea/Vomiting (PUQE-24)"),
}

VISIT_ORDER  = ["A", "C", "D", "PP"]
VISIT_LABELS = {"A": "Visit A\n(5–13 wk)", "C": "Visit C\n(20–28 wk)",
                "D": "Visit D\n(29–36 wk)", "PP": "Postpartum"}


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def kruskal_wallis(groups_data: dict[str, np.ndarray]) -> tuple[float, float]:
    """Run Kruskal-Wallis across all provided groups (skip those with n < MIN_N)."""
    arrays = [v for v in groups_data.values() if len(v) >= MIN_N]
    if len(arrays) < 2:
        return np.nan, np.nan
    h, p = stats.kruskal(*arrays)
    return float(h), float(p)


def mann_whitney_pairwise(
    df: pd.DataFrame,
    score_col: str,
    survey: str,
    visit: str,
) -> pd.DataFrame:
    """
    Pairwise Mann-Whitney U: Control vs each complication group.
    Returns a DataFrame of results with BH-corrected q-values.
    """
    rows = []
    comparisons = [(g1, g2) for g1, g2 in itertools.combinations(GROUP_ORDER, 2)
                   if g1 == "Control"]

    for g1, g2 in comparisons:
        a = df.loc[df["Group"] == g1, score_col].dropna().values
        b = df.loc[df["Group"] == g2, score_col].dropna().values
        if len(a) < MIN_N or len(b) < MIN_N:
            rows.append(dict(survey=survey, visit=visit, g1=g1, g2=g2,
                             n_g1=len(a), n_g2=len(b),
                             U=np.nan, p_value=np.nan, q_value=np.nan,
                             significant=False, excluded=True))
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(dict(survey=survey, visit=visit, g1=g1, g2=g2,
                         n_g1=len(a), n_g2=len(b),
                         U=float(u), p_value=float(p), q_value=np.nan,
                         significant=False, excluded=False))

    result = pd.DataFrame(rows)

    # BH correction across tested pairs
    tested = result[~result["excluded"]]
    if len(tested):
        reject, q_vals, _, _ = multipletests(tested["p_value"], method="fdr_bh")
        result.loc[~result["excluded"], "q_value"]    = q_vals
        result.loc[~result["excluded"], "significant"] = reject

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_distribution(
    df: pd.DataFrame,
    score_col: str,
    visit: str,
    survey_title: str,
    y_label: str,
    out_path: str,
    stats_df: pd.DataFrame,
) -> None:
    """Violin + strip plot for one survey × visit, one panel per group."""
    present_groups = [g for g in GROUP_ORDER if g in df["Group"].values]
    if not present_groups:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(present_groups) * 2), 5))

    # Violin
    sns.violinplot(
        data=df, x="Group", y=score_col,
        order=present_groups,
        palette=GROUP_PALETTE,
        inner=None, linewidth=1.2,
        ax=ax,
    )
    # Strip
    sns.stripplot(
        data=df, x="Group", y=score_col,
        order=present_groups,
        palette=GROUP_PALETTE,
        size=3, alpha=0.6, jitter=True,
        ax=ax,
    )

    # Annotate n per group
    for i, grp in enumerate(present_groups):
        n = df[df["Group"] == grp][score_col].dropna().shape[0]
        ax.text(i, ax.get_ylim()[0] - 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"n={n}", ha="center", va="top", fontsize=8)

    # Mark significant pairs
    sig = stats_df[stats_df["significant"] == True]
    y_max = df[score_col].max()
    y_range = df[score_col].max() - df[score_col].min()
    step = y_range * 0.08
    for _, row in sig.iterrows():
        if row["g1"] not in present_groups or row["g2"] not in present_groups:
            continue
        x1 = present_groups.index(row["g1"])
        x2 = present_groups.index(row["g2"])
        y_max += step
        ax.plot([x1, x1, x2, x2], [y_max - step * 0.3, y_max, y_max, y_max - step * 0.3],
                lw=1, color="black")
        q = row["q_value"]
        sig_str = "***" if q < 0.001 else ("**" if q < 0.01 else "*")
        ax.text((x1 + x2) / 2, y_max + step * 0.1, sig_str,
                ha="center", va="bottom", fontsize=9)

    visit_label = VISIT_LABELS.get(visit, visit)
    ax.set_title(f"{survey_title}  —  {visit_label}", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(y_label, fontsize=10)
    ax.tick_params(axis="x", labelsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main per-survey pipeline
# ---------------------------------------------------------------------------
def analyse_survey(survey_key: str) -> None:
    csv_file, score_col, y_label, title = SURVEYS[survey_key]
    csv_path = os.path.join(SURVEY_DIR, csv_file)
    if not os.path.exists(csv_path):
        logger.warning("Missing %s — skipping.", csv_path)
        return

    out_dir = os.path.join(OUT_ROOT, survey_key)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Find visit column
    visit_col = "Visit" if "Visit" in df.columns else None
    if visit_col is None:
        logger.error("%s: no Visit column found.", survey_key)
        return

    all_stats: list[pd.DataFrame] = []

    visits_present = [v for v in VISIT_ORDER if v in df[visit_col].values]
    logger.info("%s: visits found = %s", survey_key.upper(), visits_present)

    for visit in visits_present:
        sub = df[df[visit_col] == visit].copy()
        sub = sub[sub["Group"].isin(GROUP_ORDER)]

        if sub[score_col].dropna().empty:
            logger.info("  %s visit %s: no score data, skipping.", survey_key, visit)
            continue

        # Stats
        kw_h, kw_p = kruskal_wallis(
            {g: sub.loc[sub["Group"] == g, score_col].dropna().values
             for g in GROUP_ORDER}
        )
        logger.info(
            "  %s visit %s: KW H=%.3f p=%.4f | n=%d",
            survey_key, visit, kw_h if not np.isnan(kw_h) else -1,
            kw_p if not np.isnan(kw_p) else 1, len(sub),
        )

        pw = mann_whitney_pairwise(sub, score_col, survey_key, visit)
        pw["KW_H"] = kw_h
        pw["KW_p"] = kw_p
        all_stats.append(pw)

        # Plot
        plot_path = os.path.join(out_dir, f"{survey_key}_{visit}_distribution.png")
        plot_distribution(sub, score_col, visit, title, y_label, plot_path, pw)
        logger.info("  Saved: %s", plot_path)

    if all_stats:
        stats_df = pd.concat(all_stats, ignore_index=True)
        stats_df.to_csv(os.path.join(out_dir, f"{survey_key}_stats_results.csv"), index=False)
        sig_df = stats_df[stats_df["significant"] == True]
        sig_df.to_csv(os.path.join(out_dir, f"{survey_key}_significant_pairs.csv"), index=False)
        logger.info(
            "%s: %d significant pairs (BH q < %.2f).",
            survey_key.upper(), len(sig_df), FDR_ALPHA,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    for survey_key in SURVEYS:
        logger.info("=== %s ===", survey_key.upper())
        analyse_survey(survey_key)
    logger.info("Survey distribution analysis complete.")


if __name__ == "__main__":
    main()
