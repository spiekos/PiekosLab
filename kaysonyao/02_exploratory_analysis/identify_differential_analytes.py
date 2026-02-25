"""
Intraomics differential analysis: cross-sectional and longitudinal.

Cross-sectional:
    Mann-Whitney U test + Benjamini-Hochberg FDR (q < 0.05), all pairwise
    group comparisons derived from the Group column, min n=5 per analyte.

Longitudinal:
    Wilcoxon signed-rank test on per-participant deltas (T_later - T_earlier),
    all pairwise timepoint combinations for a specified group, min n=5 paired
    observations per analyte per delta comparison.

Usage:
    # Cross-sectional
    python identify_differential_analytes.py \
        --mode cross_sectional \
        --input cleaned.csv \
        --output-dir results

    # Longitudinal
    python identify_differential_analytes.py \
        --mode longitudinal \
        --timepoint-files t1.csv t2.csv t3.csv \
        --timepoint-labels T1 T2 T3 \
        --group Control \
        --output-dir results
"""

import os
import sys
import itertools
import argparse
import logging
import datetime
import platform

import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy import stats
import statsmodels.stats.multitest as multitest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MIN_N = 5
FDR_THRESHOLD = 0.05

# Metadata columns embedded in the cleaned wide-format CSV.
_METADATA_COLS = [
    "SubjectID", "Group", "Subgroup", "Batch",
    "GestAgeDelivery", "SampleGestAge",
]


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
    logger.info("  MIN_N          = %d  (min non-missing observations per group)", MIN_N)
    logger.info("  FDR_THRESHOLD  = %.2f", FDR_THRESHOLD)
    logger.info("  FDR method     = Benjamini-Hochberg")
    logger.info("  CS test        = Mann-Whitney U (two-sided)")
    logger.info("  Long test      = Wilcoxon signed-rank (two-sided, zero_method='wilcox')")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

# Canonical group label corrections
_GROUP_LABEL_MAP = {"sptb": "sPTB"}


def load_data(path: str) -> pd.DataFrame:
    """Load cleaned wide-format CSV (index=SampleID, columns=metadata+analytes)."""
    return pd.read_csv(path, index_col=0)


def normalise_group_labels(df: pd.DataFrame, group_col: str = "Group") -> pd.DataFrame:
    """Standardise group label capitalisation using _GROUP_LABEL_MAP (returns df for chaining)."""
    if group_col in df.columns:
        before = df[group_col].value_counts()
        df[group_col] = df[group_col].replace(_GROUP_LABEL_MAP)
        after = df[group_col].value_counts()
        for old, canonical in _GROUP_LABEL_MAP.items():
            n = before.get(old, 0)
            if n > 0:
                logger.info(
                    "  Group label fix: '%s' → '%s' (%d sample(s))", old, canonical, n
                )
    return df


def get_analyte_columns(df: pd.DataFrame) -> list:
    """Return analyte columns by excluding known metadata columns."""
    return [c for c in df.columns if c not in _METADATA_COLS]


# ---------------------------------------------------------------------------
# Cross-sectional
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
            "U_statistic": np.nan,
            "p_value": np.nan,
            "median_group1": v1.median() if len(v1) else np.nan,
            "median_group2": v2.median() if len(v2) else np.nan,
            "fold_change": np.nan,
            "n_group1": len(v1),
            "n_group2": len(v2),
            "_excluded": True,
        }
    u_stat, p_val = stats.mannwhitneyu(v1, v2, alternative="two-sided", method="auto")
    med1, med2 = v1.median(), v2.median()
    fc = (med2 / med1) if med1 != 0 else np.nan
    return {
        "U_statistic": u_stat,
        "p_value": p_val,
        "median_group1": med1,
        "median_group2": med2,
        "fold_change": fc,
        "n_group1": len(v1),
        "n_group2": len(v2),
        "_excluded": False,
    }


def run_cross_sectional(
    df: pd.DataFrame,
    analyte_cols: list,
    group_col: str = "Group",
    output_dir: str = "results/cross_sectional",
) -> None:
    """Run all pairwise Mann-Whitney U tests with per-comparison BH FDR correction.

    For each pairwise group comparison:
      - Requires >= MIN_N non-missing observations per group per analyte.
      - Applies BH FDR correction independently per comparison.
      - Saves <g1>_vs_<g2>_differential_results.csv (all analytes, sorted by q-value).
      - Saves <g1>_vs_<g2>_significant_analytes.csv (q < FDR_THRESHOLD only).
    """
    os.makedirs(output_dir, exist_ok=True)
    groups = sorted(df[group_col].dropna().unique())
    pairs = list(itertools.combinations(groups, 2))
    logger.info(
        "Cross-sectional: %d groups → %d pairwise comparisons", len(groups), len(pairs)
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

        # BH correction applied per comparison on testable analytes only.
        tested_mask = ~results_df["_excluded"]
        results_df["q_value"] = np.nan
        results_df["significant"] = False
        if tested_mask.sum() > 0:
            _, q_vals, _, _ = multitest.multipletests(
                results_df.loc[tested_mask, "p_value"], method="fdr_bh"
            )
            results_df.loc[tested_mask, "q_value"] = q_vals
            results_df.loc[tested_mask, "significant"] = q_vals < FDR_THRESHOLD

        results_df = results_df.rename(columns={"_excluded": "excluded"})
        col_order = [
            "U_statistic", "p_value", "q_value", "significant",
            "median_group1", "median_group2", "fold_change",
            "n_group1", "n_group2", "excluded",
        ]
        results_df = results_df[col_order].sort_values("q_value")

        label = f"{g1}_vs_{g2}"
        out_path = os.path.join(output_dir, f"{label}_differential_results.csv")
        results_df.to_csv(out_path)

        sig_df = results_df[results_df["significant"] == True]
        sig_path = os.path.join(output_dir, f"{label}_significant_analytes.csv")
        sig_df.to_csv(sig_path)

        n_excluded = results_df["excluded"].sum()
        logger.info(
            "  %s → %d tested, %d excluded (n < %d), %d significant (q < %.2f)",
            label, tested_mask.sum(), n_excluded, MIN_N, len(sig_df), FDR_THRESHOLD,
        )


# ---------------------------------------------------------------------------
# Longitudinal
# ---------------------------------------------------------------------------

def _compute_deltas(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    analyte_cols: list,
    subject_col: str,
) -> pd.DataFrame:
    """Compute per-participant deltas (T_b - T_a) for each analyte.

    Only participants with non-missing values at both timepoints are included
    per analyte.  Returns a DataFrame (participants × analytes).
    """
    # Ensure subject column is the index for alignment.
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

    For each pairwise timepoint combination (T_earlier, T_later):
      - Computes delta = value_T_later - value_T_earlier per participant.
      - Requires >= MIN_N paired observations per analyte.
      - Applies BH FDR correction independently per delta comparison.
      - Saves <group>_<T_later>_minus_<T_earlier>_longitudinal_results.csv.

    Args:
        timepoint_dfs:  Ordered dict mapping timepoint label → DataFrame.
        analyte_cols:   List of analyte column names.
        group:          Target group label (e.g. 'Control').
        group_col:      Column containing group labels.
        subject_col:    Column containing participant IDs.
        output_dir:     Directory for output CSVs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter each timepoint DataFrame to the target group.
    filtered = {}
    for label, df in timepoint_dfs.items():
        if group_col in df.columns:
            df = df.loc[df[group_col] == group].copy()
        elif df.index.name != subject_col and group_col not in df.columns:
            logger.warning("No group column '%s' in timepoint '%s'; using all rows.", group_col, label)
        filtered[label] = df

    labels = list(filtered.keys())
    pairs = list(itertools.combinations(range(len(labels)), 2))
    logger.info(
        "Longitudinal [%s]: %d timepoints → %d delta comparisons",
        group, len(labels), len(pairs),
    )

    for i, j in pairs:
        t_a, t_b = labels[i], labels[j]
        deltas = _compute_deltas(filtered[t_a], filtered[t_b], analyte_cols, subject_col)

        rows = []
        for analyte in analyte_cols:
            if analyte not in deltas.columns:
                continue
            d = deltas[analyte].dropna()
            if len(d) < MIN_N:
                rows.append({
                    "analyte_id": analyte,
                    "delta_comparison": f"{t_b}_minus_{t_a}",
                    "W_statistic": np.nan,
                    "p_value": np.nan,
                    "median_delta": d.median() if len(d) else np.nan,
                    "n_pairs": len(d),
                    "_excluded": True,
                })
                continue
            w_stat, p_val = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            rows.append({
                "analyte_id": analyte,
                "delta_comparison": f"{t_b}_minus_{t_a}",
                "W_statistic": w_stat,
                "p_value": p_val,
                "median_delta": d.median(),
                "n_pairs": len(d),
                "_excluded": False,
            })

        results_df = pd.DataFrame(rows).set_index("analyte_id")
        tested_mask = ~results_df["_excluded"]
        results_df["q_value"] = np.nan
        results_df["significant"] = False
        if tested_mask.sum() > 0:
            _, q_vals, _, _ = multitest.multipletests(
                results_df.loc[tested_mask, "p_value"], method="fdr_bh"
            )
            results_df.loc[tested_mask, "q_value"] = q_vals
            results_df.loc[tested_mask, "significant"] = q_vals < FDR_THRESHOLD

        results_df = results_df.rename(columns={"_excluded": "excluded"})
        col_order = [
            "delta_comparison", "W_statistic", "p_value", "q_value",
            "significant", "median_delta", "n_pairs", "excluded",
        ]
        results_df = results_df[col_order].sort_values("q_value")

        fname = f"{group}_{t_b}_minus_{t_a}_longitudinal_results.csv"
        out_path = os.path.join(output_dir, fname)
        results_df.to_csv(out_path)
        n_excluded = results_df["excluded"].sum()
        logger.info(
            "  %s−%s → %d tested, %d excluded (n < %d), %d significant (q < %.2f)",
            t_b, t_a, tested_mask.sum(), n_excluded, MIN_N,
            results_df["significant"].sum(), FDR_THRESHOLD,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Intraomics differential analysis (cross-sectional and longitudinal)."
    )
    p.add_argument(
        "--mode",
        choices=["cross_sectional", "longitudinal", "both"],
        default="cross_sectional",
        help="Analysis mode (default: cross_sectional).",
    )
    p.add_argument(
        "--input",
        help="Cleaned wide-format CSV (rows=samples, columns=metadata+analytes). "
             "Required for cross_sectional mode.",
    )
    p.add_argument(
        "--group-col",
        default="Group",
        help="Column name for group labels (default: Group).",
    )
    p.add_argument(
        "--output-dir",
        default="results",
        help="Root output directory. Cross-sectional results go to <output-dir>/cross_sectional/, "
             "longitudinal to <output-dir>/longitudinal/ (default: results).",
    )
    # Longitudinal-specific arguments.
    p.add_argument(
        "--timepoint-files",
        nargs="+",
        help="One cleaned CSV per timepoint, in chronological order. "
             "Required for longitudinal mode.",
    )
    p.add_argument(
        "--timepoint-labels",
        nargs="+",
        help="Labels for each timepoint file (e.g. T1 T2 T3). "
             "Defaults to T1, T2, ... if omitted.",
    )
    p.add_argument(
        "--group",
        help="Target group for longitudinal analysis (e.g. Control). "
             "Required for longitudinal mode.",
    )
    p.add_argument(
        "--subject-col",
        default="SubjectID",
        help="Column name for participant IDs used to pair observations (default: SubjectID).",
    )
    return p


def main():
    if len(sys.argv) == 1:
        _run_default_mode()
        return

    parser = _build_parser()
    args = parser.parse_args()

    cross_dir = os.path.join(args.output_dir, "cross_sectional")
    long_dir = os.path.join(args.output_dir, "longitudinal")

    # Start log file in the root output directory.
    _start_analysis_log(args.output_dir)

    if args.mode in ("cross_sectional", "both"):
        if not args.input:
            parser.error("--input is required for cross_sectional mode.")
        df = load_data(args.input)
        analyte_cols = get_analyte_columns(df)
        logger.info(
            "Loaded %d samples × %d analytes from %s",
            df.shape[0], len(analyte_cols), args.input,
        )
        run_cross_sectional(df, analyte_cols, group_col=args.group_col, output_dir=cross_dir)

    if args.mode in ("longitudinal", "both"):
        if not args.timepoint_files:
            parser.error("--timepoint-files is required for longitudinal mode.")
        if not args.group:
            parser.error("--group is required for longitudinal mode.")

        labels = args.timepoint_labels or [
            f"T{i + 1}" for i in range(len(args.timepoint_files))
        ]
        if len(labels) != len(args.timepoint_files):
            parser.error("--timepoint-labels count must match --timepoint-files count.")

        timepoint_dfs = {}
        for label, path in zip(labels, args.timepoint_files):
            timepoint_dfs[label] = load_data(path)

        analyte_cols = get_analyte_columns(next(iter(timepoint_dfs.values())))
        logger.info(
            "Loaded %d timepoints, %d analytes", len(timepoint_dfs), len(analyte_cols)
        )
        run_longitudinal(
            timepoint_dfs,
            analyte_cols,
            group=args.group,
            group_col=args.group_col,
            subject_col=args.subject_col,
            output_dir=long_dir,
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
        output_dir = os.path.join(wkdir, "data", "diff_analysis", "results")

        # Start log file at the root results directory.
        _start_analysis_log(output_dir)

        # ── Cross-sectional: placenta ─────────────────────────────────────
        placenta_csv = os.path.join(
            cleaned_dir_placenta, "proteomics_placenta_cleaned_with_metadata.csv"
        )
        if os.path.exists(placenta_csv):
            df = normalise_group_labels(load_data(placenta_csv))
            analyte_cols = get_analyte_columns(df)
            logger.info(
                "Cross-sectional [placenta]: %d samples x %d analytes",
                df.shape[0], len(analyte_cols),
            )
            run_cross_sectional(
                df,
                analyte_cols,
                group_col="Group",
                output_dir=os.path.join(output_dir, "placenta", "cross_sectional"),
            )
        else:
            logger.warning("Input not found, skipping cross-sectional: %s", placenta_csv)

        # ── Cross-sectional: plasma per timepoint (A, B, C, D only) ──────
        cs_plasma_timepoints = ["A", "B", "C", "D"]
        for tp in cs_plasma_timepoints:
            tp_csv = os.path.join(
                cleaned_dir_plasma, "proteomics_plasma_formatted_suffix_{}.csv".format(tp)
            )
            if not os.path.exists(tp_csv):
                logger.warning(
                    "Plasma timepoint %s not found, skipping cross-sectional: %s", tp, tp_csv
                )
                continue
            df_tp = normalise_group_labels(load_data(tp_csv))
            analyte_cols_tp = get_analyte_columns(df_tp)
            logger.info(
                "Cross-sectional [plasma %s]: %d samples x %d analytes",
                tp, df_tp.shape[0], len(analyte_cols_tp),
            )
            run_cross_sectional(
                df_tp,
                analyte_cols_tp,
                group_col="Group",
                output_dir=os.path.join(output_dir, "plasma", "cross_sectional", tp),
            )

        # ── Longitudinal: plasma all timepoints ───────────────────────────
        timepoint_files = {
            "A":  os.path.join(cleaned_dir_plasma, "proteomics_plasma_formatted_suffix_A.csv"),
            "B":  os.path.join(cleaned_dir_plasma, "proteomics_plasma_formatted_suffix_B.csv"),
            "C":  os.path.join(cleaned_dir_plasma, "proteomics_plasma_formatted_suffix_C.csv"),
            "D":  os.path.join(cleaned_dir_plasma, "proteomics_plasma_formatted_suffix_D.csv"),
            "E":  os.path.join(cleaned_dir_plasma, "proteomics_plasma_formatted_suffix_E.csv"),
            "EA": os.path.join(cleaned_dir_plasma, "proteomics_plasma_formatted_suffix_EA.csv"),
            "EB": os.path.join(cleaned_dir_plasma, "proteomics_plasma_formatted_suffix_EB.csv"),
            "EC": os.path.join(cleaned_dir_plasma, "proteomics_plasma_formatted_suffix_EC.csv"),
            "ED": os.path.join(cleaned_dir_plasma, "proteomics_plasma_formatted_suffix_ED.csv"),
            "EE": os.path.join(cleaned_dir_plasma, "proteomics_plasma_formatted_suffix_EE.csv"),
        }
        analyte_cols = get_analyte_columns(
            normalise_group_labels(load_data(next(iter(timepoint_files.values()))))
        )
        for group in ["Control"]:
            run_longitudinal(
                {k: normalise_group_labels(load_data(v)) for k, v in timepoint_files.items()},
                analyte_cols,
                group=group,
                group_col="Group",
                subject_col="SubjectID",
                output_dir=os.path.join(output_dir, "plasma", "longitudinal"),
            )

        logger.info("Differential analysis complete.")

    main()
