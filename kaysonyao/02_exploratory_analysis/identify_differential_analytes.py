"""
Intraomics differential analysis: cross-sectional and longitudinal.

Cross-sectional:
    Mann-Whitney U test + Benjamini-Hochberg FDR (q < 0.05).
    Always compares Control vs each complication group (FGR, HDP, sPTB).
    Min n=5 non-missing observations per group per analyte.

Longitudinal:
    Mann-Whitney U comparing Control deltas vs Complication deltas at each
    adjacent timepoint step (B-A, C-B, D-C, E-D). One comparison per
    Control vs FGR/HDP/sPTB pair per step. Min n=5 in each group per analyte.

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
import argparse
import logging
import pandas as pd

from utilities import (
    load_data,
    normalise_group_labels,
    get_analyte_columns,
    run_cross_sectional,
    run_longitudinal,
    write_sample_count_report,
    _start_analysis_log,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
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
        help="Complication group for longitudinal cross-group analysis (e.g. FGR, HDP, sPTB). "
             "Required for longitudinal mode. Analysis always compares Control vs this group.",
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
    args   = parser.parse_args()

    cross_dir = os.path.join(args.output_dir, "cross_sectional")
    long_dir  = os.path.join(args.output_dir, "longitudinal")

    _start_analysis_log(args.output_dir)

    if args.mode in ("cross_sectional", "both"):
        if not args.input:
            parser.error("--input is required for cross_sectional mode.")
        df          = load_data(args.input)
        analyte_cols = get_analyte_columns(df)
        logger.info(
            "Loaded %d samples × %d analytes from %s",
            df.shape[0], len(analyte_cols), args.input,
        )
        run_cross_sectional(
            df, analyte_cols,
            group_col=args.group_col,
            output_dir=cross_dir,
        )

    if args.mode in ("longitudinal", "both"):
        if not args.timepoint_files:
            parser.error("--timepoint-files is required for longitudinal mode.")
        if not args.group:
            parser.error("--group is required for longitudinal mode (complication group name).")

        labels = args.timepoint_labels or [
            f"T{i + 1}" for i in range(len(args.timepoint_files))
        ]
        if len(labels) != len(args.timepoint_files):
            parser.error("--timepoint-labels count must match --timepoint-files count.")

        timepoint_dfs = {
            label: load_data(path)
            for label, path in zip(labels, args.timepoint_files)
        }
        analyte_cols = get_analyte_columns(next(iter(timepoint_dfs.values())))
        logger.info("Loaded %d timepoints, %d analytes", len(timepoint_dfs), len(analyte_cols))
        run_longitudinal_cross_group(
            timepoint_dfs,
            analyte_cols,
            control_group="Control",
            complication_group=args.group,
            group_col=args.group_col,
            subject_col=args.subject_col,
            output_dir=long_dir,
        )


# ---------------------------------------------------------------------------
# Default (no-arg) mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    def _run_default_mode() -> None:
        wkdir               = os.getcwd()
        cleaned_dir_placenta = os.path.join(
            wkdir, "data", "cleaned", "proteomics", "normalized_full_results"
        )
        cleaned_dir_plasma  = os.path.join(
            wkdir, "data", "cleaned", "proteomics", "normalized_sliced_by_suffix"
        )
        output_dir = os.path.join(wkdir, "data", "diff_analysis", "results")

        _start_analysis_log(output_dir)

        # Helper: relabel FGR/HDP/sPTB → "Complication" (leaves Control unchanged)
        def _merge_to_complication(df: pd.DataFrame, group_col: str = "Group") -> pd.DataFrame:
            df = df.copy()
            df[group_col] = df[group_col].apply(
                lambda g: g if g == "Control" else "Complication"
            )
            return df

        # ── Sample count report ───────────────────────────────────────────
        placenta_csv = os.path.join(
            cleaned_dir_placenta, "proteomics_placenta_cleaned_with_metadata.csv"
        )
        write_sample_count_report(output_dir, cleaned_dir_plasma, placenta_csv)

        # ── Cross-sectional: placenta ─────────────────────────────────────
        if os.path.exists(placenta_csv):
            df           = normalise_group_labels(load_data(placenta_csv))
            df           = _merge_to_complication(df)
            analyte_cols = get_analyte_columns(df)
            logger.info(
                "Cross-sectional [placenta]: %d samples x %d analytes (Control vs Complication)",
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

        # ── Cross-sectional: plasma per timepoint ─────────────────────────
        # E is included because postnatal EE samples are merged into E,
        # giving complication groups representation at this timepoint.
        for tp in ["A", "B", "C", "D", "E"]:
            tp_csv = os.path.join(
                cleaned_dir_plasma, f"proteomics_plasma_formatted_suffix_{tp}.csv"
            )
            if not os.path.exists(tp_csv):
                logger.warning(
                    "Plasma timepoint %s not found, skipping cross-sectional: %s", tp, tp_csv
                )
                continue
            df_tp        = normalise_group_labels(load_data(tp_csv))
            df_tp        = _merge_to_complication(df_tp)
            analyte_cols = get_analyte_columns(df_tp)
            logger.info(
                "Cross-sectional [plasma %s]: %d samples x %d analytes (Control vs Complication)",
                tp, df_tp.shape[0], len(analyte_cols),
            )
            run_cross_sectional(
                df_tp,
                analyte_cols,
                group_col="Group",
                output_dir=os.path.join(output_dir, "plasma", "cross_sectional", tp),
            )

        # ── Longitudinal: plasma ──────────────────────────────────────────
        # Within-group Wilcoxon signed-rank for Control, FGR, HDP, sPTB
        # (individual disease deltas), plus a pooled "Complication" run.
        timepoint_files = {
            tp: os.path.join(cleaned_dir_plasma, f"proteomics_plasma_formatted_suffix_{tp}.csv")
            for tp in ["A", "B", "C", "D", "E"]
        }
        tp_dfs = {k: normalise_group_labels(load_data(v)) for k, v in timepoint_files.items()}
        analyte_cols = get_analyte_columns(next(iter(tp_dfs.values())))
        long_dir     = os.path.join(output_dir, "plasma", "longitudinal")

        for group in ["Control", "FGR", "HDP", "sPTB"]:
            run_longitudinal(
                tp_dfs,
                analyte_cols,
                group=group,
                group_col="Group",
                subject_col="SubjectID",
                output_dir=long_dir,
            )

        # Pooled "Complication" longitudinal: merge FGR/HDP/sPTB → Complication
        tp_dfs_merged = {k: _merge_to_complication(v) for k, v in tp_dfs.items()}
        run_longitudinal(
            tp_dfs_merged,
            analyte_cols,
            group="Complication",
            group_col="Group",
            subject_col="SubjectID",
            output_dir=long_dir,
        )

        logger.info("Differential analysis complete.")

    main()
