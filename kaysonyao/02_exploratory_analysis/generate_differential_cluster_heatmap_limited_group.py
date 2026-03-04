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
import argparse
import logging

from utilities import (
    load_data,
    normalise_group_labels,
    get_analyte_columns,
    plot_cross_sectional_heatmap,
    plot_pairwise_cross_sectional_heatmap,
    plot_longitudinal_heatmap,
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
    args   = parser.parse_args()

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
        diff_dir = os.path.join(wkdir, "data", "diff_analysis", "results")

        _CS_PAIRS = [
            ("Control", "FGR"),
            ("Control", "HDP"),
            ("Control", "sPTB"),
            ("FGR",     "HDP"),
            ("FGR",     "sPTB"),
            ("HDP",     "sPTB"),
        ]

        # ── Cross-sectional pairwise heatmaps: placenta ───────────────────
        placenta_csv  = os.path.join(
            cleaned_dir_placenta, "proteomics_placenta_cleaned_with_metadata.csv"
        )
        cross_results = os.path.join(diff_dir, "placenta", "cross_sectional")
        if os.path.exists(placenta_csv) and os.path.isdir(cross_results):
            logger.info("Generating placenta pairwise cross-sectional heatmaps …")
            for g1, g2 in _CS_PAIRS:
                plot_pairwise_cross_sectional_heatmap(
                    g1=g1, g2=g2,
                    data_path=placenta_csv,
                    results_dir=cross_results,
                    output_dir=cross_results,
                    group_col="Group",
                )
        else:
            logger.warning(
                "Skipping placenta cross-sectional heatmaps: CSV or results dir not found.\n"
                "  CSV:     %s\n  Results: %s", placenta_csv, cross_results,
            )

        # ── Cross-sectional pairwise heatmaps: plasma per timepoint (A–E) ─
        for tp in ["A", "B", "C", "D", "E"]:
            tp_csv     = os.path.join(
                cleaned_dir_plasma, f"proteomics_plasma_formatted_suffix_{tp}.csv"
            )
            tp_results = os.path.join(diff_dir, "plasma", "cross_sectional", tp)
            if os.path.exists(tp_csv) and os.path.isdir(tp_results):
                logger.info("Generating plasma [%s] pairwise cross-sectional heatmaps …", tp)
                for g1, g2 in _CS_PAIRS:
                    plot_pairwise_cross_sectional_heatmap(
                        g1=g1, g2=g2,
                        data_path=tp_csv,
                        results_dir=tp_results,
                        output_dir=tp_results,
                        group_col="Group",
                    )
            else:
                logger.warning(
                    "Skipping plasma cross-sectional heatmaps [%s]: CSV or results dir not found.\n"
                    "  CSV:     %s\n  Results: %s", tp, tp_csv, tp_results,
                )

        # ── Longitudinal heatmap: plasma (one heatmap per group) ─────────
        long_results = os.path.join(diff_dir, "plasma", "longitudinal")
        for group in ["Control", "FGR", "HDP", "sPTB"]:
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


# (log transformed value) fold change reach 1.5 filter
# Show every protein if the number of significant proteins are close to 500
# Report how many samples in each group in each timepoint in both plasma and placenta
# Stop filtering for non significant analytes, just show all and mark significant ones with a dot or star

# Find Optimal scaling for omics data
