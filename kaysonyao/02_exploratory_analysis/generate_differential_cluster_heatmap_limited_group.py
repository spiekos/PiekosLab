"""
Heatmap visualization for intraomics differential analysis results.

Cross-sectional heatmap:
    Rows = significant analytes (q < 0.05 in >= 1 pairwise comparison); capped at
           100, ranked by minimum q-value across comparisons.
    Columns = individual samples, clustered within each group separately then
              concatenated in alphabetical group order.
    Values = z-score of per-analyte abundance across all samples, clipped ±2.5.
    Row clustering: Ward linkage, Euclidean distance.

Longitudinal heatmap (within-group):
    Rows = analytes significant (q < 0.05 AND |median_delta| >= log2(1.5)) within
           the specified group in >= 1 adjacent comparison; capped at 500, ranked
           by number of significant comparisons.
    Columns = delta comparisons in fixed chronological order (B-A, C-B, D-C, E-D);
              not clustered.
    Values = median_delta for the specified group (median of within-subject paired
             differences: value_later - value_earlier, in log2 space), row-wise
             z-scored across the four comparisons, clipped ±2.0.
    Significance criterion = Wilcoxon signed-rank test on within-subject deltas
             vs zero (FDR-BH corrected), independently per group per comparison.
             This tests "did this metabolite change within this group?", NOT
             "did this group change differently from Control?".
    Row clustering: Ward linkage, Euclidean distance.
    Significant cells marked with a dot; non-significant cells dimmed.

    NOTE: Each group's heatmap is independent. To visualise differential change
    between groups (e.g. HDP vs Control), a separate analysis subtracting
    Control median_delta from the complication median_delta would be needed.

Usage:
    # Cross-sectional
    python generate_differential_cluster_heatmap_limited_group.py \
        --mode cross_sectional \
        --input cleaned.csv \
        --results-dir results/cross_sectional \
        --output-dir results/cross_sectional

    # Longitudinal (specify complication group)
    python generate_differential_cluster_heatmap_limited_group.py \
        --mode longitudinal \
        --results-dir results/longitudinal \
        --output-dir results/longitudinal \
        --group HDP
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
        "--omics-type",
        choices=["proteomics", "metabolomics", "metabolomics_combat", "lipids"],
        default=None,
        help="Run the full default pipeline for this omics type. "
             "When provided, --results-dir and --output-dir are inferred automatically.",
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
        default=None,
        help="Directory containing differential results CSVs.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save heatmap outputs.",
    )
    p.add_argument(
        "--group-col",
        default="Group",
        help="Column name for group labels in the cleaned CSV (default: Group).",
    )
    p.add_argument(
        "--group",
        help="Complication group for longitudinal heatmap (e.g. FGR, HDP, sPTB). "
             "Required for longitudinal mode. Heatmap shows Control vs this group.",
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

    # --omics-type triggers the full default-pipeline for that omics type
    if args.omics_type is not None:
        _run_default_mode_for_omics(args.omics_type)
        return

    if not args.results_dir:
        parser.error("--results-dir is required when --omics-type is not provided.")
    if not args.output_dir:
        parser.error("--output-dir is required when --omics-type is not provided.")

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
            parser.error("--group is required for longitudinal mode (group name, e.g. FGR, HDP, sPTB, or Complication).")
        plot_longitudinal_heatmap(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            group=args.group,
        )


# ---------------------------------------------------------------------------
# Default (no-arg) mode  +  omics-type pipeline runner
# ---------------------------------------------------------------------------

def _run_default_mode_for_omics(omics_type: str) -> None:
    """Run the full heatmap pipeline for a given omics type (non-proteomics)."""
    wkdir = os.getcwd()

    if omics_type == "metabolomics_combat":
        data_subdir  = "metabolomics_combat"
        file_prefix  = "metabolomics"
        has_placenta = False
    elif omics_type == "lipids":
        data_subdir  = "lipids"
        file_prefix  = "lipids"
        has_placenta = False
    else:
        data_subdir  = omics_type
        file_prefix  = omics_type
        has_placenta = True

    cleaned_dir_plasma = os.path.join(
        wkdir, "data", "cleaned", data_subdir, "normalized_sliced_by_suffix"
    )
    cleaned_dir_placenta = os.path.join(
        wkdir, "data", "cleaned", data_subdir, "normalized_full_results"
    )
    diff_dir    = os.path.join(
        wkdir, "04_results_and_figures", "differential_analysis", omics_type
    )
    heatmap_dir = os.path.join(
        wkdir, "04_results_and_figures", "heatmaps", omics_type
    )

    _CS_PAIRS      = [("Control", "Complication")]
    _COMPL_SOURCES = ["FGR", "HDP", "sPTB"]

    # ── Cross-sectional: placenta (if applicable) ─────────────────────────
    if has_placenta:
        placenta_csv  = os.path.join(
            cleaned_dir_placenta, f"{file_prefix}_placenta_cleaned_with_metadata.csv"
        )
        cross_results  = os.path.join(diff_dir,    "placenta", "cross_sectional")
        cross_heatmaps = os.path.join(heatmap_dir, "placenta", "cross_sectional")
        if os.path.exists(placenta_csv) and os.path.isdir(cross_results):
            logger.info("Generating placenta cross-sectional heatmap …")
            for g1, g2 in _CS_PAIRS:
                plot_pairwise_cross_sectional_heatmap(
                    g1=g1, g2=g2,
                    data_path=placenta_csv,
                    results_dir=cross_results,
                    output_dir=cross_heatmaps,
                    group_col="Group",
                    g2_source_groups=_COMPL_SOURCES,
                )
        else:
            logger.warning(
                "Skipping placenta cross-sectional heatmap: CSV or results dir not found.\n"
                "  CSV:     %s\n  Results: %s", placenta_csv, cross_results,
            )

    # ── Cross-sectional: plasma per timepoint ─────────────────────────────
    for tp in ["A", "B", "C", "D", "E"]:
        tp_csv      = os.path.join(
            cleaned_dir_plasma, f"{file_prefix}_plasma_formatted_suffix_{tp}.csv"
        )
        tp_results  = os.path.join(diff_dir,    "plasma", "cross_sectional", tp)
        tp_heatmaps = os.path.join(heatmap_dir, "plasma", "cross_sectional", tp)
        if os.path.exists(tp_csv) and os.path.isdir(tp_results):
            logger.info("Generating plasma [%s] cross-sectional heatmap …", tp)
            for g1, g2 in _CS_PAIRS:
                plot_pairwise_cross_sectional_heatmap(
                    g1=g1, g2=g2,
                    data_path=tp_csv,
                    results_dir=tp_results,
                    output_dir=tp_heatmaps,
                    group_col="Group",
                    g2_source_groups=_COMPL_SOURCES,
                )
        else:
            logger.warning(
                "Skipping plasma cross-sectional heatmap [%s]: CSV or results dir not found.\n"
                "  CSV:     %s\n  Results: %s", tp, tp_csv, tp_results,
            )

    # ── Longitudinal: individual groups + pooled Complication ─────────────
    long_results  = os.path.join(diff_dir,    "plasma", "longitudinal")
    long_heatmaps = os.path.join(heatmap_dir, "plasma", "longitudinal")
    if os.path.isdir(long_results):
        for group in ["Control", "FGR", "HDP", "sPTB", "Complication"]:
            plot_longitudinal_heatmap(
                results_dir=long_results,
                output_dir=long_heatmaps,
                group=group,
            )
    else:
        logger.warning(
            "Skipping longitudinal heatmaps: results dir not found: %s", long_results,
        )

    logger.info("Heatmap generation complete for omics_type=%s.", omics_type)


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
        diff_dir    = os.path.join(wkdir, "04_results_and_figures", "differential_analysis")
        heatmap_dir = os.path.join(wkdir, "04_results_and_figures", "heatmaps")

        # Single binary comparison: Control vs pooled Complication (FGR + HDP + sPTB)
        _CS_PAIRS         = [("Control", "Complication")]
        _COMPL_SOURCES    = ["FGR", "HDP", "sPTB"]

        # ── Cross-sectional pairwise heatmaps: placenta ───────────────────
        placenta_csv  = os.path.join(
            cleaned_dir_placenta, "proteomics_placenta_cleaned_with_metadata.csv"
        )
        cross_results  = os.path.join(diff_dir,    "placenta", "cross_sectional")
        cross_heatmaps = os.path.join(heatmap_dir, "placenta", "cross_sectional")
        if os.path.exists(placenta_csv) and os.path.isdir(cross_results):
            logger.info("Generating placenta cross-sectional heatmap (Control vs Complication) …")
            for g1, g2 in _CS_PAIRS:
                plot_pairwise_cross_sectional_heatmap(
                    g1=g1, g2=g2,
                    data_path=placenta_csv,
                    results_dir=cross_results,
                    output_dir=cross_heatmaps,
                    group_col="Group",
                    g2_source_groups=_COMPL_SOURCES,
                )
        else:
            logger.warning(
                "Skipping placenta cross-sectional heatmap: CSV or results dir not found.\n"
                "  CSV:     %s\n  Results: %s", placenta_csv, cross_results,
            )

        # ── Cross-sectional pairwise heatmaps: plasma per timepoint (A–E) ─
        for tp in ["A", "B", "C", "D", "E"]:
            tp_csv      = os.path.join(
                cleaned_dir_plasma, f"proteomics_plasma_formatted_suffix_{tp}.csv"
            )
            tp_results  = os.path.join(diff_dir,    "plasma", "cross_sectional", tp)
            tp_heatmaps = os.path.join(heatmap_dir, "plasma", "cross_sectional", tp)
            if os.path.exists(tp_csv) and os.path.isdir(tp_results):
                logger.info(
                    "Generating plasma [%s] cross-sectional heatmap (Control vs Complication) …", tp
                )
                for g1, g2 in _CS_PAIRS:
                    plot_pairwise_cross_sectional_heatmap(
                        g1=g1, g2=g2,
                        data_path=tp_csv,
                        results_dir=tp_results,
                        output_dir=tp_heatmaps,
                        group_col="Group",
                        g2_source_groups=_COMPL_SOURCES,
                    )
            else:
                logger.warning(
                    "Skipping plasma cross-sectional heatmap [%s]: CSV or results dir not found.\n"
                    "  CSV:     %s\n  Results: %s", tp, tp_csv, tp_results,
                )

        # ── Longitudinal heatmaps: individual groups + pooled Complication ─
        long_results  = os.path.join(diff_dir,    "plasma", "longitudinal")
        long_heatmaps = os.path.join(heatmap_dir, "plasma", "longitudinal")
        if os.path.isdir(long_results):
            for group in ["Control", "FGR", "HDP", "sPTB", "Complication"]:
                plot_longitudinal_heatmap(
                    results_dir=long_results,
                    output_dir=long_heatmaps,
                    group=group,
                )
        else:
            logger.warning(
                "Skipping longitudinal heatmaps: results dir not found: %s", long_results,
            )

        logger.info("Heatmap generation complete.")

    main()
