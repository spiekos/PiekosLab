"""
Title: prepare_enrichr_input.py
Author: Kayson Yao
Description:
    Prepares gene/protein lists from differential analysis significant analyte
    CSVs for upload to Enrichr (https://maayanlab.cloud/Enrichr/).

    For each pairwise comparison, splits significant analytes into directional
    lists based on fold_change:
        fold_change = median_group2 - median_group1  (NPX log2 scale)
        fold_change > 0  →  higher in group2 relative to group1
        fold_change < 0  →  higher in group1 relative to group2

    Output per comparison (saved under <output_dir>/<timepoint>/<g1>_vs_<g2>/):
        higher_in_<g2>.txt          — one gene per line, paste directly into Enrichr
        higher_in_<g1>.txt          — one gene per line, paste directly into Enrichr
        all_significant.txt         — combined list (direction-agnostic)
        significant_with_direction.csv  — full table with direction annotation

Usage:
    # Default: auto-discover all plasma cross-sectional results
    python prepare_enrichr_input.py

    # Single comparison
    python prepare_enrichr_input.py \
        --sig-csv path/to/Control_vs_HDP_significant_analytes.csv \
        --g1 Control --g2 HDP \
        --output-dir enrichr_input/Control_vs_HDP
"""

import os
import sys
import glob
import argparse
import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# All pairwise comparisons run in the differential analysis.
# Naming convention: (g1, g2) matches the <g1>_vs_<g2> CSV filename prefix,
# and fold_change = median_g2 - median_g1.
_CS_PAIRS = [
    ("Control", "FGR"),
    ("Control", "HDP"),
    ("Control", "sPTB"),
    ("FGR",     "HDP"),
    ("FGR",     "sPTB"),
    ("HDP",     "sPTB"),
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _write_gene_list(genes: list, path: str) -> None:
    """Write a plain-text gene list (one per line) ready for Enrichr paste."""
    with open(path, "w") as fh:
        fh.write("\n".join(genes))
    logger.info("    Wrote %d genes → %s", len(genes), os.path.basename(path))


def prepare_enrichr_lists(
    sig_csv_path: str,
    output_dir: str,
    g1: str,
    g2: str,
) -> dict:
    """Split significant analytes into directional Enrichr-ready lists.

    Args:
        sig_csv_path: Path to a *_significant_analytes.csv produced by
                      identify_differential_analytes.py.
        output_dir:   Directory where output files will be saved.
        g1:           Label of group 1 (reference / first group in filename).
        g2:           Label of group 2 (comparison group).

    Returns:
        dict with keys 'higher_in_g2', 'higher_in_g1', 'all_significant'
        mapping to lists of gene names.
    """
    df = pd.read_csv(sig_csv_path, index_col=0)

    if df.empty:
        logger.info("  No significant analytes in %s — skipping.", os.path.basename(sig_csv_path))
        return {"higher_in_g2": [], "higher_in_g1": [], "all_significant": []}

    if "fold_change" not in df.columns:
        logger.warning(
            "  'fold_change' column missing in %s — cannot split by direction.",
            sig_csv_path,
        )
        all_sig = df.index.tolist()
        os.makedirs(output_dir, exist_ok=True)
        _write_gene_list(all_sig, os.path.join(output_dir, "all_significant.txt"))
        return {"higher_in_g2": [], "higher_in_g1": [], "all_significant": all_sig}

    higher_in_g2 = df.index[df["fold_change"] > 0].tolist()   # g2 > g1
    higher_in_g1 = df.index[df["fold_change"] < 0].tolist()   # g1 > g2
    all_sig      = df.index.tolist()

    os.makedirs(output_dir, exist_ok=True)

    _write_gene_list(higher_in_g2, os.path.join(output_dir, f"higher_in_{g2}.txt"))
    _write_gene_list(higher_in_g1, os.path.join(output_dir, f"higher_in_{g1}.txt"))
    _write_gene_list(all_sig,      os.path.join(output_dir, "all_significant.txt"))

    # Annotated table for reference
    df["direction"] = df["fold_change"].apply(
        lambda fc: f"higher_in_{g2}" if fc > 0 else f"higher_in_{g1}"
    )
    df.to_csv(os.path.join(output_dir, "significant_with_direction.csv"))

    logger.info(
        "  %s vs %s: %d total | %d higher in %s | %d higher in %s",
        g1, g2, len(all_sig), len(higher_in_g2), g2, len(higher_in_g1), g1,
    )

    return {
        "higher_in_g2": higher_in_g2,
        "higher_in_g1": higher_in_g1,
        "all_significant": all_sig,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare Enrichr input lists from differential analysis results."
    )
    p.add_argument(
        "--sig-csv",
        default=None,
        help="Path to a single *_significant_analytes.csv file. "
             "If omitted, auto-discovers all plasma cross-sectional results.",
    )
    p.add_argument(
        "--g1",
        default=None,
        help="Group 1 label (required with --sig-csv; e.g. Control).",
    )
    p.add_argument(
        "--g2",
        default=None,
        help="Group 2 label (required with --sig-csv; e.g. HDP).",
    )
    p.add_argument(
        "--results-dir",
        default=None,
        help="Root differential results directory for auto-discovery mode "
             "(default: data/diff_analysis/results/plasma/cross_sectional).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Root output directory (default: data/enrichr_input).",
    )
    return p


def main():
    if len(sys.argv) == 1:
        _run_default_mode()
        return

    parser = _build_parser()
    args   = parser.parse_args()

    if args.sig_csv:
        if not args.g1 or not args.g2:
            parser.error("--g1 and --g2 are required when --sig-csv is provided.")
        out = args.output_dir or os.path.join(
            os.getcwd(), "data", "enrichr_input", f"{args.g1}_vs_{args.g2}"
        )
        prepare_enrichr_lists(args.sig_csv, out, args.g1, args.g2)
    else:
        _run_default_mode(
            results_root=args.results_dir,
            output_root=args.output_dir,
        )


# ---------------------------------------------------------------------------
# Default (no-arg) mode — auto-discover plasma cross-sectional results
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _run_default_mode(results_root=None, output_root=None) -> None:
        wkdir        = os.getcwd()
        results_root = results_root or os.path.join(
            wkdir, "data", "diff_analysis", "results", "plasma", "cross_sectional"
        )
        output_root  = output_root or os.path.join(wkdir, "data", "enrichr_input")

        if not os.path.isdir(results_root):
            logger.error("Results directory not found: %s", results_root)
            return

        timepoints = sorted([
            d for d in os.listdir(results_root)
            if os.path.isdir(os.path.join(results_root, d))
        ])
        if not timepoints:
            logger.warning("No timepoint subdirectories found in %s", results_root)
            return

        logger.info(
            "Auto-discovering results across %d timepoints: %s",
            len(timepoints), timepoints,
        )

        total_comparisons = 0
        for tp in timepoints:
            tp_dir = os.path.join(results_root, tp)
            logger.info("── Timepoint %s ──", tp)

            for g1, g2 in _CS_PAIRS:
                sig_csv = os.path.join(tp_dir, f"{g1}_vs_{g2}_significant_analytes.csv")
                if not os.path.exists(sig_csv):
                    logger.debug("  Not found: %s — skipping.", os.path.basename(sig_csv))
                    continue

                out_dir = os.path.join(output_root, tp, f"{g1}_vs_{g2}")
                prepare_enrichr_lists(sig_csv, out_dir, g1, g2)
                total_comparisons += 1

        logger.info(
            "Done. Processed %d comparisons. Enrichr lists saved under: %s",
            total_comparisons, output_root,
        )

    main()
