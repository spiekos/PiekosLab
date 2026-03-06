"""
Title: prepare_enrichr_input.py
Author: Kayson Yao
Description:
    Prepares gene/protein lists from differential analysis significant analyte
    CSVs and runs pathway enrichment via the Enrichr API (using gseapy).

    For the cross-sectional binary comparison (Control vs Complication), significant
    analytes are split into directional lists based on fold_change:
        fold_change = median_group2 - median_group1  (NPX log2 scale)
        fold_change > 0  →  higher in Complication relative to Control
        fold_change < 0  →  higher in Control relative to Complication

    Gene lists saved per comparison (data/enrichr_input/<timepoint>/Control_vs_Complication/):
        higher_in_Complication.txt      one gene per line, ready for manual Enrichr paste
        higher_in_Control.txt           one gene per line, ready for manual Enrichr paste
        all_significant.txt             combined list (direction-agnostic)
        significant_with_direction.csv  full table with direction annotation

    Enrichment results saved per direction (enrichment/ subfolder):
        <direction>_enrichment.csv      all databases combined, sorted by Adjusted P-value

    Databases queried:
        Default (--gene-sets):  GO_Biological_Process_2025, KEGG_2026, Reactome_Pathways_2024
        All available (--all-databases): every library currently in Enrichr (~300+ databases)

Usage:
    # Default: auto-discover all plasma cross-sectional results and run enrichment
    python prepare_enrichr_input.py

    # Query every available Enrichr database (mirrors the website behaviour)
    python prepare_enrichr_input.py --all-databases

    # Single comparison with all databases
    python prepare_enrichr_input.py \
        --sig-csv path/to/Control_vs_Complication_significant_analytes.csv \
        --g1 Control --g2 Complication \
        --all-databases

    # Skip enrichment (gene lists only)
    python prepare_enrichr_input.py --skip-enrichment

    # Custom databases
    python prepare_enrichr_input.py \
        --gene-sets GO_Biological_Process_2025 GO_Molecular_Function_2023 KEGG_2026
"""

import os
import sys
import glob
import argparse
import logging
import gseapy as gp
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CS_PAIRS = [
    ("Control", "Complication"),
]

# Default curated databases (fast; good coverage for plasma proteomics).
_DEFAULT_GENE_SETS = [
    "GO_Biological_Process_2025",
    "KEGG_2026",
    "Reactome_Pathways_2024",
]

# Minimum gene list size to attempt enrichment (too few gives meaningless results).
_MIN_GENES_FOR_ENRICHMENT = 5


def _resolve_gene_sets(gene_sets: list | None, all_databases: bool) -> list:
    """Return the list of Enrichr libraries to query.

    Priority:
      1. --all-databases  →  fetch the full list from Enrichr API (~300+ libraries)
      2. --gene-sets ...  →  use exactly what was provided
      3. (default)        →  use _DEFAULT_GENE_SETS
    """
    if all_databases:
        libs = gp.get_library_name(organism="human")
        logger.info("--all-databases: querying all %d available Enrichr libraries.", len(libs))
        return libs
    return gene_sets or _DEFAULT_GENE_SETS


# ---------------------------------------------------------------------------
# Gene list helpers
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
        g1:           Label of group 1 (reference; first group in filename).
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

    higher_in_g2 = df.index[df["fold_change"] > 0].tolist()
    higher_in_g1 = df.index[df["fold_change"] < 0].tolist()
    all_sig      = df.index.tolist()

    os.makedirs(output_dir, exist_ok=True)

    _write_gene_list(higher_in_g2, os.path.join(output_dir, f"higher_in_{g2}.txt"))
    _write_gene_list(higher_in_g1, os.path.join(output_dir, f"higher_in_{g1}.txt"))
    _write_gene_list(all_sig,      os.path.join(output_dir, "all_significant.txt"))

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
# Pathway enrichment
# ---------------------------------------------------------------------------

def run_pathway_enrichment(
    gene_lists: dict,
    g1: str,
    g2: str,
    output_dir: str,
    gene_sets: list = None,
) -> None:
    """Run Enrichr pathway enrichment for each directional gene list.

    Queries the Enrichr API for each direction (higher_in_g2, higher_in_g1)
    across all specified gene set databases, then saves one combined CSV per
    direction sorted by Adjusted P-value.

    Args:
        gene_lists:  Output dict from prepare_enrichr_lists().
        g1, g2:      Group labels — used to label output files.
        output_dir:  Parent output directory; results go into an enrichment/ subfolder.
        gene_sets:   Enrichr database names to query. Defaults to _DEFAULT_GENE_SETS.
    """

    gene_sets  = gene_sets or _DEFAULT_GENE_SETS
    enrich_dir = os.path.join(output_dir, "enrichment")
    os.makedirs(enrich_dir, exist_ok=True)

    directions = {
        f"higher_in_{g2}": gene_lists["higher_in_g2"],
        f"higher_in_{g1}": gene_lists["higher_in_g1"],
    }

    for direction_label, genes in directions.items():
        if len(genes) < _MIN_GENES_FOR_ENRICHMENT:
            logger.info(
                "  Skipping enrichment for %s: only %d gene(s) (minimum = %d).",
                direction_label, len(genes), _MIN_GENES_FOR_ENRICHMENT,
            )
            continue

        logger.info(
            "  Running Enrichr for %s (%d genes) against %d database(s) …",
            direction_label, len(genes), len(gene_sets),
        )

        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=gene_sets,
                organism="human",
                outdir=None,       # return DataFrame; don't write to disk
                verbose=False,
            )
        except Exception as exc:
            logger.error("  Enrichr API call failed for %s: %s", direction_label, exc)
            continue

        results = enr.results

        if results.empty:
            logger.info("  No enrichment results returned for %s.", direction_label)
            continue

        # Standardise column names across gseapy versions.
        results = results.rename(columns={
            "Adjusted P-value": "Adj_P_value",
            "P-value":          "P_value",
            "Combined Score":   "Combined_Score",
            "Odds Ratio":       "Odds_Ratio",
            "Overlap":          "Overlap",
        })

        if "Adj_P_value" in results.columns:
            results = results.sort_values("Adj_P_value")

        out_path = os.path.join(enrich_dir, f"{direction_label}_enrichment.csv")
        results.to_csv(out_path, index=False)

        # Summary log — concise regardless of how many databases were queried.
        if "Adj_P_value" in results.columns:
            n_sig_total = (results["Adj_P_value"] < 0.05).sum()
            n_db_with_hits = (
                results[results["Adj_P_value"] < 0.05]["Gene_set"].nunique()
                if "Gene_set" in results.columns else "n/a"
            )
            top_terms = results[results["Adj_P_value"] < 0.05]["Term"].head(5).tolist()
            logger.info(
                "    %d significant terms (adj p < 0.05) across %s database(s). "
                "Top 5: %s",
                n_sig_total,
                n_db_with_hits,
                ", ".join(top_terms) if top_terms else "none",
            )

        logger.info("    Saved → %s", os.path.basename(out_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare Enrichr gene lists and run pathway enrichment."
    )
    p.add_argument(
        "--sig-csv",
        default=None,
        help="Path to a single *_significant_analytes.csv. "
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
        help="Root differential results directory for auto-discovery "
             "(default: data/diff_analysis/results/plasma/cross_sectional).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Root output directory (default: data/enrichr_input).",
    )
    p.add_argument(
        "--gene-sets",
        nargs="+",
        default=None,
        help=f"Specific Enrichr databases to query "
             f"(default: {' '.join(_DEFAULT_GENE_SETS)}). "
             f"Ignored if --all-databases is set.",
    )
    p.add_argument(
        "--all-databases",
        action="store_true",
        help="Query every available Enrichr database (~300+), mirroring the website. "
             "Slower and produces a larger output file. Overrides --gene-sets.",
    )
    p.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Write gene list text files only; skip all Enrichr API calls.",
    )
    return p


def main():
    if len(sys.argv) == 1:
        _run_default_mode()
        return

    parser = _build_parser()
    args   = parser.parse_args()

    gene_sets = _resolve_gene_sets(args.gene_sets, args.all_databases)

    if args.sig_csv:
        if not args.g1 or not args.g2:
            parser.error("--g1 and --g2 are required when --sig-csv is provided.")
        out = args.output_dir or os.path.join(
            os.getcwd(), "data", "enrichr_input", f"{args.g1}_vs_{args.g2}"
        )
        gene_lists = prepare_enrichr_lists(args.sig_csv, out, args.g1, args.g2)
        if not args.skip_enrichment:
            run_pathway_enrichment(gene_lists, args.g1, args.g2, out, gene_sets)
    else:
        _run_default_mode(
            results_root=args.results_dir,
            output_root=args.output_dir,
            gene_sets=gene_sets,
            skip_enrichment=args.skip_enrichment,
        )


# ---------------------------------------------------------------------------
# Default (no-arg) mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _run_default_mode(
        results_root=None,
        output_root=None,
        gene_sets=None,
        skip_enrichment=False,
    ) -> None:
        wkdir        = os.getcwd()
        results_root = results_root or os.path.join(
            wkdir, "data", "diff_analysis", "results", "plasma", "cross_sectional"
        )
        output_root  = output_root or os.path.join(wkdir, "data", "enrichr_input")
        gene_sets    = gene_sets or _DEFAULT_GENE_SETS

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

                out_dir    = os.path.join(output_root, tp, f"{g1}_vs_{g2}")
                gene_lists = prepare_enrichr_lists(sig_csv, out_dir, g1, g2)

                if not skip_enrichment:
                    run_pathway_enrichment(gene_lists, g1, g2, out_dir, gene_sets)

                total_comparisons += 1

        logger.info(
            "Done. Processed %d comparisons. Output saved under: %s",
            total_comparisons, output_root,
        )

    main()
