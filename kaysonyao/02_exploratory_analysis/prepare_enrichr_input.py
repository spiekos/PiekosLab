"""
Title: prepare_enrichr_input.py
Author: Kayson Yao
Description:
    Prepares gene/protein lists from differential analysis significant analyte
    CSVs and runs pathway enrichment via the Enrichr API (using gseapy).

    Cross-sectional (Control vs Complication):
        fold_change = median_Complication - median_Control  (NPX log2 scale)
        fold_change > 0  →  higher in Complication relative to Control
        fold_change < 0  →  higher in Control relative to Complication

        Gene lists saved to 04_results_and_figures/enrichment/plasma/cross_sectional/<timepoint>/Control_vs_Complication/:
            higher_in_Complication.txt      one gene per line, ready for manual Enrichr paste
            higher_in_Control.txt           one gene per line, ready for manual Enrichr paste
            all_significant.txt             combined list (direction-agnostic)
            significant_with_direction.csv  full table with direction annotation

    Placenta cross-sectional (Control vs Complication, no timepoints):
        fold_change = median_Complication - median_Control  (NPX log2 scale)

        Gene lists saved to 04_results_and_figures/enrichment/placenta/cross_sectional/Control_vs_Complication/:
            (same file layout as plasma cross-sectional)

    Longitudinal (within-group, per adjacent timepoint step):
        median_delta = value_T_later - value_T_earlier  (per-participant, NPX log2 scale)
        median_delta > 0  →  increasing from T_earlier to T_later
        median_delta < 0  →  decreasing from T_earlier to T_later

        Gene lists saved to 04_results_and_figures/enrichment/plasma/longitudinal/<group>/<T_b>_minus_<T_a>/:
            increasing.txt                  analytes rising at this timepoint step
            decreasing.txt                  analytes falling at this timepoint step
            all_significant.txt             combined list
            significant_with_direction.csv  full table with direction annotation

    Enrichment results saved per direction (enrichment/ subfolder):
        <direction>_enrichment.csv      all databases combined, sorted by Adjusted P-value

    Databases queried:
        Default (--gene-sets):  GO_Biological_Process_2025, KEGG_2026, Reactome_Pathways_2024
        All available (--all-databases): every library currently in Enrichr (~300+ databases)

Usage:
    # Default: auto-discover CS + longitudinal results and run enrichment for all
    python prepare_enrichr_input.py

    # Query every available Enrichr database (mirrors the website behaviour)
    python prepare_enrichr_input.py --all-databases

    # Single CS comparison with all databases
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
import re
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

# Longitudinal groups to process in default mode.
_LONGITUDINAL_GROUPS = ["Control", "FGR", "HDP", "sPTB", "Complication"]

# Default curated databases
_DEFAULT_GENE_SETS = [
    "GO_Biological_Process_2025",
    "KEGG_2026",
    "Reactome_Pathways_2024",
]

# Minimum gene list size to attempt enrichment
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
# Pathway enrichment — shared core
# ---------------------------------------------------------------------------

def _run_enrichment_for_directions(
    directions: dict,
    output_dir: str,
    gene_sets: list,
) -> None:
    """Run Enrichr for each {label: [genes]} pair and save one CSV per direction.

    Args:
        directions:  Mapping of output label → gene list (e.g. {"increasing": [...]}).
        output_dir:  Parent directory; results written to an enrichment/ subfolder.
        gene_sets:   Enrichr library names to query.
    """
    enrich_dir = os.path.join(output_dir, "enrichment")
    os.makedirs(enrich_dir, exist_ok=True)

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
                outdir=None,
                verbose=False,
            )
        except Exception as exc:
            logger.error("  Enrichr API call failed for %s: %s", direction_label, exc)
            continue

        results = enr.results
        if results.empty:
            logger.info("  No enrichment results returned for %s.", direction_label)
            continue

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

        if "Adj_P_value" in results.columns:
            n_sig_total    = (results["Adj_P_value"] < 0.05).sum()
            n_db_with_hits = (
                results[results["Adj_P_value"] < 0.05]["Gene_set"].nunique()
                if "Gene_set" in results.columns else "n/a"
            )
            top_terms = results[results["Adj_P_value"] < 0.05]["Term"].head(5).tolist()
            logger.info(
                "    %d significant terms (adj p < 0.05) across %s database(s). Top 5: %s",
                n_sig_total, n_db_with_hits,
                ", ".join(top_terms) if top_terms else "none",
            )
        logger.info("    Saved → %s", os.path.basename(out_path))


def run_pathway_enrichment(
    gene_lists: dict,
    g1: str,
    g2: str,
    output_dir: str,
    gene_sets: list = None,
) -> None:
    """Run Enrichr pathway enrichment for a cross-sectional comparison.

    Args:
        gene_lists:  Output dict from prepare_enrichr_lists().
        g1, g2:      Group labels — used to label output files.
        output_dir:  Parent output directory; results go into an enrichment/ subfolder.
        gene_sets:   Enrichr database names to query. Defaults to _DEFAULT_GENE_SETS.
    """
    directions = {
        f"higher_in_{g2}": gene_lists["higher_in_g2"],
        f"higher_in_{g1}": gene_lists["higher_in_g1"],
    }
    _run_enrichment_for_directions(directions, output_dir, gene_sets or _DEFAULT_GENE_SETS)


# ---------------------------------------------------------------------------
# Longitudinal gene lists + enrichment
# ---------------------------------------------------------------------------

def prepare_enrichr_lists_longitudinal(
    results_csv: str,
    output_dir: str,
) -> dict:
    """Split significant longitudinal analytes into increasing/decreasing lists.

    Reads a <group>_<T_b>_minus_<T_a>_longitudinal_results.csv produced by
    run_longitudinal (utilities.py).  Direction is determined by median_delta:
        median_delta > 0  →  increasing (rising from T_a to T_b)
        median_delta < 0  →  decreasing (falling from T_a to T_b)

    Returns:
        dict with keys 'increasing', 'decreasing', 'all_significant'.
    """
    df = pd.read_csv(results_csv, index_col=0)

    sig_df = df[df["significant"] == True] if "significant" in df.columns else pd.DataFrame()

    if sig_df.empty:
        logger.info("  No significant analytes in %s — skipping.", os.path.basename(results_csv))
        return {"increasing": [], "decreasing": [], "all_significant": []}

    if "median_delta" not in sig_df.columns:
        logger.warning(
            "  'median_delta' column missing in %s — cannot split by direction.",
            results_csv,
        )
        all_sig = sig_df.index.tolist()
        os.makedirs(output_dir, exist_ok=True)
        _write_gene_list(all_sig, os.path.join(output_dir, "all_significant.txt"))
        return {"increasing": [], "decreasing": [], "all_significant": all_sig}

    increasing = sig_df.index[sig_df["median_delta"] > 0].tolist()
    decreasing = sig_df.index[sig_df["median_delta"] < 0].tolist()
    all_sig    = sig_df.index.tolist()

    os.makedirs(output_dir, exist_ok=True)
    _write_gene_list(increasing, os.path.join(output_dir, "increasing.txt"))
    _write_gene_list(decreasing, os.path.join(output_dir, "decreasing.txt"))
    _write_gene_list(all_sig,    os.path.join(output_dir, "all_significant.txt"))

    sig_df = sig_df.copy()
    sig_df["direction"] = sig_df["median_delta"].apply(
        lambda d: "increasing" if d > 0 else "decreasing"
    )
    sig_df.to_csv(os.path.join(output_dir, "significant_with_direction.csv"))

    logger.info(
        "  %d total significant | %d increasing | %d decreasing",
        len(all_sig), len(increasing), len(decreasing),
    )
    return {"increasing": increasing, "decreasing": decreasing, "all_significant": all_sig}


def run_pathway_enrichment_longitudinal(
    gene_lists: dict,
    output_dir: str,
    gene_sets: list = None,
) -> None:
    """Run Enrichr pathway enrichment for a longitudinal step.

    Args:
        gene_lists:  Output dict from prepare_enrichr_lists_longitudinal().
        output_dir:  Parent output directory; results go into an enrichment/ subfolder.
        gene_sets:   Enrichr database names to query. Defaults to _DEFAULT_GENE_SETS.
    """
    directions = {
        "increasing": gene_lists["increasing"],
        "decreasing": gene_lists["decreasing"],
    }
    _run_enrichment_for_directions(directions, output_dir, gene_sets or _DEFAULT_GENE_SETS)


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
        help="Path to a single *_significant_analytes.csv; triggers single CS comparison mode.",
    )
    p.add_argument(
        "--g1",
        default=None,
        help="Group 1 label (required with --sig-csv; e.g. Control).",
    )
    p.add_argument(
        "--g2",
        default=None,
        help="Group 2 label (required with --sig-csv; e.g. Complication).",
    )
    p.add_argument(
        "--results-dir",
        default=None,
        help="Root cross-sectional results directory for auto-discovery "
             "(default: 04_results_and_figures/differential_analysis/plasma/cross_sectional).",
    )
    p.add_argument(
        "--longitudinal-results-dir",
        default=None,
        help="Longitudinal results directory for auto-discovery "
             "(default: 04_results_and_figures/differential_analysis/plasma/longitudinal).",
    )
    p.add_argument(
        "--placenta-results-dir",
        default=None,
        help="Placenta cross-sectional results directory for auto-discovery "
             "(default: 04_results_and_figures/differential_analysis/placenta/cross_sectional).",
    )
    p.add_argument(
        "--skip-placenta",
        action="store_true",
        help="Skip placenta enrichment.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Root output directory (default: 04_results_and_figures/enrichment).",
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
    p.add_argument(
        "--skip-longitudinal",
        action="store_true",
        help="Skip longitudinal enrichment; run cross-sectional only.",
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
            os.getcwd(),
            "04_results_and_figures", "enrichment", f"{args.g1}_vs_{args.g2}",
        )
        gene_lists = prepare_enrichr_lists(args.sig_csv, out, args.g1, args.g2)
        if not args.skip_enrichment:
            run_pathway_enrichment(gene_lists, args.g1, args.g2, out, gene_sets)
    else:
        _run_default_mode(
            results_root=args.results_dir,
            longitudinal_root=getattr(args, "longitudinal_results_dir", None),
            placenta_root=getattr(args, "placenta_results_dir", None),
            output_root=args.output_dir,
            gene_sets=gene_sets,
            skip_enrichment=args.skip_enrichment,
            skip_longitudinal=getattr(args, "skip_longitudinal", False),
            skip_placenta=getattr(args, "skip_placenta", False),
        )


# ---------------------------------------------------------------------------
# Default (no-arg) mode
# ---------------------------------------------------------------------------

# Regex to parse longitudinal result filenames:
# <group>_<T_b>_minus_<T_a>_longitudinal_results.csv  (T_b, T_a are single uppercase letters)
_LONG_FILE_RE = re.compile(r"^(.+)_([A-E])_minus_([A-E])_longitudinal_results\.csv$")


if __name__ == "__main__":

    def _run_default_mode(
        results_root=None,
        longitudinal_root=None,
        placenta_root=None,
        output_root=None,
        gene_sets=None,
        skip_enrichment=False,
        skip_longitudinal=False,
        skip_placenta=False,
    ) -> None:
        wkdir             = os.getcwd()
        results_root      = results_root or os.path.join(
            wkdir, "04_results_and_figures", "differential_analysis", "plasma", "cross_sectional"
        )
        longitudinal_root = longitudinal_root or os.path.join(
            wkdir, "04_results_and_figures", "differential_analysis", "plasma", "longitudinal"
        )
        placenta_root     = placenta_root or os.path.join(
            wkdir, "04_results_and_figures", "differential_analysis", "placenta", "cross_sectional"
        )
        output_root  = output_root or os.path.join(wkdir, "04_results_and_figures", "enrichment")
        gene_sets    = gene_sets or _DEFAULT_GENE_SETS

        # ── Cross-sectional ───────────────────────────────────────────────
        if os.path.isdir(results_root):
            timepoints = sorted([
                d for d in os.listdir(results_root)
                if os.path.isdir(os.path.join(results_root, d))
            ])
            logger.info(
                "Cross-sectional: auto-discovering across %d timepoints: %s",
                len(timepoints), timepoints,
            )
            total_cs = 0
            for tp in timepoints:
                tp_dir = os.path.join(results_root, tp)
                logger.info("── CS Timepoint %s ──", tp)
                for g1, g2 in _CS_PAIRS:
                    sig_csv = os.path.join(tp_dir, f"{g1}_vs_{g2}_significant_analytes.csv")
                    if not os.path.exists(sig_csv):
                        logger.debug("  Not found: %s — skipping.", os.path.basename(sig_csv))
                        continue
                    out_dir    = os.path.join(output_root, "plasma", "cross_sectional", tp, f"{g1}_vs_{g2}")
                    gene_lists = prepare_enrichr_lists(sig_csv, out_dir, g1, g2)
                    if not skip_enrichment:
                        run_pathway_enrichment(gene_lists, g1, g2, out_dir, gene_sets)
                    total_cs += 1
            logger.info("Cross-sectional: processed %d comparison(s).", total_cs)
        else:
            logger.warning("Cross-sectional results directory not found: %s", results_root)

        # ── Longitudinal ──────────────────────────────────────────────────
        if skip_longitudinal:
            logger.info("Longitudinal enrichment skipped (--skip-longitudinal).")
        elif os.path.isdir(longitudinal_root):
            long_files = sorted(glob.glob(
                os.path.join(longitudinal_root, "*_longitudinal_results.csv")
            ))
            logger.info(
                "Longitudinal: found %d result file(s) in %s",
                len(long_files), longitudinal_root,
            )
            total_long = 0
            for fpath in long_files:
                fname = os.path.basename(fpath)
                m     = _LONG_FILE_RE.match(fname)
                if not m:
                    logger.debug("  Filename did not match expected pattern — skipping: %s", fname)
                    continue
                group, t_b, t_a = m.group(1), m.group(2), m.group(3)
                delta_label     = f"{t_b}_minus_{t_a}"
                logger.info("── Longitudinal [%s  %s] ──", group, delta_label)
                out_dir    = os.path.join(output_root, "plasma", "longitudinal", group, delta_label)
                gene_lists = prepare_enrichr_lists_longitudinal(fpath, out_dir)
                if not skip_enrichment:
                    run_pathway_enrichment_longitudinal(gene_lists, out_dir, gene_sets)
                total_long += 1
            logger.info("Longitudinal: processed %d file(s).", total_long)
        else:
            logger.warning("Longitudinal results directory not found: %s", longitudinal_root)

        # ── Placenta cross-sectional ──────────────────────────────────────
        if skip_placenta:
            logger.info("Placenta enrichment skipped (--skip-placenta).")
        elif os.path.isdir(placenta_root):
            logger.info("Placenta CS: processing from %s", placenta_root)
            total_placenta = 0
            for g1, g2 in _CS_PAIRS:
                sig_csv = os.path.join(placenta_root, f"{g1}_vs_{g2}_significant_analytes.csv")
                if not os.path.exists(sig_csv):
                    logger.debug("  Not found: %s — skipping.", os.path.basename(sig_csv))
                    continue
                logger.info("── Placenta CS [%s vs %s] ──", g1, g2)
                out_dir    = os.path.join(output_root, "placenta", "cross_sectional", f"{g1}_vs_{g2}")
                gene_lists = prepare_enrichr_lists(sig_csv, out_dir, g1, g2)
                if not skip_enrichment:
                    run_pathway_enrichment(gene_lists, g1, g2, out_dir, gene_sets)
                total_placenta += 1
            logger.info("Placenta CS: processed %d comparison(s).", total_placenta)
        else:
            logger.warning("Placenta results directory not found: %s", placenta_root)

        logger.info("Done. Output saved under: %s", output_root)

    main()
