"""
superset_enrichment_analysis.py

Over-representation analysis (ORA) via the Enrichr REST API on the
LASSO-selected proteins identified as important by the binary models.

Gene sets analysed
------------------
  • Per-timepoint  : proteins LASSO-selected at each plasma timepoint (A–D)
  • Placenta        : proteins LASSO-selected in the placenta model
  • Superset        : union of all the above (default: A+B+C+D+placenta)

Databases queried (Enrichr)
---------------------------
  GO_Biological_Process_2023
  GO_Molecular_Function_2023
  GO_Cellular_Component_2023
  KEGG_2021_Human
  Reactome_2022

Outputs (under --output-dir)
-----------------------------
  enrichment/
    {label}/
      {library}_enrichment.csv       (all terms, q < 0.05 highlighted)
    summary/
      {label}_top_terms.png          (top-10 terms per database, dot-plot)
  analysis_log.txt

Requires: requests, pandas, matplotlib, scipy (standard env; no gseapy needed)

Usage
-----
  python 03_model_development/superset_enrichment_analysis.py

  # Custom paths:
  python 03_model_development/superset_enrichment_analysis.py \\
      --binary-results-dir 04_results_and_figures/models/binary \\
      --output-dir 04_results_and_figures/models/binary/superset_enrichment \\
      --superset-timepoints A B C D \\
      --fdr-threshold 0.05
"""

import argparse
import datetime
import logging
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENRICHR_BASE  = "https://maayanlab.cloud/Enrichr"
ENRICHR_LIBS  = [
    "GO_Biological_Process_2025",
    "GO_Molecular_Function_2025",
    "GO_Cellular_Component_2025",
    "KEGG_2026",
    "Reactome_Pathways_2024",
]

# Enrichr result column indices
# [rank, term, p_value, z_score, combined_score, genes, adjusted_p_value, ...]
_RANK_IDX     = 0
_TERM_IDX     = 1
_PVAL_IDX     = 2
_ZSCORE_IDX   = 3
_CSCORE_IDX   = 4
_GENES_IDX    = 5
_ADJP_IDX     = 6

FDR_THRESHOLD = 0.05
MIN_GENES     = 2          # minimum gene list size; warn but still query below this
_PNG_DPI      = 300
_TOP_N        = 10         # terms shown per library in summary dot-plot


# ---------------------------------------------------------------------------
# Enrichr helpers
# ---------------------------------------------------------------------------

def _enrichr_add_list(genes: list, description: str = "") -> int | None:
    """POST a gene list to Enrichr; returns userListId or None on failure."""
    # Enrichr expects multipart/form-data, not URL-encoded
    files = {
        "list":        (None, "\n".join(genes)),
        "description": (None, description),
    }
    try:
        r = requests.post(f"{ENRICHR_BASE}/addList", files=files, timeout=30)
        r.raise_for_status()
        return r.json()["userListId"]
    except Exception as exc:
        logger.error("Enrichr addList failed: %s", exc)
        return None


def _enrichr_get_results(user_list_id: int, library: str) -> pd.DataFrame | None:
    """Fetch enrichment results for one library; returns DataFrame or None."""
    url = f"{ENRICHR_BASE}/enrich"
    params = {"userListId": user_list_id, "backgroundType": library}
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json().get(library, [])
    except Exception as exc:
        logger.error("Enrichr enrich failed (%s): %s", library, exc)
        return None

    if not data:
        return pd.DataFrame()

    rows = []
    for entry in data:
        rows.append({
            "rank":           entry[_RANK_IDX],
            "term":           entry[_TERM_IDX],
            "p_value":        entry[_PVAL_IDX],
            "z_score":        entry[_ZSCORE_IDX],
            "combined_score": entry[_CSCORE_IDX],
            "genes":          ";".join(entry[_GENES_IDX]) if isinstance(entry[_GENES_IDX], list)
                              else entry[_GENES_IDX],
            "adjusted_p_value": entry[_ADJP_IDX],
            "significant":    entry[_ADJP_IDX] < FDR_THRESHOLD,
            "library":        library,
        })
    return pd.DataFrame(rows)


def run_enrichr(
    genes: list,
    label: str,
    output_dir: str,
    fdr_threshold: float = FDR_THRESHOLD,
    retry_delay: float = 1.5,
) -> dict:
    """
    Run ORA for all ENRICHR_LIBS on *genes* and save per-library CSVs.

    Returns dict {library_name: DataFrame}.
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(genes)

    if n == 0:
        logger.warning("[%s] Empty gene list — skipping enrichment.", label)
        return {}

    if n < MIN_GENES:
        logger.warning(
            "[%s] Only %d gene(s) — enrichment results will be unreliable.", label, n
        )

    logger.info("[%s] Submitting %d genes to Enrichr …", label, n)
    user_list_id = _enrichr_add_list(genes, description=label)
    if user_list_id is None:
        return {}

    results = {}
    for lib in ENRICHR_LIBS:
        time.sleep(retry_delay)            # be polite to the API
        df = _enrichr_get_results(user_list_id, lib)
        if df is None or df.empty:
            logger.warning("  [%s] %s — no results returned.", label, lib)
            results[lib] = pd.DataFrame()
            continue

        df["fdr_threshold"] = fdr_threshold
        n_sig = (df["adjusted_p_value"] < fdr_threshold).sum()
        logger.info(
            "  [%s] %s → %d terms, %d significant (adj-p < %.2f)",
            label, lib, len(df), n_sig, fdr_threshold,
        )

        out_path = os.path.join(output_dir, f"{lib}_enrichment.csv")
        df.to_csv(out_path, index=False)
        results[lib] = df

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _clean_term_label(term: str, max_len: int = 55) -> str:
    """Shorten long GO/KEGG/Reactome term strings for axis labels."""
    # Strip trailing GO ID like ' (GO:0006355)'
    if " (" in term:
        term = term[:term.rfind(" (")]
    # Strip leading 'REACTOME_' / 'KEGG_' prefixes (some versions)
    for pfx in ("REACTOME_", "KEGG_", "WP_"):
        if term.startswith(pfx):
            term = term[len(pfx):].replace("_", " ").title()
    if len(term) > max_len:
        term = term[: max_len - 1] + "…"
    return term


def _lib_short_name(lib: str) -> str:
    mapping = {
        "GO_Biological_Process_2023": "GO:BP",
        "GO_Molecular_Function_2023": "GO:MF",
        "GO_Cellular_Component_2023": "GO:CC",
        "KEGG_2021_Human":            "KEGG",
        "Reactome_2022":              "Reactome",
    }
    return mapping.get(lib, lib)


def plot_enrichment_summary(
    all_results: dict,        # {library: DataFrame}
    label: str,
    n_genes: int,
    output_dir: str,
    top_n: int = _TOP_N,
    fdr_threshold: float = FDR_THRESHOLD,
) -> None:
    """
    Dot-plot: top-N significant terms per database.

    Dot size  = number of overlapping genes (derived from 'genes' column).
    Dot color = -log10(adjusted_p_value).
    Only significant terms (adj-p < fdr_threshold) are shown.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect top-N significant terms per library
    plot_rows = []
    for lib, df in all_results.items():
        if df.empty:
            continue
        sig = df[df["adjusted_p_value"] < fdr_threshold].copy()
        if sig.empty:
            continue
        sig = sig.nsmallest(top_n, "adjusted_p_value")
        for _, row in sig.iterrows():
            n_overlap = (
                len(str(row["genes"]).split(";")) if pd.notna(row["genes"]) and row["genes"] else 0
            )
            plot_rows.append({
                "term":      _clean_term_label(str(row["term"])),
                "neg_log10": -np.log10(max(row["adjusted_p_value"], 1e-300)),
                "n_overlap": n_overlap,
                "library":   _lib_short_name(lib),
            })

    if not plot_rows:
        logger.info("[%s] No significant terms to plot.", label)
        return

    plot_df = pd.DataFrame(plot_rows)
    # Deduplicate (same term can appear in multiple libraries)
    plot_df = plot_df.drop_duplicates(subset=["term", "library"])

    # Sort by library then by significance
    lib_order   = [_lib_short_name(l) for l in ENRICHR_LIBS]
    plot_df["lib_rank"] = plot_df["library"].map(
        {l: i for i, l in enumerate(lib_order)}
    ).fillna(99)
    plot_df = plot_df.sort_values(["lib_rank", "neg_log10"], ascending=[True, False])

    # One sub-panel per library
    libs_present = [l for l in lib_order if l in plot_df["library"].values]
    n_libs       = len(libs_present)
    if n_libs == 0:
        return

    fig, axes = plt.subplots(
        1, n_libs,
        figsize=(max(5 * n_libs, 12), max(6, top_n * 0.45)),
        constrained_layout=True,
    )
    if n_libs == 1:
        axes = [axes]

    cmap     = plt.get_cmap("YlOrRd")
    all_vals = plot_df["neg_log10"].values
    vmin, vmax = all_vals.min(), max(all_vals.max(), all_vals.min() + 0.1)

    for ax, lib_name in zip(axes, libs_present):
        sub = plot_df[plot_df["library"] == lib_name].head(top_n)
        if sub.empty:
            ax.set_visible(False)
            continue

        colors  = [cmap((v - vmin) / (vmax - vmin + 1e-9)) for v in sub["neg_log10"]]
        sizes   = np.clip(sub["n_overlap"].values, 1, 30) * 30

        y_pos = np.arange(len(sub))
        ax.scatter(sub["neg_log10"].values, y_pos, c=colors, s=sizes,
                   edgecolors="grey", linewidths=0.5, zorder=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["term"].values, fontsize=8)
        ax.set_xlabel(r"$-\log_{10}$(adj. p-value)", fontsize=9)
        ax.set_title(lib_name, fontsize=10, fontweight="bold")
        ax.axvline(-np.log10(fdr_threshold), color="grey", linestyle="--",
                   linewidth=0.8, label=f"FDR={fdr_threshold}")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    # Shared colorbar for -log10 p
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label=r"$-\log_{10}$(adj. p-value)",
                 shrink=0.5, pad=0.02)

    fig.suptitle(
        f"Enrichment: {label}  |  {n_genes} input proteins  |  "
        f"top {top_n} significant terms per database",
        fontsize=11,
    )
    out_stem = os.path.join(output_dir, f"{label}_enrichment_dotplot")
    fig.savefig(out_stem + ".png", dpi=_PNG_DPI, bbox_inches="tight")
    fig.savefig(out_stem + ".pdf",              bbox_inches="tight")
    plt.close(fig)
    logger.info("[%s] Dot-plot saved → %s.png", label, out_stem)


# ---------------------------------------------------------------------------
# Gene-set collection helpers
# ---------------------------------------------------------------------------

def _load_features(csv_path: str) -> list:
    """Load a lasso_selected_features.csv and return gene list."""
    if not os.path.exists(csv_path):
        return []
    return pd.read_csv(csv_path)["feature"].dropna().tolist()


def collect_gene_sets(binary_results_dir: str, timepoints: list) -> dict:
    """
    Return a dict of {label: [gene_list]} for each timepoint, placenta,
    and the full superset.
    """
    sets = {}

    # Per timepoint
    for tp in timepoints:
        p = os.path.join(binary_results_dir, "plasma", tp, "lasso_selected_features.csv")
        genes = _load_features(p)
        if genes:
            sets[f"plasma_{tp}"] = genes
        else:
            logger.warning("No LASSO features found for plasma %s (%s)", tp, p)

    # Placenta
    p = os.path.join(binary_results_dir, "placenta", "all", "lasso_selected_features.csv")
    genes = _load_features(p)
    if genes:
        sets["placenta"] = genes
    else:
        logger.warning("No LASSO features found for placenta (%s)", p)

    # Overall superset
    union = sorted({g for gl in sets.values() for g in gl})
    if union:
        sets["superset"] = union

    return sets


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_superset_enrichment(
    binary_results_dir: str,
    output_dir: str,
    superset_timepoints: list,
    fdr_threshold: float = FDR_THRESHOLD,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ── logging ──────────────────────────────────────────────────────────
    log_path = os.path.join(output_dir, "analysis_log.txt")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("=" * 70)
    logger.info("Superset Enrichment Analysis")
    logger.info("Date/time  : %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Timepoints : %s", " ".join(superset_timepoints))
    logger.info("Databases  : %s", ", ".join(ENRICHR_LIBS))
    logger.info("FDR thresh : %.2f (Enrichr adjusted p-value)", fdr_threshold)
    logger.info("=" * 70)

    # ── collect gene sets ─────────────────────────────────────────────────
    gene_sets = collect_gene_sets(binary_results_dir, superset_timepoints)

    if not gene_sets:
        logger.error(
            "No LASSO feature files found under %s. "
            "Run binary_classifier.py first.",
            binary_results_dir,
        )
        return

    for label, genes in gene_sets.items():
        logger.info("Gene set %-20s : %d proteins", label, len(genes))

    summary_dir = os.path.join(output_dir, "summary")

    # ── run enrichment per gene set ───────────────────────────────────────
    for label, genes in gene_sets.items():
        label_dir = os.path.join(output_dir, label)
        logger.info("-" * 60)
        results = run_enrichr(
            genes,
            label=label,
            output_dir=label_dir,
            fdr_threshold=fdr_threshold,
        )
        plot_enrichment_summary(
            results,
            label=label,
            n_genes=len(genes),
            output_dir=summary_dir,
            fdr_threshold=fdr_threshold,
        )

    logger.info("=" * 70)
    logger.info("Enrichment analysis complete. Output: %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Over-representation analysis (Enrichr) on LASSO-selected superset proteins."
        )
    )
    p.add_argument(
        "--binary-results-dir",
        default=None,
        help=(
            "Root of binary classifier outputs containing "
            "plasma/<TP>/lasso_selected_features.csv and "
            "placenta/all/lasso_selected_features.csv. "
            "Defaults to 04_results_and_figures/models/binary/."
        ),
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Root output directory. "
            "Defaults to 04_results_and_figures/models/binary/superset_enrichment/."
        ),
    )
    p.add_argument(
        "--superset-timepoints",
        nargs="+",
        default=["A", "B", "C", "D"],
        help=(
            "Plasma timepoints included in the superset "
            "(E excluded by default due to LASSO regularisation collapse). "
            "Defaults to A B C D."
        ),
    )
    p.add_argument(
        "--fdr-threshold",
        type=float,
        default=FDR_THRESHOLD,
        help=f"Adjusted p-value threshold for significance (default: {FDR_THRESHOLD}).",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    wkdir = os.getcwd()

    binary_results_dir = args.binary_results_dir or os.path.join(
        wkdir, "04_results_and_figures", "models", "binary"
    )
    output_dir = args.output_dir or os.path.join(
        wkdir, "04_results_and_figures", "models", "binary", "superset_enrichment"
    )

    run_superset_enrichment(
        binary_results_dir=binary_results_dir,
        output_dir=output_dir,
        superset_timepoints=args.superset_timepoints,
        fdr_threshold=args.fdr_threshold,
    )


if __name__ == "__main__":
    main()
