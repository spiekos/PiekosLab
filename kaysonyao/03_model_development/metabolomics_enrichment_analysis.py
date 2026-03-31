"""
metabolomics_enrichment_analysis.py

Over-representation analysis (ORA) via KEGG REST API on the significant
metabolites identified by the longitudinal and cross-sectional differential
analyses.

Efficient KEGG workflow (minimises API calls)
---------------------------------------------
  For each metabolite set:
    1. Map compound names to KEGG IDs  (1 call per compound).
    2. Get all KEGG human pathways for mapped compounds  (1 call per compound).
    3. For each pathway that has ≥1 query compound, fetch its full compound
       list from KEGG (1 call per pathway — universe comes from KEGG, not
       our dataset).
    4. Hypergeometric ORA:
         N = total unique KEGG compounds across all relevant HSA pathways
         K = compounds in pathway k (from KEGG)
         n = query compounds mapped to KEGG
         k = query compounds in pathway k
    5. BH-FDR; dot-plot of top-N by combined score (−log10p × enrichment ratio).

Results are cached to avoid redundant API calls on re-runs.

Usage
-----
  python 03_model_development/metabolomics_enrichment_analysis.py

  python 03_model_development/metabolomics_enrichment_analysis.py \\
      --diff-results-dir  04_results_and_figures/differential_analysis/metabolomics \\
      --output-dir        04_results_and_figures/models/binary/metabolomics/enrichment \\
      --top-n             15
"""

import argparse
import datetime
import glob
import json
import logging
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

KEGG_BASE     = "https://rest.kegg.jp"
REQUEST_PAUSE = 0.4       # ≤3 requests/second (KEGG policy)
FDR_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Metabolite name utilities
# ---------------------------------------------------------------------------

def clean_name(raw: str) -> str:
    """Strip _POS / _NEG polarity suffix."""
    return re.sub(r"[_ ](POS|NEG)$", "", raw, flags=re.IGNORECASE).strip()


_UNNAMED_PEAK = re.compile(r"^[pn]\d+$")
_ISTD         = re.compile(r"\bISTD\b", re.IGNORECASE)
_BRACKET      = re.compile(r"^\[Similar to:")


def is_identified(name: str) -> bool:
    return (
        not _UNNAMED_PEAK.match(name)
        and not _ISTD.search(name)
        and not _BRACKET.match(name)
    )


# ---------------------------------------------------------------------------
# KEGG REST helpers (with simple cache)
# ---------------------------------------------------------------------------

def _kegg_get(endpoint: str, cache: dict, retries: int = 3) -> str | None:
    if endpoint in cache:
        return cache[endpoint]
    url = f"{KEGG_BASE}/{endpoint}"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                cache[endpoint] = r.text
                return r.text
            if r.status_code == 404:
                cache[endpoint] = None
                return None
        except requests.RequestException:
            pass
        time.sleep(REQUEST_PAUSE * (attempt + 1))
    cache[endpoint] = None
    return None


def _simplify_name(name: str) -> str | None:
    """Generate a simplified fallback name for KEGG search.

    Removes D-(+)-, L-(+)-, stereochemistry prefixes and replaces
    Greek letters with ASCII equivalents so that e.g.
    'D-(+)-Maltose' → 'Maltose' and 'α-Lactose' → 'alpha-Lactose'.
    Returns None if no simplification is possible.
    """
    # Replace Greek letters with ASCII names
    greek = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
             "Α": "alpha", "Β": "beta", "Γ": "gamma", "Δ": "delta"}
    simplified = name
    for g, rep in greek.items():
        simplified = simplified.replace(g, rep)

    # Strip stereo/configuration prefix patterns like D-(+)-, L-(-)-
    simplified = re.sub(r"^[DL]-\([+\-±]\)-", "", simplified)
    simplified = re.sub(r"^[DL]-\(\+\)-", "", simplified)

    if simplified != name:
        return simplified.strip()

    # If name contains parentheses stereochemistry in the middle, strip it
    base = re.sub(r"\s*\([+\-±]\)", "", name).strip()
    if base != name:
        return base

    return None


def map_name_to_kegg_id(name: str, cache: dict) -> str | None:
    """Return first KEGG compound ID matching *name*.

    Falls back to a simplified version of the name if the exact match fails.
    """
    def _first_cpd(text: str) -> str | None:
        if not text:
            return None
        for line in text.strip().splitlines():
            parts = line.split("\t")
            if parts and parts[0].startswith("cpd:C"):
                return parts[0].replace("cpd:", "")
        return None

    # Try exact name first
    text = _kegg_get(f"find/compound/{requests.utils.quote(name)}", cache)
    time.sleep(REQUEST_PAUSE)
    result = _first_cpd(text)
    if result:
        return result

    # Try simplified name
    simplified = _simplify_name(name)
    if simplified and simplified != name:
        text2 = _kegg_get(f"find/compound/{requests.utils.quote(simplified)}", cache)
        time.sleep(REQUEST_PAUSE)
        result2 = _first_cpd(text2)
        if result2:
            logger.debug("  Resolved %r via simplified name %r → %s", name, simplified, result2)
            return result2

    return None


def get_hsa_pathways(cpd_id: str, cache: dict) -> list[str]:
    """Return KEGG reference pathway IDs for a compound.

    KEGG's compound→pathway link endpoint returns 'path:map####' IDs
    (reference pathways), not 'path:hsa####'.  The 'hsa####' variants
    return empty compound lists, so we use 'map####' throughout.
    """
    text = _kegg_get(f"link/pathway/{cpd_id}", cache)
    time.sleep(REQUEST_PAUSE)
    if not text:
        return []
    return [
        line.split("\t")[1].replace("path:", "")
        for line in text.strip().splitlines()
        if "\t" in line and line.split("\t")[1].startswith("path:map")
    ]


def get_pathway_compounds(path_id: str, cache: dict) -> set[str]:
    """Return all compound IDs in a KEGG pathway."""
    text = _kegg_get(f"link/compound/{path_id}", cache)
    time.sleep(REQUEST_PAUSE)
    if not text:
        return set()
    cpds = set()
    for line in text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].startswith("cpd:C"):
            cpds.add(parts[1].replace("cpd:", ""))
    return cpds


def get_pathway_name(path_id: str, cache: dict) -> str:
    """Human-readable KEGG pathway name."""
    text = _kegg_get(f"list/{path_id}", cache)
    time.sleep(REQUEST_PAUSE)
    if not text:
        return path_id
    parts = text.strip().split("\t")
    if len(parts) >= 2:
        return re.sub(r"\s*-\s*Homo sapiens.*$", "", parts[1]).strip()
    return path_id


# ---------------------------------------------------------------------------
# ORA
# ---------------------------------------------------------------------------

def run_ora(
    query_cpds: set[str],
    path_to_cpds: dict[str, set[str]],
    all_bg_cpds: set[str],
    cache: dict,
) -> pd.DataFrame:
    """Hypergeometric ORA using KEGG pathway compound universe."""
    N = len(all_bg_cpds)
    n = len(query_cpds & all_bg_cpds)
    if n == 0:
        return pd.DataFrame()

    rows = []
    for path_id, bg_cpds in path_to_cpds.items():
        K = len(bg_cpds & all_bg_cpds)
        hits = query_cpds & bg_cpds
        k = len(hits)
        if k == 0 or K == 0:
            continue
        p_val = hypergeom.sf(k - 1, N, K, n)
        enr   = (k / n) / (K / N)
        rows.append({
            "pathway_id":      path_id,
            "overlap":         k,
            "query_size":      n,
            "pathway_size":    K,
            "background_size": N,
            "p_value":         p_val,
            "enrichment_ratio": enr,
            "matched_compounds": ";".join(sorted(hits)),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    _, q_vals, _, _ = multipletests(df["p_value"], method="fdr_bh")
    df["q_value"]    = q_vals
    df["significant"] = df["q_value"] < FDR_THRESHOLD
    df["-log10p"]    = -np.log10(df["p_value"].clip(1e-300))
    df["combined_score"] = df["-log10p"] * df["enrichment_ratio"]

    # Resolve pathway names for top rows only
    df = df.head(50).copy()
    df["pathway_name"] = df["pathway_id"].apply(lambda pid: get_pathway_name(pid, cache))
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dot(df: pd.DataFrame, label: str, out_path: str, top_n: int = 15) -> None:
    if df.empty:
        return
    plot_df = (
        df.nlargest(top_n, "combined_score")
          .sort_values("combined_score", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.5)))
    colors = ["#d62728" if q < FDR_THRESHOLD else "#aec7e8" for q in plot_df["q_value"]]
    sc = ax.scatter(
        plot_df["combined_score"],
        range(len(plot_df)),
        s=np.clip(plot_df["overlap"] * 80, 40, 600),
        c=plot_df["-log10p"],
        cmap="YlOrRd",
        edgecolors="grey",
        linewidths=0.5,
        zorder=3,
    )
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["pathway_name"], fontsize=9)
    ax.set_xlabel("Combined score  (−log₁₀(p) × enrichment ratio)", fontsize=10)
    ax.set_title(
        f"KEGG Pathway Enrichment — {label}\n"
        f"(dot size ∝ overlap; red = FDR < {FDR_THRESHOLD})",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.3)
    plt.colorbar(sc, ax=ax, shrink=0.5, label="−log₁₀(p)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Plot saved → %s", out_path)


# ---------------------------------------------------------------------------
# Metabolite set collection
# ---------------------------------------------------------------------------

def collect_metabolite_sets(diff_dir: str) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}

    for fp in sorted(glob.glob(os.path.join(
        diff_dir, "plasma", "longitudinal", "*_longitudinal_results.csv"
    ))):
        df  = pd.read_csv(fp)
        sig = df[df.get("significant", pd.Series(False, index=df.index)) == True]
        if not sig.empty:
            label = os.path.basename(fp).replace("_longitudinal_results.csv", "")
            sets[label] = sig["analyte_id"].dropna().tolist()

    for tp in list("ABCDE"):
        fp = os.path.join(
            diff_dir, "plasma", "cross_sectional", tp,
            "Control_vs_Complication_differential_results.csv",
        )
        if not os.path.exists(fp):
            continue
        df  = pd.read_csv(fp)
        sig = df[df.get("significant", pd.Series(False, index=df.index)) == True]
        if not sig.empty:
            sets[f"cs_plasma_{tp}"] = sig["analyte_id"].dropna().tolist()

    fp = os.path.join(
        diff_dir, "placenta", "cross_sectional",
        "Control_vs_Complication_differential_results.csv",
    )
    if os.path.exists(fp):
        df  = pd.read_csv(fp)
        sig = df[df.get("significant", pd.Series(False, index=df.index)) == True]
        if not sig.empty:
            sets["cs_placenta"] = sig["analyte_id"].dropna().tolist()

    union = sorted({a for lst in sets.values() for a in lst})
    if union:
        sets["superset"] = union

    return sets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_metabolomics_enrichment(
    diff_results_dir: str,
    output_dir: str,
    top_n: int = 15,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    fh = logging.FileHandler(os.path.join(output_dir, "analysis_log.txt"), mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("=" * 70)
    logger.info("Metabolomics KEGG Pathway Enrichment")
    logger.info("Date/time : %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 70)

    # ── collect metabolite sets ───────────────────────────────────────────
    met_sets = collect_metabolite_sets(diff_results_dir)
    if not met_sets:
        logger.error("No significant metabolites found under %s.", diff_results_dir)
        return

    # ── deduplicate all query names across sets ───────────────────────────
    all_raw  = sorted({a for lst in met_sets.values() for a in lst})
    all_clean = {a: clean_name(a) for a in all_raw}
    all_identified = {a: c for a, c in all_clean.items() if is_identified(c)}

    for label, analytes in met_sets.items():
        named = [all_identified[a] for a in analytes if a in all_identified]
        logger.info("Set %-32s : %d analytes → %d identified", label, len(analytes), len(named))

    # ── in-memory API cache ───────────────────────────────────────────────
    cache_path = os.path.join(output_dir, "kegg_api_cache.json")
    api_cache: dict = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            api_cache = json.load(f)
        logger.info("Loaded %d cached KEGG responses.", len(api_cache))

    def _save_cache() -> None:
        with open(cache_path, "w") as f:
            json.dump(api_cache, f)

    # ── map all identified analytes to KEGG IDs ───────────────────────────
    unique_identified_names = sorted({c for c in all_identified.values()})
    logger.info("Mapping %d unique identified compound names to KEGG …", len(unique_identified_names))

    name_to_cpd: dict[str, str] = {}
    for i, name in enumerate(unique_identified_names):
        cpd = map_name_to_kegg_id(name, api_cache)
        if cpd:
            name_to_cpd[name] = cpd
        if (i + 1) % 10 == 0:
            _save_cache()
    _save_cache()

    logger.info(
        "KEGG mapping: %d / %d names resolved",
        len(name_to_cpd), len(unique_identified_names),
    )
    if not name_to_cpd:
        logger.error("No compounds mapped to KEGG — cannot run ORA.")
        return

    # ── for each mapped compound, get HSA pathways ────────────────────────
    query_cpd_ids = set(name_to_cpd.values())
    cpd_to_paths: dict[str, list[str]] = {}
    for cpd in sorted(query_cpd_ids):
        cpd_to_paths[cpd] = get_hsa_pathways(cpd, api_cache)
    _save_cache()

    # ── for each relevant pathway, get its full compound list ─────────────
    relevant_paths = sorted({p for paths in cpd_to_paths.values() for p in paths})
    logger.info("Fetching compound lists for %d relevant HSA pathways …", len(relevant_paths))

    path_to_cpds: dict[str, set[str]] = {}
    for pid in relevant_paths:
        path_to_cpds[pid] = get_pathway_compounds(pid, api_cache)
    _save_cache()

    # Background = union of all compounds in relevant pathways
    all_bg_cpds = {c for cpds in path_to_cpds.values() for c in cpds}
    logger.info("KEGG universe: %d unique compounds across %d pathways", len(all_bg_cpds), len(relevant_paths))

    # ── ORA per set ───────────────────────────────────────────────────────
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    for label, analytes in met_sets.items():
        named_query  = [all_identified[a] for a in analytes if a in all_identified]
        query_cpds   = {name_to_cpd[n] for n in named_query if n in name_to_cpd}

        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        logger.info("-" * 60)
        logger.info("ORA: %s  (%d identified → %d KEGG-mapped)", label, len(named_query), len(query_cpds))

        if not query_cpds:
            logger.warning("  No KEGG-mapped compounds — skipping.")
            continue

        result_df = run_ora(query_cpds, path_to_cpds, all_bg_cpds, api_cache)
        _save_cache()

        if result_df.empty:
            logger.info("  No pathway hits.")
            continue

        csv_path = os.path.join(label_dir, "kegg_pathway_enrichment.csv")
        result_df.to_csv(csv_path, index=False)
        logger.info(
            "  %d pathways tested, %d significant (FDR < %.2f)",
            len(result_df), result_df["significant"].sum(), FDR_THRESHOLD,
        )

        plot_dot(
            result_df, label,
            out_path=os.path.join(summary_dir, f"{label}_kegg_enrichment.png"),
            top_n=top_n,
        )

    logger.info("=" * 70)
    logger.info("Metabolomics enrichment complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    wkdir = os.getcwd()
    p = argparse.ArgumentParser(
        description="KEGG pathway ORA on differential metabolomics analytes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--diff-results-dir",
        default=os.path.join(
            wkdir, "04_results_and_figures", "differential_analysis", "metabolomics",
        ),
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join(
            wkdir, "04_results_and_figures", "models", "binary", "metabolomics", "enrichment",
        ),
    )
    p.add_argument("--top-n", type=int, default=15)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_metabolomics_enrichment(
        diff_results_dir=args.diff_results_dir,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )
