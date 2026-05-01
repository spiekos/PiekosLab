"""
Pathway over-representation analysis (ORA) for MTBL_sop significant metabolites.

Strategy:
  1. Load MTBL_plasma feature metadata → compound names for all 502 analytes (background)
  2. Load differential results (MTBL_sop) → compound names for q<0.05 analytes per TP
  3. Map compound names → HMDB IDs via MetaboAnalyst REST API (name matching)
  4. Run ORA per timepoint + union-across-TPs using KEGG PATHWAY / SMPDB via MetaboAnalyst
  5. Save per-TP result CSVs + a dot-plot summary figure

Usage (from kaysonyao/):
    python3 03_model_development/run_pathway_analysis.py
    python3 03_model_development/run_pathway_analysis.py --sig-only   # use q<0.05+FC sig set
    python3 03_model_development/run_pathway_analysis.py --timepoints A B C
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
METABOANALYST_API = "https://rest.xialab.ca/api"
KEGG_API          = "https://rest.kegg.jp"
TIMEPOINTS        = ["A", "B", "C", "D", "E"]
_METADATA_COLS    = {"SampleID","SubjectID","Group","Timepoint","Batch",
                     "MetadataCanonicalID","Age","BMI","Gestational_Age"}

# ── Data loading ─────────────────────────────────────────────────────────────

def load_feature_metadata(wkdir: str) -> pd.DataFrame:
    path = os.path.join(wkdir, "data", "cleaned", "sop_omics_pipeline_v2",
                        "MTBL_plasma", "MTBL_plasma_feature_metadata.csv")
    df = pd.read_csv(path, index_col=0)
    logger.info("Feature metadata: %d features (%d named)", len(df), df["is_named"].sum())
    return df


def load_sig_analytes(wkdir: str, use_sig_only: bool = False) -> dict:
    """
    Returns {tp: [analyte_id, ...]} for named analytes that pass threshold.
    use_sig_only=True  → q<0.05 + |log2FC|>=0.585  (the 'significant' column)
    use_sig_only=False → q<0.05 only               (what feeds the model)
    """
    result = {}
    diff_root = os.path.join(wkdir, "04_results_and_figures",
                             "differential_analysis", "MTBL_sop",
                             "plasma", "cross_sectional")
    for tp in TIMEPOINTS:
        path = os.path.join(diff_root, tp,
                            "Control_vs_Complication_differential_results.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if use_sig_only:
            hits = df[df["significant"] == True]["analyte_id"].tolist()
        else:
            hits = df[df["q_value"] < 0.05]["analyte_id"].tolist()
        result[tp] = hits
    return result


# ── Name → HMDB mapping via MetaboAnalyst ────────────────────────────────────

def _match_one(name: str) -> dict:
    """Map a single compound name via MetaboAnalyst REST. Returns a result dict."""
    url = f"{METABOANALYST_API}/mapcompounds"
    try:
        r = requests.post(url, json={"queryList": name, "inputType": "name"}, timeout=15)
        r.raise_for_status()
        d    = r.json()
        kegg = d.get("KEGG", [None])[0]
        hmdb = d.get("HMDB", [None])[0]
        return {
            "query":        name,
            "hmdb_id":      hmdb if hmdb not in (None, "NA", "null") else "",
            "kegg_id":      kegg if kegg not in (None, "NA", "null") else "",
            "match_status": d.get("Comment", [""])[0],
        }
    except Exception as e:
        logger.debug("MetaboAnalyst match failed for '%s': %s", name, e)
        return {"query": name, "hmdb_id": "", "kegg_id": "", "match_status": "error"}


def _metaboanalyst_name_match(
    compound_names: list[str],
    cache_path: str | None = None,
    n_workers: int = 20,
) -> pd.DataFrame:
    """
    Map compound names to KEGG/HMDB IDs via MetaboAnalyst REST.
    Uses parallel workers and a file-based cache with incremental saving
    so interrupted runs can resume without re-querying.

    Returns DataFrame with columns: query, hmdb_id, kegg_id, match_status
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Load cache if available
    cached = {}
    if cache_path and os.path.exists(cache_path):
        cached_df = pd.read_csv(cache_path, dtype=str).fillna("")
        cached = {row["query"]: row.to_dict() for _, row in cached_df.iterrows()}
        logger.info("  Loaded %d cached mappings from %s", len(cached), cache_path)

    to_fetch = [n for n in compound_names if n not in cached]
    logger.info("  Name matching: %d cached, %d to fetch (workers=%d)",
                len(cached), len(to_fetch), n_workers)

    # Thread-safe incremental cache writing
    lock         = threading.Lock()
    fetched      = {}
    save_counter = [0]

    def _flush_cache(new_results: dict):
        if not cache_path:
            return
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)
        all_so_far = {**cached, **fetched, **new_results}
        pd.DataFrame(list(all_so_far.values())).to_csv(cache_path, index=False)

    if to_fetch:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_match_one, name): name for name in to_fetch}
            done = 0
            for fut in as_completed(futures):
                result = fut.result()
                with lock:
                    fetched[result["query"]] = result
                    done += 1
                    save_counter[0] += 1
                    if save_counter[0] % 15 == 0:    # save every 15 results
                        _flush_cache({})
                        logger.info("    … %d / %d fetched (cache saved)", done, len(to_fetch))

        # Final save
        _flush_cache({})

    # Merge cached + fetched; return in original order
    all_rows = {**cached, **fetched}
    rows     = [all_rows[n] for n in compound_names if n in all_rows]
    df       = pd.DataFrame(rows)
    matched  = (df["kegg_id"].replace("", np.nan).notna()).sum()
    logger.info("  MetaboAnalyst name match: %d / %d mapped to KEGG", matched, len(df))
    return df


# ── KEGG pathway ORA ─────────────────────────────────────────────────────────

def _get_kegg_pathways_for_compound(kegg_cid: str, organism: str = "map") -> list[str]:
    """Return KEGG pathway IDs that contain this compound (reference map pathways).

    The KEGG link API returns reference pathway IDs in the form 'path:map00220'.
    We keep all 'map' pathways (universal reference pathways) which cover all
    metabolic reactions regardless of organism — appropriate for metabolomics ORA.
    """
    time.sleep(0.2)   # be polite to KEGG
    url = f"{KEGG_API}/link/pathway/cpd:{kegg_cid}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200 or not r.text.strip():
            return []
        pathways = []
        for line in r.text.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) == 2:
                pway = parts[1].strip()          # e.g. "path:map00220"
                # Keep reference map pathways (exclude disease/drug/brite paths > 09000)
                pway_clean = pway.replace("path:", "")
                if pway_clean.startswith("map") and pway_clean[3:].isdigit():
                    pw_num = int(pway_clean[3:])
                    if pw_num < 9000:   # metabolic + signaling pathways only
                        pathways.append(pway_clean)
        return pathways
    except Exception:
        return []


def _get_kegg_pathway_name(pathway_id: str) -> str:
    """Fetch the human-readable name for a KEGG pathway ID."""
    time.sleep(0.1)
    url = f"{KEGG_API}/get/{pathway_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return pathway_id
        for line in r.text.split("\n"):
            if line.startswith("NAME"):
                name = line.replace("NAME", "").strip()
                name = name.split(" - ")[0].strip()   # remove species suffix
                return name
        return pathway_id
    except Exception:
        return pathway_id


def build_compound_pathway_map(kegg_ids: list[str]) -> dict[str, list[str]]:
    """
    Build {kegg_id: [pathway_id, ...]} for a list of KEGG compound IDs.
    Skips empty/null IDs.
    """
    cpd_map = {}
    valid = [k for k in kegg_ids if k and k != "nan"]
    logger.info("  Fetching KEGG pathway membership for %d compounds…", len(valid))
    for cid in valid:
        pathways = _get_kegg_pathways_for_compound(cid)
        cpd_map[cid] = pathways
    mapped = sum(1 for v in cpd_map.values() if v)
    logger.info("  %d / %d compounds mapped to at least one KEGG pathway.", mapped, len(valid))
    return cpd_map


def run_ora(
    hit_kegg_ids:  list[str],
    bg_kegg_ids:   list[str],
    cpd_pathway_map: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Over-representation analysis (ORA) using Fisher's exact test.

    hit_kegg_ids : KEGG IDs of the "foreground" (significant) compounds
    bg_kegg_ids  : KEGG IDs of ALL compounds tested (background)
    cpd_pathway_map: {kegg_id: [pathway_id, ...]}
    """
    hit_set = set(hit_kegg_ids)
    bg_set  = set(bg_kegg_ids)
    n_hit   = len(hit_set)
    n_bg    = len(bg_set)

    # Invert map: pathway → set of compound IDs
    pathway_to_cpds = {}
    for cid, pathways in cpd_pathway_map.items():
        for p in pathways:
            pathway_to_cpds.setdefault(p, set()).add(cid)

    rows = []
    for pathway, pathway_cpds in pathway_to_cpds.items():
        in_hit  = len(hit_set  & pathway_cpds)
        in_bg   = len(bg_set   & pathway_cpds)
        if in_hit == 0:
            continue

        # 2×2 contingency:
        #            In pathway  Not in pathway
        # Hits              a           b
        # Background        c           d
        a = in_hit
        b = n_hit  - in_hit
        c = in_bg  - in_hit      # background only (not counting hits twice)
        d = n_bg   - n_hit - c

        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")

        rows.append({
            "pathway_id":    pathway,
            "n_hits_in_pw":  a,
            "n_hits_total":  n_hit,
            "n_bg_in_pw":    in_bg,
            "n_bg_total":    n_bg,
            "p_value":       p,
            "hit_compounds": "; ".join(sorted(hit_set & pathway_cpds)),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("p_value")
    # FDR correction
    _, q_vals, _, _ = multipletests(df["p_value"].values, method="fdr_bh")
    df["q_value"] = q_vals
    df["significant"] = df["q_value"] < 0.05

    # Add pathway names
    logger.info("  Fetching pathway names for %d pathways…", len(df))
    df["pathway_name"] = df["pathway_id"].apply(_get_kegg_pathway_name)

    return df.sort_values("q_value").reset_index(drop=True)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_pathway_dotplot(
    ora_df: pd.DataFrame,
    title: str,
    output_path: str,
    top_n: int = 20,
):
    """Dot plot: x = -log10(p), dot size = hits in pathway, colour = q-value."""
    df = ora_df.head(top_n).copy()
    if df.empty:
        logger.warning("No pathways to plot for %s", title)
        return

    df = df.sort_values("p_value", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.38 * len(df))))

    x = -np.log10(df["p_value"].clip(lower=1e-10))
    y = range(len(df))
    sizes = (df["n_hits_in_pw"] / df["n_bg_in_pw"] * 300).clip(lower=20, upper=400)
    colours = ["#ef4444" if q < 0.05 else "#94a3b8" for q in df["q_value"]]

    sc = ax.scatter(x, y, s=sizes, c=colours, alpha=0.85, zorder=3)
    ax.axvline(-np.log10(0.05), color="#64748b", linestyle="--",
               linewidth=0.9, label="p = 0.05")

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["pathway_name"].tolist(), fontsize=8.5)
    ax.set_xlabel("−log₁₀(p-value)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend for colour
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ef4444",
               markersize=9, label="q < 0.05 (FDR)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#94a3b8",
               markersize=9, label="q ≥ 0.05"),
        Line2D([0], [0], color="#64748b", linestyle="--", label="p = 0.05"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Dot plot saved: %s", output_path)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timepoints",  nargs="+", default=TIMEPOINTS)
    parser.add_argument("--sig-only",    action="store_true",
                        help="Use q<0.05+FC significant set instead of q<0.05 only.")
    args = parser.parse_args()

    wkdir    = os.getcwd()
    out_root = os.path.join(wkdir, "04_results_and_figures",
                            "pathway_analysis", "MTBL_sop")
    os.makedirs(out_root, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────
    meta     = load_feature_metadata(wkdir)
    sig_dict = load_sig_analytes(wkdir, use_sig_only=args.sig_only)

    # Background = all named analytes
    bg_named  = meta[meta["is_named"] == True]["annotation_name"].dropna().tolist()
    logger.info("Background (named analytes): %d compounds", len(bg_named))

    # ── 2. Name → HMDB/KEGG mapping for background ────────────────────────
    cache_path = os.path.join(out_root, "background_name_mapping.csv")
    logger.info("Mapping background compound names via MetaboAnalyst…")
    bg_map = _metaboanalyst_name_match(bg_named, cache_path=cache_path, n_workers=6)
    bg_map["final_feature_id"] = meta[meta["is_named"]].index.tolist()[:len(bg_map)]
    bg_map.to_csv(cache_path, index=False)

    bg_kegg_ids = bg_map["kegg_id"].dropna().tolist()
    bg_kegg_ids = [k for k in bg_kegg_ids if k and k != "nan" and k != ""]

    # ── 3. Build compound → pathway map from background KEGG IDs ──────────
    logger.info("Building KEGG pathway membership map for background…")
    cpd_pathway_map = build_compound_pathway_map(bg_kegg_ids)
    with open(os.path.join(out_root, "compound_pathway_map.json"), "w") as fh:
        json.dump({k: v for k, v in cpd_pathway_map.items() if v}, fh, indent=2)

    # ── 4. Per-timepoint ORA ──────────────────────────────────────────────
    all_pathway_results = {}
    union_hit_ids = set()

    for tp in args.timepoints:
        if tp not in sig_dict:
            logger.warning("No differential results for TP %s, skipping.", tp)
            continue

        hit_ids = [i for i in sig_dict[tp]
                   if i in meta.index and meta.loc[i, "is_named"]]

        n_hit = len(hit_ids)
        logger.info("=== TP %s: %d named hits ===", tp, n_hit)

        if n_hit < 2:
            logger.warning("  Too few named hits for ORA (n=%d), skipping.", n_hit)
            continue

        # Get compound names
        hit_names  = [meta.loc[i, "annotation_name"] for i in hit_ids]

        # Map hit names to KEGG IDs — reuse background cache first, only fetch truly new names
        logger.info("  Mapping hit compound names (reusing background cache)…")
        hit_cache = os.path.join(out_root, f"TP{tp}_hit_mapping.csv")
        # Seed hit cache with any entries already present in the background cache
        bg_cache_path = os.path.join(out_root, "background_name_mapping.csv")
        if os.path.exists(bg_cache_path) and not os.path.exists(hit_cache):
            bg_df = pd.read_csv(bg_cache_path)
            bg_for_hits = bg_df[bg_df["query"].isin(hit_names)]
            if not bg_for_hits.empty:
                bg_for_hits.to_csv(hit_cache, index=False)
                logger.info("    Pre-seeded hit cache with %d entries from background.", len(bg_for_hits))
        hit_map = _metaboanalyst_name_match(hit_names, cache_path=hit_cache, n_workers=6)
        hit_map.to_csv(hit_cache, index=False)

        hit_kegg_ids = hit_map["kegg_id"].dropna().tolist()
        hit_kegg_ids = [k for k in hit_kegg_ids if k and k != "nan" and k != ""]
        union_hit_ids.update(hit_kegg_ids)

        if not hit_kegg_ids:
            logger.warning("  No KEGG IDs returned for TP %s hits.", tp)
            continue

        # Add any new hit compounds to the pathway map
        new_ids = [k for k in hit_kegg_ids if k not in cpd_pathway_map]
        if new_ids:
            extra = build_compound_pathway_map(new_ids)
            cpd_pathway_map.update(extra)

        # ORA
        logger.info("  Running ORA for TP %s…", tp)
        ora_df = run_ora(hit_kegg_ids, bg_kegg_ids, cpd_pathway_map)

        if ora_df.empty:
            logger.warning("  No pathway overlaps found for TP %s.", tp)
            continue

        # Save
        tp_dir = os.path.join(out_root, f"plasma_TP{tp}")
        os.makedirs(tp_dir, exist_ok=True)
        ora_df.to_csv(os.path.join(tp_dir, "pathway_ora_results.csv"), index=False)

        # Plot
        n_sig = (ora_df["q_value"] < 0.05).sum()
        label = "sig only (q<0.05+FC)" if args.sig_only else "q<0.05"
        plot_pathway_dotplot(
            ora_df,
            title=f"MTBL_sop Plasma TP {tp} — Pathway ORA ({label})\n"
                  f"{n_hit} hits · {n_sig} significant pathways",
            output_path=os.path.join(tp_dir, "pathway_dotplot.png"),
        )

        all_pathway_results[tp] = ora_df
        logger.info("  TP %s: %d pathways tested, %d significant (q<0.05)", tp, len(ora_df), n_sig)

    # ── 5. Union analysis (hits across all TPs) ────────────────────────────
    if len(union_hit_ids) >= 2:
        logger.info("=== Union analysis (%d unique KEGG hit IDs across all TPs) ===",
                    len(union_hit_ids))
        new_ids = [k for k in union_hit_ids if k not in cpd_pathway_map]
        if new_ids:
            extra = build_compound_pathway_map(new_ids)
            cpd_pathway_map.update(extra)

        ora_union = run_ora(list(union_hit_ids), bg_kegg_ids, cpd_pathway_map)
        union_dir = os.path.join(out_root, "plasma_union")
        os.makedirs(union_dir, exist_ok=True)

        if not ora_union.empty:
            ora_union.to_csv(os.path.join(union_dir, "pathway_ora_results.csv"), index=False)
            n_sig = (ora_union["q_value"] < 0.05).sum()
            label = "sig only" if args.sig_only else "q<0.05"
            plot_pathway_dotplot(
                ora_union,
                title=f"MTBL_sop Plasma — Union ORA (all TPs, {label})\n"
                      f"{len(union_hit_ids)} hits · {n_sig} significant pathways",
                output_path=os.path.join(union_dir, "pathway_dotplot.png"),
            )
            all_pathway_results["union"] = ora_union
            logger.info("Union: %d pathways tested, %d significant", len(ora_union), n_sig)

    # ── 6. Cross-TP heatmap: top pathways vs timepoint significance ────────
    if len(all_pathway_results) >= 2:
        _plot_cross_tp_heatmap(all_pathway_results, out_root)

    logger.info("Pathway analysis complete. Results in: %s", out_root)


def _plot_cross_tp_heatmap(results: dict, out_root: str, top_n: int = 25):
    """
    Heatmap of −log10(p) for top pathways across timepoints.
    Rows = top pathways (by min p-value across TPs), columns = TPs.
    """
    tps = [tp for tp in TIMEPOINTS if tp in results]
    if not tps:
        return

    # Collect all pathway names and their p-values per TP
    pathway_pvals = {}    # {pathway_name: {tp: p_value}}
    for tp in tps:
        df = results[tp]
        for _, row in df.iterrows():
            pw = row["pathway_name"]
            pathway_pvals.setdefault(pw, {})
            pathway_pvals[pw][tp] = row["p_value"]

    # Rank pathways by min p-value across all TPs
    pw_min_p = {pw: min(vals.values()) for pw, vals in pathway_pvals.items()}
    top_pathways = sorted(pw_min_p, key=pw_min_p.get)[:top_n]

    # Build matrix
    mat = np.full((len(top_pathways), len(tps)), np.nan)
    for i, pw in enumerate(top_pathways):
        for j, tp in enumerate(tps):
            p = pathway_pvals[pw].get(tp, 1.0)
            mat[i, j] = -np.log10(max(p, 1e-10))

    fig, ax = plt.subplots(figsize=(max(5, len(tps) * 1.2), max(6, len(top_pathways) * 0.35)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=min(6, np.nanmax(mat)))

    ax.set_xticks(range(len(tps)))
    ax.set_xticklabels([f"TP {tp}" for tp in tps], fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(top_pathways)))
    ax.set_yticklabels(top_pathways, fontsize=8)
    ax.set_title("MTBL_sop Plasma — KEGG Pathway Enrichment Across Timepoints\n"
                 "(colour = −log₁₀ p-value, grey = not tested/p=1)",
                 fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, label="−log₁₀(p-value)", shrink=0.6)

    # Mark significant cells (q<0.05) with *
    for tp_idx, tp in enumerate(tps):
        df_tp = results[tp]
        sig_pw = set(df_tp[df_tp["q_value"] < 0.05]["pathway_name"])
        for pw_idx, pw in enumerate(top_pathways):
            if pw in sig_pw:
                ax.text(tp_idx, pw_idx, "*", ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold")

    fig.tight_layout()
    out = os.path.join(out_root, "cross_tp_pathway_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Cross-TP heatmap saved: %s", out)


if __name__ == "__main__":
    main()
