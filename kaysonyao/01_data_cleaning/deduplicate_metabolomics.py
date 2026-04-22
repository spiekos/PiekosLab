"""
deduplicate_metabolomics.py
===========================
Apply quality-score-based deduplication and compound-name mapping to the
ComBat-corrected plasma metabolomics data, following Kayla Xu's pipeline
(MTBL_datacleaning.py).

The ComBat-corrected output of clean_metabolomics_prenorm_data.py uses raw
Export Order column names (p3823_POS, n1690_NEG, etc.) and contains no
quality-score deduplication.  This script re-applies Kayla's deduplication
logic so named compounds appear once with their proper name, and unnamed
compounds that are cross-mode duplicates of each other are collapsed.

Pipeline (mirrors Kayla Xu MTBL_datacleaning.py steps 1-1, 12, 13)
--------------------------------------------------------------------
1.  Load compound metadata (pos_compounds.csv / neg_compounds.csv) from the
    collaborating lab's data directory.
2.  Calculate a composite quality score (QS) for each compound using Kayla's
    formula:
        QS = Peak Rating (max) + RSD QC Areas % (if available, else 0)
           + MS2 score + min-max scaled Area (max) + annotation confidence
3.  Resolve compound names:
        • Named compounds (Name not NaN / not "Not named") → keep their name.
        • Unnamed compounds → keep Export Order index as name (p1234, n5678).
4.  Same-polarity named deduplication: when the same compound name maps to
    multiple Export Orders within one polarity (POS or NEG), keep the Export
    Order with the highest QS.
5.  Cross-polarity named deduplication: when the same compound name appears
    in both _POS and _NEG, keep the polarity with the larger Area (Max.).
6.  Cross-polarity unnamed deduplication: for each pXXXX (POS) compound,
    search for nXXXX (NEG) compounds with a matching neutral mass (±10 ppm)
    and matching RT (±0.3 min).  The neutral mass is computed as:
        POS [M+H]+:  neutral mass = m/z − 1.00728
        NEG [M-H]−:  neutral mass = m/z + 1.00728
    For matched pairs, keep the compound with the larger Area (Max.); drop
    the other.
7.  Rename surviving named-compound columns from ExportOrder_POL (e.g.,
    p1234_POS) to CompoundName_POL (e.g., Glycine_POS).
8.  Save the deduplicated full matrix and per-timepoint (A–E) slices.

Input
-----
  data/cleaned/metabolomics_combat/normalized_full_results/
      metabolomics_plasma_cleaned_with_metadata.csv
  data/cleaned/metabolomics_combat/normalized_sliced_by_suffix/
      metabolomics_plasma_formatted_suffix_{A-E}.csv
  <kaylaxu_dir>/data/MTBL_plasma/pos_compounds.csv
  <kaylaxu_dir>/data/MTBL_plasma/neg_compounds.csv

Output
------
  data/cleaned/metabolomics_dedup/normalized_full_results/
      metabolomics_plasma_dedup_with_metadata.csv
      dedup_report.csv          – per-compound dedup decision log
  data/cleaned/metabolomics_dedup/normalized_sliced_by_suffix/
      metabolomics_plasma_dedup_suffix_{A-E}.csv

Usage
-----
  # From project root
  python 01_data_cleaning/deduplicate_metabolomics.py

  # Explicit paths
  python 01_data_cleaning/deduplicate_metabolomics.py \\
      --input-full  data/cleaned/metabolomics_combat/normalized_full_results/metabolomics_plasma_cleaned_with_metadata.csv \\
      --input-sliced data/cleaned/metabolomics_combat/normalized_sliced_by_suffix \\
      --kaylaxu-dir <path/to/kaylaxu> \\
      --output-full-dir  data/cleaned/metabolomics_dedup/normalized_full_results \\
      --output-sliced-dir data/cleaned/metabolomics_dedup/normalized_sliced_by_suffix
"""

import argparse
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
META_COLS = {"SubjectID", "Group", "Subgroup", "Batch", "GestAgeDelivery", "SampleGestAge"}

# Tolerances for unnamed cross-mode deduplication
MZ_PPM_TOL = 10.0      # ±10 ppm neutral mass tolerance
RT_MIN_TOL  = 0.30     # ±0.3 min retention time tolerance

# Proton mass for neutral-mass calculation
H_MASS = 1.007276


# ---------------------------------------------------------------------------
# Quality score — adapted from Kayla Xu's qs() in MTBL_datacleaning.py
# ---------------------------------------------------------------------------

def _quality_score(comp: pd.DataFrame) -> pd.Series:
    """
    Calculate composite quality score per compound.

    Score components (all 0–10 scaled):
      peak    : Peak Rating (Max.)
      rsd     : RSD QC Areas [%] (set to 0 if column absent)
      ms2     : MS2 evidence level
      signal  : Area (Max.), min-max scaled
      ac      : Annotation confidence from mzCloud + annotation sources
    """
    # Peak rating
    peak = pd.Series(
        [10 if x >= 7.0 else 7 if x >= 5 else 4 if x >= 3 else 1
         for x in comp["Peak Rating (Max.)"].fillna(0)],
        index=comp.index,
    )

    # RSD QC Areas (gracefully handles missing column)
    try:
        rsd = pd.Series(
            [10 if x < 10 else 8 if x < 15 else 6 if x < 20
             else 4 if x < 25 else 2 if x < 30 else 0
             for x in comp["RSD QC Areas [%]"]],
            index=comp.index,
        )
    except (KeyError, TypeError):
        rsd = pd.Series(0, index=comp.index)

    # MS2 evidence
    ms2 = pd.Series(
        [10 if x == "DDA for preferred ion"
         else 6 if x == "DDA for other ion"
         else 4 if x == "DDA available"
         else 0
         for x in comp["MS2"].fillna("")],
        index=comp.index,
    )

    # Signal intensity (min-max scaled)
    area = comp["Area (Max.)"].fillna(0).values.reshape(-1, 1).astype(float)
    if area.max() > 0:
        scaler = MinMaxScaler(feature_range=(0, 10))
        signal = pd.Series(scaler.fit_transform(area).flatten(), index=comp.index)
    else:
        signal = pd.Series(0.0, index=comp.index)

    # Annotation confidence
    annot_cols = [
        "Annot. Source: Predicted Compositions",
        "Annot. Source: mzCloud Search",
        "Annot. Source: mzVault Search",
        "Annot. Source: Metabolika Search",
        "Annot. Source: ChemSpider Search",
        "Annot. Source: MassList Search",
    ]
    present_annot = [c for c in annot_cols if c in comp.columns]
    if present_annot:
        temp = comp[present_annot]
        full    = (temp == "Full match").sum(axis=1)
        not_top = (temp == "Not the top hit").sum(axis=1)
        partial = (temp == "Partial match").sum(axis=1)
    else:
        full = not_top = partial = pd.Series(0, index=comp.index)

    mzCloud = comp["mzCloud Best Match Confidence"].fillna(-1)  # -1 → no match

    ac = []
    for i in range(len(comp)):
        mz = mzCloud.iloc[i]
        if mz >= 90:
            ac.append(10)
        elif mz >= 80:
            ac.append(9)
        elif mz >= 70:
            ac.append(8)
        elif mz >= 0:          # 0 ≤ mz < 70
            ac.append(0)
        else:                  # mz == -1 (NaN original) → fall through to annotation
            f = full.iloc[i]
            nt = not_top.iloc[i]
            pt = partial.iloc[i]
            if   f  == 6: ac.append(10)
            elif f  == 5: ac.append(9)
            elif f  == 4: ac.append(8)
            elif f  == 3: ac.append(7)
            elif f  == 2: ac.append(6)
            elif f  == 1: ac.append(5)
            elif nt >= 1: ac.append(4)
            elif pt >= 3: ac.append(3)
            elif pt == 2: ac.append(2)
            elif pt == 1: ac.append(1)
            else:         ac.append(0)

    ac_series = pd.Series(ac, index=comp.index)
    return peak + rsd + ms2 + signal + ac_series


# ---------------------------------------------------------------------------
# Build compound look-up table
# ---------------------------------------------------------------------------

def _build_compound_table(
    comp: pd.DataFrame,
    polarity: str,               # "POS" or "NEG"
) -> pd.DataFrame:
    """
    Return a table with one row per Export Order, containing:
        export_order, polarity, name, quality_score, area_max, mz, rt
    """
    out = pd.DataFrame(index=comp.index)
    out["export_order"] = comp.index
    out["polarity"]     = polarity
    out["name"]         = comp["Name"].fillna("").replace("Not named", "")
    out["quality_score"] = _quality_score(comp)
    out["area_max"]      = pd.to_numeric(comp["Area (Max.)"], errors="coerce").fillna(0.0)
    out["mz"]            = pd.to_numeric(comp["m/z"],         errors="coerce")
    out["rt"]            = pd.to_numeric(comp["RT [min]"],    errors="coerce")
    # Blank name → use export order as name (= unannotated)
    mask_blank = out["name"] == ""
    out.loc[mask_blank, "name"] = out.loc[mask_blank, "export_order"]
    out["annotated"] = ~mask_blank
    return out


# ---------------------------------------------------------------------------
# Step 4-5: named compound deduplication
# ---------------------------------------------------------------------------

def _deduplicate_named(
    pos_table: pd.DataFrame,
    neg_table: pd.DataFrame,
) -> tuple[set, set, dict]:
    """
    Returns:
      keep_pos  : set of export_orders to keep in POS
      keep_neg  : set of export_orders to keep in NEG
      rename_map: {old_col_name -> new_col_name}  (ExportOrder_POL → Name_POL)
    """
    keep_pos: set = set()
    keep_neg: set = set()
    rename_map: dict = {}
    drop_log: list = []

    named_pos = pos_table[pos_table["annotated"]].copy()
    named_neg = neg_table[neg_table["annotated"]].copy()

    # All unique names across both polarities
    all_names = set(named_pos["name"]) | set(named_neg["name"])

    for name in all_names:
        rows_pos = named_pos[named_pos["name"] == name]
        rows_neg = named_neg[named_neg["name"] == name]

        # --- Best within POS ---
        best_pos = None
        if len(rows_pos):
            best_pos = rows_pos.loc[rows_pos["quality_score"].idxmax()]
            # Drop lower-quality duplicates within POS
            for eo in rows_pos["export_order"]:
                if eo != best_pos["export_order"]:
                    drop_log.append({
                        "export_order": eo, "polarity": "POS", "name": name,
                        "reason": "same-polarity duplicate (lower QS)",
                    })

        # --- Best within NEG ---
        best_neg = None
        if len(rows_neg):
            best_neg = rows_neg.loc[rows_neg["quality_score"].idxmax()]
            for eo in rows_neg["export_order"]:
                if eo != best_neg["export_order"]:
                    drop_log.append({
                        "export_order": eo, "polarity": "NEG", "name": name,
                        "reason": "same-polarity duplicate (lower QS)",
                    })

        # --- Cross-polarity: keep the higher Area Max ---
        if best_pos is not None and best_neg is not None:
            if best_pos["area_max"] >= best_neg["area_max"]:
                winner, loser = best_pos, best_neg
                win_pol, lose_pol = "POS", "NEG"
            else:
                winner, loser = best_neg, best_pos
                win_pol, lose_pol = "NEG", "POS"

            drop_log.append({
                "export_order": loser["export_order"],
                "polarity": lose_pol, "name": name,
                "reason": "cross-polarity duplicate (lower Area Max)",
            })
            col_in  = f"{winner['export_order']}_{win_pol}"
            col_out = f"{name}_{win_pol}"
            rename_map[col_in] = col_out
            if win_pol == "POS":
                keep_pos.add(winner["export_order"])
            else:
                keep_neg.add(winner["export_order"])

        elif best_pos is not None:
            col_in  = f"{best_pos['export_order']}_POS"
            col_out = f"{name}_POS"
            rename_map[col_in] = col_out
            keep_pos.add(best_pos["export_order"])
        elif best_neg is not None:
            col_in  = f"{best_neg['export_order']}_NEG"
            col_out = f"{name}_NEG"
            rename_map[col_in] = col_out
            keep_neg.add(best_neg["export_order"])

    return keep_pos, keep_neg, rename_map, drop_log


# ---------------------------------------------------------------------------
# Step 6: unnamed cross-mode deduplication
# ---------------------------------------------------------------------------

def _deduplicate_unnamed_cross_mode(
    pos_table: pd.DataFrame,
    neg_table: pd.DataFrame,
) -> tuple[set, set, list]:
    """
    For unnamed compounds, match POS to NEG by neutral mass (±MZ_PPM_TOL ppm)
    and retention time (±RT_MIN_TOL min).  For each match, drop the compound
    with the smaller Area (Max.).

    Returns:
      unnamed_keep_pos : set of POS export_orders to keep
      unnamed_keep_neg : set of NEG export_orders to keep
      drop_log_unnamed : list of dropped compounds
    """
    unann_pos = pos_table[~pos_table["annotated"]].dropna(subset=["mz", "rt"]).copy()
    unann_neg = neg_table[~neg_table["annotated"]].dropna(subset=["mz", "rt"]).copy()

    # Neutral mass: POS [M+H]+ → subtract H; NEG [M-H]- → add H
    unann_pos["neutral_mass"] = unann_pos["mz"] - H_MASS
    unann_neg["neutral_mass"] = unann_neg["mz"] + H_MASS

    # Start with all unnamed compounds kept; subtract matched losers
    keep_pos = set(unann_pos["export_order"])
    keep_neg = set(unann_neg["export_order"])
    drop_log: list = []

    neg_masses = unann_neg["neutral_mass"].values
    neg_rts    = unann_neg["rt"].values
    neg_areas  = unann_neg["area_max"].values
    neg_orders = unann_neg["export_order"].values

    for _, pos_row in unann_pos.iterrows():
        if pos_row["export_order"] not in keep_pos:
            continue   # already dropped as loser of a previous match

        pm  = pos_row["neutral_mass"]
        prt = pos_row["rt"]
        pa  = pos_row["area_max"]
        peo = pos_row["export_order"]

        # Mass tolerance in ppm
        ppm_diff = np.abs((neg_masses - pm) / (pm + 1e-12)) * 1e6
        rt_diff  = np.abs(neg_rts - prt)
        match    = (ppm_diff <= MZ_PPM_TOL) & (rt_diff <= RT_MIN_TOL)

        for j in np.where(match)[0]:
            neo = neg_orders[j]
            if neo not in keep_neg:
                continue  # already dropped

            na = neg_areas[j]
            if pa >= na:
                # POS wins — drop NEG
                keep_neg.discard(neo)
                drop_log.append({
                    "export_order": neo, "polarity": "NEG",
                    "name": neo, "reason": "cross-mode unnamed duplicate (lower Area Max)",
                })
            else:
                # NEG wins — drop POS
                keep_pos.discard(peo)
                drop_log.append({
                    "export_order": peo, "polarity": "POS",
                    "name": peo, "reason": "cross-mode unnamed duplicate (lower Area Max)",
                })
                break   # POS compound dropped; stop searching for more NEG matches

    logger.info(
        "Unnamed cross-mode dedup: %d POS + %d NEG kept "
        "(from %d POS + %d NEG unnamed compounds; %d pairs collapsed).",
        len(keep_pos), len(keep_neg),
        len(unann_pos), len(unann_neg), len(drop_log),
    )
    return keep_pos, keep_neg, drop_log


# ---------------------------------------------------------------------------
# Main deduplication pipeline
# ---------------------------------------------------------------------------

def deduplicate(
    input_full_csv: str,
    input_sliced_dir: str,
    kaylaxu_dir: str,
    output_full_dir: str,
    output_sliced_dir: str,
) -> None:

    os.makedirs(output_full_dir,   exist_ok=True)
    os.makedirs(output_sliced_dir, exist_ok=True)

    # --- Load compound metadata ---
    pos_comp_path = os.path.join(kaylaxu_dir, "data", "MTBL_plasma", "pos_compounds.csv")
    neg_comp_path = os.path.join(kaylaxu_dir, "data", "MTBL_plasma", "neg_compounds.csv")
    logger.info("Loading compound metadata …")
    pos_comp = pd.read_csv(pos_comp_path, index_col=0)
    neg_comp = pd.read_csv(neg_comp_path, index_col=0)
    logger.info(
        "Compound metadata: POS=%d entries, NEG=%d entries", len(pos_comp), len(neg_comp)
    )

    # --- Build per-compound tables with quality scores ---
    logger.info("Computing quality scores …")
    pos_table = _build_compound_table(pos_comp, "POS")
    neg_table = _build_compound_table(neg_comp, "NEG")

    n_named_pos = pos_table["annotated"].sum()
    n_named_neg = neg_table["annotated"].sum()
    logger.info(
        "Annotated compounds: POS=%d / %d, NEG=%d / %d",
        n_named_pos, len(pos_table), n_named_neg, len(neg_table),
    )

    # --- Step 4-5: Named compound deduplication ---
    logger.info("Running named-compound deduplication …")
    named_keep_pos, named_keep_neg, rename_map, named_drop_log = _deduplicate_named(
        pos_table, neg_table
    )
    logger.info(
        "Named compounds to keep: %d POS, %d NEG (%d renamed, %d dropped)",
        len(named_keep_pos), len(named_keep_neg),
        len(rename_map), len(named_drop_log),
    )

    # --- Step 6: Unnamed cross-mode deduplication ---
    logger.info("Running unnamed cross-mode deduplication …")
    unnamed_keep_pos, unnamed_keep_neg, unnamed_drop_log = _deduplicate_unnamed_cross_mode(
        pos_table, neg_table
    )

    # Combine all decisions into a set of columns to KEEP
    final_keep_pos = named_keep_pos | unnamed_keep_pos
    final_keep_neg = named_keep_neg | unnamed_keep_neg

    all_drop_log = named_drop_log + unnamed_drop_log

    # --- Load the cleaned data ---
    logger.info("Loading ComBat-corrected data from %s …", input_full_csv)
    df_full = pd.read_csv(input_full_csv, index_col=0)

    # Identify analyte columns present in the data
    meta_present = [c for c in df_full.columns if c in META_COLS]
    analyte_present = [c for c in df_full.columns if c not in META_COLS]

    logger.info(
        "Input: %d samples × %d analytes + %d metadata columns",
        len(df_full), len(analyte_present), len(meta_present),
    )

    # --- Apply keep filter ---
    def _should_keep(col_name: str) -> bool:
        """Return True if this analyte column survives deduplication."""
        if col_name.endswith("_POS"):
            eo = col_name[:-4]   # strip "_POS"
            return eo in final_keep_pos
        elif col_name.endswith("_NEG"):
            eo = col_name[:-4]   # strip "_NEG"
            return eo in final_keep_neg
        return True   # non-Export-Order columns (should not exist, but keep safe)

    kept_analytes = [c for c in analyte_present if _should_keep(c)]
    dropped_analytes = [c for c in analyte_present if not _should_keep(c)]

    logger.info(
        "Analytes after deduplication: %d kept, %d dropped (from %d).",
        len(kept_analytes), len(dropped_analytes), len(analyte_present),
    )

    # --- Apply rename map ---
    # Only rename analyte columns that are in the full dataset
    effective_rename = {k: v for k, v in rename_map.items() if k in kept_analytes}
    logger.info("Renaming %d named compound columns (ExportOrder → CompoundName).", len(effective_rename))

    # --- Build output DataFrame ---
    df_out = df_full[meta_present + kept_analytes].copy()
    df_out = df_out.rename(columns=effective_rename)

    # --- Save dedup report ---
    report_rows = []
    for col in analyte_present:
        kept = _should_keep(col)
        new_name = effective_rename.get(col, col) if kept else None
        report_rows.append({
            "original_column": col,
            "kept": kept,
            "renamed_to": new_name,
        })
    report_df = pd.DataFrame(report_rows)
    if all_drop_log:
        drop_df = pd.DataFrame(all_drop_log)
        report_df = report_df.merge(
            drop_df.rename(columns={"export_order": "_eo"}),
            left_on="original_column", right_on="_eo", how="left"
        ).drop(columns=["_eo"], errors="ignore")
    report_path = os.path.join(output_full_dir, "dedup_report.csv")
    report_df.to_csv(report_path, index=False)
    logger.info("Dedup report → %s", report_path)

    # --- Save full matrix ---
    full_out = os.path.join(output_full_dir, "metabolomics_plasma_dedup_with_metadata.csv")
    df_out.to_csv(full_out)
    logger.info(
        "Full matrix saved → %s  (%d samples × %d cols)",
        full_out, df_out.shape[0], df_out.shape[1],
    )

    # --- Slice by timepoint and save ---
    logger.info("Slicing deduplicated data by timepoint …")
    for tp in "ABCDE":
        in_slice = os.path.join(input_sliced_dir, f"metabolomics_plasma_formatted_suffix_{tp}.csv")
        if not os.path.exists(in_slice):
            logger.warning("Slice not found, skipping: %s", in_slice)
            continue
        df_slice = pd.read_csv(in_slice, index_col=0)

        slice_analytes = [c for c in df_slice.columns if c not in META_COLS and _should_keep(c)]
        meta_slice = [c for c in df_slice.columns if c in META_COLS]
        df_slice_out = df_slice[meta_slice + slice_analytes].copy()
        df_slice_out = df_slice_out.rename(columns=effective_rename)

        out_path = os.path.join(output_sliced_dir, f"metabolomics_plasma_dedup_suffix_{tp}.csv")
        df_slice_out.to_csv(out_path)
        logger.info("  Timepoint %s: %d samples → %s", tp, len(df_slice_out), out_path)

    # --- Summary ---
    named_final = sum(
        1 for c in df_out.columns
        if c not in META_COLS
        and not re.match(r'^[pncC]\d+_(POS|NEG)$', c)
    )
    unnamed_final = sum(
        1 for c in df_out.columns
        if c not in META_COLS
        and re.match(r'^[pncC]\d+_(POS|NEG)$', c)
    )
    logger.info(
        "Final feature count — named (annotated): %d | unnamed: %d | total: %d",
        named_final, unnamed_final, named_final + unnamed_final,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    # Try to auto-detect kaylaxu directory as sibling of project root
    kaylaxu_default = os.path.join(os.path.dirname(root), "kaylaxu")

    p = argparse.ArgumentParser(
        description="Deduplicate ComBat-corrected metabolomics data following Kayla Xu's pipeline."
    )
    p.add_argument(
        "--input-full",
        default=os.path.join(
            root, "data", "cleaned", "metabolomics_combat",
            "normalized_full_results", "metabolomics_plasma_cleaned_with_metadata.csv"
        ),
        help="Path to the full ComBat-corrected metabolomics CSV.",
    )
    p.add_argument(
        "--input-sliced",
        default=os.path.join(
            root, "data", "cleaned", "metabolomics_combat",
            "normalized_sliced_by_suffix",
        ),
        help="Directory containing per-timepoint ComBat-corrected CSVs.",
    )
    p.add_argument(
        "--kaylaxu-dir",
        default=kaylaxu_default,
        help="Path to Kayla Xu's project directory (contains data/MTBL_plasma/pos_compounds.csv).",
    )
    p.add_argument(
        "--output-full-dir",
        default=os.path.join(
            root, "data", "cleaned", "metabolomics_dedup", "normalized_full_results"
        ),
    )
    p.add_argument(
        "--output-sliced-dir",
        default=os.path.join(
            root, "data", "cleaned", "metabolomics_dedup", "normalized_sliced_by_suffix"
        ),
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    deduplicate(
        input_full_csv=args.input_full,
        input_sliced_dir=args.input_sliced,
        kaylaxu_dir=args.kaylaxu_dir,
        output_full_dir=args.output_full_dir,
        output_sliced_dir=args.output_sliced_dir,
    )
