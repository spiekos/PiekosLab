"""clean_survey_data.py
=====================
Extract, filter, and clean survey data for the full DP3 survey cohort.

NOTE ON COHORT SCOPE
--------------------
The n=133 tab only covers subjects whose omics data was successfully
processed.  The raw survey files contain ~390 subjects total — many of
whom completed surveys but whose omics samples were not processed (early
loss-to-follow-up, withdrawal, excluded, or simply not yet processed).
Survey analysis does NOT require a processed omics record, so we use the
full 'clinical data' sheet in dp3 master table v2.xlsx as the group-map
source.  Only subjects whose group is one of {Control, FGR, HDP, sPTB}
are retained; subjects labelled SAB, LTFU, Withdraw, or excluded are
dropped.

Input
-----
  data/survey/epds_raw.csv
  data/survey/pss_raw.csv
  data/survey/puqe24_raw.csv
  data/survey/diet_raw.csv
  data/survey/water.csv
  data/dp3 master table v2.xlsx  (sheet: 'clinical data')
    Authoritative source for SubjectID → Group / Subgroup for all
    enrolled subjects (n=437 rows; ~364 with valid analysis groups).

Output
------
  data/survey/cleaned/epds_cleaned.csv
  data/survey/cleaned/pss_cleaned.csv
  data/survey/cleaned/puqe24_cleaned.csv
  data/survey/cleaned/diet_cleaned.csv
  data/survey/cleaned/water_cleaned.csv

Cleaning steps
--------------
  EPDS / PSS / PUQE24
    1. Rename subject-ID column to SubjectID.
    2. Keep the original REDCap event name; also map to visit label where
       possible (A/C/D/PP) — rows whose event name does not map are kept,
       Visit is left as NaN.
    3. Filter to subjects with a valid analysis group.
    4. Attach Group and Subgroup.
    5. Drop unnamed/duplicate columns.

  Diet
    1. Rename subject-ID and event columns.
    2. Keep all rows and all raw frequency-string columns as-is.
       No numeric encoding is applied — diet pattern classification
       (e.g. Mediterranean score) will be handled in a separate step.
    3. Filter to subjects with a valid analysis group.
    4. Attach Group and Subgroup.

  Water
    1. Rename Study.ID → SubjectID.
    2. Filter to subjects with a valid analysis group.
    3. Compute exceed_rate = exceed_count / num_samples for each analyte.
    4. Attach Group and Subgroup.
"""

import os
import re
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE       = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(HERE)
SURVEY_RAW = os.path.join(ROOT, "data", "survey")
SURVEY_OUT = os.path.join(ROOT, "data", "cleaned", "survey")
MASTER_XLSX = os.path.join(ROOT, "data", "dp3 master table v2.xlsx")

# Groups retained for analysis — others (SAB, LTFU, Withdraw, excluded) dropped
VALID_GROUPS = {"Control", "FGR", "HDP", "sPTB"}

os.makedirs(SURVEY_OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# REDCap event → visit label
# ---------------------------------------------------------------------------
EVENT_MAP = {
    "enrollment_513_w_arm_1":   "A",
    "3rd_visit_2028_w_arm_1":   "C",
    "4th_visit_2936_w_arm_1":   "D",
    "postpartum_arm_1":         "PP",
    # Diet sheet uses human-readable names
    "Enrollment 5-13 w":        "A",
    "3rd visit 20-28 w":        "C",
    "4th visit 29-36 w":        "D",
}


# ---------------------------------------------------------------------------
# Group-name normalisation (the metadata has one 'sptb' instead of 'sPTB')
# ---------------------------------------------------------------------------
_GROUP_NORM = {"sptb": "sPTB", "SPTB": "sPTB"}

def _normalise_group(val: str) -> str:
    return _GROUP_NORM.get(str(val).strip(), str(val).strip())


# ---------------------------------------------------------------------------
# Helper: build SubjectID → Group / Subgroup map from the master clinical sheet
# ---------------------------------------------------------------------------
def _build_group_map(master_xlsx: str) -> pd.DataFrame:
    """
    Read 'clinical data' sheet from dp3 master table v2.xlsx.
    This sheet covers all enrolled subjects (n=437), including those who
    completed surveys but whose omics data was never processed.
    Only subjects with a valid analysis group (Control/FGR/HDP/sPTB) are kept.
    """
    meta = pd.read_excel(master_xlsx, sheet_name="clinical data")
    # Drop any header-repeat rows (the sheet sometimes has a duplicate header row)
    meta = meta[meta["ID"] != "ID"].dropna(subset=["ID"])
    meta = meta[["ID", "group", "subgroup"]].copy()
    meta.columns = ["SubjectID", "Group", "Subgroup"]
    meta["SubjectID"] = meta["SubjectID"].astype(str).str.strip()
    meta["Group"]     = meta["Group"].apply(_normalise_group)
    meta["Subgroup"]  = meta["Subgroup"].astype(str).str.strip()
    # Keep only valid analysis groups
    before = len(meta)
    meta = meta[meta["Group"].isin(VALID_GROUPS)]
    logger.info(
        "Group map: %d subjects loaded (%d dropped — SAB/LTFU/Withdraw/excluded).",
        len(meta), before - len(meta),
    )
    meta = meta.drop_duplicates("SubjectID").set_index("SubjectID")
    return meta


# ---------------------------------------------------------------------------
# EPDS / PSS / PUQE24
# ---------------------------------------------------------------------------
def clean_scored_survey(
    raw_path: str,
    out_path: str,
    id_col: str,
    event_col: str,
    group_map: pd.DataFrame,
) -> pd.DataFrame:
    name = os.path.basename(raw_path).replace("_raw.csv", "").upper()
    df = pd.read_csv(raw_path)

    # Rename ID and event columns
    df = df.rename(columns={id_col: "SubjectID", event_col: "redcap_event"})

    # Map known event names → visit labels; unknown events are kept with Visit=NaN
    df["Visit"] = df["redcap_event"].map(EVENT_MAP)
    unmapped = df["Visit"].isna().sum()
    if unmapped:
        logger.info(
            "%s: %d rows have event names not in EVENT_MAP — kept with Visit=NaN.",
            name, unmapped,
        )

    # Filter to cohort
    before = len(df)
    df = df[df["SubjectID"].isin(group_map.index)].copy()
    logger.info(
        "%s: %d → %d rows after filtering to cohort (%d unique subjects).",
        name, before, len(df), df["SubjectID"].nunique(),
    )

    # Attach Group / Subgroup
    df = df.join(group_map[["Group", "Subgroup"]], on="SubjectID")

    # Drop unnamed / duplicate columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    df.to_csv(out_path, index=False)
    logger.info("%s: saved → %s", name, out_path)
    return df


# ---------------------------------------------------------------------------
# Diet
# ---------------------------------------------------------------------------
def clean_diet(raw_path: str, out_path: str, group_map: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning only — raw frequency strings are kept as-is.
    Numeric encoding and diet-pattern scoring (e.g. Mediterranean index)
    will be handled in a separate script once the classification scheme
    is finalised with the PI.
    """
    df = pd.read_csv(raw_path)

    df = df.rename(columns={"Record ID": "SubjectID", "Event Name": "redcap_event"})

    # Map known event names → visit labels; keep all rows regardless
    df["Visit"] = df["redcap_event"].map(EVENT_MAP)
    unmapped = df["Visit"].isna().sum()
    if unmapped:
        logger.info(
            "DIET: %d rows have event names not in EVENT_MAP — kept with Visit=NaN.",
            unmapped,
        )

    before = len(df)
    df = df[df["SubjectID"].isin(group_map.index)].copy()
    logger.info(
        "DIET: %d → %d rows after cohort filter (%d subjects).",
        before, len(df), df["SubjectID"].nunique(),
    )

    df = df.join(group_map[["Group", "Subgroup"]], on="SubjectID")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    df.to_csv(out_path, index=False)
    logger.info("DIET: saved → %s", out_path)
    return df


# ---------------------------------------------------------------------------
# Water
# ---------------------------------------------------------------------------
EXCEED_COLS = ["TTHM_exceed", "CHCl3_exceed", "CHBr3_exceed", "BDCM_exceed", "CDBM_exceed"]
AVG_COLS    = ["TTHM_avg", "Br.THM_avg", "CHCl3_avg", "CHBr3_avg", "BDCM_avg", "CDBM_avg"]

def clean_water(raw_path: str, out_path: str, group_map: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df = df.rename(columns={"Study.ID": "SubjectID"})

    before = len(df)
    df = df[df["SubjectID"].isin(group_map.index)].copy()
    logger.info(
        "WATER: %d → %d rows after cohort filter (%d subjects).",
        before, len(df), df["SubjectID"].nunique(),
    )

    # Compute exceed rates: count / num_samples
    for col in EXCEED_COLS:
        if col in df.columns:
            rate_col = col.replace("_exceed", "_exceed_rate")
            df[rate_col] = df[col] / df["num_samples"]

    df = df.join(group_map[["Group", "Subgroup"]], on="SubjectID")

    df.to_csv(out_path, index=False)
    logger.info("WATER: saved → %s", out_path)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    group_map = _build_group_map(MASTER_XLSX)

    clean_scored_survey(
        raw_path  = os.path.join(SURVEY_RAW, "epds_raw.csv"),
        out_path  = os.path.join(SURVEY_OUT, "epds_cleaned.csv"),
        id_col    = "id",
        event_col = "redcap_event_name",
        group_map = group_map,
    )

    clean_scored_survey(
        raw_path  = os.path.join(SURVEY_RAW, "pss_raw.csv"),
        out_path  = os.path.join(SURVEY_OUT, "pss_cleaned.csv"),
        id_col    = "record_id",
        event_col = "redcap_event_name",
        group_map = group_map,
    )

    clean_scored_survey(
        raw_path  = os.path.join(SURVEY_RAW, "puqe24_raw.csv"),
        out_path  = os.path.join(SURVEY_OUT, "puqe24_cleaned.csv"),
        id_col    = "record_id",
        event_col = "redcap_event_name",
        group_map = group_map,
    )

    clean_diet(
        raw_path  = os.path.join(SURVEY_RAW, "diet_raw.csv"),
        out_path  = os.path.join(SURVEY_OUT, "diet_cleaned.csv"),
        group_map = group_map,
    )

    clean_water(
        raw_path  = os.path.join(SURVEY_RAW, "water.csv"),
        out_path  = os.path.join(SURVEY_OUT, "water_cleaned.csv"),
        group_map = group_map,
    )

    logger.info("All survey cleaning complete.")


if __name__ == "__main__":
    main()
