"""Create the single locked participant-level 70/30 development / test split.

Why one split, written once
---------------------------
Aim F.2 requires a development set (70%) and a held-out test set (30%) that is
"locked away and used only once", with splitting done at the participant level
so no individual's samples cross the boundary. Two properties follow that the
previous per-model `train_test_split` calls did not provide:

1. **Participant-level.** Assignment is by `SubjectID`, never by `SampleID`, so
   every sample from one participant lands in the same set.
2. **Shared across every base model.** One assignment is written to disk and
   read by all datasets, windows, and modalities. Previously each per-timepoint
   model drew its own independent split: among subjects present in all five
   timepoint models, 20 of 27 received *different* assignments. That does not
   corrupt any single base model, but it contaminates the Aim 3B meta-model,
   whose inputs are base-model predictions - a participant's risk score from a
   base model that trained on them would feed the meta-model's held-out test.

Stratification
--------------
Stratified on the primary outcome ("any complication": HDP, FGR, or sPTB vs
Control) so both sets preserve the outcome balance. Syndrome is recorded in the
output for auditing but is not stratified on - at 133 participants, stratifying
on 4 levels leaves cells too small to split stably.

Output
------
`data/cleaned/locked_split.csv` with columns:
    SubjectID, split ("dev"|"test"), Group, any_complication
"""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

COMPLICATIONS = ["HDP", "FGR", "sPTB"]
TEST_FRACTION = 0.30
RANDOM_STATE = 42

# Subject roster is taken from the master table so the split covers every
# participant, not only those with a given assay.
ROSTER = "data/dp3 master table v2.xlsx"
ROSTER_SHEET = "variables of interest"


def build_roster(repo_root: str) -> pd.DataFrame:
    path = os.path.join(repo_root, ROSTER)
    df = pd.read_excel(path, sheet_name=ROSTER_SHEET)
    df = df[["ID", "group"]].rename(columns={"ID": "SubjectID", "group": "Group"})
    df = df.dropna(subset=["SubjectID"])

    # Normalise casing ("sptb" -> "sPTB") before matching.
    canon = {"sptb": "sPTB", "hdp": "HDP", "fgr": "FGR", "control": "Control"}
    df["Group"] = df["Group"].astype(str).str.strip()
    df["Group"] = df["Group"].apply(lambda g: canon.get(g.lower(), g))

    keep = ["Control"] + COMPLICATIONS
    dropped = df[~df["Group"].isin(keep)]["Group"].value_counts().to_dict()
    if dropped:
        logger.info("Excluding non-outcome statuses: %s", dropped)
    df = df[df["Group"].isin(keep)].drop_duplicates(subset=["SubjectID"])

    df["any_complication"] = df["Group"].isin(COMPLICATIONS).astype(int)
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the locked participant-level 70/30 split.")
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing locked split. Refuses by default: the test "
             "set is meant to be fixed, and redrawing it after seeing results "
             "is how a held-out set stops being held out.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.repo_root)
    out_path = os.path.join(root, "data", "cleaned", "locked_split.csv")

    if os.path.exists(out_path) and not args.force:
        logger.error(
            "Locked split already exists at %s. Refusing to overwrite - pass --force "
            "only if you genuinely intend to void the held-out test set.", out_path
        )
        raise SystemExit(1)

    roster = build_roster(root)
    logger.info("Roster: %d participants | %s", len(roster), roster["Group"].value_counts().to_dict())

    dev, test = train_test_split(
        roster,
        test_size=args.test_fraction,
        random_state=args.random_state,
        stratify=roster["any_complication"],
    )
    dev = dev.assign(split="dev")
    test = test.assign(split="test")
    out = pd.concat([dev, test], ignore_index=True).sort_values("SubjectID")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out[["SubjectID", "split", "Group", "any_complication"]].to_csv(out_path, index=False)

    for name, part in (("dev", dev), ("test", test)):
        n = len(part)
        pos = int(part["any_complication"].sum())
        logger.info(
            "%-4s: %3d participants | any-complication %3d (%.1f%%) | %s",
            name, n, pos, 100 * pos / n, part["Group"].value_counts().to_dict(),
        )
    logger.info("Locked split saved -> %s", out_path)


if __name__ == "__main__":
    main()
