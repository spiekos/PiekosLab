"""Slice plasma data into the project's five visit timepoints (A-E).

These are the timepoints the study has always used, keyed on the SampleID
suffix rather than on gestational age. They are *relative* windows: visit B
spans 13.6-33.6 gestational weeks, so two participants labelled B can be
20 weeks apart. That is a known property, not an error - see
`bin_echo_windows.py` for the gestational-age binnings used where absolute
timing matters.

Late-enrolment series
---------------------
SampleIDs carry ten distinct suffixes: A-E and EA-EE. The E-prefixed codes are
the later-enrolment series and map onto the base visit (EA->A, EB->B, ...).
Merging reproduces the participant counts of the original `_suffix_A..E`
files exactly:

    A+EA = 133, B+EB = 133, C+EC = 128, D+ED = 109, E+EE = 27

Failing to merge would silently drop 23 participants from A and B.

Each participant contributes at most one sample per visit, so no
midpoint-selection rule is needed here - unlike the gestational-age binnings,
where re-binning creates within-window duplicates.
"""

from __future__ import annotations

import argparse
import logging
import os
import re

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

VISITS = ["A", "B", "C", "D", "E"]
SUFFIX_RE = re.compile(r"(E?[A-E])$")


def base_visit(sample_id: str) -> str | None:
    """DP3-0005A -> 'A'; DP3-0140EA -> 'A' (late-enrolment series)."""
    m = SUFFIX_RE.search(str(sample_id).strip())
    if not m:
        return None
    s = m.group(1)
    return s[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Slice plasma data into visit timepoints A-E.")
    ap.add_argument("--repo-root", default=os.getcwd())
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--datasets", nargs="+", default=["MTBL_plasma", "LIPD_plasma"])
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    root = os.path.abspath(args.repo_root)
    out_root = os.path.join(root, args.output_root)
    os.makedirs(out_root, exist_ok=True)

    log_rows = []
    for ds in args.datasets:
        path = os.path.join(root, args.source_root, ds, f"{ds}_cleaned_with_metadata.csv")
        if not os.path.exists(path):
            logger.warning("%s: not found (%s)", ds, path)
            continue
        df = pd.read_csv(path, low_memory=False)
        df["_visit"] = df["SampleID"].map(base_visit)
        unmapped = int(df["_visit"].isna().sum())
        if unmapped:
            logger.warning("%s: %d sample(s) had no A-E suffix and were dropped", ds, unmapped)

        out_dir = os.path.join(out_root, ds)
        os.makedirs(out_dir, exist_ok=True)
        for v in VISITS:
            sub = df[df["_visit"] == v]
            if sub.empty:
                logger.warning("%s / %s: no samples", ds, v)
                continue
            dup = len(sub) - sub["SubjectID"].nunique()
            if dup:
                logger.warning("%s / %s: %d duplicate participant row(s) - unexpected for "
                               "visit slicing; investigate before modelling", ds, v, dup)
            ga = pd.to_numeric(sub.get("SampleGestAge"), errors="coerce")
            logger.info("%s / %s: %d samples, %d participants, GA %.1f-%.1f wks",
                        ds, v, len(sub), sub["SubjectID"].nunique(), ga.min(), ga.max())
            out = sub.drop(columns=["_visit"])
            out.to_csv(os.path.join(out_dir, f"{ds}_echo_{v}.csv"), index=False)
            log_rows.append({"dataset": ds, "visit": v, "n_samples": len(sub),
                             "n_participants": sub["SubjectID"].nunique(),
                             "ga_min": ga.min(), "ga_max": ga.max()})

    if log_rows:
        pd.DataFrame(log_rows).to_csv(os.path.join(out_root, "visit_slicing_log.csv"), index=False)
    logger.info("Visit slicing complete -> %s", out_root)


if __name__ == "__main__":
    main()
