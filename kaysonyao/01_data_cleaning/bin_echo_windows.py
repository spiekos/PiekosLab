"""Re-bin plasma and urine samples into gestational windows.

Background
----------
Visit letters A-E are *relative* windows, not fixed timepoints: two participants
both labelled "B" can be 20 gestational weeks apart (observed B range
13.6-33.6 wks). Modelling them as one timepoint is not meaningful, so samples
are re-binned on actual `SampleGestAge`.

Two schemes, selected with --scheme (see WINDOW_SCHEMES):

    echo2  W1_early 6.0-19.9 wks (mid 13.0), W2_third >=28.0 wks (mid 35.0)
           Matches the Penn-CHOP ECHO cohort so model (3) in Aim F.2 can be
           externally validated. Samples in 20.0-27.9 wks fall outside both
           windows and are dropped from this view (~22% of samples, though no
           participant is lost entirely). They remain in the source files.

    dp3_5t T1 6-14, T2 14-22, T3 22-32, T4 32-37, T5 37-42 wks
           The project's five currently defined timepoints. Uses every sample
           and gives five ordered timepoints for F.4's rolling CV and F.6's
           "stable across gestational windows" check.

The grant text defines no window boundaries outside the ECHO pair.

Selection rule (per the PI)
---------------------------
One sample per participant per window. Where a participant has several samples
in a window, keep the one closest to the window midpoint. Ties break toward the
earlier draw, which is the more clinically useful direction for an early-risk
model. Re-binning is what creates this choice: under visit-letter slicing each
participant already had exactly one sample per letter.

Plasma and urine (both longitudinal, with SampleGestAge) can be binned; placenta
is excluded - it is collected once at delivery and carries no SampleGestAge.

Outputs
-------
For each dataset, per window:
    <out>/<dataset>/<dataset>_echo_<window>.csv
plus a single `echo_binning_log.csv` recording every sample, its window, its
distance from the midpoint, and whether it was kept or dropped.
"""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# (name, lower_inclusive, upper_exclusive, midpoint)
# Upper bound 42.0 is post-term; used only to define a midpoint.
#
# echo2  - matches the Penn-CHOP ECHO cohort, required for the externally
#          validated arm (model 3 in F.2). Discards 20.0-27.9 wks, which ECHO
#          does not sample: ~22% of DP3 samples, though no participant is lost
#          entirely.
# dp3_5t - the project's five defined timepoints. Uses every sample and gives
#          five ordered timepoints for F.4's rolling CV and F.6's "stable across
#          gestational windows" check. For models 1 and 2, which carry no ECHO
#          matching constraint.
WINDOW_SCHEMES = {
    "echo2": [
        ("W1_early", 6.0, 20.0, 13.0),
        ("W2_third", 28.0, 42.0, 35.0),
    ],
    # The project's five defined timepoints. Boundaries read off the T1-T5 files
    # supplied 2026-08-12; reproduced here so the midpoint rule can be applied,
    # which those files do not do (46 duplicate participant-rows across T1-T5 in
    # MTBL plasma alone).
    "dp3_5t": [
        ("T1", 6.0, 14.0, 10.0),
        ("T2", 14.0, 22.0, 18.0),
        ("T3", 22.0, 32.0, 27.0),
        ("T4", 32.0, 37.0, 34.5),
        ("T5", 37.0, 42.0, 39.5),
    ],
}

# Active scheme; overridden by --scheme.
ECHO_WINDOWS = WINDOW_SCHEMES["echo2"]

GA_COL = "SampleGestAge"
SUBJ_COL = "SubjectID"
SAMPLE_COL = "SampleID"

DATASETS = {
    "MTBL_plasma": "data/processed/MTBL/plasma/MTBL_plasma_cleaned_with_metadata.csv",
    "LIPD_plasma": "data/processed/LIPD/plasma/LIPD_plasma_cleaned_with_metadata.csv",
    "MTBL_urine": "data/processed/MTBL/urine/MTBL_urine_cleaned_with_metadata.csv",
    "proteomics_plasma": "data/processed/proteomics/proteomics_plasma_cleaned_with_metadata.csv",
}


def assign_window(ga: float) -> str | None:
    """Return the ECHO window label for a gestational age, or None if outside."""
    if pd.isna(ga):
        return None
    for name, lo, hi, _ in ECHO_WINDOWS:
        if lo <= ga < hi:
            return name
    return None


def midpoint_of(window: str) -> float:
    return next(mid for name, _, _, mid in ECHO_WINDOWS if name == window)


def bin_dataset(name: str, path: str, out_root: str) -> pd.DataFrame:
    """Bin one dataset and write per-window CSVs. Returns the audit log."""
    df = pd.read_csv(path, low_memory=False)
    for col in (GA_COL, SUBJ_COL, SAMPLE_COL):
        if col not in df.columns:
            raise ValueError(f"{name}: missing required column '{col}'")

    df = df.copy()
    df["_ga"] = pd.to_numeric(df[GA_COL], errors="coerce")
    df["_window"] = df["_ga"].map(assign_window)

    log_rows = df[[SAMPLE_COL, SUBJ_COL, "_ga", "_window"]].copy()
    log_rows["dataset"] = name
    log_rows["dist_to_midpoint"] = [
        abs(ga - midpoint_of(w)) if w else pd.NA
        for ga, w in zip(log_rows["_ga"], log_rows["_window"])
    ]
    log_rows["kept"] = False

    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    for window, lo, hi, mid in ECHO_WINDOWS:
        sub = df[df["_window"] == window].copy()
        if sub.empty:
            logger.warning("%s / %s: no samples in window", name, window)
            continue

        sub["_dist"] = (sub["_ga"] - mid).abs()
        # Ties break toward the earlier draw: sort by distance, then by GA.
        sub = sub.sort_values(["_dist", "_ga"], kind="mergesort")
        kept = sub.drop_duplicates(subset=[SUBJ_COL], keep="first")

        n_before, n_after = len(sub), len(kept)
        multi = sub.groupby(SUBJ_COL).size()
        logger.info(
            "%s / %s [%.1f-%.1f wks, mid %.1f]: %d samples -> %d participants "
            "(%d dropped; %d participants had >1 sample, max %d)",
            name, window, lo, hi, mid, n_before, n_after, n_before - n_after,
            int((multi > 1).sum()), int(multi.max()),
        )

        log_rows.loc[log_rows[SAMPLE_COL].isin(kept[SAMPLE_COL]), "kept"] = True

        out = kept.drop(columns=["_ga", "_window", "_dist"])
        out_path = os.path.join(out_dir, f"{name}_echo_{window}.csv")
        out.to_csv(out_path, index=False)
        logger.info("    saved -> %s", out_path)

    outside = int(log_rows["_window"].isna().sum())
    if outside:
        logger.info(
            "%s: %d sample(s) fell outside every window in this scheme and were dropped.",
            name, outside,
        )
    return log_rows.rename(columns={"_ga": "gest_age_wks", "_window": "echo_window"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-bin plasma and urine samples into gestational windows.")
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument(
        "--scheme", choices=sorted(WINDOW_SCHEMES), default="echo2",
        help="echo2 = the two Penn-CHOP ECHO windows; dp3_5t = the project's "
             "five defined timepoints (T1-T5), which use every sample.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["MTBL_plasma", "LIPD_plasma"],
        help="Datasets to bin. With --source-root: folder names under it. "
             "Without: keys of DATASETS (MTBL_plasma, LIPD_plasma, MTBL_urine, proteomics_plasma).",
    )
    parser.add_argument(
        "--source-root", default=None,
        help="sop_omics_pipeline.py --output-root to bin (e.g. data/processed_devfit); files "
             "are read from <root>/<MTBL|LIPD>/<tissue>/<dataset>_cleaned_with_metadata.csv. "
             "Default: the paths in DATASETS (data/processed/...).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Default: <repo-root>/data/processed/windows",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.repo_root)

    global ECHO_WINDOWS
    ECHO_WINDOWS = WINDOW_SCHEMES[args.scheme]
    logger.info("Window scheme '%s': %s", args.scheme,
                ", ".join(f"{n} [{lo}-{hi})" for n, lo, hi, _ in ECHO_WINDOWS))

    # Only bin what was asked for. Previously the no --source-root branch binned
    # every entry in DATASETS regardless of --datasets, which silently included
    # the label-aware MTBL/LIPD outputs whenever proteomics was requested.
    datasets = {k: v for k, v in DATASETS.items() if k in args.datasets}
    if args.source_root:
        datasets = {
            name: os.path.join(args.source_root, *name.split("_", 1),
                               f"{name}_cleaned_with_metadata.csv")
            for name in args.datasets
        }

    out_root = args.output_root or os.path.join(root, "data", "processed", "windows")
    os.makedirs(out_root, exist_ok=True)

    logs = []
    for name, rel in datasets.items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            logger.warning("%s: source not found, skipping (%s)", name, path)
            continue
        logger.info("=== %s ===", name)
        logs.append(bin_dataset(name, path, out_root))

    if logs:
        log = pd.concat(logs, ignore_index=True)
        log_path = os.path.join(out_root, "echo_binning_log.csv")
        log.to_csv(log_path, index=False)
        logger.info("Binning log saved -> %s", log_path)
    logger.info("ECHO window binning complete.")


if __name__ == "__main__":
    main()
