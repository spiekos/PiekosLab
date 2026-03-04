import logging
import os
import re
import argparse
import pandas as pd

logger = logging.getLogger(__name__)

_POSTNATAL_TO_PRENATAL_MAP: dict[str, str] = {
    "EA": "A",
    "EB": "B",
    "EC": "C",
    "ED": "D",
    "EE": "E",
}

# Metadata columns carried through the wide matrix (non-analyte).
_META_COLS: frozenset[str] = frozenset(
    {"Group", "Subgroup", "GestAgeDelivery", "SampleGestAge"}
)


def remove_internal_whitespace(s: pd.Series) -> pd.Series:
    """Remove all whitespace characters from a string series."""
    return s.astype(str).str.replace(r"\s+", "", regex=True)


def extract_suffix(sample_id: str, subject_id: str) -> str:
    """
    Extract longitudinal suffix by comparing SampleID and SubjectID.

    Expected patterns include:
    - SampleID='DP3-0005A', SubjectID='DP3-0005'  -> 'A'
    - SampleID='DP3-0005AB', SubjectID='DP3-0005' -> 'AB'
    """
    if sample_id.startswith(subject_id):
        return sample_id[len(subject_id):]

    # Fallback: trailing letters, if prefix is not cleanly aligned.
    # Log a warning so ID formatting issues are visible in the run log.
    m = re.search(r"([A-Za-z]+)$", sample_id)
    if m:
        logger.warning(
            "SampleID %r does not start with SubjectID %r; "
            "falling back to trailing-letter extraction (got %r). "
            "Check for ID formatting mismatches.",
            sample_id, subject_id, m.group(1),
        )
        return m.group(1)
    return ""


def format_proteomics_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Format longitudinal plasma proteomics output into per-suffix dataframes.

    Steps:
    1) Remove all internal whitespace in SampleID and SubjectID.
    2) Derive longitudinal suffix from SampleID vs SubjectID (stored as OriginalLabel).
    3) Warn and exclude rows with no recognizable longitudinal suffix.
    4) Merge postnatal suffixes into their prenatal equivalents using
       _POSTNATAL_TO_PRENATAL_MAP (EA→A, EB→B, EC→C, ED→D, EE→E).
       The merged label is stored as LongitudinalLabel.
    5) Drop Batch column.
    6) Split rows by LongitudinalLabel.
    7) For each split dataframe:
       - Strip the ORIGINAL suffix from SampleID to restore the base subject ID.
         (Crucial: stripping must use the original "EA"/"A" suffix, not the
          remapped label, so "DP3-0005EA" → "DP3-0005", not "DP3-0005E".)
       - Drop SubjectID, OriginalLabel, LongitudinalLabel.
       - If a subject contributed both a prenatal and a postnatal sample that
         land in the same merged bucket, aggregate duplicate SampleIDs:
         analyte columns → median, metadata columns → first value.

    Returns:
        dict mapping merged suffix label (e.g. "A") -> dataframe
    """
    required = {"SampleID", "SubjectID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["SampleID"] = remove_internal_whitespace(out["SampleID"])
    out["SubjectID"] = remove_internal_whitespace(out["SubjectID"])

    # Step 2: extract original suffix (e.g. "A", "EA").
    out["OriginalLabel"] = [
        extract_suffix(sid, subid) for sid, subid in zip(out["SampleID"], out["SubjectID"])
    ]

    # Step 3: drop rows with no suffix.
    no_suffix = out[out["OriginalLabel"] == ""]
    if not no_suffix.empty:
        logger.warning(
            "%d sample(s) have no longitudinal suffix and will be excluded: %s",
            len(no_suffix),
            no_suffix["SampleID"].tolist(),
        )
    out = out[out["OriginalLabel"] != ""].copy()

    # Step 4: merge postnatal → prenatal.
    out["LongitudinalLabel"] = out["OriginalLabel"].replace(_POSTNATAL_TO_PRENATAL_MAP)
    n_merged = (out["OriginalLabel"] != out["LongitudinalLabel"]).sum()
    if n_merged > 0:
        logger.info(
            "Merged %d postnatal sample(s) (EA–EE → A–E) into prenatal equivalents.",
            n_merged,
        )

    # Step 5: drop Batch.
    if "Batch" in out.columns:
        out = out.drop(columns=["Batch"])

    grouped: dict[str, pd.DataFrame] = {}
    for suffix, g in out.groupby("LongitudinalLabel", sort=True):
        g2 = g.copy()

        # Step 7a: strip the ORIGINAL suffix from each SampleID individually.
        # This correctly handles mixed groups (e.g. "A" and "EA" → both become base ID).
        g2["SampleID"] = [
            sid[: -len(orig)] if orig and sid.endswith(orig) else sid
            for sid, orig in zip(g2["SampleID"], g2["OriginalLabel"])
        ]
        g2 = g2.drop(columns=["SubjectID", "LongitudinalLabel", "OriginalLabel"])

        # Step 7b: aggregate duplicates caused by pre+postnatal merge.
        dup_mask = g2["SampleID"].duplicated(keep=False)
        if dup_mask.any():
            dup_ids = g2.loc[dup_mask, "SampleID"].unique().tolist()
            logger.warning(
                "Suffix '%s': %d subject(s) have both prenatal and postnatal samples "
                "after merge; aggregating analytes by median, metadata by first: %s",
                suffix, len(dup_ids), dup_ids,
            )
            present_meta = [c for c in g2.columns if c != "SampleID" and c in _META_COLS]
            analyte_cols = [c for c in g2.columns if c != "SampleID" and c not in _META_COLS]
            agg_dict: dict[str, str] = {c: "first" for c in present_meta}
            agg_dict.update({c: "median" for c in analyte_cols})
            g2 = g2.groupby("SampleID", sort=False).agg(agg_dict).reset_index()

        grouped[suffix] = g2

    return grouped


def save_sliced_outputs(sliced: dict[str, pd.DataFrame], output_dir: str, base_name: str) -> list[str]:
    """Save per-suffix dataframes to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for suffix, sdf in sliced.items():
        suffix_norm = suffix.upper()
        out_path = os.path.join(output_dir, f"{base_name}_suffix_{suffix_norm}.csv")
        sdf.to_csv(out_path, index=False)
        written.append(out_path)
    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Format plasma proteomics output into per-suffix files.")
    p.add_argument(
        "--input-csv",
        default=None,
        help="Path to final plasma output CSV. Defaults to normalized_full_results output.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for per-suffix output files. Defaults under normalized_full_results.",
    )
    p.add_argument(
        "--base-name",
        default="proteomics_plasma_formatted",
        help="Base output filename prefix (default: proteomics_plasma_formatted).",
    )
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    wkdir = os.getcwd()
    input_csv = args.input_csv or os.path.join(
        wkdir,
        "data",
        "cleaned",
        "proteomics",
        "normalized_full_results",
        "proteomics_plasma_cleaned_with_metadata.csv",
    )
    output_dir = args.output_dir or os.path.join(
        wkdir,
        "data",
        "cleaned",
        "proteomics",
        "normalized_sliced_by_suffix",
    )
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    sliced = format_proteomics_data(df)
    paths = save_sliced_outputs(sliced, output_dir, args.base_name)

    logger.info("Formatted suffix groups: %d", len(sliced))
    for p in paths:
        logger.info(p)


if __name__ == "__main__":
    main()
