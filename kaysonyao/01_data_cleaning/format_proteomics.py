import logging
import os
import re
import argparse
import pandas as pd

logger = logging.getLogger(__name__)


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
    2) Derive longitudinal suffix from SampleID vs SubjectID.
    3) Drop Batch column.
    4) Warn and exclude rows with no recognizable longitudinal suffix.
    5) Split rows by suffix.
    6) For each split dataframe:
       - Drop SubjectID
       - Remove suffix from SampleID (leave base subject ID)

    Returns:
        dict mapping suffix label -> dataframe
    """
    required = {"SampleID", "SubjectID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["SampleID"] = remove_internal_whitespace(out["SampleID"])
    out["SubjectID"] = remove_internal_whitespace(out["SubjectID"])

    out["LongitudinalLabel"] = [
        extract_suffix(sid, subid) for sid, subid in zip(out["SampleID"], out["SubjectID"])
    ]

    # Warn about rows dropped due to missing suffix (issue #10).
    no_suffix = out[out["LongitudinalLabel"] == ""]
    if not no_suffix.empty:
        logger.warning(
            "%d sample(s) have no longitudinal suffix and will be excluded: %s",
            len(no_suffix),
            no_suffix["SampleID"].tolist(),
        )
    out = out[out["LongitudinalLabel"] != ""].copy()

    if "Batch" in out.columns:
        out = out.drop(columns=["Batch"])

    grouped: dict[str, pd.DataFrame] = {}
    for suffix, g in out.groupby("LongitudinalLabel", sort=True):
        g2 = g.copy()
        # Remove suffix from SampleID to restore base subject ID.
        g2["SampleID"] = g2["SampleID"].str.replace(rf"{re.escape(suffix)}$", "", regex=True)
        g2 = g2.drop(columns=["SubjectID", "LongitudinalLabel"])
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
