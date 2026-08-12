"""

options:
  -h, --help            show this help message and exit
  --input-csv INPUT_CSV
                        Path to final plasma output CSV. Defaults to normalized_full_results output.
  --output-dir OUTPUT_DIR
                        Directory for per-suffix output files. Defaults under normalized_full_results.
  --base-name BASE_NAME
                        Base output filename prefix (default: proteomics_plasma_formatted).

"""
import logging
import os
import re
import argparse
import pandas as pd

logger = logging.getLogger(__name__)

# Metadata columns carried through the wide matrix (non-analyte).
_META_COLS: frozenset[str] = frozenset(
    {"Group", "Subgroup", "GestAgeDelivery", "SampleGestAge"}
)

# define timepoints
def categorize_sample_time(gest_age_samp):
    if gest_age_samp is None:
        return None
    if gest_age_samp<6.0:
        return None
    if gest_age_samp<=13.9:
        return '1'
    if gest_age_samp<=21.9:
        return '2'
    if gest_age_samp<=31.9:
        return '3'
    if gest_age_samp<=36.9:
        return '4'
    return '5'

def remove_internal_whitespace(s: pd.Series) -> pd.Series:
    """Remove all whitespace characters from a string series."""
    return s.astype(str).str.replace(r"\s+", "", regex=True)


def format_proteomics_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Format longitudinal plasma proteomics output into per-suffix dataframes.

    Steps:
    1) Add Timepoint column
    2) Drop Batch column.
    3) Split rows by timepoint.

    Returns:
        dict mapping merged suffix label (e.g. "A") -> dataframe
    """
    required = {"SampleGestAge"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["SampleID"] = remove_internal_whitespace(out["SampleID"])
    out["SubjectID"] = remove_internal_whitespace(out["SubjectID"])


    # Step 1: add Timepoint column
    out["Timepoint"] = [categorize_sample_time(x) for x in out["SampleGestAge"]]

    # Step 2: drop Batch.
    if "Batch" in out.columns:
        out = out.drop(columns=["Batch"])

    grouped: dict[str, pd.DataFrame] = {}
    for suffix, g in out.groupby("Timepoint", sort=True):
        grouped[suffix] = g

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
