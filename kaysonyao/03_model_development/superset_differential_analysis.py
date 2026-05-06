import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT  = os.path.dirname(_SCRIPT_DIR)

sys.path.insert(0, _SCRIPT_DIR)

_EA_PATH = os.path.join(_PROJ_ROOT, "02_exploratory_analysis")
sys.path.insert(0, _EA_PATH)

from utilities import (
    TIMEPOINTS,
    load_data,
    normalise_group_labels,
    get_analyte_columns,
    collect_superset_features,
)

import importlib.util as _ilu

def _import_ea_utilities():
    """Import 02_exploratory_analysis/utilities.py under an explicit name."""
    spec   = _ilu.spec_from_file_location(
        "ea_utilities",
        os.path.join(_EA_PATH, "utilities.py"),
    )
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_ea_utils = _import_ea_utilities()
run_cross_sectional = _ea_utils.run_cross_sectional
run_longitudinal    = _ea_utils.run_longitudinal
_start_analysis_log = _ea_utils._start_analysis_log

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _merge_to_complication(df: pd.DataFrame, group_col: str = "Group") -> pd.DataFrame:
    df = df.copy()
    df[group_col] = df[group_col].apply(
        lambda g: g if str(g).strip() == "Control" else "Complication"
    )
    return df


def _filter_to_superset(df: pd.DataFrame, superset: list) -> tuple[pd.DataFrame, list]:
    all_cols     = set(df.columns)
    analyte_cols = get_analyte_columns(df)
    available    = set(analyte_cols)

    present  = [f for f in superset if f in available]
    missing  = len(superset) - len(present)
    if missing:
        logger.warning(
            "  %d / %d superset features absent from this dataset; using %d.",
            missing, len(superset), len(present),
        )

    meta_cols = [c for c in df.columns if c not in set(analyte_cols)]
    keep_cols = meta_cols + present
    return df[keep_cols], present


def run_superset_differential(
    binary_results_dir: str,
    plasma_dir: str,
    placenta_csv: str,
    output_dir: str,
    superset_timepoints: list,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    _start_analysis_log(output_dir)

    superset = collect_superset_features(binary_results_dir, superset_timepoints)
    if not superset:
        logger.error(
            "No superset features found in %s for timepoints %s. "
            "Check that binary_classifier.py has been run and "
            "lasso_selected_features.csv files exist.",
            binary_results_dir, superset_timepoints,
        )
        return

    logger.info(
        "Superset (%s + placenta): %d features",
        " ".join(superset_timepoints), len(superset),
    )

    if os.path.exists(placenta_csv):
        df_plac = normalise_group_labels(load_data(placenta_csv))
        df_plac = _merge_to_complication(df_plac)
        df_plac, feat_plac = _filter_to_superset(df_plac, superset)
        logger.info(
            "Cross-sectional [placenta]: %d samples × %d superset features "
            "(Control vs Complication)",
            df_plac.shape[0], len(feat_plac),
        )
        run_cross_sectional(
            df_plac,
            feat_plac,
            group_col="Group",
            output_dir=os.path.join(output_dir, "placenta", "cross_sectional"),
        )
    else:
        logger.warning("Placenta CSV not found, skipping: %s", placenta_csv)

    tp_dfs: dict[str, pd.DataFrame] = {}
    for tp in TIMEPOINTS:
        tp_csv = os.path.join(
            plasma_dir, f"proteomics_plasma_formatted_suffix_{tp}.csv"
        )
        if not os.path.exists(tp_csv):
            logger.warning("Plasma timepoint %s not found, skipping: %s", tp, tp_csv)
            continue

        df_tp = normalise_group_labels(load_data(tp_csv))
        tp_dfs[tp] = df_tp            # keep originals for longitudinal

        df_tp_merged = _merge_to_complication(df_tp)
        df_tp_filt, feat_tp = _filter_to_superset(df_tp_merged, superset)

        logger.info(
            "Cross-sectional [plasma %s]: %d samples × %d superset features "
            "(Control vs Complication)",
            tp, df_tp_filt.shape[0], len(feat_tp),
        )
        run_cross_sectional(
            df_tp_filt,
            feat_tp,
            group_col="Group",
            output_dir=os.path.join(output_dir, "plasma", "cross_sectional", tp),
        )

    if len(tp_dfs) < 2:
        logger.warning(
            "Fewer than 2 plasma timepoints loaded (%d); skipping longitudinal.",
            len(tp_dfs),
        )
    else:
        long_dir = os.path.join(output_dir, "plasma", "longitudinal")

        def _prep_tp(df: pd.DataFrame) -> pd.DataFrame:
            df_f, _ = _filter_to_superset(df, superset)
            return df_f

        tp_dfs_filt = {tp: _prep_tp(df) for tp, df in tp_dfs.items()}
        _feat_long = [f for f in superset
                      if f in get_analyte_columns(next(iter(tp_dfs_filt.values())))]

        logger.info(
            "Longitudinal [plasma]: %d timepoints, %d superset features",
            len(tp_dfs_filt), len(_feat_long),
        )

        run_longitudinal(
            tp_dfs_filt,
            _feat_long,
            group="Control",
            group_col="Group",
            subject_col="SubjectID",
            output_dir=long_dir,
        )

        tp_dfs_merged = {tp: _merge_to_complication(df) for tp, df in tp_dfs_filt.items()}
        run_longitudinal(
            tp_dfs_merged,
            _feat_long,
            group="Complication",
            group_col="Group",
            subject_col="SubjectID",
            output_dir=long_dir,
        )

    logger.info("Superset differential analysis complete. Output: %s", output_dir)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Differential analysis (cross-sectional + longitudinal) restricted to "
            "the LASSO-selected superset proteins."
        )
    )
    p.add_argument(
        "--binary-results-dir",
        default=None,
        help=(
            "Root of binary classifier outputs containing "
            "plasma/<TP>/lasso_selected_features.csv and "
            "placenta/all/lasso_selected_features.csv. "
            "Defaults to 04_results_and_figures/models/binary/."
        ),
    )
    p.add_argument(
        "--plasma-dir",
        default=None,
        help=(
            "Directory of per-suffix plasma CSVs "
            "(proteomics_plasma_formatted_suffix_<TP>.csv). "
            "Defaults to data/cleaned/proteomics/normalized_sliced_by_suffix/."
        ),
    )
    p.add_argument(
        "--placenta-csv",
        default=None,
        help=(
            "Path to placenta cleaned CSV. "
            "Defaults to data/cleaned/proteomics/normalized_full_results/"
            "proteomics_placenta_cleaned_with_metadata.csv."
        ),
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Root output directory. "
            "Defaults to 04_results_and_figures/models/binary/superset_differential/."
        ),
    )
    p.add_argument(
        "--superset-timepoints",
        nargs="+",
        default=["A", "B", "C", "D"],
        help=(
            "Plasma timepoints whose LASSO-selected features contribute to the "
            "superset.  Placenta is always included.  Defaults to A B C D "
            "(E excluded because its LASSO regularisation was too weak, "
            "selecting ~1600 features)."
        ),
    )
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    wkdir = os.getcwd()

    binary_results_dir = args.binary_results_dir or os.path.join(
        wkdir, "04_results_and_figures", "models", "binary"
    )
    plasma_dir = args.plasma_dir or os.path.join(
        wkdir, "data", "cleaned", "proteomics", "normalized_sliced_by_suffix"
    )
    placenta_csv = args.placenta_csv or os.path.join(
        wkdir,
        "data", "cleaned", "proteomics", "normalized_full_results",
        "proteomics_placenta_cleaned_with_metadata.csv",
    )
    output_dir = args.output_dir or os.path.join(
        wkdir, "04_results_and_figures", "models", "binary", "superset_differential"
    )

    run_superset_differential(
        binary_results_dir=binary_results_dir,
        plasma_dir=plasma_dir,
        placenta_csv=placenta_csv,
        output_dir=output_dir,
        superset_timepoints=args.superset_timepoints,
    )


if __name__ == "__main__":
    main()
