import os
import argparse
import numpy as np
import pandas as pd

from clean_proteomics_data import (
    CUTOFF_PERCENT_MISSING,
    load_metadata_with_batch,
    process_single_file,
    combine_batches,
    apply_panel_normalization_long,
    is_olink_control_sample,
    missingness_filter_and_group_check,
    half_min_impute_wide,
)


def discover_placenta_files(data_dir: str) -> list[str]:
    files = []
    for fn in os.listdir(data_dir):
        if not fn.endswith('.csv'):
            continue
        low = fn.lower()
        if 'placenta' in low or 'tissue' in low or 'lysate' in low:
            files.append(os.path.join(data_dir, fn))
    return sorted(files)


def _to_dataframe_like(arr, index, columns) -> pd.DataFrame:
    if isinstance(arr, pd.DataFrame):
        out = arr.copy()
        out = out.reindex(index=index, columns=columns)
        return out
    out = pd.DataFrame(arr, index=index, columns=columns)
    return out


def combat_correct_python(X: pd.DataFrame, batch: pd.Series) -> tuple[pd.DataFrame, str]:
    """
    Try available Python ComBat implementations.
    Returns corrected matrix and method name.
    """
    b = batch.reindex(X.index).astype(str)

    # pycombat package
    try:
        from pycombat import Combat  # type: ignore

        model = Combat()
        corrected = model.fit_transform(X, b.values)
        return _to_dataframe_like(corrected, X.index, X.columns), 'pycombat.Combat'
    except Exception:
        pass

    # neuroCombat package
    try:
        from neuroCombat import neuroCombat  # type: ignore

        covars = pd.DataFrame({'batch': b.values}, index=X.index)
        result = neuroCombat(dat=X.T.values, covars=covars, batch_col='batch')
        corrected = pd.DataFrame(result['data'].T, index=X.index, columns=X.columns)
        return corrected, 'neuroCombat.neuroCombat'
    except Exception:
        pass

    # combat package (combat.pycombat)
    try:
        from combat.pycombat import pycombat  # type: ignore

        corrected = pycombat(X.T, b.values)
        if isinstance(corrected, pd.DataFrame):
            corrected = corrected.T
        else:
            corrected = np.asarray(corrected).T
        return _to_dataframe_like(corrected, X.index, X.columns), 'combat.pycombat'
    except Exception:
        pass

    raise ImportError(
        "No supported Python ComBat package found. "
        "Install one of: pycombat, neuroCombat, or combat."
    )


def run_placenta_combat_test(files: list[str], metadata_path: str, output_csv: str) -> None:
    print(f"[combat-test] placenta start | files={len(files)}")
    if len(files) == 0:
        raise ValueError('No placenta files provided/found.')

    metadata = load_metadata_with_batch(metadata_path, meta_type='placenta')

    dfs = []
    for fp in files:
        df_single = process_single_file(fp)
        df_single['SourceFile'] = os.path.basename(fp)
        dfs.append(df_single)

    df_long = combine_batches(dfs)
    df_long = apply_panel_normalization_long(df_long)
    df_long_bio = df_long.loc[~is_olink_control_sample(df_long)].copy()

    X = df_long_bio.pivot_table(index='SampleID', columns='Assay', values='NPX', aggfunc='median')

    groups = metadata['Group'].reindex(X.index)
    has_metadata = groups.notna()
    if not has_metadata.all():
        dropped = int((~has_metadata).sum())
        print(f"[combat-test] warning: filtering {dropped} samples without metadata group")
        X = X.loc[has_metadata].copy()
        groups = groups.loc[has_metadata]

    groups_binary = pd.Series(
        np.where(groups.astype(str).str.strip().str.upper() == 'CONTROL', 'Control', 'Complication'),
        index=groups.index,
        name='GroupBinary'
    )

    X_kept, dropped_report = missingness_filter_and_group_check(
        X, groups_binary, cutoff=CUTOFF_PERCENT_MISSING, alpha_bh=0.05
    )
    if not dropped_report.empty:
        rep_path = os.path.splitext(output_csv)[0] + '_dropped_missingness_report.csv'
        dropped_report.to_csv(rep_path, index=False)
        print(f"[combat-test] missingness report: {rep_path}")

    metadata = metadata.reindex(X_kept.index)
    batch = metadata['Batch'].reindex(X_kept.index)
    has_batch = batch.notna()
    if not has_batch.all():
        dropped = int((~has_batch).sum())
        print(f"[combat-test] warning: filtering {dropped} samples without batch labels")
        X_kept = X_kept.loc[has_batch].copy()
        metadata = metadata.reindex(X_kept.index)
        batch = batch.loc[has_batch]

    # Most Python ComBat implementations require complete matrix.
    X_for_combat = X_kept.apply(lambda col: col.fillna(col.median(skipna=True)), axis=0)
    X_combat, method_name = combat_correct_python(X_for_combat, batch)
    print(f"[combat-test] ComBat method: {method_name}")

    X_final = half_min_impute_wide(X_combat)
    X_final_linear = np.power(2.0, X_final)

    metadata_aligned = metadata.reindex(X_final_linear.index)
    final_output = pd.concat([metadata_aligned, X_final_linear], axis=1)
    final_output.to_csv(output_csv, index=True)

    print(
        f"[combat-test] done | samples={final_output.shape[0]} | assays={X_final_linear.shape[1]} "
        f"| output={output_csv}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Placenta-only ComBat test run (separate from main cleaner).')
    p.add_argument('--files', nargs='+', default=None, help='Placenta Olink CSV files. If omitted, discover from --data-dir.')
    p.add_argument('--data-dir', default=None, help='Folder to auto-discover placenta CSV files.')
    p.add_argument('--metadata-path', default=None, help='Path to metadata Excel file.')
    p.add_argument('--output-csv', default=None, help='Output CSV path for ComBat test result.')
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    wkdir = os.getcwd()
    data_dir = args.data_dir or os.path.join(wkdir, 'data', 'proteomics')
    metadata_path = args.metadata_path or os.path.join(wkdir, 'data', 'dp3 master table v2.xlsx')
    output_csv = args.output_csv or os.path.join(
        wkdir, 'data', 'cleaned', 'proteomics', 'proteomics_placenta_cleaned_with_metadata_combat_test.csv'
    )

    files = args.files if args.files is not None else discover_placenta_files(data_dir)
    run_placenta_combat_test(files, metadata_path, output_csv)


if __name__ == '__main__':
    main()
