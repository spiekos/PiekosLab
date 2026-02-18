"""
Unit tests and integration test for the DP3 proteomics cleaning pipeline.

Unit tests cover every utility function called by clean_proteomics_data.py.
The integration test validates a completed cleaned output CSV with tighter
statistical thresholds than the original broad sanity checks.

Usage
-----
Run unit tests only:
    python test_proteomics.py

Include integration tests on one or more cleaned CSVs:
    python test_proteomics.py --integration path/to/cleaned.csv [more/paths...]

Verbose integration output:
    python test_proteomics.py --integration path/to/cleaned.csv --verbose
"""

import os
import tempfile

import numpy as np
import pandas as pd
import scipy.stats as stats

from utilities import (
    CUTOFF_PERCENT_MISSING,
    collect_olink_files,
    standardize_missing_npx,
    qc_mask,
    process_single_file,
    combine_batches,
    is_olink_control_sample,
    is_olink_control_assay,
    apply_panel_normalization_long,
    benjamini_hochberg_rejections,
    missingness_filter_and_group_check,
    combat_normalize_wide,
    half_min_impute_wide,
)

# Metadata columns in the final cleaned CSV
_METADATA_COLS = [
    "SampleID", "SubjectID", "Group", "Subgroup",
    "Batch", "GestAgeDelivery", "SampleGestAge",
]

# Fixed seed for all random sampling in the integration test.
_RNG_SEED = 42


# ============================================================
# Unit tests
# ============================================================

def test_standardize_missing_npx():
    """NaN sentinels, zero, and valid numerics are handled correctly."""
    df = pd.DataFrame({
        "NPX": ["", " ", "NA", "N/A", "na", "n/a", "NaN", "nan", None, "0", "3.5", 2.0],
    })
    result = standardize_missing_npx(df)

    # First 10 entries (all sentinel values including zero) must become NaN.
    nan_mask = result["NPX"].isna()
    assert nan_mask.iloc[:10].all(), (
        f"Expected first 10 entries to be NaN; got: {result['NPX'].values[:10]}"
    )
    assert abs(result["NPX"].iloc[10] - 3.5) < 1e-9, "String '3.5' should parse to float 3.5"
    assert abs(result["NPX"].iloc[11] - 2.0) < 1e-9, "Float 2.0 should be preserved"

    # DataFrames without an NPX column are returned unchanged.
    df_no_npx = pd.DataFrame({"Other": [0, "NA", None]})
    assert standardize_missing_npx(df_no_npx).equals(df_no_npx)

    print("PASS: test_standardize_missing_npx")


def test_qc_mask():
    """QC_Warning and Assay_Warning != 'PASS' (case-insensitive) set NPX to NaN."""
    df = pd.DataFrame({
        "NPX":           [1.0,   2.0,    3.0,       4.0,       5.0],
        "QC_Warning":    ["Pass","PASS", "Warning", "PASS",    "PASS"],
        "Assay_Warning": ["Pass","PASS", "PASS",    "Warning", "PASS"],
    })
    out = qc_mask(df)

    assert abs(out["NPX"].iloc[0] - 1.0) < 1e-9, "Both PASS → value preserved"
    assert abs(out["NPX"].iloc[1] - 2.0) < 1e-9, "Both PASS → value preserved"
    assert pd.isna(out["NPX"].iloc[2]),            "QC_Warning != PASS → NaN"
    assert pd.isna(out["NPX"].iloc[3]),            "Assay_Warning != PASS → NaN"
    assert abs(out["NPX"].iloc[4] - 5.0) < 1e-9,  "Both PASS → value preserved"

    # No NPX column: DataFrame returned unchanged.
    df_no_npx = pd.DataFrame({"Other": [1, 2]})
    assert qc_mask(df_no_npx).equals(df_no_npx)

    print("PASS: test_qc_mask")


def test_combine_batches():
    """Concatenation preserves all rows and resets the index."""
    df1 = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    df2 = pd.DataFrame({"A": [3, 4], "B": ["z", "w"]})
    out = combine_batches([df1, df2])

    assert list(out["A"]) == [1, 2, 3, 4],        "All rows should be present"
    assert list(out.index) == [0, 1, 2, 3],        "Index should be reset after concat"
    assert combine_batches([df1]).shape == df1.shape

    print("PASS: test_combine_batches")


def test_is_olink_control_sample():
    """Samples with CONTROL / NEG / PLATE prefixes (case-insensitive) are flagged."""
    df = pd.DataFrame({"SampleID": [
        "CONTROL_SAMPLE_001", "control_sample_002",   # CONTROL prefix
        "NEG_CTRL_01",        "neg_ctrl_02",           # NEG prefix
        "PLATE_001",          "plate_002",             # PLATE prefix
        "DP3-0005A",          "DP3-0005B",             # biological samples
    ]})
    mask = is_olink_control_sample(df)

    assert mask.iloc[:6].all(),    "First 6 entries should be flagged as controls"
    assert not mask.iloc[6:].any(), "Last 2 entries should NOT be flagged"

    print("PASS: test_is_olink_control_sample")


def test_is_olink_control_assay():
    """Assays containing 'control' as a whole word (\\bcontrol\\b) are flagged."""
    df = pd.DataFrame({"Assay": [
        "Incubation Control",        # whole-word match
        "Amplification control",     # lowercase, whole-word match
        "This is a control assay",   # surrounded by spaces
        "GAPDH",                     # biological assay
        "IL6",                       # biological assay
        "uncontrolled",              # 'control' not a whole word → must NOT match
    ]})
    mask = is_olink_control_assay(df)

    assert mask.iloc[0],      "'Incubation Control' should match"
    assert mask.iloc[1],      "'Amplification control' should match"
    assert mask.iloc[2],      "'control' surrounded by spaces should match"
    assert not mask.iloc[3],  "'GAPDH' should not match"
    assert not mask.iloc[4],  "'IL6' should not match"
    assert not mask.iloc[5],  "'uncontrolled' should not match (word boundary)"

    print("PASS: test_is_olink_control_assay")


def test_apply_panel_normalization_long():
    """Panel adjustments align control-sample medians to a global reference."""
    # Assay "X" measured on two panels.
    # Panel A controls: NPX = 4, 6  → panel median = 5.
    # Panel B controls: NPX = 8, 10 → panel median = 9.
    # Global reference for X: median(5, 9) = 7.
    # Adjustments: Panel A += (7 − 5) = +2 ; Panel B += (7 − 9) = −2.
    # Biological samples both start at NPX = 3:
    #   Panel A bio → 3 + 2 = 5,  Panel B bio → 3 − 2 = 1.
    df = pd.DataFrame({
        "SampleID": ["CONTROL_S1", "CONTROL_S2", "CONTROL_S3", "CONTROL_S4", "BIO_001", "BIO_002"],
        "Panel":    ["PanelA",     "PanelA",     "PanelB",     "PanelB",     "PanelA",  "PanelB"],
        "Assay":    ["X",          "X",          "X",          "X",          "X",       "X"],
        "NPX":      [4.0,          6.0,          8.0,          10.0,         3.0,       3.0],
    })
    out = apply_panel_normalization_long(df)

    bio_a = out.loc[out["SampleID"] == "BIO_001", "NPX"].values[0]
    bio_b = out.loc[out["SampleID"] == "BIO_002", "NPX"].values[0]

    assert abs(bio_a - 5.0) < 1e-9, f"Panel A bio sample: expected 5.0, got {bio_a}"
    assert abs(bio_b - 1.0) < 1e-9, f"Panel B bio sample: expected 1.0, got {bio_b}"

    # No control samples present → must raise ValueError.
    df_no_ctrl = df[~df["SampleID"].str.startswith("CONTROL")].copy()
    try:
        apply_panel_normalization_long(df_no_ctrl)
        assert False, "Expected ValueError when no controls are present"
    except ValueError:
        pass

    print("PASS: test_apply_panel_normalization_long")


def test_benjamini_hochberg_rejections():
    """Small p-values are rejected; large ones are not; empty input returns empty."""
    pvals = pd.Series([0.001, 0.002, 0.900, 0.950])
    result = benjamini_hochberg_rejections(pvals, alpha=0.05)

    assert result.iloc[0] and result.iloc[1],   "p=0.001 and p=0.002 should be rejected"
    assert not result.iloc[2] and not result.iloc[3], "p=0.9 and p=0.95 should not be rejected"

    # Empty series → empty result.
    assert benjamini_hochberg_rejections(pd.Series(dtype=float)).empty

    # NaN values are dropped before correction.
    pvals_nan = pd.Series([0.001, np.nan, 0.900])
    result_nan = benjamini_hochberg_rejections(pvals_nan)
    assert len(result_nan) == 2, "NaN entry should be dropped (2 non-NaN values)"

    print("PASS: test_benjamini_hochberg_rejections")


def test_missingness_filter_and_group_check():
    """Assays at or above the missingness cutoff are dropped; a report is returned."""
    # 4 samples, 3 assays.
    # Assay A: 1/4 NaN = 0.25  → at cutoff, dropped (keep requires strictly < cutoff).
    # Assay B: 2/4 NaN = 0.50  → above cutoff, dropped.
    # Assay C: 0/4 NaN = 0.00  → kept.
    X = pd.DataFrame({
        "A": [1.0, 2.0, 3.0, np.nan],
        "B": [1.0, 2.0, np.nan, np.nan],
        "C": [1.0, 2.0, 3.0, 4.0],
    })
    groups = pd.Series(["Control", "Control", "Complication", "Complication"])
    X_kept, report = missingness_filter_and_group_check(X, groups, cutoff=CUTOFF_PERCENT_MISSING)

    assert "C" in X_kept.columns,     "Assay C (0% missing) should be kept"
    assert "A" not in X_kept.columns, "Assay A (25% missing = cutoff) should be dropped"
    assert "B" not in X_kept.columns, "Assay B (50% missing) should be dropped"
    assert len(report) == 2,          f"Report should have 2 rows, got {len(report)}"

    # No assays to drop → report is empty and all columns are kept.
    X_clean = pd.DataFrame({"P1": [1.0, 2.0], "P2": [3.0, 4.0]})
    g_clean = pd.Series(["Control", "Complication"])
    X_k2, rep2 = missingness_filter_and_group_check(X_clean, g_clean)
    assert rep2.empty,                          "No assays should be dropped"
    assert sorted(X_k2.columns) == ["P1", "P2"]

    print("PASS: test_missingness_filter_and_group_check")


def test_combat_normalize_wide():
    """ComBat preserves shape; missingness pattern is restored; single batch → copy."""
    rng = np.random.default_rng(_RNG_SEED)

    # Single batch: ComBat cannot correct, function returns a copy.
    X_single = pd.DataFrame(
        rng.standard_normal((10, 5)),
        columns=[f"P{i}" for i in range(5)],
    )
    batch_single = pd.Series(["batch1"] * 10, index=X_single.index)
    out_single = combat_normalize_wide(X_single, batch_single)
    assert out_single.shape == X_single.shape, "Single-batch output shape must match input"

    # Two batches: shape preserved, missingness pattern fully restored.
    X = pd.DataFrame(
        rng.standard_normal((20, 8)),
        columns=[f"P{i}" for i in range(8)],
    )
    batch = pd.Series(["A"] * 10 + ["B"] * 10, index=X.index)
    X.iloc[0, 0] = np.nan
    X.iloc[5, 3] = np.nan
    missing_mask = X.isna()

    out = combat_normalize_wide(X, batch)
    assert out.shape == X.shape,                  "Output shape must match input"
    assert out.isna().equals(missing_mask),        "Missingness pattern must be preserved"

    # Missing batch label → ValueError.
    batch_bad = batch.copy()
    batch_bad.iloc[0] = np.nan
    try:
        combat_normalize_wide(X, batch_bad)
        assert False, "Expected ValueError for missing batch label"
    except ValueError:
        pass

    print("PASS: test_combat_normalize_wide")


def test_half_min_impute_wide():
    """NaN is imputed with (observed min − 1) per assay; non-NaN values are unchanged."""
    X = pd.DataFrame({
        "A": [1.0, 2.0, np.nan],        # min = 1.0 → imputed = 0.0
        "B": [np.nan, np.nan, np.nan],  # all NaN → no observed value → stays NaN
        "C": [3.0, np.nan, 5.0],        # min = 3.0 → imputed = 2.0
    })
    out = half_min_impute_wide(X)

    # Assay A
    assert abs(out.loc[2, "A"] - 0.0) < 1e-9, f"Assay A imputed: expected 0.0, got {out.loc[2, 'A']}"
    assert abs(out.loc[0, "A"] - 1.0) < 1e-9, "Assay A observed value should be unchanged"

    # Assay B (all NaN)
    assert out["B"].isna().all(), "All-NaN column should remain NaN (no min to impute from)"

    # Assay C
    assert abs(out.loc[1, "C"] - 2.0) < 1e-9, f"Assay C imputed: expected 2.0, got {out.loc[1, 'C']}"
    assert abs(out.loc[0, "C"] - 3.0) < 1e-9, "Assay C observed value should be unchanged"

    print("PASS: test_half_min_impute_wide")


def test_process_single_file():
    """process_single_file loads a CSV, standardizes NPX sentinels, and applies QC mask."""
    csv_content = (
        "SampleID,Panel,Assay,NPX,QC_Warning,Assay_Warning\n"
        "S1,PanelA,Prot1,3.5,Pass,Pass\n"         # valid row
        "S2,PanelA,Prot1,NA,Pass,Pass\n"           # NA → NaN
        "S3,PanelA,Prot1,2.0,Warning,Pass\n"       # QC fail → NaN
        "S4,PanelA,Prot1,2.0,Pass,Warning\n"       # Assay fail → NaN
        "S5,PanelA,Prot1,0,Pass,Pass\n"            # zero → NaN
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        tmp = f.name

    try:
        df = process_single_file(tmp)

        def npx(sid):
            return df.loc[df["SampleID"] == sid, "NPX"].values[0]

        assert abs(npx("S1") - 3.5) < 1e-9, "S1: valid NPX should be preserved"
        assert pd.isna(npx("S2")),           "S2: NA should become NaN"
        assert pd.isna(npx("S3")),           "S3: QC_Warning fail should set NaN"
        assert pd.isna(npx("S4")),           "S4: Assay_Warning fail should set NaN"
        assert pd.isna(npx("S5")),           "S5: zero NPX should become NaN"
    finally:
        os.unlink(tmp)

    print("PASS: test_process_single_file")


def test_collect_olink_files():
    """Files are classified as plasma or placenta; unrecognised .csv files trigger a warning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_names = [
            "batch1_plasma.csv",
            "batch2_plasma.csv",
            "batch1_placenta.csv",
            "batch1_tissue.csv",
            "batch1_lysate.csv",
            "unknown.csv",   # unclassified → logged as warning, not returned
            "readme.txt",    # not .csv → ignored
        ]
        for fn in file_names:
            open(os.path.join(tmpdir, fn), "w").close()

        plasma, placenta = collect_olink_files(tmpdir)

        assert len(plasma) == 2,   f"Expected 2 plasma files, got {len(plasma)}"
        assert len(placenta) == 3, f"Expected 3 placenta files, got {len(placenta)}"
        assert all("plasma" in f for f in plasma)
        assert all(any(kw in f for kw in ("placenta", "tissue", "lysate")) for f in placenta)

        # Lists should be sorted alphabetically.
        assert plasma == sorted(plasma),     "Plasma list should be sorted"
        assert placenta == sorted(placenta), "Placenta list should be sorted"

    print("PASS: test_collect_olink_files")


# ============================================================
# Integration test
# ============================================================

def test_cleaned_output(result_path: str, verbose: bool = False) -> None:
    """
    Validate a completed cleaned proteomics CSV.

    Checks structural correctness, distribution stability after ComBat
    normalization, KS similarity between random sample pairs, and shape
    of the per-sample distribution (skewness, kurtosis).

    Thresholds are deliberately tighter than the broad sanity bounds used
    during development; a failure here indicates a normalization regression.

    Args:
        result_path: Path to the cleaned wide-format CSV (index = SampleID).
        verbose:     Print per-metric statistics alongside assertions.
    """
    df = pd.read_csv(result_path, index_col=0)

    meta_cols = [c for c in _METADATA_COLS if c in df.columns]
    X = np.log2(df.drop(columns=meta_cols))

    if verbose:
        print(f"\nFile : {os.path.basename(result_path)}")
        print(f"Shape: {df.shape}  (samples × [metadata + assays])")

    # --- Structure --------------------------------------------------------
    assert df.shape[0] > 0 and df.shape[1] > 0, "Cleaned dataframe is empty."
    assert X.index.is_unique,   "Row indices are not unique."
    assert X.columns.is_unique, "Column indices are not unique."
    assert not X.isnull().any().any(), (
        "Cleaned dataframe contains NaN values — imputation may have failed."
    )

    # --- Distribution stability -------------------------------------------
    sample_stds = X.std(axis=1)
    sample_medians = X.median(axis=1)

    cv_stds = sample_stds.std() / sample_stds.mean()
    cv_medians = (
        sample_medians.std() / abs(sample_medians.mean())
        if abs(sample_medians.mean()) > 0.01
        else float("inf")
    )

    if verbose:
        print(f"CV(sample stds):    {cv_stds:.4f}   [threshold < 0.50]")
        print(f"CV(sample medians): {cv_medians:.4f}  [threshold < 1.00]")

    assert np.isfinite(cv_stds), "Sample std CV is non-finite — normalization likely failed."
    assert cv_stds < 0.50, (
        f"Sample stds are too variable after normalization (CV={cv_stds:.4f}; threshold 0.50)."
    )
    if abs(sample_medians.mean()) > 0.1:
        assert cv_medians < 1.0, (
            f"Sample medians are too variable (CV={cv_medians:.4f}; threshold 1.00)."
        )

    # --- KS test between random sample pairs -----------------------------
    n_samples = X.shape[0]
    n_tests = min(10, n_samples - 1)
    ks_stats = []

    if n_tests > 0:
        rng = np.random.default_rng(_RNG_SEED)
        i_idx = rng.choice(n_samples, size=n_tests, replace=False)
        offsets = rng.integers(1, n_samples, size=n_tests)
        j_idx = (i_idx + offsets) % n_samples

        for i, j in zip(i_idx, j_idx):
            stat, _ = stats.ks_2samp(X.iloc[i].values, X.iloc[j].values)
            ks_stats.append(stat)

    mean_ks = np.mean(ks_stats) if ks_stats else 0.0
    if verbose:
        print(f"Mean KS (random pairs): {mean_ks:.4f}  [threshold < 0.60]")

    assert mean_ks < 0.60, (
        f"Sample distributions are too dissimilar (mean KS={mean_ks:.4f}; threshold 0.60). "
        "Check normalization."
    )

    # --- ComBat batch uniformity check -----------------------------------
    # After ComBat the mean-of-per-sample-medians should be tightly
    # aligned across batches.  A high CV here means ComBat did not
    # successfully remove the batch effect.
    if "Batch" in df.columns:
        batches = df["Batch"]
        unique_batches = batches.dropna().unique()
        if len(unique_batches) >= 2:
            batch_medians = pd.Series({
                b: X.loc[batches == b].median(axis=1).mean()
                for b in unique_batches
            })
            batch_cv = (
                batch_medians.std() / abs(batch_medians.mean())
                if abs(batch_medians.mean()) > 0.01
                else float("inf")
            )
            if verbose:
                print(f"Per-batch median CV: {batch_cv:.4f}  [threshold < 0.10]")
            assert np.isfinite(batch_cv), "Per-batch median CV is non-finite."
            assert batch_cv < 0.10, (
                f"Per-batch sample medians not well aligned after ComBat "
                f"(CV={batch_cv:.4f}; threshold 0.10). ComBat may have failed."
            )

    # --- Distribution shape -----------------------------------------------
    skewness = X.apply(stats.skew, axis=1)
    kurtosis = X.apply(stats.kurtosis, axis=1)

    if verbose:
        print(f"Skewness std:  {skewness.std():.4f}  [threshold < 0.70]")
        print(f"Kurtosis std:  {kurtosis.std():.4f}  [threshold < 1.50]")

    assert skewness.std() < 0.70, (
        f"Per-sample skewness is too heterogeneous (std={skewness.std():.4f}; threshold 0.70)."
    )
    assert kurtosis.std() < 1.50, (
        f"Per-sample kurtosis is too heterogeneous (std={kurtosis.std():.4f}; threshold 1.50)."
    )

    print(f"PASS: test_cleaned_output ({os.path.basename(result_path)})")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Run unit tests for the proteomics cleaning pipeline. "
            "Optionally validate one or more cleaned CSV outputs with integration tests."
        )
    )
    parser.add_argument(
        "--integration",
        nargs="+",
        metavar="CSV",
        default=None,
        help="Path(s) to cleaned CSV files to run the integration test against.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed statistics during the integration test.",
    )
    args = parser.parse_args()

    _unit_tests = [
        test_standardize_missing_npx,
        test_qc_mask,
        test_combine_batches,
        test_is_olink_control_sample,
        test_is_olink_control_assay,
        test_apply_panel_normalization_long,
        test_benjamini_hochberg_rejections,
        test_missingness_filter_and_group_check,
        test_combat_normalize_wide,
        test_half_min_impute_wide,
        test_process_single_file,
        test_collect_olink_files,
    ]

    print("=" * 60)
    print("UNIT TESTS")
    print("=" * 60)
    n_pass = n_fail = 0
    for fn in _unit_tests:
        try:
            fn()
            n_pass += 1
        except Exception as e:
            print(f"FAIL: {fn.__name__}: {e}")
            n_fail += 1

    print(f"\nUnit tests: {n_pass} passed, {n_fail} failed.")

    if args.integration:
        print("\n" + "=" * 60)
        print("INTEGRATION TESTS")
        print("=" * 60)
        for csv_path in args.integration:
            if not os.path.exists(csv_path):
                print(f"SKIP: file not found: {csv_path}")
                continue
            try:
                test_cleaned_output(csv_path, verbose=args.verbose)
            except AssertionError as e:
                print(f"FAIL: {e}")
                n_fail += 1
    else:
        print(
            "\nTo validate cleaned output files, run:\n"
            "    python test_proteomics.py --integration path/to/cleaned.csv"
        )

    sys.exit(n_fail)
