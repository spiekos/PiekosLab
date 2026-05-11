"""
Shared utilities for the DP3 omics preprocessing pipeline.

Imported by:
    - clean_proteomics_data.py
    - proteomics_diagnostics.py
"""

import logging
import os
import re

import numpy as np
import pandas as pd
from pycombat import Combat
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

CUTOFF_PERCENT_MISSING = 0.20
CONTROL_SAMPLE_PREFIXES = ("CONTROL", "NEG", "PLATE")

# Canonical group label corrections — applied to the Group column at metadata load time
# so every downstream output CSV carries consistent labels.
_GROUP_LABEL_MAP = {"sptb": "sPTB"}

# Columns produced by metadata loaders — used downstream to separate metadata
# from analyte/feature columns when both share the same DataFrame.
METADATA_COLS = [
    "SubjectID", "Group", "Subgroup", "Batch", "GestAgeDelivery", "SampleGestAge",
    "MetadataCanonicalID",
]


# -----------------------------
# File discovery
# -----------------------------
def collect_olink_files(data_dir: str) -> tuple[list[str], list[str]]:
    """
    Scan a directory for Olink CSV files and classify them as plasma or placenta.

    Returns:
        (plasma_files, placenta_files) - both lists sorted alphabetically.
    """
    plasma_files: list[str] = []
    placenta_files: list[str] = []

    for fn in os.listdir(data_dir):
        if not fn.endswith(".csv"):
            continue
        full_path = os.path.join(data_dir, fn)
        fn_lower = fn.lower()
        if "plasma" in fn_lower:
            plasma_files.append(full_path)
        elif "placenta" in fn_lower or "tissue" in fn_lower or "lysate" in fn_lower:
            placenta_files.append(full_path)
        else:
            logger.warning("Cannot determine sample type for file: %s — skipping.", fn)

    return sorted(plasma_files), sorted(placenta_files)


# -----------------------------
# Metadata Loading
# -----------------------------
def load_metadata_with_batch(
    metadata_path: str,
    meta_type: str = "proteomics",
    sample_id_col: str = None,
    batch_col: str = "omics set#",
) -> pd.DataFrame:
    """
    Load metadata from the DP3 master Excel file and return a clean DataFrame.

    Supported meta_type values and their index semantics
    -----------------------------------------------------
    "proteomics"   → sheet "n=133 proteomics", index = SampleID (DP3-XXXXZ)
    "placenta"     → sheet "n=133 placenta",   index = SampleID (DP3-XXXXZ)
    "metabolomics" → sheet "n=133 metabolomics", index = SampleID (DP3-XXXXZ)
    "lipids"       → sheet "Sheet1" (positional columns), index = SubjectID (DP3-XXXX)
                     Postpartum-only IDs stored as "DP3-XXXXE" are stripped to
                     "DP3-XXXX" so they can be joined on bare subject ID.

    Common output columns: Batch, Group, Subgroup, GestAgeDelivery
    Proteomics-only extra column: SampleGestAge

    The "lipids" branch uses positional column access because Sheet1 has no
    standardised named columns. Column layout assumed:
        col 0  → subject/sample ID
        col 5  → gest age del
        col 7  → group
        col 8  → subgroup
        col 16 → omics set# / batch

    Args:
        metadata_path: path to the metadata Excel file
        meta_type: one of "proteomics", "placenta", "metabolomics", "lipids"
        sample_id_col: (non-lipids only) column name for sample IDs;
                       default None = auto-detect from meta_type
        batch_col: (non-lipids only) column name for batch; default "omics set#"

    Returns:
        DataFrame indexed by SampleID (proteomics/placenta/metabolomics)
        or SubjectID (lipids), with columns Batch, Group, Subgroup, GestAgeDelivery
        (and SampleGestAge for proteomics).
    """
    # -----------------------------------------------------------------
    # Lipids: Sheet1 with positional column access
    # -----------------------------------------------------------------
    if meta_type == "lipids":
        return _load_lipids_metadata(metadata_path)

    # -----------------------------------------------------------------
    # Named-sheet datasets: proteomics / placenta / metabolomics
    # -----------------------------------------------------------------
    if meta_type == "proteomics":
        sheet_name = "n=133 proteomics"
        if sample_id_col is None:
            sample_id_col = "sample Id"
    elif meta_type == "placenta":
        sheet_name = "n=133 placenta"
        if sample_id_col is None:
            sample_id_col = "ID"
    elif meta_type == "metabolomics":
        sheet_name = "n=133 metabolomics"
        if sample_id_col is None:
            sample_id_col = "Sample ID"
    else:
        raise ValueError(
            f"meta_type must be 'proteomics', 'placenta', 'metabolomics', or 'lipids', "
            f"got '{meta_type}'"
        )

    meta = pd.read_excel(metadata_path, sheet_name=sheet_name)

    required_cols = [sample_id_col, batch_col, "group", "subgroup", "gest age del"]

    if meta_type == "proteomics":
        if "sample gest Age" in meta.columns:
            required_cols.append("sample gest Age")
        else:
            logger.warning("'sample gest Age' column not found in proteomics sheet.")

    missing_cols = [col for col in required_cols if col not in meta.columns]
    if missing_cols:
        raise ValueError(
            f"Metadata missing required columns: {missing_cols}. "
            f"Available columns: {meta.columns.tolist()}"
        )

    meta_subset = meta[required_cols].copy()

    sid_clean = meta_subset[sample_id_col].astype(str).str.strip()
    meta_subset = meta_subset[
        meta_subset[sample_id_col].notna() & (sid_clean != "")
    ].copy()

    rename_dict = {
        sample_id_col: "SampleID",
        batch_col: "Batch",
        "group": "Group",
        "subgroup": "Subgroup",
        "gest age del": "GestAgeDelivery",
    }

    if meta_type == "proteomics" and "sample gest Age" in meta_subset.columns:
        rename_dict["sample gest Age"] = "SampleGestAge"

    meta_subset = meta_subset.rename(columns=rename_dict)

    meta_subset["SampleID"] = (
        meta_subset["SampleID"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    meta_subset = meta_subset.set_index("SampleID")
    meta_subset = meta_subset[~meta_subset.index.duplicated(keep="first")]

    # Standardise Group label capitalisation.
    if "Group" in meta_subset.columns:
        for old, canonical in _GROUP_LABEL_MAP.items():
            n = (meta_subset["Group"] == old).sum()
            if n > 0:
                meta_subset["Group"] = meta_subset["Group"].replace({old: canonical})
                logger.info(
                    "Group label fix: '%s' → '%s' (%d sample(s)) in %s metadata",
                    old, canonical, n, meta_type,
                )

    return meta_subset


def _load_lipids_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load subject-level metadata from the master table's "Sheet1".

    Uses positional column access (Sheet1 has no standardised named columns):
        col 0  → subject/sample ID   (DP3-XXXX or DP3-XXXXE for postpartum-only)
        col 5  → gest age del
        col 7  → group
        col 8  → subgroup
        col 16 → omics set# / batch

    Returns a DataFrame indexed by bare SubjectID (DP3-XXXX) — without the
    timepoint letter.  Postpartum-only IDs stored as "DP3-XXXXE" are stripped
    to "DP3-XXXX" so SampleIDs like "DP3-0029E" from the feature matrix can
    be joined by stripping the suffix.

    Columns: Batch, Group, Subgroup, GestAgeDelivery
    """
    logger.info("Loading lipids metadata from %s", metadata_path)
    wb = pd.read_excel(metadata_path, sheet_name="Sheet1", header=0, engine="openpyxl")
    rows = []
    n_suffixed = 0
    for _, row in wb.iterrows():
        subject_id = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else None
        if not subject_id or not subject_id.startswith("DP3-"):
            continue
        gest_age_del = row.iloc[5] if pd.notna(row.iloc[5]) else None
        group        = str(row.iloc[7]).strip() if pd.notna(row.iloc[7]) else None
        subgroup     = str(row.iloc[8]).strip() if pd.notna(row.iloc[8]) else None
        batch        = row.iloc[16] if pd.notna(row.iloc[16]) else None
        if group:
            for old, canonical in _GROUP_LABEL_MAP.items():
                group = group.replace(old, canonical)
        # Strip timepoint suffix from postpartum-only subjects ("DP3-0029E" → "DP3-0029")
        m = re.match(r"^(DP3-\d{4})[A-Ea-e]$", subject_id)
        if m:
            lookup_key = m.group(1)
            n_suffixed += 1
        else:
            lookup_key = subject_id
        rows.append({
            "SubjectID":       lookup_key,
            "Batch":           batch,
            "Group":           group,
            "Subgroup":        subgroup,
            "GestAgeDelivery": gest_age_del,
        })
    meta_df = pd.DataFrame(rows).set_index("SubjectID")
    meta_df = meta_df[~meta_df.index.duplicated(keep="first")]
    logger.info(
        "Loaded lipids metadata for %d subjects (%d with timepoint suffix in ID column)",
        len(meta_df), n_suffixed,
    )
    return meta_df


# -----------------------------
# Shared analyte/group helpers
# (re-exported for use by 02_exploratory_analysis and 03_model_development)
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load a cleaned wide-format CSV (index=SampleID, columns=metadata+analytes)."""
    return pd.read_csv(path, index_col=0)


def get_analyte_columns(df: pd.DataFrame) -> list:
    """Return analyte column names by excluding known metadata columns."""
    return [c for c in df.columns if c not in METADATA_COLS]


def normalise_group_labels(
    df: pd.DataFrame,
    group_col: str = "Group",
) -> pd.DataFrame:
    """Standardise group label capitalisation using _GROUP_LABEL_MAP (returns df for chaining)."""
    if group_col in df.columns:
        before = df[group_col].value_counts()
        df = df.copy()
        df[group_col] = df[group_col].replace(_GROUP_LABEL_MAP)
        for old, canonical in _GROUP_LABEL_MAP.items():
            n = before.get(old, 0)
            if n > 0:
                logger.info("Group label fix: '%s' → '%s' (%d sample(s))", old, canonical, n)
    return df


# -----------------------------
# Standardize missing + QC mask
# -----------------------------
def standardize_missing_npx(df: pd.DataFrame) -> pd.DataFrame:
    """Convert various missing value representations to NaN."""
    df = df.copy()
    if "NPX" not in df.columns:
        return df

    s = df["NPX"]
    if s.dtype == object:
        s = s.replace(["", " ", "NA", "N/A", "na", "n/a", "NaN", "nan", None], np.nan)
    s = pd.to_numeric(s, errors="coerce")
    s = s.mask(s == 0, np.nan)
    df["NPX"] = s
    return df


def qc_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Set NPX to NaN for samples that fail QC warnings."""
    df = df.copy()
    if "NPX" not in df.columns:
        return df

    if "QC_Warning" in df.columns:
        bad = df["QC_Warning"].astype(str).str.upper() != "PASS"
        df.loc[bad, "NPX"] = np.nan

    if "Assay_Warning" in df.columns:
        bad = df["Assay_Warning"].astype(str).str.upper() != "PASS"
        df.loc[bad, "NPX"] = np.nan

    return df


def process_single_file(file_input: str) -> pd.DataFrame:
    """
    Process a single Olink CSV file:
    1. Load data (auto-detect separator for robustness)
    2. Standardize missing values
    3. Apply QC masking
    4. Keep SampleID as-is (including letter suffixes for longitudinal samples)
    """
    df = pd.read_csv(file_input, sep=None, engine="python")
    df = standardize_missing_npx(df)
    df = qc_mask(df)

    if "SampleID" in df.columns:
        df["SampleID"] = (
            df["SampleID"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    return df


def combine_batches(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate multiple long-format dataframes."""
    return pd.concat(df_list, axis=0, ignore_index=True)


def is_olink_control_sample(df_long: pd.DataFrame) -> pd.Series:
    """Identify Olink technical control samples from SampleID prefixes."""
    sid = df_long["SampleID"].astype(str).str.strip().str.upper()
    return sid.str.startswith(CONTROL_SAMPLE_PREFIXES)


def is_olink_control_assay(df_long: pd.DataFrame) -> pd.Series:
    """Identify technical control assays (e.g., incubation/amplification controls)."""
    assay = df_long["Assay"].astype(str).str.strip().str.lower()
    return assay.str.contains(r"\bcontrol\b", regex=True)


# -----------------------------
# Panel normalization (long)
# -----------------------------
def apply_panel_normalization_long(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Panel normalization using Olink internal control samples.

    For each assay, computes the per-panel control median, then shifts
    all panel NPX values by (global_median - panel_median).
"""
    df = df_long.copy()
    required = {"SampleID", "Panel", "Assay", "NPX"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Panel normalization requires columns: {sorted(required)}; "
            f"missing: {sorted(missing)}"
        )

    ctrl = is_olink_control_sample(df)

    panel_medians = (
        df.loc[ctrl]
        .groupby(["Panel", "Assay"])["NPX"]
        .median()
        .rename("panel_median")
        .reset_index()
    )
    if panel_medians.empty:
        raise ValueError(
            "No Olink control samples found "
            f"(SampleID startswith one of {CONTROL_SAMPLE_PREFIXES})."
        )

    global_ref = (
        panel_medians.groupby("Assay")["panel_median"]
        .median()
        .rename("global_median")
        .reset_index()
    )

    adj = panel_medians.merge(global_ref, on="Assay", how="left")
    adj["adjustment"] = adj["global_median"] - adj["panel_median"]

    df = df.merge(adj[["Panel", "Assay", "adjustment"]], on=["Panel", "Assay"], how="left")
    df["adjustment"] = df["adjustment"].fillna(0.0)
    df["NPX"] = df["NPX"] + df["adjustment"]
    df = df.drop(columns=["adjustment"])

    return df


# -----------------------------
# Missingness + group check
# -----------------------------
def benjamini_hochberg_rejections(pvals: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Apply Benjamini-Hochberg FDR correction via statsmodels multipletests."""
    pvals = pvals.dropna().astype(float).clip(0, 1)
    if pvals.empty:
        return pd.Series(dtype=bool)

    rejected, _, _, _ = multipletests(pvals.values, alpha=alpha, method="fdr_bh")
    return pd.Series(rejected, index=pvals.index, dtype=bool)


def missingness_filter_and_group_check(
    X: pd.DataFrame,
    groups: pd.Series | None,
    cutoff: float = CUTOFF_PERCENT_MISSING,
    alpha_bh: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter out assays with high missingness and test for group imbalance.
    """
    miss_frac = X.isna().mean(axis=0)
    keep = miss_frac[miss_frac < cutoff].index
    drop = miss_frac[miss_frac >= cutoff].index

    X_kept = X.loc[:, keep].copy()

    if len(drop) == 0:
        return X_kept, pd.DataFrame(columns=["Assay", "missing_frac", "p_value", "bh_reject"])

    report_rows = []
    pvals = {}

    if groups is None:
        for assay in drop:
            report_rows.append(
                {
                    "Assay": assay,
                    "missing_frac": float(miss_frac[assay]),
                    "p_value": np.nan,
                    "bh_reject": False,
                }
            )
        return X_kept, pd.DataFrame(report_rows).sort_values("missing_frac", ascending=False)

    g = groups.reindex(X.index)
    valid = g.notna()
    g = g[valid]
    Xg = X.loc[valid, drop]

    uniq = pd.unique(g)
    if len(uniq) != 2:
        logger.warning(
            "Expected 2 groups for Fisher's test, but got %d: %s. "
            "This may be because samples didn't match metadata properly. "
            "Skipping group-wise missingness testing.",
            len(uniq),
            uniq,
        )
        for assay in drop:
            report_rows.append(
                {
                    "Assay": assay,
                    "missing_frac": float(miss_frac[assay]),
                    "p_value": np.nan,
                    "bh_reject": False,
                }
            )
        return X_kept, pd.DataFrame(report_rows).sort_values("missing_frac", ascending=False)

    g0, g1 = uniq[0], uniq[1]

    for assay in drop:
        miss = Xg[assay].isna()
        a = int(((g == g0) & miss).sum())
        b = int(((g == g0) & (~miss)).sum())
        c = int(((g == g1) & miss).sum())
        d = int(((g == g1) & (~miss)).sum())

        if (a + b == 0) or (c + d == 0):
            p = 1.0
        else:
            _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")

        pvals[assay] = p
        report_rows.append(
            {
                "Assay": assay,
                "missing_frac": float(miss_frac[assay]),
                f"{g0}_missing": a,
                f"{g0}_observed": b,
                f"{g1}_missing": c,
                f"{g1}_observed": d,
                "p_value": p,
            }
        )

    report = pd.DataFrame(report_rows).set_index("Assay")
    reject = benjamini_hochberg_rejections(pd.Series(pvals), alpha=alpha_bh)
    report["bh_reject"] = reject.reindex(report.index).fillna(False).astype(bool)
    report = report.sort_values(["bh_reject", "p_value"], ascending=[False, True]).reset_index()

    return X_kept, report


# -----------------------------
# Normalization (wide)
# -----------------------------
def combat_normalize_wide(X: pd.DataFrame, batch_labels: pd.Series) -> pd.DataFrame:
    """
    Apply ComBat batch normalization using Python pycombat.
    Preserves original missingness pattern after correction.
    """
    b = batch_labels.reindex(X.index)
    if b.isna().any():
        raise ValueError("ComBat requires non-missing batch labels for all samples.")

    if b.nunique() < 2:
        return X.copy()

    missing_mask = X.isna()

    X_filled = X.copy()
    med = X_filled.median(axis=0, skipna=True)
    X_filled = X_filled.fillna(med).fillna(0.0)

    model = Combat()
    # pycombat can error with pandas slicing internals; pass ndarray explicitly.
    corrected = model.fit_transform(X_filled.to_numpy(dtype=float), b.values)
    Xc = pd.DataFrame(corrected, index=X.index, columns=X.columns)

    Xc = Xc.mask(missing_mask)
    return Xc


# -----------------------------
# Imputation
# -----------------------------
def half_min_impute_wide(X: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values on log2 scale using (minimum observed - 1) per assay.

    Subtracting 1 in log2 space is equivalent to halving in linear space,
    placing imputed values just below the minimum detected concentration.
    """
    Xi = X.copy()
    for col in Xi.columns:
        s = Xi[col]
        if s.notna().any():
            Xi[col] = s.fillna(s.min(skipna=True) - 1.0)
    return Xi
