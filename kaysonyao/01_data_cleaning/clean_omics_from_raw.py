"""
clean_omics_from_raw.py
=======================
Full preprocessing pipeline for untargeted metabolomics (MTBL) and
lipidomics (LIPD) data, starting from the raw extracted CSVs produced by
Kayla Xu's MTBL_extraction.py.

Follows the Kayla Xu pipeline (README.md) step-by-step, with ComBat
batch correction added after biological-replicate normalization (as
directed by the Apr 1 notes).

Datasets processed
------------------
  MTBL_plasma   – untargeted polar metabolomics, maternal plasma (3 batches)
  MTBL_placenta – untargeted polar metabolomics, placental tissue (2 batches)
  LIPD_plasma   – untargeted lipidomics, maternal plasma (3 batches)
  LIPD_placenta – untargeted lipidomics, placental tissue (2 batches)

Pipeline (README step numbers preserved)
-----------------------------------------
  1.  Convert 0 / blank / "NA" → np.nan (standard missing value).
  2.  Separate QC pools from biological samples; split both by batch.
  3.  MAD filter: drop biological samples whose internal-standard value
      (c1 OR c2) is outside median ± 5 × MAD.  Log every dropped sample.
  4.  Sample missingness filter: drop samples with > 50% missing analytes.
  5.  RSD filter: drop analytes with RSD > 30% in any QC-pool batch.
  6.  Analyte missingness filter: drop analytes with > 20% missing across
      biological samples in any batch.  Run Fisher exact + BH FDR test on
      dropped analytes to flag differential missingness by group.
  7.  PCA plot (pre-normalization), colored by batch.
  8.  Biological-replicate batch normalization (ratio method):
        a. Identify samples present in ≥ 2 batches (cross-batch replicates).
        b. For each analyte: compute ratio = non-reference / reference.
           Exclude outlier ratios (> 5 MAD from the median ratio).
        c. Correction factor = 1 / median ratio per analyte.
        d. Multiply all samples in the non-reference batch by the CF.
  9.  PCA plot (post-normalization), colored by batch.
  10. Average cross-batch replicates → one row per unique sample.
  12. Log2(x + 1) transformation.
  [C] ComBat batch correction on the log2 matrix.
  [C] PCA plot pre- and post-ComBat.
  13. Combine POS + NEG:
        a. Named compounds present in both polarities → keep the polarity
           with the higher Area (Max.) from compound metadata.
        b. Unnamed compounds with matching neutral mass (±10 ppm) and RT
           (±0.3 min) across polarities → keep the higher-Area polarity.
        c. Quality-score deduplication for named compounds (within each
           polarity, keep the best Export Order for each compound name).
        d. Surviving unnamed analytes keep their Export Order column name
           (e.g., p3823_POS / n1690_NEG).
  14. Half-minimum imputation: NaN → min_log2_observed − 1 per analyte.
  16. Attach metadata (Group, Subgroup, Batch, GestAgeDelivery,
      SampleGestAge [plasma only]).
  17. Rename Export Order columns to compound names where available.
  18. Split plasma output by timepoint (A–E); placenta is a single matrix.

Input
-----
  kaylaxu_dir/data/{MTBL,LIPD}_{plasma,placenta}/
      pos_expression.csv, pos_batch.csv, pos_compounds.csv
      neg_expression.csv, neg_batch.csv, neg_compounds.csv
  data/dp3 master table v2.xlsx
      sheet "n=133 metabolomics"  (plasma, Sample ID = "DP3-XXXX T")
      sheet "n=133 placenta"      (placenta, ID = "DP3-XXXX")

Output
------
  data/cleaned/omics_pipeline/{datatype}_{tissue}/
      {datatype}_{tissue}_cleaned_with_metadata.csv
      {datatype}_{tissue}_formatted_suffix_{tp}.csv  (plasma only)
      pca_pre_normalization.png
      pca_post_normalization.png
      pca_post_combat.png
      qc_failures.csv
      dropped_analytes_report.csv

Usage
-----
  python 01_data_cleaning/clean_omics_from_raw.py

  python 01_data_cleaning/clean_omics_from_raw.py \\
      --kaylaxu-dir  ../../kaylaxu \\
      --metadata     data/dp3 master table v2.xlsx \\
      --output-dir   data/cleaned/omics_pipeline
"""

import argparse
import logging
import os
import re
import sys
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global QC thresholds (mirror Kayla Xu MTBL_datacleaning.py)
# ---------------------------------------------------------------------------
MAD_THRESHOLD      = 5.0   # step 3: internal-std outlier cutoff
SAMPLE_MISSING_PCT = 0.50  # step 4: max fraction of missing per sample
RSD_THRESHOLD      = 30.0  # step 5: max % RSD in QC pools
ANALYTE_MISS_PCT   = 0.20  # step 6: max fraction of missing per analyte
BH_ALPHA           = 0.05  # step 6: FDR threshold for differential missingness

# Tolerances for cross-mode unnamed deduplication (step 13b)
MZ_PPM_TOL = 10.0   # ±10 ppm neutral mass
RT_MIN_TOL  = 0.30  # ±0.30 min retention time
H_MASS      = 1.007276  # proton mass (Da)

VALID_TIMEPOINTS = set("ABCDE")


# ===========================================================================
# STEP 1 — Convert missing / zero to NaN
# ===========================================================================

def _to_float(x):
    """Convert a single cell to float, returning NaN for 0 or non-numeric."""
    try:
        val = float(x)
        return np.nan if val == 0 else val
    except (TypeError, ValueError):
        return np.nan


def convert_missing(exp: pd.DataFrame) -> pd.DataFrame:
    """Step 1: replace 0 / blank / 'NA' with np.nan."""
    return exp.map(_to_float)


# ===========================================================================
# STEP 2 — Separate QC pools from biological samples; split by batch
# ===========================================================================

def _normalise_sample_id(sid: str) -> str:
    """Strip internal whitespace between subject number and timepoint letter."""
    return re.sub(r"\s+", "", str(sid).strip())


def split_by_batch(
    exp: pd.DataFrame,
    batch: pd.DataFrame,
) -> tuple[dict, dict]:
    """
    Step 2: separate QC (Pooled) from biological samples, then split both
    by batch.

    Returns
    -------
    samples_dict : {batch_key: DataFrame}   biological samples
    pooled_dict  : {batch_key: DataFrame}   QC pool samples
    """
    # Normalise sample IDs in expression index
    exp = exp.copy()
    exp.index = [_normalise_sample_id(s) for s in exp.index]
    batch = batch.copy()
    batch.index = [_normalise_sample_id(s) for s in batch.index]

    is_pooled = exp.index.str.startswith("Pooled")
    pooled_exp = exp[is_pooled]
    sample_exp = exp[~is_pooled]

    pooled_batch = batch.loc[is_pooled, "batch"]
    sample_batch = batch.loc[~is_pooled, "batch"]

    unique_batches = sorted(batch["batch"].unique().tolist())

    samples_dict: dict = {}
    pooled_dict:  dict = {}

    for b in unique_batches:
        bkey = str(b)
        samples_dict[bkey] = sample_exp.loc[sample_batch == b].copy()
        pooled_dict[bkey]  = pooled_exp.loc[pooled_batch == b].copy()

    return samples_dict, pooled_dict, unique_batches


# ===========================================================================
# STEP 3 — MAD filter on internal standards
# ===========================================================================

def mad_filter_samples(
    samples_dict: dict,
    std_cols: list,
    qc_failures: list,
    label: str,
) -> dict:
    """
    Step 3: For each batch, flag samples where any internal-standard column
    is outside median ± 5 × MAD.  Remove flagged samples from ALL batches
    (instrument failure affects the sample, not just that batch).

    Parameters
    ----------
    std_cols : list of internal-standard column names (e.g. ["c1","c2"])
    qc_failures : list to append failure records to (in-place)
    label : dataset label for logging
    """
    failed_samples: set = set()

    for bkey, df in samples_dict.items():
        for col in std_cols:
            if col not in df.columns:
                continue
            vals = df[col].dropna()
            if len(vals) < 3:
                continue
            med = vals.median()
            mad = stats.median_abs_deviation(vals, nan_policy="omit")
            if mad == 0:
                continue
            lo, hi = med - MAD_THRESHOLD * mad, med + MAD_THRESHOLD * mad
            fail_mask = (df[col] < lo) | (df[col] > hi)
            for sid in df.index[fail_mask]:
                if sid not in failed_samples:
                    logger.info(
                        "[%s] Step 3 MAD fail: %s in batch %s (std=%s, value=%.3e)",
                        label, sid, bkey, col, df.at[sid, col],
                    )
                    qc_failures.append({
                        "step": 3, "sample": sid, "batch": bkey,
                        "reason": f"internal-std {col} outside 5×MAD",
                    })
                failed_samples.add(sid)

    # Drop from all batches
    for bkey in samples_dict:
        before = len(samples_dict[bkey])
        samples_dict[bkey] = samples_dict[bkey].drop(
            index=[s for s in failed_samples if s in samples_dict[bkey].index]
        )
        n_dropped = before - len(samples_dict[bkey])
        if n_dropped:
            logger.info(
                "[%s] Step 3: dropped %d samples from batch %s (MAD fail).",
                label, n_dropped, bkey,
            )

    return samples_dict


# ===========================================================================
# STEP 4 — Sample missingness filter (> 50% missing)
# ===========================================================================

def sample_missingness_filter(
    samples_dict: dict,
    qc_failures: list,
    label: str,
) -> dict:
    """Step 4: drop samples with > 50% missing analytes in any batch."""
    for bkey, df in samples_dict.items():
        # Only count analyte columns (exclude internal standards and metadata)
        analyte_cols = [c for c in df.columns if not (c.startswith("c") and c[1:].isdigit())]
        miss_rate = df[analyte_cols].isna().sum(axis=1) / max(len(analyte_cols), 1)
        fail_mask = miss_rate > SAMPLE_MISSING_PCT
        for sid in df.index[fail_mask]:
            logger.info(
                "[%s] Step 4: dropping %s in batch %s (%.1f%% missing).",
                label, sid, bkey, miss_rate[sid] * 100,
            )
            qc_failures.append({
                "step": 4, "sample": sid, "batch": bkey,
                "reason": f">50% missingness ({miss_rate[sid]*100:.1f}%)",
            })
        samples_dict[bkey] = df[~fail_mask]

    return samples_dict


# ===========================================================================
# STEP 5 — RSD filter on QC pools (> 30% → drop analyte)
# ===========================================================================

def rsd_filter_analytes(
    samples_dict: dict,
    pooled_dict: dict,
    label: str,
) -> tuple[dict, dict]:
    """
    Step 5: compute RSD = SD / Mean × 100 in each batch's QC pools.
    Drop analytes with RSD > 30% in ANY batch from both samples and pools.
    """
    # Analyte columns = all columns that are not internal standards or batch
    all_analytes: set = set()
    for df in pooled_dict.values():
        all_analytes.update(df.columns)
    std_pattern = re.compile(r"^c\d+$")
    analyte_cols = [c for c in all_analytes if not std_pattern.match(c)]

    failed_analytes: set = set()

    for bkey, pool_df in pooled_dict.items():
        for col in analyte_cols:
            if col not in pool_df.columns:
                continue
            vals = pool_df[col].dropna()
            if len(vals) < 2:
                continue
            mean_val = vals.mean()
            if mean_val == 0 or np.isnan(mean_val):
                continue
            rsd = (vals.std() / mean_val) * 100
            if rsd > RSD_THRESHOLD:
                if col not in failed_analytes:
                    logger.info(
                        "[%s] Step 5: analyte %s dropped (RSD=%.1f%% in batch %s).",
                        label, col, rsd, bkey,
                    )
                failed_analytes.add(col)

    # Drop failed analytes from samples and pools
    for bkey in samples_dict:
        samples_dict[bkey] = samples_dict[bkey].drop(
            columns=[c for c in failed_analytes if c in samples_dict[bkey].columns],
            errors="ignore",
        )
    for bkey in pooled_dict:
        pooled_dict[bkey] = pooled_dict[bkey].drop(
            columns=[c for c in failed_analytes if c in pooled_dict[bkey].columns],
            errors="ignore",
        )

    logger.info("[%s] Step 5: dropped %d analytes (RSD > 30%%).", label, len(failed_analytes))
    return samples_dict, pooled_dict


# ===========================================================================
# STEP 6 — Analyte missingness filter (> 20%) + differential check
# ===========================================================================

def analyte_missingness_filter(
    samples_dict: dict,
    groups: dict,
    label: str,
) -> tuple[dict, pd.DataFrame]:
    """
    Step 6: drop analytes with > 20% missing in any batch.
    Run Fisher exact + BH FDR on dropped analytes to detect differential
    missingness between Control and Complication.

    groups : dict {sample_id: group_label}
    Returns (filtered samples_dict, dropped_analytes report DataFrame)
    """
    # Collect analyte columns across all batches
    std_pattern = re.compile(r"^c\d+$")
    all_cols: set = set()
    for df in samples_dict.values():
        all_cols.update(c for c in df.columns if not std_pattern.match(c))

    failed_analytes: set = set()
    miss_rates: dict = {}

    for bkey, df in samples_dict.items():
        n = len(df)
        if n == 0:
            continue
        for col in all_cols:
            if col not in df.columns:
                continue
            miss_frac = df[col].isna().sum() / n
            if miss_frac > ANALYTE_MISS_PCT:
                failed_analytes.add(col)
                miss_rates[col] = miss_rates.get(col, {})
                miss_rates[col][bkey] = miss_frac

    # Differential missingness: Fisher exact per dropped analyte
    report_rows = []
    # Build combined group-coded df for Fisher test
    combined = pd.concat(list(samples_dict.values()))
    combined["_group"] = combined.index.map(
        lambda s: "Control" if groups.get(s, "") == "Control" else "Complication"
    )

    for col in failed_analytes:
        if col not in combined.columns:
            report_rows.append({"analyte": col, "reason": ">20% missing", "fisher_p": np.nan, "bh_q": np.nan})
            continue
        is_missing = combined[col].isna()
        ctrl_miss   = (is_missing & (combined["_group"] == "Control")).sum()
        ctrl_obs    = (~is_missing & (combined["_group"] == "Control")).sum()
        comp_miss   = (is_missing & (combined["_group"] == "Complication")).sum()
        comp_obs    = (~is_missing & (combined["_group"] == "Complication")).sum()
        contingency = [[ctrl_miss, ctrl_obs], [comp_miss, comp_obs]]
        try:
            _, fisher_p = stats.fisher_exact(contingency)
        except Exception:
            fisher_p = np.nan
        report_rows.append({
            "analyte": col,
            "reason": ">20% missing",
            "miss_rates_by_batch": str(miss_rates.get(col, {})),
            "ctrl_missing": ctrl_miss, "ctrl_observed": ctrl_obs,
            "comp_missing": comp_miss, "comp_observed": comp_obs,
            "fisher_p": fisher_p,
            "bh_q": np.nan,
        })

    report_df = pd.DataFrame(report_rows)
    if len(report_df) and "fisher_p" in report_df.columns:
        valid_p = report_df["fisher_p"].notna()
        if valid_p.sum() > 0:
            _, q_vals, _, _ = multipletests(
                report_df.loc[valid_p, "fisher_p"], method="fdr_bh"
            )
            report_df.loc[valid_p, "bh_q"] = q_vals
            n_diff = (q_vals < BH_ALPHA).sum()
            if n_diff:
                logger.warning(
                    "[%s] Step 6: %d dropped analytes show differential missingness "
                    "(BH q < %.2f) — review dropped_analytes_report.csv.",
                    label, n_diff, BH_ALPHA,
                )

    # Drop from samples
    for bkey in samples_dict:
        samples_dict[bkey] = samples_dict[bkey].drop(
            columns=[c for c in failed_analytes if c in samples_dict[bkey].columns],
            errors="ignore",
        )
    logger.info("[%s] Step 6: dropped %d analytes (>20%% missing).", label, len(failed_analytes))
    return samples_dict, report_df


# ===========================================================================
# STEP 7 / 9 / [C] — PCA plots
# ===========================================================================

def generate_pca(
    samples_dict: dict,
    out_path: str,
    title: str,
    color_by_batch: bool = True,
    batch_labels_override: pd.Series = None,
) -> None:
    """Combine all batch DataFrames and generate a 2-D PCA plot."""
    frames = []
    for bkey, df in samples_dict.items():
        tmp = df.copy()
        tmp["_batch"] = bkey
        frames.append(tmp)
    if not frames:
        return
    combined = pd.concat(frames)
    batch_col = combined["_batch"]
    numeric = combined.drop(columns=["_batch"], errors="ignore").select_dtypes(include=np.number)

    if batch_labels_override is not None:
        batch_col = batch_labels_override.reindex(numeric.index).fillna("unknown")

    # Fill NaN with column median for PCA
    numeric = numeric.apply(lambda col: col.fillna(col.median(skipna=True)), axis=0)
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[0] < 3 or numeric.shape[1] < 2:
        logger.warning("[PCA] Not enough data for %s — skipping.", title)
        return

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric.values)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(coords, index=numeric.index, columns=["PC1", "PC2"])
    pca_df["batch"] = batch_col.reindex(pca_df.index).values

    fig, ax = plt.subplots(figsize=(8, 6))
    for b in sorted(pca_df["batch"].dropna().unique()):
        sub = pca_df[pca_df["batch"] == b]
        ax.scatter(sub["PC1"], sub["PC2"], label=str(b), alpha=0.7, s=50)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("PCA plot saved → %s", out_path)


def generate_pca_pre_post(
    X_pre: pd.DataFrame,
    X_post: pd.DataFrame,
    batch_labels: pd.Series,
    out_path: str,
    title: str,
) -> None:
    """Side-by-side pre/post ComBat PCA."""
    shared = X_pre.index.intersection(X_post.index)
    if len(shared) < 3:
        return

    def _pca2d(X):
        Xp = X.loc[shared].apply(lambda c: c.fillna(c.median(skipna=True)), axis=0)
        Xp = Xp.dropna(axis=1, how="all")
        if Xp.shape[0] < 3 or Xp.shape[1] < 2:
            return None, None
        sc = StandardScaler()
        pca = PCA(n_components=2)
        coords = pca.fit_transform(sc.fit_transform(Xp.values))
        return pd.DataFrame(coords, index=Xp.index, columns=["PC1", "PC2"]), pca.explained_variance_ratio_

    pre_c, pre_v   = _pca2d(X_pre)
    post_c, post_v = _pca2d(X_post)
    if pre_c is None or post_c is None:
        return

    labs = batch_labels.reindex(shared).astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, coords, var, subtitle in [
        (axes[0], pre_c,  pre_v,  "Pre-ComBat"),
        (axes[1], post_c, post_v, "Post-ComBat"),
    ]:
        for b in sorted(labs.unique()):
            m = labs == b
            ax.scatter(coords.loc[m, "PC1"], coords.loc[m, "PC2"], label=b, alpha=0.7, s=50)
        ax.set_title(f"{subtitle} — colored by batch")
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Pre/post ComBat PCA → %s", out_path)


# ===========================================================================
# STEP 8 — Biological-replicate batch normalization (ratio method)
# ===========================================================================

def _correction_factor(rep_ref: pd.DataFrame, rep_other: pd.DataFrame) -> dict:
    """
    Compute per-analyte correction factor using median ratio method.
    Outlier ratios (> 5 MAD from median) are excluded before taking the median.
    Returns {analyte: correction_factor} where CF = 1 / median_ratio.
    """
    ratios = rep_other / rep_ref
    ratios = ratios.dropna(how="all")
    if ratios.empty:
        return {}

    cf = {}
    for col in ratios.columns:
        col_ratios = ratios[col].dropna()
        if len(col_ratios) < 1:
            continue
        mad = stats.median_abs_deviation(col_ratios, nan_policy="omit")
        med = col_ratios.median()
        if mad > 0:
            col_ratios = col_ratios[(col_ratios > med - 5 * mad) & (col_ratios < med + 5 * mad)]
        med_ratio = col_ratios.median()
        if np.isfinite(med_ratio) and med_ratio != 0:
            cf[col] = 1.0 / med_ratio
    return cf


def batch_normalize(
    samples_dict: dict,
    unique_batches: list,
    ref_batch: str,
    label: str,
) -> dict:
    """
    Step 8: biological-replicate batch normalization.
    Reference batch stays unchanged; all other batches are corrected.
    """
    unique_batches_str = [str(b) for b in unique_batches]
    ref = str(ref_batch)

    # Collect all sample IDs across batches
    all_ids: dict[str, list] = defaultdict(list)
    for bkey in unique_batches_str:
        if bkey not in samples_dict:
            continue
        for sid in samples_dict[bkey].index:
            all_ids[sid].append(bkey)

    # Identify cross-batch replicates (sample in ≥ 2 batches)
    replicates = {sid for sid, batches in all_ids.items() if len(batches) > 1}
    logger.info(
        "[%s] Step 8: %d cross-batch replicates found (reference batch = %s).",
        label, len(replicates), ref,
    )

    if not replicates or ref not in samples_dict:
        logger.warning("[%s] Step 8: skipping normalization (no replicates or missing ref batch).", label)
        return samples_dict

    ref_df = samples_dict[ref]
    ref_rep = ref_df.loc[[s for s in replicates if s in ref_df.index]].select_dtypes(include=np.number)

    for bkey in unique_batches_str:
        if bkey == ref or bkey not in samples_dict:
            continue
        other_df = samples_dict[bkey]
        shared = [s for s in replicates if s in other_df.index]
        if not shared:
            logger.warning("[%s] Step 8: no shared replicates with batch %s.", label, bkey)
            continue

        other_rep = other_df.loc[shared].select_dtypes(include=np.number)
        # Align columns
        common_cols = ref_rep.columns.intersection(other_rep.columns)
        cf = _correction_factor(ref_rep[common_cols], other_rep[common_cols])

        corrected = other_df.copy()
        for col, factor in cf.items():
            if col in corrected.columns:
                corrected[col] = corrected[col] * factor
        samples_dict[bkey] = corrected
        logger.info(
            "[%s] Step 8: batch %s corrected (%d analyte CFs applied).",
            label, bkey, len(cf),
        )

    return samples_dict


# ===========================================================================
# STEP 10 — Average cross-batch replicates
# ===========================================================================

def average_replicates(samples_dict: dict, label: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Step 10: combine all batches into one DataFrame, then average rows that
    share the same sample ID (cross-batch replicates).

    Returns (combined_df, batch_series) where batch_series maps each
    unique sample ID to its primary (reference) batch.
    """
    frames = []
    batch_records = {}

    for bkey, df in samples_dict.items():
        tmp = df.copy()
        tmp["_batch"] = bkey
        frames.append(tmp)
        for sid in df.index:
            if sid not in batch_records:
                batch_records[sid] = bkey   # first seen = primary batch

    if not frames:
        return pd.DataFrame(), pd.Series(dtype=str)

    combined = pd.concat(frames)
    batch_col = combined.pop("_batch")

    # Numeric analyte columns
    meta_pattern = re.compile(r"^c\d+$")
    analyte_cols = [c for c in combined.columns if not meta_pattern.match(c)]

    # Average duplicates
    numeric = combined[analyte_cols].groupby(level=0).mean()
    batch_series = pd.Series({sid: batch_records[sid] for sid in numeric.index}, name="Batch")

    logger.info(
        "[%s] Step 10: %d combined rows → %d unique samples (averaged duplicates).",
        label, len(combined), len(numeric),
    )
    return numeric, batch_series


# ===========================================================================
# STEP 12 — Log2(x + 1) transformation
# ===========================================================================

def log2_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Step 12: apply log2(x + 1) to all values."""
    return pd.DataFrame(
        np.log2(df.values.astype(float) + 1.0),
        index=df.index, columns=df.columns,
    )


# ===========================================================================
# STEP [C] — ComBat batch correction
# ===========================================================================

def combat_correct(
    df_log2: pd.DataFrame,
    batch_labels: pd.Series,
    label: str,
) -> pd.DataFrame:
    """
    Step [C]: apply ComBat on the log2 matrix.
    Uses pyComBat (inmoose package) or falls back to neuroComBat.
    """
    try:
        from inmoose.pycombat import pycombat_norm
        batch_list = batch_labels.reindex(df_log2.index).fillna("unknown").tolist()
        corrected = pycombat_norm(df_log2.T, batch_list).T
        logger.info("[%s] ComBat correction applied (inmoose.pycombat).", label)
        return corrected
    except ImportError:
        pass

    try:
        import combat
        batch_list = batch_labels.reindex(df_log2.index).fillna("unknown").tolist()
        corrected_arr = combat.combat(df_log2.values.T, batch_list)
        corrected = pd.DataFrame(corrected_arr.T, index=df_log2.index, columns=df_log2.columns)
        logger.info("[%s] ComBat correction applied (combat package).", label)
        return corrected
    except (ImportError, Exception):
        pass

    # Fallback: use the existing utilities.py combat function if available
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, here)
        from utilities import combat_normalize_wide
        corrected = combat_normalize_wide(df_log2, batch_labels)
        logger.info("[%s] ComBat correction applied (utilities.combat_normalize_wide).", label)
        return corrected
    except Exception as e:
        logger.error("[%s] ComBat failed (%s). Returning uncorrected log2 data.", label, e)
        return df_log2


# ===========================================================================
# STEP 13 — Quality score + combine POS + NEG
# ===========================================================================

def _is_lipid_comp(comp: pd.DataFrame) -> bool:
    """Return True if the compound DataFrame uses LIPD-style columns (LipidID)."""
    return "LipidID" in comp.columns and "Name" not in comp.columns


def _quality_score(comp: pd.DataFrame) -> pd.Series:
    """
    Composite quality score (Kayla Xu MTBL_datacleaning.py qs() function).
    Returns all-zero series for LIPD compound tables (which lack QS columns).
    """
    if _is_lipid_comp(comp):
        return pd.Series(0.0, index=comp.index)

    try:
        peak = pd.Series(
            [10 if x >= 7.0 else 7 if x >= 5 else 4 if x >= 3 else 1
             for x in comp["Peak Rating (Max.)"].fillna(0)],
            index=comp.index,
        )
    except (KeyError, TypeError):
        peak = pd.Series(1, index=comp.index)

    try:
        rsd = pd.Series(
            [10 if x < 10 else 8 if x < 15 else 6 if x < 20
             else 4 if x < 25 else 2 if x < 30 else 0
             for x in comp["RSD QC Areas [%]"]],
            index=comp.index,
        )
    except (KeyError, TypeError):
        rsd = pd.Series(0, index=comp.index)

    try:
        ms2 = pd.Series(
            [10 if x == "DDA for preferred ion"
             else 6 if x == "DDA for other ion"
             else 4 if x == "DDA available"
             else 0
             for x in comp["MS2"].fillna("")],
            index=comp.index,
        )
    except (KeyError, TypeError):
        ms2 = pd.Series(0, index=comp.index)

    try:
        area = comp["Area (Max.)"].fillna(0).values.reshape(-1, 1).astype(float)
        if area.max() > 0:
            signal = pd.Series(MinMaxScaler((0, 10)).fit_transform(area).flatten(), index=comp.index)
        else:
            signal = pd.Series(0.0, index=comp.index)
    except (KeyError, TypeError):
        signal = pd.Series(0.0, index=comp.index)

    annot_cols = [c for c in [
        "Annot. Source: Predicted Compositions", "Annot. Source: mzCloud Search",
        "Annot. Source: mzVault Search", "Annot. Source: Metabolika Search",
        "Annot. Source: ChemSpider Search", "Annot. Source: MassList Search",
    ] if c in comp.columns]
    if annot_cols:
        tmp = comp[annot_cols]
        full    = (tmp == "Full match").sum(axis=1)
        not_top = (tmp == "Not the top hit").sum(axis=1)
        partial = (tmp == "Partial match").sum(axis=1)
    else:
        full = not_top = partial = pd.Series(0, index=comp.index)

    try:
        mzCloud = comp["mzCloud Best Match Confidence"].fillna(-1)
    except (KeyError, TypeError):
        mzCloud = pd.Series(-1, index=comp.index)
    ac = []
    for i in range(len(comp)):
        mz = mzCloud.iloc[i]
        if mz >= 90:   ac.append(10)
        elif mz >= 80: ac.append(9)
        elif mz >= 70: ac.append(8)
        elif mz >= 0:  ac.append(0)
        else:
            f, nt, pt = full.iloc[i], not_top.iloc[i], partial.iloc[i]
            ac.append(10 if f==6 else 9 if f==5 else 8 if f==4 else 7 if f==3
                      else 6 if f==2 else 5 if f==1 else 4 if nt>=1
                      else 3 if pt>=3 else 2 if pt==2 else 1 if pt==1 else 0)
    return peak + rsd + ms2 + signal + pd.Series(ac, index=comp.index)


def combine_pos_neg(
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    pos_comp: pd.DataFrame,
    neg_comp: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """
    Step 13: combine POS and NEG expression matrices.

    Handles both MTBL (Name / m/z / RT [min] / Area (Max.) columns) and
    LIPD (LipidID / CalcMz / BaseRt columns) compound table formats.

    a. Named compounds: compute quality scores (MTBL) or use mean expression
       as proxy signal (LIPD); for each compound name keep the best Export
       Order within each polarity, then keep the higher-signal polarity.
    b. Unnamed compounds: match by neutral mass (±10 ppm) and RT (±0.3 min);
       keep the one with higher mean expression.
    c. All retained columns get a _POS or _NEG suffix.
    d. Compound names replace Export Order labels for named analytes.
    """
    is_lipid = _is_lipid_comp(pos_comp)

    # -- column name aliases for MTBL vs LIPD --
    name_col = "LipidID" if is_lipid else "Name"
    mz_col   = "CalcMz"   if is_lipid else "m/z"
    rt_col   = "BaseRt"   if is_lipid else "RT [min]"

    # -- compute quality scores --
    pos_comp = pos_comp.copy()
    neg_comp = neg_comp.copy()
    pos_comp["qs"] = _quality_score(pos_comp)
    neg_comp["qs"] = _quality_score(neg_comp)

    # For LIPD / any case without Area column: compute mean expression per
    # Export Order across all samples as the signal-intensity proxy.
    if is_lipid or "Area (Max.)" not in pos_comp.columns:
        def _mean_expr(exp_df, comp_idx):
            cols = [c for c in comp_idx if c in exp_df.columns]
            return exp_df[cols].mean(axis=0, skipna=True).rename("_mean_expr")
        pos_mean_expr = _mean_expr(pos_df, pos_comp.index)
        neg_mean_expr = _mean_expr(neg_df, neg_comp.index)
        pos_comp["_area"] = pos_comp.index.map(lambda x: pos_mean_expr.get(x, 0))
        neg_comp["_area"] = neg_comp.index.map(lambda x: neg_mean_expr.get(x, 0))
    else:
        pos_comp["_area"] = pd.to_numeric(pos_comp["Area (Max.)"], errors="coerce").fillna(0)
        neg_comp["_area"] = pd.to_numeric(neg_comp["Area (Max.)"], errors="coerce").fillna(0)

    # -- resolve names (blank / NaN → use Export Order as fallback) --
    def _resolve_name(comp_df):
        names = comp_df[name_col].fillna("").astype(str)
        names = names.replace("Not named", "").replace("nan", "")
        blank = names == ""
        names = names.copy()
        names[blank] = comp_df.index[blank].astype(str)
        return names, ~blank   # (name_series, annotated_mask)

    pos_names, pos_ann = _resolve_name(pos_comp)
    neg_names, neg_ann = _resolve_name(neg_comp)

    # ------------------------------------------------------------------
    # a. Named compound deduplication + cross-polarity selection
    # ------------------------------------------------------------------
    keep_pos: set = set()
    keep_neg: set = set()
    rename_map: dict = {}   # old col name → new col name

    all_named = set(pos_names[pos_ann]) | set(neg_names[neg_ann])

    for name in all_named:
        # Best within POS (highest QS, tie-break by _area)
        pos_candidates = pos_comp[pos_names == name]
        if len(pos_candidates):
            best_pos = pos_candidates.sort_values(
                ["qs", "_area"], ascending=False
            ).iloc[0]
        else:
            best_pos = None

        # Best within NEG
        neg_candidates = neg_comp[neg_names == name]
        if len(neg_candidates):
            best_neg = neg_candidates.sort_values(
                ["qs", "_area"], ascending=False
            ).iloc[0]
        else:
            best_neg = None

        if best_pos is not None and best_neg is not None:
            # Cross-polarity: keep the one with higher signal (_area)
            if float(best_pos["_area"]) >= float(best_neg["_area"]):
                winner_eo, win_pol = best_pos.name, "POS"
            else:
                winner_eo, win_pol = best_neg.name, "NEG"
        elif best_pos is not None:
            winner_eo, win_pol = best_pos.name, "POS"
        else:
            winner_eo, win_pol = best_neg.name, "NEG"

        col_old = f"{winner_eo}_{win_pol}"
        col_new = f"{name}_{win_pol}"
        rename_map[col_old] = col_new
        if win_pol == "POS":
            keep_pos.add(winner_eo)
        else:
            keep_neg.add(winner_eo)

    # ------------------------------------------------------------------
    # b. Unnamed cross-mode deduplication (neutral mass + RT)
    # ------------------------------------------------------------------
    unann_pos = pos_comp[~pos_ann].copy()
    unann_neg = neg_comp[~neg_ann].copy()

    # Neutral mass from measured m/z (POS: [M+H]+, NEG: [M-H]-)
    unann_pos["neutral_mass"] = pd.to_numeric(unann_pos[mz_col], errors="coerce") - H_MASS
    unann_neg["neutral_mass"] = pd.to_numeric(unann_neg[mz_col], errors="coerce") + H_MASS
    unann_pos["rt"]   = pd.to_numeric(unann_pos[rt_col], errors="coerce")
    unann_neg["rt"]   = pd.to_numeric(unann_neg[rt_col], errors="coerce")
    unann_pos["area"] = unann_pos["_area"]
    unann_neg["area"] = unann_neg["_area"]

    unnamed_keep_pos = set(unann_pos.index)
    unnamed_keep_neg = set(unann_neg.index)

    neg_masses = unann_neg["neutral_mass"].values
    neg_rts    = unann_neg["rt"].values
    neg_areas  = unann_neg["area"].values
    neg_eos    = unann_neg.index.values

    for _, row in unann_pos.iterrows():
        eo_p = row.name
        if eo_p not in unnamed_keep_pos:
            continue
        pm = row["neutral_mass"]
        if np.isnan(pm):
            continue
        ppm_diff = np.abs((neg_masses - pm) / (pm + 1e-12)) * 1e6
        rt_diff  = np.abs(neg_rts - row["rt"])
        match    = (ppm_diff <= MZ_PPM_TOL) & (rt_diff <= RT_MIN_TOL)
        for j in np.where(match)[0]:
            eo_n = neg_eos[j]
            if eo_n not in unnamed_keep_neg:
                continue
            if row["area"] >= neg_areas[j]:
                unnamed_keep_neg.discard(eo_n)
            else:
                unnamed_keep_pos.discard(eo_p)
                break

    n_cross = (len(unann_pos) + len(unann_neg)) - (len(unnamed_keep_pos) + len(unnamed_keep_neg))
    logger.info(
        "[%s] Step 13: %d named compounds kept (%d renamed); "
        "%d unnamed cross-mode pairs collapsed.",
        label, len(rename_map), len(rename_map), n_cross,
    )

    # ------------------------------------------------------------------
    # c. Build the combined expression matrix
    # ------------------------------------------------------------------
    final_keep_pos = keep_pos | unnamed_keep_pos
    final_keep_neg = keep_neg | unnamed_keep_neg

    # Filter expression columns (convert to str to match compound index type)
    pos_cols = [c for c in pos_df.columns if str(c) in {str(x) for x in final_keep_pos}]
    neg_cols = [c for c in neg_df.columns if str(c) in {str(x) for x in final_keep_neg}]

    # Add polarity suffix
    pos_sub = pos_df[pos_cols].copy()
    pos_sub.columns = [f"{c}_POS" for c in pos_sub.columns]
    neg_sub = neg_df[neg_cols].copy()
    neg_sub.columns = [f"{c}_NEG" for c in neg_sub.columns]

    # Outer join on sample index
    combined = pos_sub.join(neg_sub, how="outer")

    # Apply rename map
    combined = combined.rename(columns=rename_map)

    logger.info(
        "[%s] Step 13: combined matrix %d samples × %d analytes.",
        label, combined.shape[0], combined.shape[1],
    )
    return combined


# ===========================================================================
# STEP 14 — Half-minimum imputation (log2 space)
# ===========================================================================

def half_min_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 14: for each analyte, fill NaN with (min_observed − 1).
    In log2 space log2(x/2) = log2(x) − 1, so this equals half the minimum.
    """
    result = df.copy()
    for col in result.columns:
        col_min = result[col].min(skipna=True)
        if pd.isna(col_min):
            continue
        result[col] = result[col].fillna(col_min - 1.0)
    return result


# ===========================================================================
# STEP 16 — Attach metadata
# ===========================================================================

def _load_metadata(meta_path: str, sheet: str, id_col: str, is_plasma: bool) -> pd.DataFrame:
    """Load metadata from the master Excel table, indexed by normalised sample ID."""
    cols_wanted = [id_col, "group", "subgroup", "gest age del", "omics set#"]
    if is_plasma:
        cols_wanted.append("sample gest Age")

    wb = pd.read_excel(meta_path, sheet_name=sheet)
    available = [c for c in cols_wanted if c in wb.columns]
    meta = wb[available].dropna(subset=[id_col]).copy()
    meta[id_col] = meta[id_col].astype(str).str.strip().str.replace(r"\s+", "", regex=True)

    rename = {
        id_col:          "SampleID",
        "group":         "Group",
        "subgroup":      "Subgroup",
        "gest age del":  "GestAgeDelivery",
        "omics set#":    "Batch",
    }
    if is_plasma:
        rename["sample gest Age"] = "SampleGestAge"

    meta = meta.rename(columns=rename).set_index("SampleID")
    meta["Group"] = meta["Group"].replace({"sptb": "sPTB", "SPTB": "sPTB"})
    return meta[~meta.index.duplicated(keep="first")]


def attach_metadata(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Step 16: prepend metadata columns; drop samples without a group match."""
    matched = df.join(meta, how="left")
    n_unmatched = matched["Group"].isna().sum()
    if n_unmatched:
        logger.warning(
            "[%s] Step 16: %d samples have no metadata match — dropped.",
            label, n_unmatched,
        )
    matched = matched[matched["Group"].notna()].copy()

    meta_cols_present = [c for c in ["Group", "Subgroup", "Batch", "GestAgeDelivery", "SampleGestAge"]
                         if c in matched.columns]
    analyte_cols = [c for c in matched.columns if c not in set(meta_cols_present)]

    # SubjectID from index (strip timepoint suffix for plasma)
    subject_ids = matched.index.to_series().str.replace(r"[A-E]$", "", regex=True)
    out = pd.concat(
        [subject_ids.rename("SubjectID"), matched[meta_cols_present], matched[analyte_cols]],
        axis=1,
    )
    logger.info(
        "[%s] Step 16: %d samples × %d analytes with metadata.",
        label, len(out), len(analyte_cols),
    )
    return out


# ===========================================================================
# STEP 18 — Split plasma output by timepoint
# ===========================================================================

def split_by_timepoint(df: pd.DataFrame, out_dir: str, prefix: str) -> None:
    """Step 18: slice the full matrix by timepoint suffix (A–E) and save CSVs."""
    for tp in "ABCDE":
        mask = df.index.str.endswith(tp)
        df_tp = df[mask].copy()
        if df_tp.empty:
            continue
        # Strip timepoint letter from SampleID
        df_tp.index = df_tp.index.str[:-1]
        df_tp.index.name = "SampleID"
        out_path = os.path.join(out_dir, f"{prefix}_formatted_suffix_{tp}.csv")
        df_tp.to_csv(out_path)
        logger.info("  Timepoint %s: %d samples → %s", tp, len(df_tp), out_path)


# ===========================================================================
# Main pipeline orchestrator
# ===========================================================================

def run_pipeline(
    datatype: str,        # "MTBL" or "LIPD"
    tissue: str,          # "plasma" or "placenta"
    kaylaxu_dir: str,
    meta_path: str,
    out_dir: str,
) -> None:

    label = f"{datatype}_{tissue}"
    data_dir = os.path.join(kaylaxu_dir, "data", f"{datatype}_{tissue}")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting pipeline: %s", label)
    logger.info("=" * 60)

    # --- Load raw extracted CSVs ---
    logger.info("[%s] Loading raw extracted CSV files …", label)
    pos_exp   = pd.read_csv(os.path.join(data_dir, "pos_expression.csv"),  index_col=0)
    pos_batch = pd.read_csv(os.path.join(data_dir, "pos_batch.csv"),        index_col=0)
    pos_comp  = pd.read_csv(os.path.join(data_dir, "pos_compounds.csv"),    index_col=0)
    neg_exp   = pd.read_csv(os.path.join(data_dir, "neg_expression.csv"),  index_col=0)
    neg_batch = pd.read_csv(os.path.join(data_dir, "neg_batch.csv"),        index_col=0)
    neg_comp  = pd.read_csv(os.path.join(data_dir, "neg_compounds.csv"),    index_col=0)

    # Internal standard columns (c1, c2)
    std_cols = [c for c in pos_exp.columns if re.match(r"^c\d+$", c)]
    logger.info("[%s] Internal standard columns: %s", label, std_cols)

    qc_failures: list = []

    # -----------------------------------------------------------------------
    # Step 1: convert missing
    # -----------------------------------------------------------------------
    logger.info("[%s] Step 1: converting 0/blank/NA to NaN …", label)
    pos_exp = convert_missing(pos_exp)
    neg_exp = convert_missing(neg_exp)

    # -----------------------------------------------------------------------
    # Step 2: split QC pools from biological samples, by batch
    # -----------------------------------------------------------------------
    logger.info("[%s] Step 2: splitting by sample type and batch …", label)
    pos_samples, pos_pooled, pos_batches = split_by_batch(pos_exp, pos_batch)
    neg_samples, neg_pooled, neg_batches = split_by_batch(neg_exp, neg_batch)
    unique_batches = sorted(set(pos_batches) | set(neg_batches), key=str)
    logger.info("[%s] Batches found: %s", label, unique_batches)

    # -----------------------------------------------------------------------
    # Step 3: MAD filter on internal standards
    # -----------------------------------------------------------------------
    logger.info("[%s] Step 3: MAD filter on internal standards …", label)
    pos_samples = mad_filter_samples(pos_samples, std_cols, qc_failures, label + "/POS")
    neg_samples = mad_filter_samples(neg_samples, std_cols, qc_failures, label + "/NEG")

    # -----------------------------------------------------------------------
    # Step 4: sample missingness filter (>50%)
    # -----------------------------------------------------------------------
    logger.info("[%s] Step 4: sample missingness filter (>50%%) …", label)
    pos_samples = sample_missingness_filter(pos_samples, qc_failures, label + "/POS")
    neg_samples = sample_missingness_filter(neg_samples, qc_failures, label + "/NEG")

    # -----------------------------------------------------------------------
    # Step 5: RSD filter on QC pools (>30%)
    # -----------------------------------------------------------------------
    logger.info("[%s] Step 5: RSD filter on QC pools (>30%%) …", label)
    pos_samples, pos_pooled = rsd_filter_analytes(pos_samples, pos_pooled, label + "/POS")
    neg_samples, neg_pooled = rsd_filter_analytes(neg_samples, neg_pooled, label + "/NEG")

    # -----------------------------------------------------------------------
    # Step 6: analyte missingness filter (>20%) + differential check
    # -----------------------------------------------------------------------
    logger.info("[%s] Step 6: analyte missingness filter (>20%%) …", label)
    # Build group map from metadata (for differential missingness test)
    is_plasma = tissue == "plasma"
    meta_sheet = "n=133 metabolomics" if is_plasma else "n=133 placenta"
    id_col_name = "Sample ID" if is_plasma else "ID"
    meta = _load_metadata(meta_path, meta_sheet, id_col_name, is_plasma)
    groups = meta["Group"].to_dict()

    pos_samples, pos_drop_report = analyte_missingness_filter(pos_samples, groups, label + "/POS")
    neg_samples, neg_drop_report = analyte_missingness_filter(neg_samples, groups, label + "/NEG")

    drop_report = pd.concat([pos_drop_report, neg_drop_report], ignore_index=True)
    drop_report_path = os.path.join(out_dir, "dropped_analytes_report.csv")
    drop_report.to_csv(drop_report_path, index=False)
    logger.info("[%s] Dropped analytes report → %s", label, drop_report_path)

    # -----------------------------------------------------------------------
    # Step 7: PCA pre-normalization
    # -----------------------------------------------------------------------
    logger.info("[%s] Step 7: generating pre-normalization PCA …", label)
    generate_pca(
        pos_samples,
        os.path.join(out_dir, "pca_pre_normalization_POS.png"),
        f"{label} POS — pre-normalization (colored by batch)",
    )
    generate_pca(
        neg_samples,
        os.path.join(out_dir, "pca_pre_normalization_NEG.png"),
        f"{label} NEG — pre-normalization (colored by batch)",
    )

    # -----------------------------------------------------------------------
    # Step 8: biological-replicate batch normalization
    # -----------------------------------------------------------------------
    logger.info("[%s] Step 8: biological-replicate batch normalization …", label)
    # Reference batch: 110123 for plasma, 32425 for placenta
    ref_batch = "110123" if is_plasma else "32425"
    # Fallback if reference batch not present
    str_batches = [str(b) for b in unique_batches]
    if ref_batch not in str_batches:
        ref_batch = str_batches[0]
        logger.warning("[%s] Default reference batch not found; using %s.", label, ref_batch)

    pos_samples = batch_normalize(pos_samples, unique_batches, ref_batch, label + "/POS")
    neg_samples = batch_normalize(neg_samples, unique_batches, ref_batch, label + "/NEG")

    # -----------------------------------------------------------------------
    # Step 9: PCA post-normalization
    # -----------------------------------------------------------------------
    logger.info("[%s] Step 9: generating post-normalization PCA …", label)
    generate_pca(
        pos_samples,
        os.path.join(out_dir, "pca_post_normalization_POS.png"),
        f"{label} POS — post-normalization (colored by batch)",
    )
    generate_pca(
        neg_samples,
        os.path.join(out_dir, "pca_post_normalization_NEG.png"),
        f"{label} NEG — post-normalization (colored by batch)",
    )

    # -----------------------------------------------------------------------
    # Step 10: average cross-batch replicates
    # -----------------------------------------------------------------------
    logger.info("[%s] Step 10: averaging cross-batch replicates …", label)
    pos_avg, pos_batch_series = average_replicates(pos_samples, label + "/POS")
    neg_avg, neg_batch_series = average_replicates(neg_samples, label + "/NEG")

    # -----------------------------------------------------------------------
    # Step 12: log2(x + 1) transformation
    # -----------------------------------------------------------------------
    logger.info("[%s] Step 12: log2(x + 1) transformation …", label)
    # Drop internal-standard columns before log2 (not analytes of interest)
    pos_analytes_only = pos_avg.drop(columns=std_cols, errors="ignore")
    neg_analytes_only = neg_avg.drop(columns=std_cols, errors="ignore")

    pos_log2 = log2_transform(pos_analytes_only)
    neg_log2 = log2_transform(neg_analytes_only)

    # -----------------------------------------------------------------------
    # Step [C]: ComBat batch correction
    # -----------------------------------------------------------------------
    logger.info("[%s] Step [C]: ComBat batch correction …", label)
    # Use batch_series aligned to pos_log2 index
    pos_batch_for_combat = pos_batch_series.reindex(pos_log2.index).fillna(ref_batch)
    neg_batch_for_combat = neg_batch_series.reindex(neg_log2.index).fillna(ref_batch)

    pos_pre_combat = pos_log2.copy()
    neg_pre_combat = neg_log2.copy()

    pos_combat = combat_correct(pos_log2, pos_batch_for_combat, label + "/POS")
    neg_combat = combat_correct(neg_log2, neg_batch_for_combat, label + "/NEG")

    # PCA pre/post ComBat
    generate_pca_pre_post(
        pos_pre_combat, pos_combat, pos_batch_for_combat,
        os.path.join(out_dir, "pca_combat_POS.png"),
        f"{label} POS — Pre/Post ComBat",
    )
    generate_pca_pre_post(
        neg_pre_combat, neg_combat, neg_batch_for_combat,
        os.path.join(out_dir, "pca_combat_NEG.png"),
        f"{label} NEG — Pre/Post ComBat",
    )

    # -----------------------------------------------------------------------
    # Step 13: combine POS + NEG (dedup by quality score + signal intensity)
    # -----------------------------------------------------------------------
    logger.info("[%s] Step 13: combining POS and NEG …", label)
    combined = combine_pos_neg(pos_combat, neg_combat, pos_comp, neg_comp, label)

    # -----------------------------------------------------------------------
    # Step 14: half-minimum imputation
    # -----------------------------------------------------------------------
    logger.info("[%s] Step 14: half-minimum imputation …", label)
    n_missing_before = int(combined.isna().sum().sum())
    combined_imputed = half_min_impute(combined)
    logger.info("[%s] Step 14: imputed %d missing values.", label, n_missing_before)

    # -----------------------------------------------------------------------
    # Step 16: attach metadata
    # -----------------------------------------------------------------------
    logger.info("[%s] Step 16: attaching metadata …", label)
    full_matrix = attach_metadata(combined_imputed, meta, label)

    # -----------------------------------------------------------------------
    # Save full matrix
    # -----------------------------------------------------------------------
    full_csv = os.path.join(out_dir, f"{label}_cleaned_with_metadata.csv")
    full_matrix.to_csv(full_csv)
    logger.info(
        "[%s] Full matrix saved → %s  (%d samples × %d cols)",
        label, full_csv, full_matrix.shape[0], full_matrix.shape[1],
    )

    # -----------------------------------------------------------------------
    # Step 18: split by timepoint (plasma only)
    # -----------------------------------------------------------------------
    if is_plasma:
        logger.info("[%s] Step 18: splitting by timepoint …", label)
        split_by_timepoint(full_matrix, out_dir, label)

    # -----------------------------------------------------------------------
    # Save QC failure log
    # -----------------------------------------------------------------------
    if qc_failures:
        qc_df = pd.DataFrame(qc_failures)
        qc_path = os.path.join(out_dir, "qc_failures.csv")
        qc_df.to_csv(qc_path, index=False)
        logger.info("[%s] QC failures log → %s (%d entries).", label, qc_path, len(qc_df))

    logger.info("[%s] Pipeline complete.", label)


# ===========================================================================
# Entry point
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    kaylaxu_default = os.path.join(os.path.dirname(root), "kaylaxu")

    p = argparse.ArgumentParser(
        description="Full omics preprocessing pipeline from raw extracted CSVs."
    )
    p.add_argument(
        "--kaylaxu-dir", default=kaylaxu_default,
        help="Path to Kayla Xu project directory (contains data/MTBL_plasma/ etc.).",
    )
    p.add_argument(
        "--metadata", default=os.path.join(root, "data", "dp3 master table v2.xlsx"),
        help="Path to dp3 master table v2.xlsx",
    )
    p.add_argument(
        "--output-dir", default=os.path.join(root, "data", "cleaned", "omics_pipeline"),
        help="Root output directory (one subdirectory per dataset).",
    )
    p.add_argument(
        "--datasets", nargs="+",
        default=["MTBL_plasma", "MTBL_placenta", "LIPD_plasma", "LIPD_placenta"],
        help="Datasets to process (e.g. MTBL_plasma LIPD_plasma).",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    for ds in args.datasets:
        parts = ds.split("_", 1)
        if len(parts) != 2:
            logger.error("Invalid dataset name %r — expected DATATYPE_TISSUE.", ds)
            continue
        datatype, tissue = parts
        out_subdir = os.path.join(args.output_dir, ds)
        run_pipeline(
            datatype=datatype,
            tissue=tissue,
            kaylaxu_dir=args.kaylaxu_dir,
            meta_path=args.metadata,
            out_dir=out_subdir,
        )
