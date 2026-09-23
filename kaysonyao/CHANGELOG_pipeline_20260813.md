# DP3 preprocessing & modelling — change summary
**As of 2026-08-13**

> **Note on versions.** This repository contains SOP **v3** (2026-04-20, archived
> to `_legacy/20260811/docs/`) and SOP **v4** (2026-04-30, current). There is no
> v2 document here. Section 1 covers the documented v3→v4 change; sections 2
> onward cover implementation changes, which are where most of the movement has
> been.

---

## 1. SOP v3 → v4 (documented, April 2026)

33 steps → 35 steps. Nothing was dropped; the substantive change is that three
QC filters moved to **after** batch correction, and one new filter was added.

| Change | v3 | v4 | Rationale (per SOP) |
|---|---|---|---|
| RSD filter on QC pools | Step 8 (pre-ComBat) | **Step 17 (post-ComBat)** | On uncorrected data a valid feature can show inflated RSD purely from batch scale differences |
| Sample QC via ISTD MAD | Step 10 (pre-ComBat) | **Step 16 (post-ComBat)** | Bad samples inflate per-feature variance, making RSD and IQR misclassify features; remove them first |
| Sample-level missingness | Step 9 | **Step 8**, before log2 | Ordering only |
| **IQR filter (within-timepoint)** | — | **Step 18 (new)** | Removes near-constant and hypervariable features. Computed *within* timepoint so longitudinal biology (e.g. progesterone rising across pregnancy) is not mistaken for artefact. Dual threshold: a feature must be extreme in both percentile and absolute terms |
| Trajectory plots / final log | Step 32–33 combined | Steps 34–35 split | Bookkeeping |

---

## 2. Batch correction — the largest change

### 2.1 SOP Step 13 was never satisfied, by anyone

The SOP mandates **reference-batch (`ref.batch`) ComBat**, anchoring correction
to the 20 bridge samples spanning all three batches. It was never used.

The pipeline attempted `sva::ComBat` via R, failed silently because **R, rpy2
and sva were all absent**, and fell back to `pycombat` — which has no
`ref.batch` capability at all (`Combat.__init__(self, mode='p', conv=1e-4)` is
its entire API). Every log recorded `pycombat_standard` and nobody noticed,
because the run still succeeded. **This affected the collaborator's
2026-08-12 delivery as well.**

Fixed 2026-08-13. R + sva installed; `_combat_r_ref_batch` now succeeds and logs
`sva_combat_ref_batch`. The silent fallback now emits a large
`SOP STEP 13 NOT SATISFIED` error block.

### 2.2 Outcome labels were encoded in the predictors

The ComBat design matrix contained
`protected = ['Subgroup', 'SampleGestAge', 'GestAgeDelivery']`.

`Subgroup` is the diagnosis; `GestAgeDelivery` *defines* preterm birth. Every
feature value was therefore adjusted using knowledge of each participant's
outcome. Random forest and XGBoost reached PR-AUC ≈ 1.00 and ROC-AUC = 1.000 on
fewer than 90 participants — the signature of label information in the
features, not biology.

SOP Step 12 explicitly permits this (*"consider using ComBat with biological
covariates protected"*), and it is correct **for differential analysis**. It is
not usable for prediction: the correction cannot be reproduced on a patient
whose outcome is unknown, which rules out ECHO external validation (F.2 model 3)
and any prospective use (F.5 lead time).

New `--blind-combat` flag drops `Group`/`Subgroup`/`GestAgeDelivery` and retains
`SampleGestAge`, which is a collection-time nuisance covariate, not an outcome.

### 2.3 Held-out participants shaped the correction

ComBat was fitted on all 530 samples, so the locked test set influenced the
values the model trained on.

New `--fit-split dev` estimates ComBat and half-minimum imputation floors on the
268 development participants and *applies* them to everyone.

### 2.4 ref.batch and fit/transform are mutually exclusive in every package

| | ref.batch | fit / transform |
|---|---|---|
| `pycombat` | ✗ | ✓ |
| `inmoose` | ✓ | ✗ (single-shot function) |
| `sva::ComBat` (R) | ✓ | ✗ — returns only the corrected matrix, discards γ\*/δ\* |

Resolved by `combat_fit_transform.R`, which splits sva's own estimator into
`combat_fit()` / `combat_apply()`, retaining γ\*, δ\*, grand mean and pooled
variance instead of discarding them. **Validated against `sva::ComBat` to
max |diff| = 3.6e-15.** A pure-Python equivalent (`combat_ref.py`) matches to
4.97e-14.

### 2.5 How much it mattered

pycombat vs sva ref.batch, both blind and dev-fit:

| dataset | features (pycombat) | features (sva) | mean abs diff | max abs diff |
|---|---:|---:|---:|---:|
| MTBL_plasma | 664 | **1399** | 0.106 | 4.473 |
| LIPD_plasma | 1386 | 1327 | 0.022 | 8.350 |
| MTBL_placenta | 1295 | **1588** | 0.139 | 5.092 |
| LIPD_placenta | 1822 | 2006 | 0.003 | 3.100 |

MTBL_plasma retains more than twice as many features. Not cosmetic.

---

## 3. Timepoint definitions

Visit letters A–E are **relative** windows, not fixed timepoints: visit B spans
**13.6–33.6 gestational weeks**, so two participants both labelled B can be
20 weeks apart. Modelling them as one timepoint is not meaningful.

Three binnings now exist, selectable rather than hard-coded:

| Scheme | Windows | Use |
|---|---|---|
| Visit letters A–E | by SampleID suffix | The project's five defined timepoints (the PI's intent) |
| `echo2` | 6.0–19.9, ≥28 wks | Penn-CHOP ECHO matching, for external validation (F.2 model 3) |
| `dp3_5t` | 6–14, 14–22, 22–32, 32–37, 37–42 | Gestational-age bins (superseded by A–E per the PI) |

**Late-enrolment series.** SampleIDs carry ten suffixes: `A`–`E` and `EA`–`EE`.
The E-prefixed codes map onto the base visit (`EA`→A, …). Merging reproduces the
original counts exactly (133/133/128/109/27); failing to merge silently drops
23 participants from A and B.

**Midpoint rule (per the PI).** One sample per participant per window; where
several fall in a window, keep the one closest to the window midpoint. Ties
break toward the earlier draw. Needed only for gestational-age binnings —
visit-letter slicing already has one sample per participant per letter. The
collaborator's T1–T5 files did **not** apply it (46 duplicate participant-rows
in MTBL plasma alone).

---

## 4. Modelling (Aim 3A)

### 4.1 Elastic net was not elastic net — or lasso

`lasso_feature_selection_binary()` called `LogisticRegressionCV(l1_ratios=...)`
**without `penalty="elasticnet"`**. sklearn defaults to `penalty="l2"` and
silently ignores `l1_ratios`. Because L2 never drives a coefficient to exactly
zero, the downstream `coef != 0` test selected **every** feature.

Fingerprint in the archived outputs: identical selection counts at every
timepoint (`metabolomics/plasma` = 1887 at A–E; `MTBL_sop_nodiff/plasma` = 502
at A–E). Real selection cannot produce the same count five times.

**No dimensionality reduction ever occurred.** All `lasso_selected_features.csv`
files and any biology read from them are unreliable.

### 4.2 Splitting

| | Before | Now |
|---|---|---|
| Ratio | 70/15/15 | **70/30** |
| Unit | SampleID | **SubjectID** (participant-level) |
| Consistency | each timepoint model drew its own split — 20 of 27 subjects got different assignments across models | **one locked split**, shared by every dataset, window and modality |

`make_locked_split.py` refuses to redraw without `--force`.

### 4.3 Leakage inside the modelling pipeline

- Feature selection was fitted **once** on the whole development set, so it saw
  every inner validation fold before tuning used them. Now inside the `Pipeline`,
  refit per fold.
- Imputation was fitted on the whole development set. Now inside the `Pipeline`.
- Imputation strategy was **median** (missing-at-random) where the SOP uses
  **half-minimum** (below-detection). Material: 4.5% of LIPD_plasma cells reach
  the model still missing, across every feature. Now a fold-wise
  `HalfMinimumImputer`.

### 4.4 Nested CV

Single train/val/test with Optuna scored on one validation split →
**nested CV**, 5 outer × 3 inner, `StratifiedGroupKFold` grouped by participant
in both loops. `l1_ratio` tuned in the inner loop, per F.3.

### 4.5 Model roster and metrics

- **SVM restored** (had been silently dropped from the four-model roster).
- **XGBoost `scale_pos_weight` restored** — it had no imbalance handling at all
  while the other three used `class_weight="balanced"`.
- **Optuna search spaces restored** to the previous ranges (`min_samples_split`,
  `gamma`, `reg_alpha` had been dropped); `n_trials` 50 → 40 for runtime.
- **Confidence intervals** changed from a normal approximation over 5 folds to a
  **1000× percentile bootstrap over pooled out-of-fold predictions**, matching
  the previous pipeline's approach.
- Added specificity and Brier score (F.5 calibration).
- Placenta coverage restored (had been dropped when binning required
  `SampleGestAge`, which placenta lacks — it is collected once at delivery).

---

## 5. Data corrections

- **DP3-0343 reclassified** `sptb` → `FGR` in the updated master table.
  Consequence: sPTB 19 → 18, FGR 23 → 24. Every downstream count and figure
  shifts; sPTB was already the least-powered group.
- **Export-footer contamination.** The May-2026 placenta metabolomics export
  appends summary rows that survive extraction as `Unnamed:` feature columns
  (2 per polarity). Now dropped on read, with a warning.
- **Placenta injection order.** `MTBL_placenta` and `LIPD_placenta` had no
  `raw_workbook` configured and silently fell back to row order. Now wired up;
  all four datasets report a real injection-order source.
- **Trailing-apostrophe bug.** The placenta lipidomics export writes raw-file
  names ending in `'`, which defeated the injection-order regex.
- **`n=133 placenta` sheet regression** in the new master table: 54 `#ERROR!`
  cells in `TP# rnalater`, 52 in `TP# tissue`. Not read by the pipeline, but the
  data manager should know.

---

## 6. Known deviations from the SOP

| Item | Status |
|---|---|
| Step 12 — protected biological covariates | **Deliberately not used for modelling.** Retained as correct for differential analysis. Two parallel outputs are the intended resolution |
| Step 18 — IQR computed within timepoints A–E | Still keyed on visit letters while modelling now uses gestational windows. Inconsistent; unresolved |
| Split stratification | F.2 asks for balance across data types and gestational windows; the split is stratified on any-complication only. Within the n=133 omics subset this left **HDP 23 dev / 17 test** and **sPTB 11 dev / 3 test** — per-syndrome sPTB test evaluation is not possible |

---

## 7. Outstanding

- **Test set opened five times.** Each was a specified methodological change, not
  a reaction to results, but F.5 requires a single look. Current plan:
  development-set only until preprocessing is frozen, then one final evaluation.
- **Superseded results.** Everything predating 2026-08-13 used non-compliant
  ComBat, including both comparison zips sent to the PI. The biased-vs-corrected
  contrast in them remains qualitatively valid (it concerned *label* leakage,
  independent of ref.batch) but the absolute numbers are superseded.
- **Aim 3B (meta-model), calibration curves, and the routine-EHR benchmark** are
  not yet built.
- **`feature_interpretation.py` / `run_permutation_test.py`** were orphaned when
  artefact writing changed; the required files are written again, but SHAP and
  permutation importance (Aim 3D / F.6) have not been re-run.
