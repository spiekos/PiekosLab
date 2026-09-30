# In-fold preprocessing plan

Version 2, 30 Sep 2026. Replaces the 21 Sep plan (`DP3_infold_CV_plan.docx`, and the short
version sent to the PI, `DP3_fold_design.docx`). Nothing in this plan has been built yet
except where the status column says so.

## Goal

From the PI (17 Sep): run the whole preprocessing pipeline inside every fold of the nested CV,
on the raw data, keeping the SOP's step order. Every step that learns something from the data
learns it from that fold's training participants only, and the held-out participants are
processed with those learned settings. Different folds will keep different features; that is
expected and must not be "fixed" by taking a union or intersection of feature lists.
QC pools and internal standards must be available in every fold.

## What changed since the 21 Sep plan

1. **Step 17 (QC-pool RSD filter) moved in-fold.** The first plan marked it "arguably need not";
   at your request the version sent to the PI listed it, so seven steps are in-fold:
   5, 7, 10, 13, 16, 17, 18.
2. **New finding from Jenn's code:** Steps 5 and 17 learn only from QC pools. QC pools are in
   every training set, so these two steps give identical results in every fold. They stay on the
   list (it costs nothing), but they need no code change.
3. **The pipeline is now Jenn's SOP v7 version** (`01_data_cleaning/sop_omics_pipeline.py`).
   Step numbers are unchanged.
4. **Step 13 (ComBat) is already fold-ready:** `combat_ref.ComBatRef` learns and applies
   separately, and `--fit-subjects` / `--blind-combat` exist. Today they are used only to keep
   the locked test set out, not per fold.
5. **Windows are now T1–T5 gestational bins**, not visit letters A–E. The old risk about window E
   no longer applies; T4 is now the smallest window (27–30 development participants).
6. **Urine metabolomics and proteomics are included.**
7. **One set of folds is shared by every assay and every window**, so each fold's preprocessing
   is run once per dataset and reused for all five windows.

## 1. Folds

- Drawn once over the ~90 development participants who have omics data, written to
  `data/processed/cv_folds.csv` by a new `make_cv_folds.py` (refuses to overwrite, like
  `make_locked_split.py`). The locked 30% test set is never in any fold.
- 5 outer folds; inside each outer training set, 3 inner folds. Columns:
  `SubjectID, outer_fold, inner_fold`.
- Grouped by participant: all of a participant's samples (every window, every bridge replicate)
  land on the same side.
- Stratified by omics set (master table `omics set#`, 1–3) × any complication. Every participant
  belongs to exactly one omics set. Current development counts (controls / cases):
  set 1 = 12 / 13, set 2 = 12 / 12, set 3 = 15 / 26.
- Typical sizes: outer training ≈ 72, held out ≈ 18; inner training ≈ 48, inner quiz ≈ 24.
  Per window the numbers are smaller (T4: ≈ 24 / 6 outer, ≈ 16 / 8 inner).
- **QC pools** are not participants and are never split: they are added to the training side of
  every fold. Plasma metabolomics has 42 (10 / 12 / 20 across its 3 batches), plasma lipidomics
  39, urine 43 across 5 batches.
- **Internal standards** are compound columns measured in every sample, so they are present in
  every fold automatically. What is learned per fold is the Step 16 cutoff.
- **Bridge samples** (plasma: 15 samples run in two batches, 12 of those participants in the
  development set) stay with their participant through the grouping. Step 19 averages the
  replicates inside the fold. Urine has no bridge samples, but most urine participants' samples
  are spread over several of its 5 batches, so every training set contains every urine batch.

**Checks** (the fold generator stops if any fail): each participant on one side only; both
classes on both sides; QC pools in every training set; for every dataset, every run batch and the
reference batch present among the training participants plus QC pools.

## 2. What runs inside each fold

The whole SOP pipeline runs once per fold. Seven steps learn from the data and are restricted to
the fold's training participants plus QC pools; their learned settings are then applied to
everyone.

| Step | What it learns | Learns from | Code status |
|---|---|---|---|
| 5 Median fold-change batch normalization | Batch scaling factors | QC pools only | No change needed |
| 7 Feature missingness filter | Which features survive (20%, within case/control and window) | Training participants **and their labels** | To build |
| 10 Half-minimum imputation | Per-feature fill value | Training participants + QC | To build |
| 13 ComBat (reference batch, outcome-blind) | Batch location/scale, pooled variance | Training participants + QC | Done (`--fit-subjects`) |
| 16 Sample QC via internal standards | Per-batch median and MAD cutoffs | Training participants | To build |
| 17 RSD filter on QC pools | Which features survive | QC pools only | No change needed |
| 18 IQR filter (within window) | Percentile cutoffs | Training participants | To build |

Step 7 matters most: it is the only step that looks at case/control labels, so today it uses the
labels of the held-out fold and of the locked test set.

Everything else does not learn from other samples, so it is the same in or out of a fold:
Steps 1–2 (bookkeeping), 4 (each sample divided by its own internal standards), 8 (per-sample
missingness), 9 (log2), 19 (average a participant's bridge replicates), 20–33 (feature
deduplication, annotation, merging modes, metadata), and the diagnostic plots (3, 6, 11, 12, 14,
15, 34).

**Proteomics** (`clean_proteomics_data.py`) has three learning steps, all to build: ComBat,
the missingness filter, and half-minimum imputation. Its ComBat is currently fitted on all
participants, including the locked test set.

**Accepted limits** (cannot be refitted, state in the methods): values computed by the vendors
across all injections — Olink's plate normalization, and Compound Discoverer's `Area (Max.)` and
`RSD QC Areas` columns used by the Step 24 quality score. None of these use outcome labels.

## 3. Flow

```
for each outer fold k (1..5):
    preprocess, fitted on outer-train(k) + QC   -> apply to all rows -> bin T1–T5
    for each inner fold j (1..3):
        preprocess, fitted on inner-train(k,j) + QC -> apply to all rows -> bin T1–T5
    tune (Optuna, 40 settings): train on inner-train(k,j), score on inner-quiz(k,j),
        each using the (k,j) matrices
    refit the winning setting on outer-train(k) with the (k) matrices
    score outer-held-out(k) with the (k) matrices
pool the 5 held-out predictions -> pr_auc_oof, roc_auc, bootstrap CIs
```

20 preprocessing runs per dataset (5 outer + 15 inner), shared by all windows and models.

## 4. Code changes

1. `make_cv_folds.py`: draw and check the folds (Section 1).
2. `sop_omics_pipeline.py`: extend the existing `--fit-subjects` hook from Step 13 to Steps 7,
   10, 16 and 18. Keep `--blind-combat` required.
3. `clean_proteomics_data.py`: add the same `--fit-subjects` option for ComBat, missingness and
   imputation.
4. New driver `run_infold_preprocessing.py`: for each of the 20 training sets, write the subject
   list, run steps 2–3 into `data/processed_folds/o{k}/{all|i{j}}/`, then bin.
5. `nested_cv.py` / `run_base_models.py`: read folds from `cv_folds.csv` instead of drawing new
   ones per window, and load the matching (k) or (k,j) matrix for each fit and score. Feature
   lists are per fold.

## 5. Reporting (PI rules)

- No union or intersection of features across folds.
- `selected_features.csv` becomes per fold; `feats_in` becomes a range.
- Feature stability reported separately: the fraction of folds in which each feature survives.
- After CV, one separate run fitted on all development participants gives the final model and
  the reportable feature list. That run is also what the one-time locked test evaluation uses.

## 6. Runtime

One pipeline run of all five SOP datasets took about 4.5 minutes in testing, so the 20 fold runs
take roughly 1.5 hours; proteomics adds seconds. Model fitting time stays about the same as now.

## 7. Risks and open decisions

- **Decide before the full run** (each changes the matrices): delivery and post-delivery samples
  in T4/T5 (SOP v7 Step 8b/8c); ComBat over-compression (mean-only or non-parametric);
  Step 16 neutral-lipid-only failures (review, not exclude); urine dilution normalization.
- **Small windows:** T4 has ≈ 24 training participants per outer fold and ≈ 16 per inner fold.
  ComBat and Steps 5, 10, 16 learn from all of a training participant's samples across windows,
  so they are fine; Steps 7 and 18 work within a window and will be noisy in T4.
- **Batch and outcome are mildly linked** (participant level, Cramér's V 0.15–0.26; placenta
  p ≈ 0.03). Stratifying folds on omics set × outcome keeps them balanced, but outcome-blind
  ComBat may still remove a little real signal.
- **Expect lower scores than now.** That is the point: they will be the honest estimate.

## 8. Order of work

1. `make_cv_folds.py` and its checks.
2. Extend `--fit-subjects` in the SOP pipeline (Steps 7, 10, 16, 18) and in proteomics.
3. Leakage test on one dataset: change the held-out rows' values and confirm no learned setting
   and no training row's output changes (the same test ComBat already passed).
4. Dry run: one dataset, one outer fold; confirm feature lists differ between folds.
5. Update the model code to use the shared folds and per-fold matrices.
6. Full run, after the Section 7 decisions.
7. Final run fitted on all development participants; then, once, the locked test evaluation.
