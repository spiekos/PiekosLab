---
marp: true
---

# DP3 Multi-Omics Pipeline: Cleaning, Differential Analysis, and Modeling

---

## Proteomics Cleaning

### Input and metadata
- Auto-discovered Olink files and split them into plasma and placenta groups.
- Loaded metadata from the master table.
- Standardized metadata label `sptb -> sPTB`.

### Raw preprocessing
- Loaded Olink exports in long format.
- Converted invalid NPX entries to `NaN`.
- Masked NPX when `QC_Warning != PASS` or `Assay_Warning != PASS`.
- Combined all three sets before downstream processing.
---
### Normalization and cleanup
- Performed panel normalization using internal control samples.
- Treated `CONTROL`, `NEG`, and `PLATE` sample IDs as technical controls for normalization.
- Removed control assays after normalization.
- Pivoted to wide format and matched metadata.
- Applied ComBat batch correction.
- Applied missingness filter after ComBat.
- For dropped assays, ran Fisher exact test on missingness and Benjamini-Hochberg FDR.
- Imputed remaining missing values with `min_log2 - 1`.
---
### Latest printed results
- `files | plasma=3 | placenta=3`
- `plasma: start`
- `[plasma] removed 25344 rows from 12 control assays (assay name contains 'control').`
- `plasma: done | samples=530 | assays=2815 | dropped_assays=92 | metadata_matched=530/530`
- `placenta: start`
- `[placenta] removed 4992 rows from 12 control assays (assay name contains 'control').`
- `placenta: done | samples=114 | assays=2789 | dropped_assays=117 | metadata_matched=114/114`
- `All processing complete.`
---
### Output summary
- Plasma cleaned matrix: `530 samples x 2815 assays`
- Placenta cleaned matrix: `114 samples x 2789 assays`
- Plasma dropped assays: `92`, of which `86` were BH-rejected in the missingness report
- Placenta dropped assays: `117`, of which `81` were BH-rejected in the missingness report
---
### Cleaning plots to show
- PCA before vs after ComBat:

![Proteomics plasma PCA pre/post ComBat](./data_cleaning/plasma/pca/comparison/pca_pre_post_shared_axes.png)


---

## Metabolomics Cleaning — Original Pipeline (`clean_metabolomics_data.py`)

### Prerequisite
- Used cleaned data from Kayla
- This is the only pipeline with placenta data.

### Steps
- Loaded all `Samples_*.csv` files.
- Removed internal whitespace from sample IDs.
- Reconstructed plasma `SampleID = patient_ID + timepoint` when needed.
- Averaged duplicate sample IDs across files.

---
- Removed embedded metadata columns from the assay matrix and replaced them with authoritative metadata from the master sheet.
- Applied missingness filter at `<20% missing`.
- Re-imputed remaining NaN values with half-minimum logic.
- Saved full cleaned matrices for plasma and placenta.

### Latest printed results
- `Averaged 19 duplicate SampleID row(s) → 523 unique SampleIDs.`
- `[plasma] Done — 523 samples × 1887 analytes`
- `Averaged 8 duplicate SampleID row(s) → 106 unique SampleIDs.`
- `[placenta] Done — 106 samples × 2039 analytes`
- `All metabolomics processing complete.`
---
### Output summary
- Plasma cleaned matrix: `523 samples x 1887 analytes`
- Placenta cleaned matrix: `106 samples x 2039 analytes`


---

## Metabolomics Cleaning — ComBat Pipeline (`clean_metabolomics_prenorm_data.py`)

### Why this pipeline exists
- Input is raw peak areas (not log-transformed, not normalized) from instrument files named `Samples_{batch}_{timepoint}_preNorm.csv`.
- Three instrument batches: `51223`, `110123`, `112524`.
- Plasma only — placenta has one batch only so used original pipeline
---
### Steps
- Loaded all `Samples_*_preNorm.csv` files from `data/metabolomics_prenormalized/`; extracted instrument batch date and timepoint from filename.
- Constructed canonical `SampleID = patient_ID + filename_timepoint`; special subjects (patient_ID ending in a timepoint letter) kept as-is.
- Combined all files; averaged analyte values for duplicate `SampleID`s (cross-batch technical replicates).
- Applied `log2(x + 1)` transform to all analyte columns.
- Built per-sample batch labels from instrument date strings.
- Filtered samples missing group or batch labels.
- Applied ComBat batch correction using instrument date strings as batch labels.
- Generated pre/post ComBat PCA plots (colored by batch and by group).
---
- Applied missingness filter: dropped analytes with `≥20%` missing; Fisher's exact + Benjamini-Hochberg group-imbalance test; saved dropped-analyte report.
- Imputed remaining missing values with `min_log2 − 1`.
- Attached metadata columns (`SubjectID`, `Batch`, `Group`, `Subgroup`, `GestAgeDelivery`, `SampleGestAge`).
- Saved full matrix CSV and sliced per-timepoint files.
---
### Latest printed results
- `Combined matrix: 550 samples × 2105 analytes`
- `98 duplicate SampleID row(s) found — averaging analyte values.`
- `Group labels matched: 452 / 452 samples`
- `Applying log2(x + 1) transformation …`
- `Applying ComBat batch correction (batches: ['110123', '112524', '51223']) …`
- `Generating pre/post ComBat PCA plots …`
- `PCA plot saved → data/cleaned/metabolomics_combat/normalized_full_results/metabolomics_pca_pre_post_combat_batch.png`
- `PCA plot saved → data/cleaned/metabolomics_combat/normalized_full_results/metabolomics_pca_pre_post_combat_group.png`
---
- `Applying missingness filter (cutoff = 20%) …`
- `Analytes after missingness filter: 2105 (dropped 0)`
- `Done. Final matrix: 452 samples × 2105 analytes | dropped: 0 | batches: ['110123', '112524', '51223']`
- Timepoint slices saved: A: `108`, B: `108`, C: `104`, D: `88`, E: `44`
---
### Output summary
- Plasma cleaned matrix: `452 samples x 2105 analytes`
- Dropped analytes: `0`

---
### Cleaning plots to show
- Pre/post ComBat PCA colored by batch:

![Metabolomics ComBat PCA pre/post by batch](../data/cleaned/metabolomics_combat/normalized_full_results/metabolomics_pca_pre_post_combat_batch.png)

---
- Pre/post ComBat PCA colored by group:

![Metabolomics ComBat PCA pre/post by group](../data/cleaned/metabolomics_combat/normalized_full_results/metabolomics_pca_pre_post_combat_group.png)

---

## Lipidomics Cleaning

### How this pipeline differs from the other two
- Excel input file
- Used raw `3Sets` QC sheets to recover the authoritative `Rej` flags.
- Merged positive- and negative-ion mode features into one matrix.
- Combat is applied but different from the other two

---
### Steps
- Read raw QC rejection flags from the two `3Sets` sheets.
- Read processed POS and NEG sheets.
- Dropped ISTD rows.
- Dropped pooled control columns.
- Normalized sample IDs and collapsed replicate labels to canonical suffixes.
- Kept only primary-batch columns for cross-batch re-injections.
- Averaged duplicate sample columns before log transform.
- Merged POS and NEG matrices.
- Applied `log2(area + 1)`.
- Merged authoritative metadata.

---
- Converted omics set numbers from metadata (1→051223, 2→110123, 3→041625) to instrument run date strings; applied ComBat using those date strings as batch labels.
- Generated pre/post ComBat PCA plots.
- Applied missingness filter and saved dropped-feature report.
- Imputed remaining missing values with `min_log2 - 1`.
- Saved the full matrix and sliced timepoint-specific files.

---
### Latest printed results
- `[041625_Sadovsky_Plasma_3Sets_Po] Non-rejected LipidIDs: 1986 | Truly-rejected-only: 379`
- `[Sadovsky_Plasma_3Sets_Neg] Non-rejected LipidIDs: 511 | Truly-rejected-only: 237`
- `[Plasma POS Lipids] Discarded 20 cross-batch re-injection column(s)`
- `[Plasma POS Lipids] Patient columns: 530 → 457 canonical SampleIDs`
- `[Plasma POS Lipids] Output matrix: 457 samples × 1986 features (before log2)`

---
- `[Plasma NEG Lipids] Discarded 20 cross-batch re-injection column(s)`
- `[Plasma NEG Lipids] Patient columns: 530 → 457 canonical SampleIDs`
- `[Plasma NEG Lipids] Output matrix: 457 samples × 511 features (before log2)`
- `Combined matrix: 457 samples × 2497 features`
- `Metadata matched: 457 / 457 samples`
- `Dropped 61 features (missingness ≥ 20%)`
- `Saved full matrix → ... lipids_plasma_cleaned_with_metadata.csv (457 rows × 2441 cols)`
- `Done. Final matrix: 457 samples × 2436 lipid features | dropped features: 61`

---
### Output summary
- Plasma cleaned matrix: `457 samples x 2436 lipid features`
- Dropped features: `61`
- BH-rejected dropped features: `0`
- Timepoint slices saved:
  - A: `110`
  - B: `110`
  - C: `105`
  - D: `89`
  - E: `43`
---
### Cleaning plots to show
- Pre/post ComBat PCA colored by batch:

![Lipids PCA pre/post ComBat by batch](../data/cleaned/lipids/normalized_full_results/lipids_pca_pre_post_combat_batch.png)

---
- Pre/post ComBat PCA colored by group:

![Lipids PCA pre/post ComBat by group](../data/cleaned/lipids/normalized_full_results/lipids_pca_pre_post_combat_group.png)

---
### Main takeaway
- Lipidomics required the heaviest raw-data reconstruction step because we had to recover QC rejection from the raw workbook and harmonize replicated sample columns before modeling.

---

## Slide 4. Differential Analysis

### Cross-sectional analysis
- Applied to placenta and to each plasma timepoint separately.
- Compared `Control` vs pooled `Complication`.
- Test: Mann-Whitney U, two-sided.
- FDR: Benjamini-Hochberg.
- Significance threshold: `q < 0.05`.
- Additional effect-size threshold: `|log2 FC| >= log2(1.5)`.
- Minimum non-missing per group: `n >= 5`.
---
### Longitudinal analysis
- Applied to plasma only.
- Built adjacent within-subject deltas: `B-A`, `C-B`, `D-C`, `E-D`.
- Ran Wilcoxon signed-rank, two-sided, `zero_method='wilcox'`.
- Performed within each group: `Control`, `FGR`, `HDP`, `sPTB`, and pooled `Complication`.
---
### Proteomics differential results
- Placenta cross-sectional:
  - `2789 tested`, `1 significant`
- Plasma cross-sectional:
  - A: `0 significant`
  - B: `0 significant`
  - C: `149 significant`
  - D: `0 significant`
  - E: `0 significant`
---
- Plasma longitudinal:
  - Control: `60`, `62`, `115`, `0`
  - FGR: `76`, `48`, `38`, `0`
  - HDP: `66`, `94`, `87`, `0`
  - sPTB: `56`, `60`, `0`, `skipped for E-D`
  - Complication: `69`, `67`, `58`, `0`
---
### Metabolomics differential results
- Placenta cross-sectional:
  - `2039 tested`, `0 significant`
- Plasma cross-sectional:
  - A-E: `0 significant` at every timepoint
- Plasma longitudinal:
  - Control: `4`, `4`, `3`, `0`
  - FGR: `0`, `1`, `0`, `0`
  - HDP: `5`, `2`, `12`, `0`
  - sPTB: `0`, `0`, `0`, `skipped for E-D`
  - Complication: `3`, `2`, `10`, `0`
---
### Lipidomics differential results
- Plasma cross-sectional:
  - A: `0`
  - B: `0`
  - C: `0`
  - D: `0`
  - E: `1`
- Plasma longitudinal:
  - Control: `475`, `81`, `55`, `0`
  - FGR: `23`, `0`, `0`, `0 tested / 2436 excluded at E-D` (FGR has $n<5$ co-exist in D&E)
  - HDP: `347`, `59`, `18`, `0`
  - sPTB: `378`, `0`, `0`, `skipped for E-D`
  - Complication: `513`, `88`, `23`, `0`
---
### Representative differential-analysis plots
- Proteomics longitudinal heatmap:

![Proteomics complication longitudinal heatmap](./heatmaps/plasma/longitudinal/Complication_longitudinal_heatmap.png)

---
- Metabolomics longitudinal heatmap:

![Metabolomics complication longitudinal heatmap](./heatmaps/metabolomics/plasma/longitudinal/Complication_longitudinal_heatmap.png)

---
- Lipidomics example cross-sectional boxplot:

![Lipidomics example boxplot](./differential_analysis/lipids/plasma/cross_sectional_boxplots/NEG__LPC_20_3__CH3COO_boxplot.png)

---

## Modeling

### Preprocessing before modeling
- Started from cleaned wide matrices.
- Removed metadata columns from predictors.
- Encoded binary outcome as `Control` vs pooled `Complication`.
- For multilabel models, modeled `HDP`, `FGR`, and `sPTB` simultaneously.
- Used stratified `70 / 15 / 15` split into train, validation, and test.
- Applied `RobustScaler` inside folds only.
- Generated Pearson correlation matrices before and after feature selection.
---
- Feature selection:
  - binary: `LogisticRegressionCV` with elastic net penalty (`l1_ratio` grid 0.1→1.0, `solver='saga'`)
  - multilabel: `MultiTaskElasticNetCV` (`l1_ratio` grid 0.1→1.0)
- Hyperparameter tuning:
  - Optuna TPE
  - `n_trials = 50`
- Compared `LogisticRegression`, `RandomForest`, `XGBoost`, and `SVM`.
- Primary evaluation metric: `PR-AUC`.
- Final evaluation done on a held-out test set.

---
### Best binary results by omics
- Proteomics binary:
  - Plasma A: `PR-AUC 0.8827` with LogisticRegression
  - Plasma B: `PR-AUC 0.8906` with LogisticRegression
  - Placenta: `PR-AUC 0.8795` with RandomForest
- Metabolomics binary:
  - Plasma B: `PR-AUC 0.8791` with LogisticRegression
  - Placenta: `PR-AUC 0.7910` with LogisticRegression
- Lipidomics binary:
  - Plasma B: `PR-AUC 0.7746` with XGBoost
- Note:
  - Timepoint E removed from interpretation
---
### Feature-selection behavior
- Proteomics plasma B:
  - `2815 -> 3` features after elastic net
- Proteomics placenta:
  - `2789 -> 6` features after elastic net
- Metabolomics plasma B:
  - `1887 -> 1887` features after elastic net
  - elastic net did not become sparse; model kept the full feature set
- Metabolomics placenta:
  - `2039 -> 2039` features after elastic net
- Lipidomics plasma B:
  - `2436 -> 1075` features after elastic net
---
### Best multilabel results
- Proteomics multilabel:
  - Best strong class-level result was HDP at plasma D with XGBoost, `PR-AUC 0.8875`
  - HDP at plasma B with LogisticRegression `PR-AUC 0.8357`
- Metabolomics multilabel:
  - Best class-level result was FGR at plasma C with RandomForest, `PR-AUC 0.6436`
- Lipidomics multilabel:
  - HDP at plasma C with RandomForest reached `PR-AUC 1.0`, should be overfitting
  - HDP at plasma A with RandomForest  `PR-AUC 0.7497`
---
### Metabolomics ComBat vs. Original pipeline
- Comparing best binary PR-AUC by timepoint:
  - A: standard metabolomics `0.6948` vs ComBat `0.5776`
  - B: standard `0.8791` vs ComBat `0.7466`
  - C: standard `0.7913` vs ComBat `0.7425`
  - D: standard `0.7659` vs ComBat `0.3668`
---
### Representative modeling plots
- Proteomics plasma B PR curve:

![w:500 h:500 Proteomics binary plasma B LogisticRegression PR curve](./models/binary/plasma/B/LogisticRegression_pr_curve.png)

---

- Proteomics plasma B correlation matrix after LASSO:

![w:500 Proteomics binary plasma B correlation matrix post-LASSO](./models/binary/plasma/B/correlation_matrix_postlasso.png)

---
- Metabolomics plasma B feature importance:

![w:500 Metabolomics binary plasma B LogisticRegression feature importance](./models/binary/metabolomics/plasma/B/LogisticRegression_feature_importance.png)

---
- Lipidomics plasma B feature importance:

![w:500 Lipidomics binary plasma B XGBoost feature importance](./models/binary/lipids/plasma/B/XGBoost_feature_importance.png)
