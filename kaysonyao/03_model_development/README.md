# 03 Model Development

Machine learning classification pipeline for DP3 multi-omics and survey data. Predicts pregnancy
complications (binary: Control vs. Complication; multilabel: HDP, FGR, sPTB simultaneously) from
plasma and placenta profiles.

## Scripts

| Script | Description |
|---|---|
| `utilities.py` | Shared helpers: data loading, 70/15/15 split, LASSO feature selection, CV runners, evaluation metrics, plotting, superset feature collection |
| `binary_classifier.py` | Binary classifiers (Control vs. pooled Complication) per tissue and timepoint; proteomics default |
| `multilabel_classifier.py` | Joint multi-label classifier (HDP + FGR + sPTB) per tissue and timepoint; proteomics default |
| `run_sop_models.py` | Binary classification on SOP v4 outputs (MTBL_sop / LIPD_sop) with differential pre-filtering |
| `run_sop_nodiff.py` | Same as `run_sop_models.py` but skips differential pre-filtering (ablation) |
| `run_survey_models.py` | Binary + multilabel classification on survey/environmental data (EPDS, PSS, PUQE-24, water) |
| `run_permutation_test.py` | Permutation test (n=1000) on saved binary model PR-AUC |
| `feature_interpretation.py` | SHAP, LIME, and Gini importance for trained binary models |
| `superset_differential_analysis.py` | Differential analysis restricted to the LASSO superset |
| `superset_enrichment_analysis.py` | Enrichr enrichment (GO:BP/MF/CC, KEGG, Reactome) on LASSO-selected proteomics features |
| `metabolomics_enrichment_analysis.py` | KEGG REST API pathway enrichment on significant metabolomics analytes |
| `run_pathway_analysis.py` | HMDB/KEGG pathway analysis for metabolomics via MetaboAnalyst-style name matching |

`utilities.py` imports the shared data helpers (`load_data`, `get_analyte_columns`,
`normalise_group_labels`, `METADATA_COLS`) from `01_data_cleaning/utilities.py` via `importlib`.

---

## Pipeline overview

```
Cleaned CSV
    │
    ├─ 70 / 15 / 15 stratified split
    │
    ├─ Pearson correlation matrix (training features, pre-LASSO)
    │
    ├─ LASSO feature selection
    │       binary:     L1 LogisticRegressionCV (saga solver)
    │       multilabel: MultiTaskLassoCV
    │
    ├─ Pearson correlation matrix (post-LASSO features)
    │
    ├─ Optuna TPE hyperparameter tuning (n_trials=50 per model)
    │       Trains on X_train, scores PR-AUC on X_val
    │
    ├─ 10-fold cross-validation on train set (tuned hyperparameters)
    │       LogisticRegression | RandomForest | XGBoost | SVM
    │
    ├─ Best model (val PR-AUC) retrained on train+val
    │
    └─ Final evaluation on held-out test set
```

---

## Usage

Run all scripts from the **project root**:

```bash
# ── Proteomics ──────────────────────────────────────────────────────────────

# Binary classifiers — all tissues, all timepoints
python 03_model_development/binary_classifier.py

# Multi-label — all tissues, all timepoints
python 03_model_development/multilabel_classifier.py

# Plasma only, timepoints A and B
python 03_model_development/binary_classifier.py --timepoints A B --skip-placenta

# Superset enrichment (proteomics LASSO features → Enrichr)
python 03_model_development/superset_enrichment_analysis.py

# ── SOP v4 (metabolomics / lipidomics) ──────────────────────────────────────

# Binary classifiers with differential pre-filtering
python 03_model_development/run_sop_models.py --dataset MTBL_sop
python 03_model_development/run_sop_models.py --dataset LIPD_sop

# Binary classifiers without pre-filtering (ablation)
python 03_model_development/run_sop_nodiff.py

# Permutation test on saved SOP model results
python 03_model_development/run_permutation_test.py --dataset MTBL_sop

# ── Survey / Environmental ───────────────────────────────────────────────────

python 03_model_development/run_survey_models.py

# ── Enrichment ───────────────────────────────────────────────────────────────

# KEGG enrichment on metabolomics differential analytes
python 03_model_development/metabolomics_enrichment_analysis.py

# HMDB/KEGG pathway analysis (MetaboAnalyst-style)
python 03_model_development/run_pathway_analysis.py
```

---

## Input data

| Omics | Tissue | Path | Notes |
|---|---|---|---|
| Proteomics | Plasma | `data/cleaned/proteomics/normalized_sliced_by_suffix/proteomics_plasma_formatted_suffix_{A-E}.csv` | One CSV per timepoint |
| Proteomics | Placenta | `data/cleaned/proteomics/normalized_full_results/proteomics_placenta_cleaned_with_metadata.csv` | Single CSV |
| MTBL_sop | Plasma | `data/cleaned/sop_omics_pipeline_v2/MTBL_plasma/MTBL_plasma_suffix_{A-E}.csv` | One CSV per timepoint |
| MTBL_sop | Placenta | `data/cleaned/sop_omics_pipeline_v2/MTBL_placenta/MTBL_placenta_cleaned_with_metadata.csv` | Single CSV |
| LIPD_sop | Plasma | `data/cleaned/sop_omics_pipeline_v2/LIPD_plasma/LIPD_plasma_suffix_{A-E}.csv` | One CSV per timepoint |

All inputs are wide-format CSV with `index_col=0` (SampleID), metadata columns (`Group`,
`Subgroup`, etc.), and one column per analyte. All values are in log2 scale.

---

## Output structure

```
04_results_and_figures/models/
│
├── binary/
│   ├── plasma/                         (proteomics)
│   │   └── <timepoint A-E>/
│   │       └── <outcome HDP|FGR|sPTB>/
│   │           ├── sample_splits.csv
│   │           ├── correlation_matrix_pretrain.png
│   │           ├── correlation_matrix_postlasso.png
│   │           ├── lasso_selected_features.csv
│   │           ├── tuned_hyperparams.json
│   │           ├── cv_results.csv
│   │           ├── test_results.csv
│   │           ├── summary.json
│   │           ├── <BestModel>_pr_curve.png
│   │           ├── <BestModel>_roc_curve.png
│   │           └── <Model>_feature_importance.png
│   ├── placenta/                        (proteomics)
│   ├── all_results_summary.csv
│   ├── superset_enrichment/             ← superset_enrichment_analysis.py
│   │   ├── superset_features.csv
│   │   └── <database>_enrichment.csv
│   └── metabolomics/                   ← metabolomics_enrichment_analysis.py enrichment outputs
│
├── multilabel/
│   ├── plasma/ placenta/               (proteomics)
│   └── all_results_summary.csv
│
├── sop_models/                         ← run_sop_models.py
│   ├── MTBL_sop/
│   │   ├── plasma/<timepoint>/<outcome>/
│   │   └── placenta/all/<outcome>/
│   └── LIPD_sop/
│       └── plasma/<timepoint>/<outcome>/
│
├── sop_nodiff/                         ← run_sop_nodiff.py
│
└── survey/                             ← run_survey_models.py
    └── binary/ multilabel/
```

---

## Evaluation metrics

Primary: **PR-AUC** (average precision), per the R21 grant specification.

Also reported: ROC-AUC, Accuracy, Precision, Recall, F1.

Class imbalance is handled via `class_weight='balanced'` (sklearn) and `scale_pos_weight` (XGBoost).

Scaling: **RobustScaler** (median / IQR) is applied inside every CV fold, tuning trial, and final
evaluation — never fit on val or test data.

---

## CLI flags

### `binary_classifier.py` / `multilabel_classifier.py`

| Flag | Default | Description |
|---|---|---|
| `--plasma-dir` | `data/cleaned/proteomics/normalized_sliced_by_suffix/` | Per-timepoint plasma CSVs |
| `--placenta-csv` | `data/cleaned/proteomics/normalized_full_results/proteomics_placenta_cleaned_with_metadata.csv` | Placenta CSV |
| `--output-dir` | `04_results_and_figures/models/binary/` | Root output directory |
| `--file-prefix` | `proteomics` | Filename prefix for plasma CSVs |
| `--timepoints` | `A B C D E` | Plasma timepoints |
| `--complications` | `HDP FGR sPTB` | Labels pooled as Complication |
| `--n-trials` | `50` | Optuna trials per model |
| `--skip-plasma` | False | Skip plasma |
| `--skip-placenta` | False | Skip placenta |

### `run_sop_models.py` / `run_sop_nodiff.py`

| Flag | Default | Description |
|---|---|---|
| `--dataset` | both | `MTBL_sop` or `LIPD_sop` |
| `--n-trials` | `50` | Optuna trials per model |
| `--skip-plasma` | False | Skip plasma |
| `--skip-placenta` | False | Skip placenta |

### `run_survey_models.py`

| Flag | Default | Description |
|---|---|---|
| `--surveys` | `epds pss puqe24 water` | Survey datasets to include |
| `--n-trials` | `10` | Optuna trials per model |

### `run_permutation_test.py`

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `MTBL_sop` | Dataset to test |
| `--n-perm` | `1000` | Number of permutations |

### `feature_interpretation.py`

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `proteomics` | Dataset name (determines input paths) |
| `--timepoints` | `A B C D E` | Plasma timepoints |
| `--skip-placenta` | False | Skip placenta |
| `--n-lime-samples` | `500` | LIME neighbourhood sample size |

### `superset_enrichment_analysis.py`

| Flag | Default | Description |
|---|---|---|
| `--binary-results-dir` | `04_results_and_figures/models/binary/` | Binary model results root |
| `--output-dir` | `04_results_and_figures/models/binary/superset_enrichment/` | Enrichment output directory |
| `--superset-timepoints` | `A B C D` | Timepoints in the superset |
| `--fdr-threshold` | `0.05` | Adjusted p-value threshold |

### `metabolomics_enrichment_analysis.py`

| Flag | Default | Description |
|---|---|---|
| `--diff-results-dir` | `04_results_and_figures/differential_analysis/metabolomics` | Metabolomics differential results |
| `--output-dir` | `04_results_and_figures/models/binary/metabolomics/enrichment/` | Enrichment output directory |
| `--top-n` | `15` | Maximum pathways per dot-plot |
