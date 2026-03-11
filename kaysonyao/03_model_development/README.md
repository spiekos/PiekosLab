# 03_model_development

Machine learning classification pipeline for DP3 proteomics data. Predicts pregnancy complications (HDP, FGR, sPTB) from Olink NPX plasma and placenta proteomic profiles.

## Scripts

| Script | Description |
|--------|-------------|
| `utilities.py` | Shared helpers: data loading, 70/15/15 split, LASSO feature selection, CV runners, evaluation metrics, plotting |
| `binary_classifier.py` | Per-outcome binary classifiers (Control vs. HDP, Control vs. FGR, Control vs. sPTB) |
| `multilabel_classifier.py` | Joint multi-label classifier (HDP + FGR + sPTB simultaneously) |

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
    │       binary:     PR-AUC objective (single binary outcome)
    │       multilabel: macro-average PR-AUC across all outcomes
    │
    ├─ 10-fold cross-validation on train set (with tuned hyperparameters)
    │       LogisticRegression | RandomForest | XGBoost | SVM
    │
    ├─ Best model (val PR-AUC from Optuna) retrained on train+val
    │
    └─ Final evaluation on held-out test set
```

## Usage

Run all scripts from the **project root** (where `data/` lives):

```bash
# Binary classifiers — all tissues, all timepoints, all outcomes
python 03_model_development/binary_classifier.py

# Multi-label — all tissues, all timepoints
python 03_model_development/multilabel_classifier.py

# Run only plasma, only timepoints A and B
python 03_model_development/binary_classifier.py --timepoints A B --skip-placenta

# Override data paths
python 03_model_development/binary_classifier.py \
    --plasma-dir  data/cleaned/proteomics/normalized_sliced_by_suffix/ \
    --placenta-csv data/cleaned/proteomics/normalized_full_results/proteomics_placenta_cleaned_with_metadata.csv \
    --output-dir  04_results_and_figures/models/binary/
```

## Input data

| Tissue | Path | Notes |
|--------|------|-------|
| Plasma | `data/cleaned/proteomics/normalized_sliced_by_suffix/proteomics_plasma_formatted_suffix_{A-E}.csv` | One CSV per timepoint |
| Placenta | `data/cleaned/proteomics/normalized_full_results/proteomics_placenta_cleaned_with_metadata.csv` | Single CSV |

Expected format: wide-format CSV with `index_col=0` (SampleID), metadata columns (`SubjectID`, `Group`, `Subgroup`, `Batch`, `GestAgeDelivery`, `SampleGestAge`), and one column per analyte.

## Output structure

```
04_results_and_figures/models/
│
├── binary/
│   ├── plasma/
│   │   └── <timepoint A-E>/
│   │       └── <outcome HDP|FGR|sPTB>/
│   │           ├── sample_splits.csv
│   │           ├── correlation_matrix_pretrain.png
│   │           ├── correlation_matrix_postlasso.png
│   │           ├── lasso_selected_features.csv
│   │           ├── cv_results.csv
│   │           ├── test_results.csv
│   │           ├── summary.json
│   │           ├── <BestModel>_pr_curve.png
│   │           ├── <BestModel>_roc_curve.png
│   │           └── <Model>_feature_importance.png
│   ├── placenta/
│   │   └── all/
│   │       └── <outcome>/
│   │           └── (same as above)
│   └── all_results_summary.csv        ← aggregated across all conditions
│
└── multilabel/
    ├── plasma/
    │   └── <timepoint A-E>/
    │       ├── sample_splits.csv
    │       ├── correlation_matrix_pretrain.png
    │       ├── correlation_matrix_postlasso.png
    │       ├── lasso_selected_features.csv
    │       ├── cv_results.csv
    │       ├── test_results.csv
    │       ├── summary.json
    │       ├── <BestModel>_<outcome>_pr_curve.png
    │       ├── <BestModel>_<outcome>_roc_curve.png
    │       └── <Model>_feature_importance_avg.png
    ├── placenta/
    │   └── all/
    │       └── (same as above)
    └── all_results_summary.csv
```

## Evaluation metrics

Primary: **PR-AUC** (average precision), per the R21 grant specification.

Also reported: ROC-AUC, Accuracy, Precision, Recall, F1.

Class imbalance is handled via `class_weight='balanced'` for sklearn models and `scale_pos_weight` for XGBoost.

Scaling: **RobustScaler** (median / IQR) is applied inside every CV fold, tuning trial, and final evaluation — never fit on val or test data.

## Hyperparameter tuning

Both classifiers use **Optuna TPE** (Tree-structured Parzen Estimator) Bayesian optimisation on the validation set. The objective is PR-AUC (binary) or macro-average PR-AUC across all outcomes (multilabel). Tuned params are saved to `tuned_hyperparams.json` per condition.

## CLI flags

### `binary_classifier.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--plasma-dir` | `data/cleaned/proteomics/normalized_sliced_by_suffix/` | Per-timepoint plasma CSVs |
| `--placenta-csv` | `data/cleaned/proteomics/normalized_full_results/proteomics_placenta_cleaned_with_metadata.csv` | Placenta CSV |
| `--output-dir` | `04_results_and_figures/models/binary/` | Root output directory |
| `--timepoints` | `A B C D E` | Plasma timepoints |
| `--complications` | `HDP FGR sPTB` | Labels pooled as "Complication" |
| `--n-trials` | `50` | Optuna trials per model |
| `--skip-plasma` | False | Skip plasma |
| `--skip-placenta` | False | Skip placenta |

### `multilabel_classifier.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--plasma-dir` | `data/cleaned/proteomics/normalized_sliced_by_suffix/` | Per-timepoint plasma CSVs |
| `--placenta-csv` | `data/cleaned/proteomics/normalized_full_results/proteomics_placenta_cleaned_with_metadata.csv` | Placenta CSV |
| `--output-dir` | `04_results_and_figures/models/multilabel/` | Root output directory |
| `--timepoints` | `A B C D E` | Plasma timepoints |
| `--outcomes` | `HDP FGR sPTB` | Outcomes modelled jointly |
| `--n-trials` | `50` | Optuna trials per model |
| `--skip-plasma` | False | Skip plasma |
| `--skip-placenta` | False | Skip placenta |
