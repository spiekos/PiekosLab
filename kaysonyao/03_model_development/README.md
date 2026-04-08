# 03_model_development

Machine learning classification pipeline for DP3 multi-omics data. Predicts pregnancy complications (binary: Control vs Complication; multilabel: HDP, FGR, sPTB simultaneously) from plasma and placenta profiles. Supports both proteomics (Olink NPX) and metabolomics (LC-MS) inputs.

## Scripts

| Script | Description |
|--------|-------------|
| `utilities.py` | Shared helpers: data loading, 70/15/15 split, LASSO feature selection, CV runners, evaluation metrics, plotting |
| `binary_classifier.py` | Binary classifiers (Control vs pooled Complication) per tissue and timepoint |
| `multilabel_classifier.py` | Joint multi-label classifier (HDP + FGR + sPTB simultaneously) per tissue and timepoint |
| `superset_enrichment_analysis.py` | Enrichr pathway enrichment on the union of LASSO-selected proteomics features across all significant timepoints |
| `metabolomics_enrichment_analysis.py` | KEGG REST API pathway enrichment (ORA) on significant metabolomics analytes from differential analysis |
| `superset_differential_analysis.py`| Same differential analysis pipeline can be conducted again on the superset if necessary|

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
# ── Proteomics ──────────────────────────────────────────────────────────────

# Binary classifiers — all tissues, all timepoints
python 03_model_development/binary_classifier.py

# Multi-label — all tissues, all timepoints
python 03_model_development/multilabel_classifier.py

# Run only plasma, only timepoints A and B
python 03_model_development/binary_classifier.py --timepoints A B --skip-placenta

# Superset enrichment (proteomics LASSO features → Enrichr)
python 03_model_development/superset_enrichment_analysis.py

# ── Metabolomics ─────────────────────────────────────────────────────────────

# Binary classifiers (use --file-prefix to resolve metabolomics_plasma_* filenames)
python 03_model_development/binary_classifier.py \
    --plasma-dir  data/cleaned/metabolomics/normalized_sliced_by_suffix/ \
    --placenta-csv data/cleaned/metabolomics/normalized_full_results/metabolomics_placenta_cleaned_with_metadata.csv \
    --output-dir  04_results_and_figures/models/binary/metabolomics/ \
    --file-prefix metabolomics

# Multi-label
python 03_model_development/multilabel_classifier.py \
    --plasma-dir  data/cleaned/metabolomics/normalized_sliced_by_suffix/ \
    --placenta-csv data/cleaned/metabolomics/normalized_full_results/metabolomics_placenta_cleaned_with_metadata.csv \
    --output-dir  04_results_and_figures/models/multilabel/metabolomics/ \
    --file-prefix metabolomics

# KEGG pathway enrichment (metabolomics — uses KEGG REST API, not Enrichr)
python 03_model_development/metabolomics_enrichment_analysis.py
```

## Input data

| Omics | Tissue | Path | Notes |
|-------|--------|------|-------|
| Proteomics | Plasma | `data/cleaned/proteomics/normalized_sliced_by_suffix/proteomics_plasma_formatted_suffix_{A-E}.csv` | One CSV per timepoint |
| Proteomics | Placenta | `data/cleaned/proteomics/normalized_full_results/proteomics_placenta_cleaned_with_metadata.csv` | Single CSV |
| Metabolomics | Plasma | `data/cleaned/metabolomics/normalized_sliced_by_suffix/metabolomics_plasma_formatted_suffix_{A-E}.csv` | One CSV per timepoint |
| Metabolomics | Placenta | `data/cleaned/metabolomics/normalized_full_results/metabolomics_placenta_cleaned_with_metadata.csv` | Single CSV |

Expected format: wide-format CSV with `index_col=0` (SampleID), metadata columns (`Group`, `Subgroup`, etc.), and one column per analyte. Both proteomics and metabolomics values are in log2 scale.

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
│   │   └── all/
│   │       └── <outcome>/
│   │           └── (same as above)
│   ├── all_results_summary.csv          ← aggregated across all conditions
│   ├── enrichment/                      ← superset_enrichment_analysis.py
│   │   ├── superset_features.csv
│   │   ├── superset_enrichment_*.png
│   │   └── <database>_enrichment.csv
│   └── metabolomics/                   ← binary_classifier.py --file-prefix metabolomics
│       ├── plasma/
│       │   └── <timepoint A-E>/
│       │       └── <outcome>/
│       │           └── (same per-condition structure as proteomics)
│       ├── placenta/
│       │   └── all/ …
│       ├── all_results_summary.csv
│       └── enrichment/                 ← metabolomics_enrichment_analysis.py
│           ├── kegg_api_cache.json
│           ├── analysis_log.txt
│           ├── summary/
│           │   └── <set>_kegg_enrichment.png
│           └── <set>/
│               └── kegg_pathway_enrichment.csv
│
└── multilabel/
    ├── plasma/                          (proteomics)
    │   └── <timepoint A-E>/
    │       ├── sample_splits.csv
    │       ├── correlation_matrix_pretrain.png
    │       ├── correlation_matrix_postlasso.png
    │       ├── lasso_selected_features.csv
    │       ├── tuned_hyperparams.json
    │       ├── cv_results.csv
    │       ├── test_results.csv
    │       ├── summary.json
    │       ├── <BestModel>_<outcome>_pr_curve.png
    │       ├── <BestModel>_<outcome>_roc_curve.png
    │       └── <Model>_feature_importance_avg.png
    ├── placenta/
    │   └── all/ …
    ├── all_results_summary.csv
    └── metabolomics/                   ← multilabel_classifier.py --file-prefix metabolomics
        └── (same structure as proteomics multilabel)
```

## Evaluation metrics

Primary: **PR-AUC** (average precision), per the R21 grant specification.

Also reported: ROC-AUC, Accuracy, Precision, Recall, F1.

Class imbalance is handled via `class_weight='balanced'` for sklearn models and `scale_pos_weight` for XGBoost.

Scaling: **RobustScaler** (median / IQR) is applied inside every CV fold, tuning trial, and final evaluation — never fit on val or test data.

## Hyperparameter tuning

Both classifiers use **Optuna TPE** (Tree-structured Parzen Estimator) Bayesian optimisation on the validation set. The objective is PR-AUC (binary) or macro-average PR-AUC across all outcomes (multilabel). Tuned params are saved to `tuned_hyperparams.json` per condition.

Requires `optuna` (`pip install optuna`).

## Notes on metabolomics models

- **LASSO collapse:** All metabolomics plasma timepoints select all 1,887 features (LASSO does not find a sparse solution). This reflects the absence of strong cross-sectional signal — the elastic-net regularisation path does not converge to a sparse subset. Enrichment is therefore run on significant differential analytes from the differential analysis, not on LASSO features.
- **No placenta metabolomics enrichment:** Metabolomics enrichment is plasma-only (matching where longitudinal signal was found).
- **KEGG vs Enrichr:** Metabolomics enrichment uses the KEGG REST API because Enrichr requires gene symbols. Results are not directly comparable to the proteomics Enrichr output.

## CLI flags

### `binary_classifier.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--plasma-dir` | `data/cleaned/proteomics/normalized_sliced_by_suffix/` | Per-timepoint plasma CSVs |
| `--placenta-csv` | `data/cleaned/proteomics/normalized_full_results/proteomics_placenta_cleaned_with_metadata.csv` | Placenta CSV |
| `--output-dir` | `04_results_and_figures/models/binary/` | Root output directory |
| `--file-prefix` | `proteomics` | Filename prefix for plasma CSVs. Use `metabolomics` when running on metabolomics data (resolves `metabolomics_plasma_formatted_suffix_{tp}.csv`). |
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
| `--file-prefix` | `proteomics` | Filename prefix for plasma CSVs. Use `metabolomics` when running on metabolomics data. |
| `--timepoints` | `A B C D E` | Plasma timepoints |
| `--outcomes` | `HDP FGR sPTB` | Outcomes modelled jointly |
| `--n-trials` | `50` | Optuna trials per model |
| `--skip-plasma` | False | Skip plasma |
| `--skip-placenta` | False | Skip placenta |

### `superset_enrichment_analysis.py`

Runs Enrichr pathway enrichment on the union of LASSO-selected proteomics features across all plasma timepoints and placenta where the binary model achieved PR-AUC ≥ 0.6.

| Flag | Default | Description |
|------|---------|-------------|
| `--models-dir` | `04_results_and_figures/models/binary/` | Root binary model output directory |
| `--output-dir` | `04_results_and_figures/models/binary/enrichment/` | Enrichment output directory |
| `--gene-sets` | `GO_Biological_Process_2025 KEGG_2026 Reactome_Pathways_2024` | Enrichr databases to query |
| `--pr-auc-threshold` | `0.6` | Minimum PR-AUC for a timepoint to contribute to the superset |

### `metabolomics_enrichment_analysis.py`

Runs KEGG REST API over-representation analysis (ORA) on significant metabolites from the metabolomics differential analysis. Uses only named/identified metabolites (unnamed `p####`/`n####` peaks are excluded). Caches API responses to `kegg_api_cache.json` to avoid redundant calls on re-runs.

| Flag | Default | Description |
|------|---------|-------------|
| `--diff-results-dir` | `04_results_and_figures/differential_analysis/metabolomics` | Metabolomics differential results directory |
| `--output-dir` | `04_results_and_figures/models/binary/metabolomics/enrichment/` | Enrichment output directory |
| `--top-n` | `15` | Maximum pathways shown per dot-plot |

**Note:** Enrichr (used for proteomics) requires gene symbols and does not support metabolite names. The KEGG REST API is used instead. Background set = all compounds in KEGG pathways that contain at least one query metabolite (not the full KEGG compound universe).
