# 03_model_development

Machine learning classification pipeline for DP3 multi-omics data. Predicts pregnancy complications (binary: Control vs Complication; multilabel: HDP, FGR, sPTB simultaneously) from plasma and placenta profiles. Supports both proteomics (Olink NPX) and metabolomics (LC-MS) inputs.

## Scripts

| Script | Description |
|--------|-------------|
| `utilities.py` | Shared helpers: data loading, 70/15/15 split, LASSO feature selection, CV runners, evaluation metrics, plotting, superset feature collection |
| `binary_classifier.py` | Binary classifiers (Control vs pooled Complication) per tissue and timepoint; supports proteomics and metabolomics via `--file-prefix` |
| `multilabel_classifier.py` | Joint multi-label classifier (HDP + FGR + sPTB simultaneously) per tissue and timepoint |
| `run_sop_models.py` | Binary classification on SOP v4 pipeline outputs (MTBL_sop / LIPD_sop); plasma pre-filter uses cross-sectional differential results, placenta pre-filter uses CS ∪ longitudinal union |
| `run_sop_nodiff.py` | Same as `run_sop_models.py` but skips differential pre-filtering — all analyte columns are passed directly to LASSO |
| `run_survey_models.py` | Binary + multilabel classification on survey/environmental data (EPDS, PSS, PUQE-24, water disinfection by-products) at visits A, C, D, PP |
| `run_permutation_test.py` | Permutation test (n=1000) on saved binary model results; validates PR-AUC significance by shuffling training labels |
| `feature_interpretation.py` | SHAP, LIME, and Gini feature importance for trained binary models; generates combined panel plots per tissue/timepoint |
| `superset_differential_analysis.py` | Runs differential analysis restricted to the LASSO superset (union of selected features across timepoints) |
| `superset_enrichment_analysis.py` | Enrichr pathway enrichment (GO:BP, GO:MF, GO:CC, KEGG 2026, Reactome) on the union of LASSO-selected proteomics features across timepoints |
| `metabolomics_enrichment_analysis.py` | KEGG REST API pathway enrichment (ORA) on significant metabolomics analytes from differential analysis |
| `run_pathway_analysis.py` | HMDB/KEGG pathway analysis for metabolomics; alternative enrichment approach using MetaboAnalyst-style name matching |

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

Runs Enrichr pathway enrichment (GO:BP, GO:MF, GO:CC, KEGG 2026, Reactome 2024) on per-set gene lists: one per plasma timepoint, one for placenta, and a union superset. Input is `lasso_selected_features.csv` from binary classifier outputs.

| Flag | Default | Description |
|------|---------|-------------|
| `--binary-results-dir` | `04_results_and_figures/models/binary/` | Root binary model output directory containing `plasma/<TP>/lasso_selected_features.csv` |
| `--output-dir` | `04_results_and_figures/models/binary/superset_enrichment/` | Enrichment output directory |
| `--superset-timepoints` | `A B C D` | Plasma timepoints included in the superset (E excluded by default due to LASSO regularisation collapse) |
| `--fdr-threshold` | `0.05` | Adjusted p-value threshold for significance |

### `run_sop_models.py`

Binary classification on SOP v4 pipeline outputs (MTBL_sop / LIPD_sop). Plasma uses cross-sectional differential pre-filter at each timepoint; placenta uses the union of cross-sectional and longitudinal significant analytes.

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | both | `MTBL_sop` or `LIPD_sop`; if omitted, runs both |
| `--n-trials` | `50` | Optuna trials per model |
| `--skip-plasma` | False | Skip plasma timepoints |
| `--skip-placenta` | False | Skip placenta |

### `run_sop_nodiff.py`

Same pipeline as `run_sop_models.py` but without differential pre-filtering — all analyte columns are passed to LASSO. Useful for ablation comparison.

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | both | `MTBL_sop` or `LIPD_sop` |
| `--n-trials` | `50` | Optuna trials per model |
| `--skip-plasma` | False | Skip plasma |
| `--skip-placenta` | False | Skip placenta |

### `run_survey_models.py`

Runs binary and multilabel classification on survey/environmental datasets. Prepares model-ready matrices from cleaned survey CSVs, then passes each through the standard binary and multilabel pipelines.

| Flag | Default | Description |
|------|---------|-------------|
| `--surveys` | `epds pss puqe24 water` | Survey datasets to include |
| `--n-trials` | `10` | Optuna trials per model (lower than omics default) |

### `run_permutation_test.py`

Validates trained binary model PR-AUC by permuting training labels 1000 times and comparing the null distribution to the observed score.

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `MTBL_sop` | Dataset whose saved results to test (`MTBL_sop` or `LIPD_sop`) |
| `--n-perm` | `1000` | Number of permutations per condition |

### `feature_interpretation.py`

Generates SHAP, LIME, and Gini importance plots for trained binary models, optionally restricted to the LASSO superset. Produces combined panel plots per tissue/timepoint.

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `proteomics` | Dataset name (determines input paths) |
| `--timepoints` | `A B C D E` | Plasma timepoints to interpret |
| `--skip-placenta` | False | Skip placenta interpretation |
| `--n-lime-samples` | `500` | LIME neighbourhood sample size |

### `metabolomics_enrichment_analysis.py`

Runs KEGG REST API over-representation analysis (ORA) on significant metabolites from the metabolomics differential analysis. Uses only named/identified metabolites (unnamed `p####`/`n####` peaks are excluded). Caches API responses to `kegg_api_cache.json` to avoid redundant calls on re-runs.

| Flag | Default | Description |
|------|---------|-------------|
| `--diff-results-dir` | `04_results_and_figures/differential_analysis/metabolomics` | Metabolomics differential results directory |
| `--output-dir` | `04_results_and_figures/models/binary/metabolomics/enrichment/` | Enrichment output directory |
| `--top-n` | `15` | Maximum pathways shown per dot-plot |

**Note:** Enrichr (used for proteomics) requires gene symbols and does not support metabolite names. The KEGG REST API is used instead. Background set = all compounds in KEGG pathways that contain at least one query metabolite (not the full KEGG compound universe).
