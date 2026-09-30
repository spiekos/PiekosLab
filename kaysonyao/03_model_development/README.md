# 03 Model Development

Run from the project root.

## Current pipeline (Aim 3A base models, gestational windows)

| Script | Purpose |
|---|---|
| `run_base_models.py` | Base models per dataset x window (plus placenta), development participants only |
| `nested_cv.py` | Nested CV: 5 outer x 3 inner `StratifiedGroupKFold` by SubjectID; imputation, scaling and elastic-net selection inside the Pipeline; pooled out-of-fold PR/ROC with 1000x bootstrap CI |
| `optuna_tuning.py` | Optuna TPE search (40 trials) inside each outer fold, scored on inner folds only |
| `run_holdout_evaluation.py` | One-look evaluation on the locked 30% test set; refuses to overwrite a previous result |
| `utilities.py` | Shared helpers (sklearn estimators use `n_jobs=1`, see comments) |

```bash
# Windows produced by 01_data_cleaning/bin_echo_windows.py --scheme dp3_5t
python 03_model_development/run_base_models.py --windows T1 T2 T3 T4 T5 \
    --datasets MTBL_plasma LIPD_plasma MTBL_urine proteomics_plasma \
    --out 04_results_and_figures/models_dp3_5t

# Final held-out evaluation (needs --fit-split dev preprocessing; run once)
python 03_model_development/run_holdout_evaluation.py \
    --windows-root data/processed/windows_devfit --placenta-root data/processed_devfit \
    --windows T1 T2 T3 T4 T5 --out 04_results_and_figures/holdout_dp3_5t
```

`run_base_models.py` defaults: `--windows-root data/processed/windows`,
`--placenta-root data/processed` (reads `<root>/<MTBL|LIPD>/placenta/<DS>_cleaned_with_metadata.csv`),
`--windows W1_early W2_mid W3_third` (pass `--windows` explicitly), `--skip-placenta` to omit placenta.
Split: `data/processed/locked_split.csv`.

Per dataset/window it writes `summary.json` (headline metrics; `roc_auc_oof` is the pooled
value), `cv_results.csv`, `oof_predictions.csv`, `tuned_hyperparams.json`, `selected_features.csv`,
fitted models and PR/ROC/importance plots; `<out>/model_metrics_all.csv` and `model_metrics_<assay>.csv` summarise all runs.

## Retired

SVM was removed from the model list on 2026-09-30. The A-E-era tools (`binary_classifier.py`,
`multilabel_classifier.py`, `run_survey_models.py`, `run_permutation_test.py`,
`feature_interpretation.py`, the superset / enrichment / pathway scripts and this folder's
`utilities.py`) were retired the same day and are in `_legacy/scripts/a_e_tools/`. Earlier
superseded scripts (`run_sop_models.py`, `run_sop_nodiff.py`, `run_echo_base_models.py`) are in
`_legacy/scripts/`.
