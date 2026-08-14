# 03_model_development

This directory contains the modeling workflow used to test associations between Fitbit-derived predictors and clinical/demographic outcomes.

## Contents

- `run_fitbit_glm_interactions.py` — GLM pipeline that fits models for combinations of Fitbit metrics (predictors) and clinical outcomes (responses).

## Summary

`run_fitbit_glm_interactions.py` loads cleaned Fitbit, placental, and clinical tables from `anushas/01_data_cleaning/processed_data/`, merges them on `id`, applies basic placeholder cleaning, and fits a separate generalized linear model for each Fitbit metric vs each clinical outcome while adjusting for a small set of covariates.

## Exact inputs

- `anushas/01_data_cleaning/processed_data/processed_fitbit_data.csv`
- `anushas/01_data_cleaning/processed_data/processed_placental_data.csv`
- `anushas/01_data_cleaning/processed_data/processed_clinical_data.csv`

## Fitbit metrics considered (explicit list in the script)

- `activities_summary_activitycalories`
- `activities_summary_caloriesout`
- `activities_summary_fairlyactiveminutes`
- `activities_summary_lightlyactiveminutes`
- `activities_summary_sedentaryminutes`
- `heart_rate_resting_heart_rate`
- `heart_rate_zone:_fat_burn_minutes`
- `heart_rate_zone:_cardio_minutes`
- `heart_rate_zone:_peak_minutes`
- `sleep_summary_stages_deep`
- `sleep_summary_stages_light`
- `sleep_summary_stages_rem`
- `sleep_summary_stages_wake`
- `sleep_summary_total_sleep_records`
- `sleep_summary_total_time_in_bed`
- `activities_summary_steps`
- `activities_summary_totaldistances`
- `activities_summary_veryactiveminutes`
- `sleep_summary_total_minutes_asleep`

Note: The script filters this list at runtime and only includes metrics with >10 non-missing numeric observations in the source Fitbit sheet.

## Clinical/outcome variables used (example list in the script)

- `maternal_age`
- `weight_(kg)`
- `prepregnancy_weight_self_or_record`
- `prepregnancy_bmi_self_or_record`
- `gravida`
- `parity`
- `gest_age_del`
- `birthweight`
- `apgar_1`
- `apgar_5`
- `nicu_days`

## Modeling strategy (details)

- Merge order: Fitbit <- placental <- clinical (inner merges on `id`). Columns with `_x`/`_y` suffixes are reconciled by filling `_x` with `_y` when present, then dropping suffixed columns.
- Placeholder cleaning: converts string placeholders like `"no value"` and sentinel values such as `-1.0` to `NaN` for numeric Fitbit features prior to numeric conversion.
- Covariates: the base covariate set is `['maternal_age','prepregnancy_bmi_self_or_record','race','infant_sex','smoking','parity']`. The target outcome variable is excluded from the covariate set when applicable.
- Missingness: each model is fit on the complete-case subset for the predictor, outcome, and selected covariates; models with fewer than 15 rows are skipped.
- Outcome family selection:
  - If the outcome is binary (values subset of {0,1}) a Binomial family with Logit link is used.
  - Otherwise, a Gaussian family with identity link is the default.
  - If the continuous outcome is heavily right-skewed (absolute skew > 1.5 or KS-test p < 0.05) and non-negative, the script switches to a Gamma family with Log link. Zero values are offset with a small positive value before Gamma fitting.
- Outlier handling: if extreme outliers are detected in the outcome using a 3*IQR rule, the GLM is fit with heteroskedasticity-robust standard errors (`cov_type='hc0'`).

## Output files

- `anushas/04_results_and_figures/models/glm_fitbit_analysis.log` — detailed info and exception messages captured during model looping.
- `anushas/04_results_and_figures/models/final_glm_results.csv` — aggregated results table. Columns include at minimum:
  - `Fitbit_Metric`, `Outcome`, `N`, `Model_Family`, `Link_Function`, `Extreme_Outliers`, `Converged`, `Pseudo_R_Squared`
  - `coef_<param>` for each estimated parameter
  - `p_<param>` for each parameter p-value

## Run instructions

From the repository root run:

```bash
python anushas/03_model_development/run_fitbit_glm_interactions.py
```

The script has no CLI arguments; edit the script directly to change the metric/outcome lists or thresholds.

## Dependencies (recommended)

- Python 3.8+ (the environment used for development: conda/miniforge)
- pandas
- numpy
- scipy
- statsmodels

Install with pip:

```bash
pip install pandas numpy scipy statsmodels
```

## Troubleshooting & notes

- If `final_glm_results.csv` is empty or missing many combinations, check that the processed input CSVs have overlapping `id` values and that Fitbit metrics contain enough non-missing numeric values.
- Permission errors when writing results usually indicate that the `anushas/04_results_and_figures/models/` folder does not exist or lacks write permission — create the folder before running if needed.
- The model list and clinical variables are defined in the script as Python lists; modify them there to run a different set.
