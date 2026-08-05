# 03_model_development

This directory contains the modeling workflow for Fitbit-derived predictors and placental histopathology outcomes.

## Contents

- `run_fitbit_glm_interactions.py` — runs a generalized linear modeling pipeline that tests Fitbit metrics against placental outcomes, adjusting for clinical covariates and interaction terms.

## Workflow

### `run_fitbit_glm_interactions.py`

* **Purpose:** Fit GLMs for Fitbit activity/sleep/heart rate metrics with placental pathology and spontaneous preterm birth outcomes.
* **Inputs:**
  * `01_data_cleaning/processed_data/processed_fitbit_data.csv`
  * `01_data_cleaning/processed_data/processed_placental_data.csv`
  * `01_data_cleaning/processed_data/processed_clinical_data.csv`
* **Modeling strategy:**
  * Merges Fitbit, placental, and clinical datasets by `id`.
  * Converts placeholder values such as `"no value"` and `-1.0` to `NaN` for numeric Fitbit features.
  * Uses `maternal_age` as the primary predictor and fits separate GLMs for each Fitbit metric and each placental outcome.
  * Adjusts for covariates: prepregnancy_bmi_self_or_record, race, infant sex, smoking, and parity.
  * Determines the GLM family automatically based on the outcome distribution:
    * `Binomial` with logit link for binary outcomes.
    * `Gamma` with log link for non-negative, skewed continuous outcomes.
    * `Gaussian` with identity link for continuous outcomes that are not heavily skewed.
  * Uses heteroskedasticity-robust standard errors when extreme outcome outliers are detected.

## Outputs

- `04_results_and_figures/models/glm_fitbit_analysis.log` — logging details for model fitting failures.
- `04_results_and_figures/models/final_glm_results.csv` — aggregated model results including coefficients, p-values, convergence status, pseudo R-squared, and selected family/link information.

## Run instructions

From the repository root, execute:

```bash
python anushas/03_model_development/run_fitbit_glm_interactions.py
```

## Notes

- The script currently uses a hard-coded set of Fitbit metrics and placental outcomes.
- Only Fitbit metrics with at least 10 non-missing values are retained for modeling.
