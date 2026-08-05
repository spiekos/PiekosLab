## 02_exploratory_analysis

This directory contains the exploratory data analysis scripts for the cleaned Fitbit, clinical, and placental datasets. The folder currently includes analysis workflows for clinical missingness, Fitbit longitudinal data quality, correlation tests, and pregnancy timeline histograms.

## Directory Structure

```text
02_exploratory_analysis/
├── README.md
├── analyze_clinical.py
├── analyze_fitbit.py
└── correlation.py
```

## Output Locations

The scripts in this folder write results into the project-level `04_results_and_figures/` directory.

- `04_results_and_figures/data_analysis/clinical/`
- `04_results_and_figures/data_analysis/fitbit/`
- `04_results_and_figures/correlations/`

## Script Overview

### `analyze_clinical.py`

* **Purpose:** Summarize clinical demographic data and track missingness patterns.
* **Inputs:**
  * `01_data_cleaning/processed_data/processed_clinical_data.csv`
  * `01_data_cleaning/processed_data/processed_placental_data.csv`
  * `01_data_cleaning/processed_data/processed_fitbit_data.csv`
  * `00_raw_data/dp3 master table v2.xlsx - clinical data.csv`
  * `00_raw_data/dp3 master table v2.xlsx - Sheet1.csv`
* **Key operations:**
  * Normalizes split header structure from the raw clinical data file.
  * Computes feature-level missingness for maternal age, infant sex, prepregnancy BMI, race, ethnicity, and smoking.
  * Builds race/ethnicity intersection counts and stratified control/adverse tables.
  * Generates summary tables for continuous and categorical clinical features.
* **Outputs:**
  * `04_results_and_figures/data_analysis/clinical/clinical_summary_tables.txt`
  * `04_results_and_figures/data_analysis/clinical/clinical_data_analysis.txt`

### `analyze_fitbit.py`

* **Purpose:** Profile Fitbit data completeness, feature-level missingness, and per-patient data density.
* **Inputs:**
  * `01_data_cleaning/processed_data/processed_fitbit_data.csv`
  * `01_data_cleaning/processed_data/processed_clinical_data.csv`
* **Key operations:**
  * Filters for pregnancy-period Fitbit data and excludes general metadata rows.
  * Splits valid tracking days into five bins: `first`, `early_second`, `late_second_early_third`, `mid_third`, `late_third`.
  * Computes per-patient missing-day counts, maximum consecutive missing streaks, and data coverage matrices.
  * Creates pregnancy-week histograms of patient Fitbit data availability.
  * Generates violin/box plot summaries for first-trimester Fitbit metrics by control status.
* **Outputs:**
  * `04_results_and_figures/data_analysis/fitbit/fitbit_data_analysis.txt`
  * `04_results_and_figures/data_analysis/fitbit/pregnancy_plots_report.pdf`
  * `04_results_and_figures/data_analysis/fitbit/violin_box_plots.pdf`
  * `04_results_and_figures/data_analysis/fitbit/patients_per_feature_per_bin.csv`
  * `04_results_and_figures/data_analysis/fitbit/omics_patients_per_feature_per_bin.csv`
  * `04_results_and_figures/data_analysis/fitbit/missing_per_feature_per_bin.csv`

### `correlation.py`

* **Purpose:** Run correlation and distribution tests across placental, clinical, and Fitbit variables.
* **Inputs:**
  * `01_data_cleaning/processed_data/processed_placental_data.csv`
  * `00_raw_data/dp3 master table v2.xlsx - variables of interest.csv`
  * `01_data_cleaning/processed_data/processed_clinical_data.csv`
  * `01_data_cleaning/processed_data/processed_fitbit_data.csv`
  * `01_data_cleaning/processed_data/master_fitbit_clinical_correlation_data.csv`
* **Key operations:**
  * Test 1: Spearman correlations between placental histopathology and delivery features.
  * Test 2: Spearman correlations between Fitbit metrics and delivery/placental outcomes.
  * Test 3: Demographics collinearity analysis with a heatmap and VIF scores.
  * Test 4: Distribution comparisons of Fitbit features between control and complication groups during pregnancy.
* **Outputs:**
  * `04_results_and_figures/correlations/test1/full_correlation_table_placental.txt`
  * `04_results_and_figures/correlations/test1/filtered_correlation_table_placental.txt`
  * `04_results_and_figures/correlations/test1/positively_associated_vars_placental.txt`
  * `04_results_and_figures/correlations/test1/negatively_associated_vars_placental.txt`
  * `04_results_and_figures/correlations/test2/full_correlation_table_fitbit.txt`
  * `04_results_and_figures/correlations/test2/filtered_correlation_table_fitbit.txt`
  * `04_results_and_figures/correlations/test2/positively_associated_vars_fitbit.txt`
  * `04_results_and_figures/correlations/test2/negatively_associated_vars_fitbit.txt`
  * `04_results_and_figures/correlations/test3/demographics_correlation_heatmap.png`
  * `04_results_and_figures/correlations/test3/demographics_vif.txt`
  * `04_results_and_figures/correlations/test3/demographics_correlation_matrix.csv`
  * `04_results_and_figures/correlations/test4/fitbit_differential_distribution.csv`
  * `04_results_and_figures/correlations/test4/fitbit_differential_distribution_significant.csv`

## Recommended Execution Order

1. Run the `01_data_cleaning` preprocessing pipeline first.
2. Run `02_exploratory_analysis/analyze_clinical.py`.
3. Run `02_exploratory_analysis/analyze_fitbit.py`.
4. Run `02_exploratory_analysis/correlation.py`.

## Notes

* The exploratory scripts rely on cleaned files from `01_data_cleaning/processed_data/` and a few raw `00_raw_data/` sources.
