## 01_data_cleaning

This directory contains the data preprocessing pipeline for the project. It produces cleaned and harmonized clinical, Fitbit, and placental datasets that feed downstream exploratory analysis and modeling.

## Directory Structure

```
01_data_cleaning/
├── README.md
├── preprocess_clinical.py
├── preprocess_fitbit.py
├── preprocess_placental.py
├── preprocess_correlation.py
└── processed_data/
    ├── master_fitbit_clinical_correlation_data.csv
    ├── processed_clinical_data.csv
    ├── processed_fitbit_data.csv
    └── processed_placental_data.csv
```

## Scripts & Inputs

### `preprocess_clinical.py`

* **Purpose:** Clean and encode clinical metadata for cohort-level analysis.
* **Inputs:**
  * `00_raw_data/dp3 master table v2.xlsx - variables of interest.csv`
  * `00_raw_data/DP3_playset.csv`
* **Key operations:**
  * Standardizes column names and text values to lowercase with underscore separators.
  * Fixes known typos (for example, `africian american` → `african american`).
  * Keeps only patients with `status == delivered`.
  * Restricts records to the DP3 playset patient IDs.
  * Adds missingness indicator columns for `race`, `ethnicity`, and `smoking`.
  * Encodes smoking status and imputes median prepregnancy BMI.
  * One-hot encodes key demographic categories.
* **Output:** `processed_data/processed_clinical_data.csv`

### `preprocess_fitbit.py`

* **Purpose:** Clean Fitbit longitudinal tracking data and apply validity filters.
* **Inputs:**
  * `00_raw_data/DP3_playset.csv`
  * `00_raw_data/DP3-FitbitFullReport_DATA_LABELS_2025-02-18_1356.csv`
  * `00_raw_data/implausible_value_bounds.csv`
* **Key operations:**
  * Standardizes column names and text values.
  * Merges Fitbit sheets on `id` and `date`.
  * Filters to `fitbit data` events.
  * Removes deprecated/redundant Fitbit metric columns.
  * Applies day-level validity checks for wear time and implausible values.
  * Flags or nulls invalid sleep and activity values.
* **Output:** `processed_data/processed_fitbit_data.csv`

### `preprocess_placental.py`

* **Purpose:** Clean, merge, and encode placental histopathology recordings.
* **Inputs:**
  * `00_raw_data/dp3 3rd set gest age for Tony assessed final.xlsx - Sheet1.csv`
  * `00_raw_data/DP3 slides Tony's analysis batches 1-2.xlsx - Sheet2.csv`
  * `01_data_cleaning/processed_data/processed_clinical_data.csv`
  * `00_raw_data/DP3_playset.csv`
* **Key operations:**
  * Standardizes column names and merges the two placental source sheets.
  * Drops patients with missing slides or with all slide indicators marked `x`.
  * Restricts the data to the DP3 playset patient IDs.
  * Drops empty and low-variance pathology features.
  * Encodes text pathology labels into numeric values.
  * Adds a `spontaneous_preterm_birth` indicator derived from the clinical sheet.
* **Output:** `processed_data/processed_placental_data.csv`
* **Additional log:** `04_results_and_figures/data_analysis/placental/sum_placental_histo_features.txt`

### `preprocess_correlation.py`

* **Purpose:** Build the integrated Fitbit–clinical–placental correlation dataset.
* **Inputs:**
  * `01_data_cleaning/processed_data/processed_fitbit_data.csv`
  * `01_data_cleaning/processed_data/processed_placental_data.csv`
  * `01_data_cleaning/processed_data/processed_clinical_data.csv`
* **Key operations:**
  * Filters Fitbit rows to pregnancy-time `fitbit data` events.
  * Buckets longitudinal Fitbit records into five pregnancy windows.
  * Applies per-patient coverage thresholds and averages valid days by trimester.
  * Merges aggregated Fitbit features with clinical delivery variables and placental features.
  * Drops rows missing all placental metrics.
* **Output:** `processed_data/master_fitbit_clinical_correlation_data.csv`

## Execution

Run the full data cleaning pipeline from the repository root:

```bash
python 01_data_cleaning/preprocess_clinical.py
python 01_data_cleaning/preprocess_fitbit.py
python 01_data_cleaning/preprocess_placental.py
python 01_data_cleaning/preprocess_correlation.py
```

> Note: Each script uses raw files from `00_raw_data/` and expects those files to exist in the repository before running.
