# 01_data_cleaning

This directory houses the source code and resulting outputs for the data preprocessing pipeline. It handles data ingestion, schema harmonization, missing data filtration, and categorical encoding for both Fitbit and placental histopathology datasets.

## Directory Structure

```text
01_data_cleaning/
├── README.md                          # This file
├── preprocess_fitbit.py               # Script to clean and filter Fitbit data
├── preprocess_placental.py            # Script to clean and encode histopathology data
└── processed_data/                    # Output directory for cleaned datasets
    ├── README.md                      # Detailed data dictionary and pipeline log
    ├── processed_fitbit_data.csv
    └── processed_placental_data.csv
```

---

## Scripts Overview

### `preprocess_fitbit.py`

* **Purpose:** Cleans raw Fitbit tracking data and enforces quality control based on missingness thresholds.
* **Key Operations:** Merges multiple data sheets, standardizes column nomenclature, and references an auxiliary missing-data table.
* **Quality Gate:** Drops any patient cohort record containing 2 or more consecutive days of missing data for any tracked feature.
* **Output Destination:** `processed_data/processed_fitbit_data.csv`

### `preprocess_placental.py`

* **Purpose:** Prepares raw placental histopathology records for downstream statistical modeling.
* **Key Operations:** Harmonizes column names across sheets, filters out patients missing critical slide metadata, and drops zero-variance features (columns where no patients tested positive).
* **Encoding:** Transforms qualitative clinical observations into discrete numerical labels (`0`, `1`, `2`, `3`, etc.).
* **Output Destination:** `processed_data/processed_placental_histopathology.csv`

---

## Execution

To re-run the entire data cleaning pipeline and refresh the files in `processed_data/`, execute the scripts from the root of this directory:

```bash
python preprocess_fitbit.py
python preprocess_placental.py
```

> ⚠️ **Prerequisites:** Ensure all raw data dependencies are placed in their expected input paths before running these scripts.
