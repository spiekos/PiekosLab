## 01_data_cleaning

This directory contains the data preprocessing pipeline used to produce the cleaned, analysis-ready datasets for clinical, Fitbit, and placental data. The outputs are consumed by the exploratory analysis and modeling steps in `anushas/02_exploratory_analysis` and `anushas/03_model_development`.

## Directory structure

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

## Overview of scripts

`preprocess_clinical.py` — Cleans and encodes clinical metadata. Key steps: standardize column names, filter to delivered patients in the DP3 playset, harmonize race/ethnicity/smoking, impute prepregnancy BMI when appropriate, and export `processed_clinical_data.csv` to `processed_data/`.

`preprocess_fitbit.py` — Cleans Fitbit longitudinal data. Key steps: merge Fitbit sheets, filter to `event_name == "fitbit data"`, apply day-level validity and implausible-value bounds, and export `processed_fitbit_data.csv` to `processed_data/`.

`preprocess_placental.py` — Merges and encodes placental histopathology recordings. Key steps: standardize and merge placental source sheets, drop empty/low-variance features, encode pathology labels, attach clinical indicators (e.g., `spontaneous_preterm_birth`), export `processed_placental_data.csv`, and write a summary log to `04_results_and_figures/data_analysis/placental/sum_placental_histo_features.txt`.

`preprocess_correlation.py` — Builds the integrated correlation dataset. Key steps: aggregate longitudinal Fitbit days into pregnancy windows, apply per-patient coverage thresholds, average valid days by bin, and merge Fitbit features with clinical and placental variables. Output: `master_fitbit_clinical_correlation_data.csv`.

## Primary inputs

- Raw clinical and codebook files in `00_raw_data/` (examples used by scripts):
  - `dp3 master table v2.xlsx - variables of interest.csv`
  - `DP3_playset.csv`
  - `dp3 3rd set gest age for Tony assessed final.xlsx - Sheet1.csv`
  - `DP3 slides Tony's analysis batches 1-2.xlsx - Sheet2.csv`
  - `DP3-FitbitFullReport_DATA_LABELS_2025-02-18_1356.csv`
  - `implausible_value_bounds.csv`

## Outputs

- `processed_data/processed_clinical_data.csv`
- `processed_data/processed_fitbit_data.csv`
- `processed_data/processed_placental_data.csv`
- `processed_data/master_fitbit_clinical_correlation_data.csv`
- Supporting log: `04_results_and_figures/data_analysis/placental/sum_placental_histo_features.txt`

These processed CSVs are the canonical inputs for the exploratory and modeling steps and should be committed to analysis snapshots if you need reproducible results.

## How to run

Run each script from the repository root. Typical order:

```bash
python anushas/01_data_cleaning/preprocess_fitbit.py
python anushas/01_data_cleaning/preprocess_clinical.py
python anushas/01_data_cleaning/preprocess_placental.py
python anushas/01_data_cleaning/preprocess_correlation.py
```

Notes:

- Scripts expect the raw files listed above to be present in `anushas/00_raw_data/`.
- The correlation preprocessing buckets Fitbit data into five pregnancy windows and applies per-patient coverage thresholds; consult `preprocess_correlation.py` for exact thresholds.

## Environment

Recommended packages:

- Python 3.9+
- pandas
- numpy
- scipy
- statsmodels
- matplotlib
- seaborn

Install with pip (example):

```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn
```

## Reproducibility & notes

- Keep raw files in `anushas/00_raw_data/` unchanged to maintain reproducibility.
- If you change preprocessing logic, re-run the full `01_data_cleaning` pipeline and version the resulting `processed_data` CSVs used for downstream analysis.
