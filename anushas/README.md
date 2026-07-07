# anushas

## Pregnancy Data Pipeline & Exploratory Analysis

This project provides a complete end-to-end data processing and exploratory analysis pipeline for analyzing the longitudinal relationships between maternal Fitbit metrics, placental histopathology features, and clinical delivery outcomes.

```
---

## Project Architecture

anushas/
├── README.md                           # This file (Global Project Overview)
│
├── 00_raw_data/                        # Immutable raw datasets (Inputs)
│   ├── dp3 3rd set gest age for Tony assessed final.xlsx - Sheet1.csv  
│   ├── dp3 master table v2.xlsx - variables of interest.csv  
│   ├── DP3 slides Tony's analysis batches 1-2.xlsx - Sheet2.csv
│   ├── DP3_playset_PE.csv
│   ├── DP3_playset.csv
│   └── DP3-FitbitFullReport_DATA_LABELS_2025-02-18_1356.csv   
│
├── 01_data_cleaning/                   # Stage 1: Preprocessing & Harmonization
│   ├── README.md
│   ├── preprocess_fitbit.py
│   ├── preprocess_placental.py
│   └── processed_data/                 # Cleaned output datasets
│       ├── master_fitbit_clinical_correlation_data.csv
│       ├── processed_fitbit_data.csv
│       └── processed_placental_data.csv
│
└── 02_exploratory_analysis/            # Stage 2: Statistical Modeling & Profiling
    ├── README.md
    ├── analyze_fitbit.py
    ├── correlation.py
    ├── histogram.py
    └── outputs/                        # Final logs, plots, and statistical tables
        ├── [prefix]filtered_correlation_table.txt
        ├── [prefix]full_correlation_table.txt
        ├── [prefix]negatively_associated_vars.txt
        ├── [prefix]positively_associated_vars.txt
        ├── fitbit_data_analysis.txt
        └── pregnancy_plots_report.pdf
```

## Module Directory Breakdown

### 00_raw_data

Contains the baseline clinical spreadsheet registries, patient biometric timeseries data, and histopathology sheets. **Files in this folder must be kept immutable to ensure analysis reproducibility.**

### 01_data_cleaning

Ingests the raw data configurations to clean, filter, and structure features ready for downstream compute.

* **Fitbit Quality Control:** Rather than dropping full patient cohorts, individual tracking features are screened chronologically. If a patient has **7 or more consecutive days of missing data** for a specific activity, sleep, or heart rate feature, that specific metric is nulled out (**$\text{NaN}$**) for that patient.
* **Placental Categorization:** Eliminates zero-variance clinical features (where no patients in the cohort tested positive) and converts qualitative pathology observations into discrete numerical labels (`0`, `1`, `2`, `3`, etc.).

### 02_exploratory_analysis

Takes clean intermediate artifacts and executes the longitudinal statistical profiling layer.

* **Trimester-Level Aggregation:** Collapses multi-row tracking streams into single-row patient **means** across 4 discrete pregnancy trimesters, compiling them alongside clinical variables into a master table.
* **Hypothesis Testing:** Calculates pairwise **Spearman Rank Correlation (**$\rho$**)** coefficients between cross-set pairs (e.g., Placental Architecture vs. Delivery Metrics; Trimester Fitbit habits vs. Outcomes). It automatically controls for multiple comparisons using the **Benjamini-Hochberg False Discovery Rate (FDR)** procedure (**$FDR \le 0.05$**).

## Getting Started & Execution Order

To run or replicate this pipeline from scratch, execute the following commands sequentially from the root of the `anushas` workspace:

### Step 1: Preprocess and Clean Raw Data

Run the data cleaning scripts to extract raw inputs, handle schemas, and enforce missingness quality gates:

```
python 01_data_cleaning/preprocess_fitbit.py
python 01_data_cleaning/preprocess_placental.py
```

### Step 2: Generate Exploratory Statistics, Plots, and Correlation Tables

Run the exploratory analysis scripts to aggregate metrics into trimester windows, generate tracking plots, and execute multi-hypothesis correlation engines:

```
python 02_exploratory_analysis/analyze_fitbit.py
python 02_exploratory_analysis/correlation.py
python 02_exploratory_analysis/histogram.py
```

> **Pipeline Note:** The intermediate dataset `01_data_cleaning/processed_data/master_fitbit_clinical_correlation_data.csv` is compiled during Step 2 by `analyze_fitbit.py` and is immediately consumed by `correlation.py`. Ensure you run the scripts in the exact sequence specified above.

## System Requirements & Environment

* **Runtime Environment:** Python 3.8+
* **Core Dependencies:** `numpy`, `pandas`, `scipy`, `matplotlib`

*(For exact package version specifications, please refer to the local script headers or environment configuration files where applicable).*
