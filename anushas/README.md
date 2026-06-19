# anushas

## Pregnancy Data Pipeline & Exploratory Analysis

This project provides a complete end-to-end data processing and exploratory analysis pipeline for analyzing the longitudinal relationships between maternal Fitbit metrics, placental histopathology features, and clinical delivery outcomes.

---

## Project Architecture

```text

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

│   └── processed_data/                 # Cleaned datasets

│       ├── processed_fitbit_data.csv

│       └── processed_placental_data.csv

│

└── 02_exploratory_analysis/            # Stage 2: Statistical Modeling & Profiling

    ├── README.md

    ├── analyze_fitbit.py

    ├── correlation.py

    ├── histogram.py

    └── outputs/                        # Final logs, plots, and statistical tables

        ├── README.md

        ├── dropped_patients_fitbit_log.txt

        ├── filtered_correlation_table.txt

        ├── fitbit_data_analysis.txt

        ├── full_correlation_table.txt

        ├── negatively_associated_delivery_vars.txt

        ├── positively_associated_delivery_vars.txt

        ├── pregnancy_plots_report.pdf

        └── sum_placental_histo_features.txt



```

## End-to-End Workflow Pipeline

```

  [ 01_data_cleaning ] 

         │

         ├──► (preprocess_fitbit.py)    ──► QC Quality Gate (Drop if missing continuous days ≥ 2) ──► processed_fitbit_data.csv

         └──► (preprocess_placental.py) ──► Filter metadata & encode categorical labels    ──► processed_placental_data.csv

                                                                                                            │

 ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┘

 │

 ▼

  [ 02_exploratory_analysis ]

         │

         ├──► (analyze_fitbit.py)  ──► Metric summaries & pregnancy missingness logs ──► fitbit_data_analysis.txt

         ├──► (histogram.py)       ──► Trimester-mapped data density visualization  ──► pregnancy_plots_report.pdf

         └──► (correlation.py)     ──► Cross-set Spearman Correlation + FDR Testing  ──► full & filtered tables

```

## Module Directory Breakdown

### 00_raw_data

Contains the baseline clinical spreadsheet registries, patient biometric timeseries data, and histopathology sheets. **Files in this folder must be kept immutable to ensure analysis reproducibility.**

### 01_data_cleaning

Ingests the raw data configurations to clean, filter, and structure features ready for downstream compute.

* **Fitbit Quality Control:** Filters patient trajectories and filters records showing a prolonged loss of coverage (&ge; 2 consecutive days missing).
* **Placental Categorization:** Eliminates zero-variance clinical features and converts narrative/qualitative scores into categorical integers (`0`, `1`, `2`, `3`).

### 02_exploratory_analysis

Takes clean intermediate artifacts and executes the statistical profiling layer.

* **Missingness Profiling:** Restricts longitudinal data frames exclusively to the calculated pregnancy windows to capture accurate physiological baselines.
* **Hypothesis Testing:** Calculates pairwise **Spearman Rho (&rho;)** coefficients between independent feature vectors (Placental Architecture vs. Delivery Metrics) and controls for type-1 family-wise errors using a **False Discovery Rate (FDR)** multi-test correction.

## Getting Started & Execution Order

To run or replicate this pipeline from scratch, execute the following commands sequentially from the root of the workspace:

### Step 1: Preprocess and Clean Raw Data

**Bash**

```

cd 01_data_cleaning

python preprocess_fitbit.py

python preprocess_placental.py

cd ..

```

### Step 2: Generate Exploratory Statistics, Plots, and Correlation Tables

**Bash**

```

cd 02_exploratory_analysis

python analyze_fitbit.py

python histogram.py

python correlation.py

cd ..

```

## System Requirements & Environment

* **Runtime Environment:** Python 3.8+
* **Core Dependencies:**`numpy`, `pandas`, `scipy`, `matplotlib` (Refer to local script headers/requirements files for exact package version specifications).
