# anushas

## Pregnancy data pipeline and exploratory analysis

This project contains an end-to-end workflow for cleaning and analyzing maternal Fitbit data, placental histopathology findings, and clinical delivery outcomes. The pipeline standardizes raw inputs, generates cleaned intermediate datasets, and produces summary tables and figures for downstream analysis.

## What this project does

- Cleans and harmonizes clinical, Fitbit, and placental data
- Applies quality-control rules for missingness and feature completeness
- Builds analysis-ready datasets for correlation and exploratory work
- Produces logs, plots, and summary files in the outputs folders

## Project structure

```text
anushas/
├── 00_raw_data/                      # Immutable source files
├── 01_data_cleaning/                 # Data cleaning and preprocessing
│   ├── preprocess_clinical.py
│   ├── preprocess_fitbit.py
│   ├── preprocess_placental.py
│   ├── preprocess_correlation.py
│   └── processed_data/               # Cleaned intermediate datasets
├── 02_exploratory_analysis/          # Analysis, correlation, and plotting
│   ├── analyze_clinical.py
│   ├── analyze_fitbit.py
│   ├── correlation.py
│   ├── histogram.py
│   └── outputs/                      # Final logs, tables, and figures
└── README.md                         # Project overview
```

## Workflow

Run the scripts from the `anushas` project directory in the following order.

### 1. Clean and preprocess the raw data

```bash
python 01_data_cleaning/preprocess_clinical.py
python 01_data_cleaning/preprocess_fitbit.py
python 01_data_cleaning/preprocess_placental.py
python 01_data_cleaning/preprocess_correlation.py
```

These steps generate the cleaned files in `01_data_cleaning/processed_data/`.

### 2. Run exploratory analysis and generate outputs

```bash
python 02_exploratory_analysis/analyze_clinical.py
python 02_exploratory_analysis/analyze_fitbit.py
python 02_exploratory_analysis/correlation.py
python 02_exploratory_analysis/histogram.py
```

These steps create the summary outputs, correlation tables, and figures in `02_exploratory_analysis/outputs/`.

## Key outputs

- Processed clinical, Fitbit, and placental datasets in `01_data_cleaning/processed_data/`
- Correlation results and supporting lists in `02_exploratory_analysis/outputs/`
- Figures in `02_exploratory_analysis/outputs/figures/`

## Environment requirements

Recommended Python environment:

- Python 3.9+
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- statsmodels

> The raw files in `00_raw_data/` should remain unchanged so the pipeline stays reproducible.
