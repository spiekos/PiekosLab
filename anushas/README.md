# anushas

## Pregnancy data pipeline and exploratory analysis

This project contains an end-to-end workflow for cleaning and analyzing maternal Fitbit data, placental histopathology findings, and clinical delivery outcomes. The pipeline standardizes raw inputs, builds analysis-ready datasets, and produces summary tables, figures, and model outputs.

## What this project does

- Cleans and harmonizes clinical, Fitbit, and placental data
- Applies quality-control rules for missingness and feature completeness
- Builds integrated datasets for correlation and exploratory analysis
- Generates visualizations and correlation tables
- Runs GLM modeling of Fitbit metrics versus clinical and placental outcomes

## Project structure

```text
anushas/
├── 00_raw_data/                      # Immutable source files
├── 01_data_cleaning/                 # Data preprocessing pipeline
│   ├── README.md
│   ├── preprocess_clinical.py
│   ├── preprocess_fitbit.py
│   ├── preprocess_placental.py
│   ├── preprocess_correlation.py
│   └── processed_data/               # Cleaned intermediate datasets
├── 02_exploratory_analysis/          # Exploratory analysis scripts
│   ├── README.md
│   ├── analyze_clinical.py
│   ├── analyze_fitbit.py
│   └── correlation.py
├── 03_model_development/             # Modeling workflow
│   ├── README.md
│   └── run_fitbit_glm_interactions.py
├── 04_results_and_figures/           # Final outputs for results and figures
│   ├── README.md
│   ├── correlations/
│   ├── data_analysis/
│   └── models/
└── README.md                         # Project overview
```

Run the scripts from the repository root in the following order.

### 1. Clean and preprocess raw data

```bash
python anushas/01_data_cleaning/preprocess_fitbit.py
python anushas/01_data_cleaning/preprocess_clinical.py
python anushas/01_data_cleaning/preprocess_placental.py
python anushas/01_data_cleaning/preprocess_correlation.py
```

These steps generate cleaned files in `anushas/01_data_cleaning/processed_data/`.

### 2. Run exploratory analysis

```bash
python anushas/02_exploratory_analysis/analyze_clinical.py
python anushas/02_exploratory_analysis/analyze_fitbit.py
python anushas/02_exploratory_analysis/correlation.py
```

These scripts write their outputs to `anushas/04_results_and_figures/`.

### 3. Run modeling

```bash
python anushas/03_model_development/run_fitbit_glm_interactions.py
```

Model results are saved in `anushas/04_results_and_figures/models/`.

## Key outputs

- Cleaned clinical, Fitbit, and placental datasets in `anushas/01_data_cleaning/processed_data/`
- Exploratory outputs and correlation tables in `anushas/04_results_and_figures/correlations/`
- Clinical and Fitbit analysis summaries in `anushas/04_results_and_figures/data_analysis/`
- Modeling results and logs in `anushas/04_results_and_figures/models/`

## Environment requirements

Recommended Python environment:

- Python 3.9+
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- statsmodels
- statsmodels

> Keep raw files in `anushas/00_raw_data/` unchanged to maintain reproducibility.

## Quick checks & troubleshooting

- If a script fails with missing columns or empty outputs, confirm the processed CSVs exist in `anushas/01_data_cleaning/processed_data/` and that `id` values overlap across datasets.
