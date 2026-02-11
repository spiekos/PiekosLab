# Proteomics Cleaning Script

This README documents how to run:

- `clean_proteomics_data.py`

## What It Does

- Loads Olink proteomics CSV files.
- Applies QC masking and panel normalization.
- Removes Olink control samples.
- Filters assays by missingness and writes a dropped-assay report.
- Quantile-normalizes, imputes, converts to linear scale, and merges metadata.
- Saves cleaned output CSV.

## Required Inputs

- Olink CSV files folder path (for example: `$ROOT_DIR/data/proteomics`)
- Metadata Excel file path (for example: `$ROOT_DIR/data/dp3 master table v2.xlsx`)

## Quick Run

Run from terminal:

```bash
ROOT_DIR=/path/to/repo
python "$ROOT_DIR/01_data_cleaning/clean_proteomics_data.py"
```

What this does:

- Auto-detects plasma and placenta files from `data/proteomics`.
- Writes outputs to `data/cleaned/proteomics`.

## CLI Modes

### 1) Auto Mode

```bash
ROOT_DIR=/path/to/repo
python "$ROOT_DIR/01_data_cleaning/clean_proteomics_data.py" \
  --mode auto \
  --data-dir "$ROOT_DIR/data/proteomics" \
  --metadata-path "$ROOT_DIR/data/dp3 master table v2.xlsx" \
  --output-dir "$ROOT_DIR/data/cleaned/proteomics"
```

### 2) Single Mode (Run One Dataset Type)

Plasma example:

```bash
ROOT_DIR=/path/to/repo
python "$ROOT_DIR/01_data_cleaning/clean_proteomics_data.py" \
  --mode single \
  --meta-type proteomics \
  --metadata-path "$ROOT_DIR/data/dp3 master table v2.xlsx" \
  --output-csv "$ROOT_DIR/data/cleaned/proteomics/proteomics_plasma_cleaned_with_metadata.csv" \
  --files "$ROOT_DIR/data/proteomics/Q-04558_Barak_EDTAPlasma_NPX_2022-12-28.csv" "$ROOT_DIR/data/proteomics/Q-07626_Barak_EDTAPlasma_NPX_2023-06-12.csv"
```

Placenta example:

```bash
ROOT_DIR=/path/to/repo
python "$ROOT_DIR/01_data_cleaning/clean_proteomics_data.py" \
  --mode single \
  --meta-type placenta \
  --metadata-path "$ROOT_DIR/data/dp3 master table v2.xlsx" \
  --output-csv "$ROOT_DIR/data/cleaned/proteomics/proteomics_placenta_cleaned_with_metadata.csv" \
  --files "$ROOT_DIR/data/proteomics/Q-04558_Barak_PlacentalTissue_NPX_2022-12-28.csv" "$ROOT_DIR/data/proteomics/Q-07626_Barak_TissueLysate_NPX_2023-06-12.csv"
```

## Outputs

Main cleaned outputs:

- `$ROOT_DIR/data/cleaned/proteomics/proteomics_plasma_cleaned_with_metadata.csv`
- `$ROOT_DIR/data/cleaned/proteomics/proteomics_placenta_cleaned_with_metadata.csv`

Missingness reports (created when assays are dropped):

- `*_dropped_missingness_report.csv`

## Help

```bash
ROOT_DIR=/path/to/repo
python "$ROOT_DIR/01_data_cleaning/clean_proteomics_data.py" --help
```
