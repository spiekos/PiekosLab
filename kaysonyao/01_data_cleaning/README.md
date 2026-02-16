# Proteomics Cleaning Script

This README documents how to run:

- `clean_proteomics_data.py`
- `format_proteomics.py`

## What It Does

- Loads Olink proteomics CSV files.
- Applies QC masking and panel normalization.
- Removes Olink control samples.
- Applies ComBat batch normalization using batch labels from metadata.
- Filters assays by missingness after ComBat and writes a dropped-assay report.
- Imputes, converts to linear scale, and merges metadata.
- Saves cleaned output CSV.

## Required Inputs

- Olink CSV files folder path (for example: `$ROOT_DIR/data/proteomics`)
- Metadata Excel file path (for example: `$ROOT_DIR/data/<metadata_file>.xlsx`)

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
  --metadata-path "$ROOT_DIR/data/<metadata_file>.xlsx" \
  --output-dir "$ROOT_DIR/data/cleaned/proteomics"
```

### 2) Single Mode (Run One Dataset Type)

Plasma example:

```bash
ROOT_DIR=/path/to/repo
python "$ROOT_DIR/01_data_cleaning/clean_proteomics_data.py" \
  --mode single \
  --meta-type proteomics \
  --metadata-path "$ROOT_DIR/data/<metadata_file>.xlsx" \
  --output-csv "$ROOT_DIR/data/cleaned/proteomics/proteomics_plasma_cleaned_with_metadata.csv" \
  --files "$ROOT_DIR/data/proteomics/<plasma_file_1>.csv" "$ROOT_DIR/data/proteomics/<plasma_file_2>.csv"
```

Placenta example:

```bash
ROOT_DIR=/path/to/repo
python "$ROOT_DIR/01_data_cleaning/clean_proteomics_data.py" \
  --mode single \
  --meta-type placenta \
  --metadata-path "$ROOT_DIR/data/<metadata_file>.xlsx" \
  --output-csv "$ROOT_DIR/data/cleaned/proteomics/proteomics_placenta_cleaned_with_metadata.csv" \
  --files "$ROOT_DIR/data/proteomics/<placenta_file_1>.csv" "$ROOT_DIR/data/proteomics/<placenta_file_2>.csv"
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

## Format Plasma Output By Longitudinal Suffix

Use `format_proteomics.py` to split the final plasma output into one file per suffix label.

What it does:

- Removes internal whitespace in `SampleID` and `SubjectID`
- Extracts suffix label by comparing `SampleID` vs `SubjectID`
- Drops `Batch`
- Writes one CSV per suffix label
- In each output file:
  - drops `SubjectID`
  - removes suffix from `SampleID`

Run:

```bash
ROOT_DIR=/path/to/repo
python "$ROOT_DIR/01_data_cleaning/format_proteomics.py" \
  --input-csv "$ROOT_DIR/data/cleaned/proteomics/proteomics_plasma_cleaned_with_metadata.csv" \
  --output-dir "$ROOT_DIR/data/cleaned/proteomics/sliced_by_suffix" \
  --base-name proteomics_plasma
```

Format script help:

```bash
ROOT_DIR=/path/to/repo
python "$ROOT_DIR/01_data_cleaning/format_proteomics.py" --help
```
