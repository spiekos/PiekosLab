# 01 Data Cleaning

This README documents how to run:

**Proteomics**
- `clean_proteomics_data.py`
- `format_proteomics.py`
- `test_proteomics.py`
- `proteomics_diagnostics.py`

**Metabolomics**
- `clean_metabolomics_data.py`

---

## Proteomics — `clean_proteomics_data.py`

### What It Does

- Loads Olink proteomics CSV files.
- Applies QC masking and panel normalization.
- Removes Olink control samples.
- Applies ComBat batch normalization using batch labels from metadata.
- Filters assays by missingness after ComBat (cutoff + Fisher's exact test for group-imbalanced missingness) and writes a dropped-assay report.
- Imputes below-LOD values with half-minimum per analyte (log2 scale).
- Merges metadata and saves cleaned output CSV.

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
- Writes outputs to `data/cleaned/proteomics/normalized_full_results`.

## CLI Modes

### 1) Auto Mode

```bash
ROOT_DIR=/path/to/repo
python "$ROOT_DIR/01_data_cleaning/clean_proteomics_data.py" \
  --mode auto \
  --data-dir "$ROOT_DIR/data/proteomics" \
  --metadata-path "$ROOT_DIR/data/<metadata_file>.xlsx" \
  --output-dir "$ROOT_DIR/data/cleaned/proteomics/normalized_full_results"
```

### 2) Single Mode (Run One Dataset Type)

Plasma example:

```bash
ROOT_DIR=/path/to/repo
python "$ROOT_DIR/01_data_cleaning/clean_proteomics_data.py" \
  --mode single \
  --meta-type proteomics \
  --metadata-path "$ROOT_DIR/data/<metadata_file>.xlsx" \
  --output-csv "$ROOT_DIR/data/cleaned/proteomics/normalized_full_results/proteomics_plasma_cleaned_with_metadata.csv" \
  --files "$ROOT_DIR/data/proteomics/<plasma_file_1>.csv" "$ROOT_DIR/data/proteomics/<plasma_file_2>.csv"
```

Placenta example:

```bash
ROOT_DIR=/path/to/repo
python "$ROOT_DIR/01_data_cleaning/clean_proteomics_data.py" \
  --mode single \
  --meta-type placenta \
  --metadata-path "$ROOT_DIR/data/<metadata_file>.xlsx" \
  --output-csv "$ROOT_DIR/data/cleaned/proteomics/normalized_full_results/proteomics_placenta_cleaned_with_metadata.csv" \
  --files "$ROOT_DIR/data/proteomics/<placenta_file_1>.csv" "$ROOT_DIR/data/proteomics/<placenta_file_2>.csv"
```

## Outputs

Main cleaned outputs:

- `$ROOT_DIR/data/cleaned/proteomics/normalized_full_results/proteomics_plasma_cleaned_with_metadata.csv`
- `$ROOT_DIR/data/cleaned/proteomics/normalized_full_results/proteomics_placenta_cleaned_with_metadata.csv`

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
  --input-csv "$ROOT_DIR/data/cleaned/proteomics/normalized_full_results/proteomics_plasma_cleaned_with_metadata.csv" \
  --output-dir "$ROOT_DIR/data/cleaned/proteomics/normalized_full_results/sliced_by_suffix" \
  --base-name proteomics_plasma
```

Format script help:

```bash
ROOT_DIR=/path/to/repo
python "$ROOT_DIR/01_data_cleaning/format_proteomics.py" --help
```

## Run Diagnostics

Use `proteomics_diagnostics.py` to generate duplicate-combo report and PCA diagnostics.

```bash
ROOT_DIR=/path/to/repo
python "$ROOT_DIR/01_data_cleaning/proteomics_diagnostics.py"
```

Outputs are written under:

- `$ROOT_DIR/data/cleaned/proteomics/normalized_full_results` (cleaned outputs and reports)
- `$ROOT_DIR/data/cleaned/proteomics/normalized_full_results/diagnostics` (all plots and diagnostics)

## Run QA Tests

Use `test_proteomics.py` to run validation checks and produce QA plots:

```bash
ROOT_DIR=/path/to/repo
python "$ROOT_DIR/01_data_cleaning/test_proteomics.py"
```

Validate a specific cleaned output file:

```bash
ROOT_DIR=/path/to/repo
python "$ROOT_DIR/01_data_cleaning/test_proteomics.py" \
  --integration "$ROOT_DIR/data/cleaned/proteomics/normalized_full_results/proteomics_plasma_cleaned_with_metadata.csv" \
  --verbose
```

QA report outputs are written under:

- `$ROOT_DIR/data/cleaned/proteomics/qa_reports`

---

## Metabolomics — `clean_metabolomics_data.py`

### What It Does

- Loads per-batch wide-format CSV files provided by the collaborator (plasma and placenta separately).
- Strips non-analyte metadata columns (`patient_ID`, `group`, `subgroup`, `gestational_age`, `gestational_age_at_collection`) that are embedded alongside analyte intensities in the raw files.
- Merges batch files and attaches clinical metadata (Group, Subgroup) from the master table.
- Applies the same collaborator batch normalization already embedded in the data (per-analyte median/MAD scaling using control samples — a constant factor per analyte applied uniformly to all samples, preserving between-group fold changes).
- Runs missingness filter (cutoff only — **Fisher's exact test for group-imbalanced missingness is omitted** because the collaborator-provided data contains 0% missing values across all plasma timepoints).
- Imputes any residual missing values with half-minimum per analyte (log2 scale).
- Splits plasma output by timepoint suffix (A–E) using `format_metabolomics.py`.

### Key properties of metabolomics data

| Property | Value |
|---|---|
| Scale | Already log2-transformed by collaborator |
| Missing values | 0% across all plasma timepoints |
| Plasma analytes | 1,887 (after removing metadata columns and unnamed-peak filtering) |
| Placenta analytes | 2,039 |
| Plasma samples | 523 (across all timepoints) |
| Placenta samples | 106 |

### Quick Run

```bash
ROOT_DIR=/path/to/repo
python "$ROOT_DIR/01_data_cleaning/clean_metabolomics_data.py"
```

Auto-detects plasma and placenta raw files from `data/cleaned/metabolomics/plasma/` and
`data/cleaned/metabolomics/placenta/`, writes outputs to
`data/cleaned/metabolomics/normalized_full_results/` and
`data/cleaned/metabolomics/normalized_sliced_by_suffix/`.

### Outputs

| File | Description |
|---|---|
| `data/cleaned/metabolomics/normalized_full_results/metabolomics_plasma_cleaned_with_metadata.csv` | Full plasma matrix (all timepoints merged) |
| `data/cleaned/metabolomics/normalized_full_results/metabolomics_placenta_cleaned_with_metadata.csv` | Full placenta matrix |
| `data/cleaned/metabolomics/normalized_sliced_by_suffix/metabolomics_plasma_formatted_suffix_{A-E}.csv` | One CSV per plasma timepoint |
