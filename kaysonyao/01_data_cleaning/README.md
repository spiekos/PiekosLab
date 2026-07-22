# 01 Data Cleaning

Scripts for QC, normalization, batch correction, and imputation across all DP3 omics and survey modalities.

| Script | Description |
|---|---|
| `utilities.py` | Shared helpers imported by all cleaning scripts (see below) |
| `clean_proteomics_data.py` | Olink proteomics: QC, ComBat, imputation |
| `format_proteomics.py` | Split proteomics plasma output by timepoint suffix |
| `sop_omics_pipeline.py` | SOP v4 pipeline for metabolomics + lipidomics (Compound Discoverer exports) |
| `clean_survey_data.py` | Extract and clean survey data (EPDS, PSS, PUQE-24, diet, water quality) |

---

## `utilities.py` — Shared helpers

**Do not run directly** — imported by all other scripts in this folder and by `02_exploratory_analysis/utilities.py` and `03_model_development/utilities.py` via `importlib`.

Key exports:

| Symbol | Description |
|---|---|
| `METADATA_COLS` | Canonical list of metadata column names: `[SubjectID, Group, Subgroup, Batch, GestAgeDelivery, SampleGestAge, MetadataCanonicalID]` |
| `_GROUP_LABEL_MAP` | Label corrections applied at metadata load time (e.g. `"sptb"` → `"sPTB"`) |
| `load_metadata_with_batch` | Load and merge metadata from the master Excel workbook. Supports `meta_type` values: `"proteomics"`, `"placenta"`, `"metabolomics"`, `"lipids"` |
| `load_data` | Load a cleaned wide-format CSV with `index_col=0` |
| `get_analyte_columns` | Return all non-metadata column names from a cleaned DataFrame |
| `normalise_group_labels` | Apply `_GROUP_LABEL_MAP` corrections to the Group column |
| `half_min_impute_wide` | Impute missing values with per-column minimum − 1 (log2 space = half-min in linear space) |
| `standardize_missing_npx` | Replace Olink-style missing flags with `NaN` |
| `qc_mask` | Mask NPX values below Olink LOD |
| `apply_panel_normalization_long` | Olink panel normalization using internal control samples |
| `benjamini_hochberg_rejections` | BH FDR correction via `statsmodels.multipletests` with NaN-safe index handling |
| `missingness_filter_and_group_check` | Filter assays by missingness + Fisher's exact test for group-imbalanced missing |
| `combat_normalize_wide` | ComBat batch correction (pycombat) on wide-format DataFrames |

---

## `clean_proteomics_data.py` — Proteomics

### What it does

- Loads Olink proteomics CSV files.
- Applies QC masking (`qc_mask`) and panel normalization (`apply_panel_normalization_long`).
- Removes Olink control samples.
- Applies ComBat batch normalization using batch labels from metadata.
- Filters assays by missingness (cutoff + Fisher's exact test for group-imbalanced missingness).
- Imputes below-LOD values with half-minimum per analyte (log2 scale).
- Merges metadata and saves cleaned output CSV.

### Quick run

```bash
python 01_data_cleaning/clean_proteomics_data.py
```

Auto-detects plasma and placenta files from `data/proteomics/`.
Writes outputs to `data/cleaned/proteomics/normalized_full_results/`.

### CLI modes

**Auto mode** (default):
```bash
python 01_data_cleaning/clean_proteomics_data.py \
  --mode auto \
  --data-dir data/proteomics \
  --metadata-path "data/dp3 master table v2.xlsx" \
  --output-dir data/cleaned/proteomics/normalized_full_results
```

**Single mode** (one dataset type):
```bash
# Plasma
python 01_data_cleaning/clean_proteomics_data.py \
  --mode single --meta-type proteomics \
  --metadata-path "data/dp3 master table v2.xlsx" \
  --output-csv data/cleaned/proteomics/normalized_full_results/proteomics_plasma_cleaned_with_metadata.csv \
  --files data/proteomics/<file1>.csv data/proteomics/<file2>.csv

# Placenta
python 01_data_cleaning/clean_proteomics_data.py \
  --mode single --meta-type placenta \
  --metadata-path "data/dp3 master table v2.xlsx" \
  --output-csv data/cleaned/proteomics/normalized_full_results/proteomics_placenta_cleaned_with_metadata.csv \
  --files data/proteomics/<file1>.csv
```

### Outputs

- `data/cleaned/proteomics/normalized_full_results/proteomics_plasma_cleaned_with_metadata.csv`
- `data/cleaned/proteomics/normalized_full_results/proteomics_placenta_cleaned_with_metadata.csv`
- `*_dropped_missingness_report.csv` (when assays are dropped)

---

## `format_proteomics.py` — Split plasma by timepoint

Splits the full plasma cleaned CSV into one file per suffix label (A–E).

What it does:
- Removes internal whitespace in `SampleID` and `SubjectID`
- Extracts suffix label by comparing `SampleID` vs `SubjectID`
- Drops `Batch`; writes one CSV per suffix
- In each output file: drops `SubjectID`, removes suffix from `SampleID`

```bash
python 01_data_cleaning/format_proteomics.py \
  --input-csv data/cleaned/proteomics/normalized_full_results/proteomics_plasma_cleaned_with_metadata.csv \
  --output-dir data/cleaned/proteomics/normalized_sliced_by_suffix \
  --base-name proteomics_plasma
```

---

## `sop_omics_pipeline.py` — SOP v4 metabolomics + lipidomics

Implements the April 2026 DP3 SOP from raw Compound Discoverer exports. Handles all four
dataset configurations: `MTBL_plasma`, `MTBL_placenta`, `LIPD_plasma`, `LIPD_placenta`.

### Processing steps (in order)

1. Missing-value standardization
2. Sample type / batch / injection-order parsing
3. Pre-normalization drift diagnostics
4. ISTD normalization
5. Median fold-change batch normalization
6. Post-normalization drift diagnostics
7. Feature missingness filter (per-polarity)
8. Sample missingness filter
9. Log2 transformation
10. Half-minimum imputation
11. Pre-correction PCA
12. Batch-confounding checks
13. ComBat batch correction
14. Post-correction PCA
15. Post-ComBat intensity check + sample-level ISTD MAD QC
16. QC-pool RSD filter
17. IQR filter (within-timepoint)
18. Bridge-sample averaging
19. Deduplication, annotation, metadata integration
20. Trajectory plots + human-readable pipeline log

### Inputs

Raw data lives in the sibling `kaylaxu/` repository:

| Dataset | Input directory | Tissue |
|---|---|---|
| MTBL_plasma | `kaylaxu/data/MTBL_plasma/` | Plasma metabolomics (Compound Discoverer CSV) |
| MTBL_placenta | `kaylaxu/data/MTBL_placenta/` | Placenta metabolomics |
| LIPD_plasma | `kaylaxu/data/LIPD_plasma/` | Plasma lipidomics |
| LIPD_placenta | `kaylaxu/data/LIPD_placenta/` | Placenta lipidomics |

### Quick run

```bash
# All four datasets
python 01_data_cleaning/sop_omics_pipeline.py

# One dataset only
python 01_data_cleaning/sop_omics_pipeline.py --datasets MTBL_plasma

# Multiple specific datasets
python 01_data_cleaning/sop_omics_pipeline.py --datasets MTBL_plasma MTBL_placenta
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--datasets` | all four | Space-separated list: `MTBL_plasma`, `MTBL_placenta`, `LIPD_plasma`, `LIPD_placenta` |
| `--kayla-root` | `../kaylaxu` | Path to the Kayla Xu raw-export repository |
| `--metadata` | `data/dp3 master table v2.xlsx` | Master metadata workbook |
| `--output-root` | `data/cleaned/sop_omics_pipeline` | Root directory for cleaned outputs |

### Outputs

Per dataset under `data/cleaned/sop_omics_pipeline/<DATASET_ID>/`:

```
MTBL_plasma/
├── MTBL_plasma_cleaned_with_metadata.csv    Full plasma matrix (all timepoints)
├── MTBL_plasma_suffix_{A-E}.csv             One CSV per plasma timepoint
├── MTBL_plasma_feature_metadata.csv         m/z, RT, annotation per feature
├── MTBL_plasma_metadata_audit.csv           Per-sample metadata match log
├── MTBL_plasma_drop_log.csv                 All dropped features/samples with reason
├── pipeline_log.txt                         Human-readable step-by-step log
└── diagnostics/
    ├── pos/ neg/                            Per-polarity diagnostic plots
    └── post_combat_intensity_check/
```

Placenta output (`MTBL_placenta/`) has the same structure minus the per-timepoint suffix files.

---

## `clean_survey_data.py` — Survey data

Extracts, filters, and cleans survey instruments for the full DP3 survey cohort (≈390 subjects;
broader than the n=133 omics cohort).

### Inputs

| File | Description |
|---|---|
| `data/survey/epds_raw.csv` | Edinburgh Postnatal Depression Scale |
| `data/survey/pss_raw.csv` | Perceived Stress Scale |
| `data/survey/puqe24_raw.csv` | Pregnancy-Unique Quantification of Emesis |
| `data/survey/diet_raw.csv` | Diet frequency questionnaire |
| `data/survey/water.csv` | Drinking water THM/disinfection by-product data |
| `data/dp3 master table v2.xlsx` (sheet: `clinical data`) | Group/Subgroup source for all enrolled subjects |

### Outputs

```
data/survey/cleaned/
├── epds_cleaned.csv
├── pss_cleaned.csv
├── puqe24_cleaned.csv
├── diet_cleaned.csv
└── water_cleaned.csv
```

### Quick run

```bash
python 01_data_cleaning/clean_survey_data.py
```
