# 01 Data Cleaning

Run from the project root. Paths below are defaults; see `data/README.md` for the layout.

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `sop_omics_pipeline.py` | SOP v7 metabolomics + lipidomics pipeline (Jenn Ko's version + fixes) | `data/raw/extracted/`, `data/raw/original/` | `data/processed/<MTBL\|LIPD>/<tissue>/`, `04_results_and_figures/{pre_post_combat_pca,trajectory_plots}/` |
| `combat_ref.py` | Reference-batch ComBat (Python port of `sva::ComBat`, separate fit/transform) | imported | - |
| `clean_proteomics_data.py` | Olink proteomics QC, panel normalisation, reference-batch ComBat, missingness, imputation | `data/raw/original/proteomics/npx/` | `data/processed/proteomics/` |
| `bin_echo_windows.py` | Re-bin plasma into gestational windows, one sample per participant per window (midpoint rule) | `data/processed/...` | `data/processed/windows/` |
| `make_locked_split.py` | The one participant-level 70/30 dev/test split (refuses to overwrite) | master table | `data/processed/locked_split.csv` |
| `format_proteomics.py` | Older A-E visit-letter slicing of proteomics plasma (used by the older A-E tools) | `data/processed/proteomics/` | `data/processed/proteomics/normalized_sliced_by_suffix/` |
| `clean_survey_data.py` | EPDS, PSS, PUQE-24, diet, water | `data/raw/original/survey/` | `data/processed/survey/` |
| `utilities.py` | Shared helpers (metadata loading, Olink QC, `combat_normalize_wide` via ComBatRef) | imported | - |
| `extraction/MTBL_extraction.py` | Kayla's metabolomics extraction (plasma, placenta). `python 01_data_cleaning/extraction/MTBL_extraction.py "<workbook>.xlsx" <out_dir>` | `data/raw/original/MTBL/<tissue>/` | `data/raw/extracted/MTBL/<tissue>/` |
| `extraction/MTBL_extraction_urine.py` | Urine version (different sheet layout; batch pools BatchNPoolN, CumulativePool excluded). Same arguments | `data/raw/original/MTBL/urine/` | `data/raw/extracted/MTBL/urine/` |
| `extraction/LIPD_extraction.py` | Lipid workbook -> `{pos,neg}_{batch,compounds,expression}.csv` (Kayla's extraction, lipid part, adapted to this layout). `python 01_data_cleaning/extraction/LIPD_extraction.py <plasma\|placenta> "<workbook>.xlsx"` | `data/raw/original/LIPD/<tissue>/` | `data/raw/extracted/LIPD/<tissue>/` |
| `combat_validation/` | `validate_combat.R` (sva ground truth) + `check_python_vs_R.py`; run both from inside that folder | - | `validate_combat_*.csv` |

## `sop_omics_pipeline.py`

Steps 1-35 follow SOP v7 (Parts 1-5; list in the module docstring). Changes made on 2026-09-29
to Jenn's version (the as-received copy is in `_legacy/scripts/`):

- ISTDs identified from the export's own label first (`LipidGroup` "POS/NEG ISTD"; metabolite
  names ending in "ISTD"), then the configured name lists with a spelling-tolerant key.
  Class-matched lipid normalisation (Step 4aii) now resolves standards to feature IDs.
- Step 13 uses `combat_ref.ComBatRef` (reference batch = first batch, passes through unchanged;
  zero-variance features left unadjusted, as sva does; no silent fallback).

```bash
python 01_data_cleaning/sop_omics_pipeline.py --datasets MTBL_plasma MTBL_placenta LIPD_plasma LIPD_placenta

# For held-out evaluation: ComBat fit on dev participants + QC pools only, outcome-blind design
python 01_data_cleaning/sop_omics_pipeline.py --blind-combat \
    --fit-subjects data/processed/locked_split.csv --fit-split dev \
    --output-root data/processed_devfit
```

| Flag | Default | |
|---|---|---|
| `--inputs-root` | `data/raw/extracted` | Kayla's extraction CSVs |
| `--metadata` | `data/raw/original/dp3 master table v2.xlsx` | |
| `--output-root` | `data/processed` | |
| `--datasets` | all five (MTBL plasma/placenta/urine, LIPD plasma/placenta) | |
| `--blind-combat` | off | drop Group/Subgroup and GestAgeDelivery from the ComBat design |
| `--fit-subjects`, `--fit-split` | none | fit ComBat on these participants + QC; requires `--blind-combat` |

Outputs per dataset: `<DS>_cleaned_with_metadata.csv`, `<DS>_feature_metadata.csv`,
`<DS>_T1..T5.csv` (plasma/urine; no midpoint de-duplication, use `bin_echo_windows.py` for that),
sample/feature filter logs, dedup log, comprehensive drop log, metadata audit, `pipeline_log.txt`.

Changed 2026-09-30: LIPD_placenta reads the `060525` placenta export (sheets "POS/NEG Lipids"); Jenn's
config had named a `072925` placenta workbook that does not exist. Modification lists (SOP Appendix A)
are read from `data/raw/original/{MTBL,LIPD}/common_*_modification_list.csv`.

## `clean_proteomics_data.py`

```bash
python 01_data_cleaning/clean_proteomics_data.py            # auto: plasma + placenta
python 01_data_cleaning/clean_proteomics_data.py --mode single --meta-type placenta \
    --output-csv data/processed/proteomics/proteomics_placenta_cleaned_with_metadata.csv \
    --files "data/raw/original/proteomics/npx/<file>.csv"
```

Auto mode reads every `.csv` in `--data-dir` and classifies it as plasma or placenta by name, so
keep only NPX exports in `npx/`. SubjectID = SampleID with whitespace removed and one trailing
visit letter A-E stripped.

## `bin_echo_windows.py`

```bash
python 01_data_cleaning/bin_echo_windows.py --scheme dp3_5t \
    --datasets MTBL_plasma LIPD_plasma MTBL_urine proteomics_plasma
python 01_data_cleaning/bin_echo_windows.py --scheme dp3_5t --source-root data/processed_devfit \
    --output-root data/processed/windows_devfit
```

Schemes: `dp3_5t` (T1 6-14, T2 14-22, T3 22-32, T4 32-37, T5 37-42 wk) and `echo2` (default;
ECHO-matched W1 6-20, W2 28-42). Writes `<out>/<DS>/<DS>_echo_<window>.csv` and `echo_binning_log.csv`.

## `combat_validation/`

```bash
cd 01_data_cleaning/combat_validation
Rscript validate_combat.R          # tests 1-5 against sva::ComBat, writes validate_combat_*.csv
python check_python_vs_R.py        # combat_ref.py vs the R output (last run: MATCH, ~5e-14)
```
