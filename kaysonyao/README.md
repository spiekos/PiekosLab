# DP3 Proteomics Analysis

Proteomics preprocessing, differential analysis, and visualization pipeline for the DP3 cohort
(plasma Olink Explore + placental tissue lysate).

All scripts are intended to be run from the **project root** — the directory containing this file
and the `data/` folder.

---

## Project overview

The pipeline is divided into four numbered stages. Each stage has its own folder and README.

| Stage | Folder | Purpose |
|---|---|---|
| 1 | `01_data_cleaning/` | QC, panel normalization, ComBat batch correction, imputation |
| 2 | `02_exploratory_analysis/` | Differential analysis (cross-sectional & longitudinal), heatmaps, Enrichr lists |
| 3 | `03_model_development/` | Predictive model development *(in progress)* |
| 4 | `04_results_and_figures/` | Final publication figures *(in progress)* |

---

## Data

```
data/
├── proteomics/                          Raw Olink NPX CSV exports (plasma + placenta)
├── dp3 master table v2.xlsx             Master metadata (Group, Subgroup, Batch, GestAge, etc.)
├── dp3 n=133 clinical and metadata.xlsx Clinical metadata
├── cleaned/
│   └── proteomics/
│       ├── normalized_full_results/     Full cleaned matrices (plasma + placenta)
│       └── normalized_sliced_by_suffix/ Plasma split by timepoint suffix (A–E)
├── diff_analysis/
│   └── results/
│       ├── plasma/
│       │   ├── cross_sectional/         Per-timepoint (A–E) pairwise differential results + heatmaps
│       │   └── longitudinal/            Adjacent-step delta results + heatmaps
│       ├── placenta/
│       │   └── cross_sectional/         Pairwise differential results + heatmaps
│       └── sample_counts_per_group_timepoint.csv
└── enrichr_input/                       Directional gene lists ready for Enrichr
    └── <timepoint>/<g1>_vs_<g2>/
```

---

## Groups and timepoints

**Clinical groups:** Control, FGR (Fetal Growth Restriction), HDP (Hypertensive Disorders of
Pregnancy), sPTB (spontaneous Preterm Birth)

**Plasma timepoints:** A, B, C, D, E (longitudinal blood draws; postnatal EA–EE merged into A–E
during data cleaning)

---

## Environment

```bash
conda activate PiekosLab
```

Python dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`, `seaborn`, `matplotlib`,
`neurocombat_sklearn` (or `neuroCombat`).

---

## Recommended run order

```bash
# 1. Clean raw Olink files
python 01_data_cleaning/clean_proteomics_data.py

# 2. Split plasma output by timepoint suffix
python 01_data_cleaning/format_proteomics.py \
    --input-csv data/cleaned/proteomics/normalized_full_results/proteomics_plasma_cleaned_with_metadata.csv \
    --output-dir data/cleaned/proteomics/normalized_sliced_by_suffix \
    --base-name proteomics_plasma

# 3. Run differential analysis (cross-sectional + longitudinal)
python 02_exploratory_analysis/identify_differential_analytes.py

# 4. Generate heatmaps
python 02_exploratory_analysis/generate_differential_cluster_heatmap_limited_group.py

# 5. Prepare Enrichr input lists
python 02_exploratory_analysis/prepare_enrichr_input.py
```

See each folder's `README.md` for full CLI documentation and output details.

---

## Change log

| Date | Change |
|---|---|
| 2026-03-04 | Added `02_exploratory_analysis/utilities.py`; refactored both differential analysis scripts to import from it. |
| 2026-03-04 | Added `02_exploratory_analysis/prepare_enrichr_input.py` for directional Enrichr gene list generation. |
| 2026-03-04 | Created this top-level README and `02_exploratory_analysis/README.md`. |
