# DP3 Multi-Omics Analysis

Multi-omics preprocessing, differential analysis, and predictive modelling pipeline for the DP3 cohort.
Covers two omics modalities:

- **Proteomics** — plasma Olink Explore + placental tissue lysate (NPX values, log2 scale)
- **Metabolomics** — plasma untargeted LC-MS (collaborator-normalized, log2 scale, plasma only)

All scripts are intended to be run from the **project root** — the directory containing this file
and the `data/` folder.

---

## Project overview

The pipeline is divided into four numbered stages. Each stage has its own folder and README.

| Stage | Folder | Purpose |
|---|---|---|
| 1 | `01_data_cleaning/` | QC, normalization, batch correction, imputation |
| 2 | `02_exploratory_analysis/` | Differential analysis (cross-sectional & longitudinal), heatmaps, enrichment input |
| 3 | `03_model_development/` | Predictive model development and pathway enrichment |
| 4 | `04_results_and_figures/` | All outputs — differential results, heatmaps, model outputs, enrichment |

---

## Data

```
data/
├── proteomics/                          Raw Olink NPX CSV exports (plasma + placenta)
├── dp3 master table v2.xlsx             Master metadata (Group, Subgroup, Batch, GestAge, etc.)
├── dp3 n=133 clinical and metadata.xlsx Clinical metadata
└── cleaned/
    ├── proteomics/
    │   ├── normalized_full_results/     Full cleaned matrices (plasma + placenta)
    │   └── normalized_sliced_by_suffix/ Plasma split by timepoint suffix (A–E)
    └── metabolomics/
        ├── plasma/                      Raw per-batch metabolomics CSV files (from collaborator)
        ├── placenta/                    Raw placenta metabolomics CSV files (from collaborator)
        ├── normalized_full_results/     Full cleaned matrices (plasma + placenta)
        └── normalized_sliced_by_suffix/ Plasma split by timepoint suffix (A–E)
```

---

## Groups and timepoints

**Clinical groups:** Control, FGR (Fetal Growth Restriction), HDP (Hypertensive Disorders of
Pregnancy), sPTB (spontaneous Preterm Birth)

**Plasma timepoints:** A, B, C, D, E (longitudinal blood draws)

---

## Environment

```bash
conda activate PiekosLab
```

Python dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`, `seaborn`, `matplotlib`,
`neurocombat_sklearn` (or `neuroCombat`), `scikit-learn`, `xgboost`, `optuna`, `requests`,
`gseapy` (optional, for Enrichr enrichment).

---

## Recommended run order

### Proteomics

```bash
# 1. Clean raw Olink files
python 01_data_cleaning/clean_proteomics_data.py

# 2. Split plasma output by timepoint suffix
python 01_data_cleaning/format_proteomics.py \
    --input-csv data/cleaned/proteomics/normalized_full_results/proteomics_plasma_cleaned_with_metadata.csv \
    --output-dir data/cleaned/proteomics/normalized_sliced_by_suffix \
    --base-name proteomics_plasma

# 3. Differential analysis (cross-sectional + longitudinal)
python 02_exploratory_analysis/identify_differential_analytes.py

# 4. Heatmaps
python 02_exploratory_analysis/generate_differential_cluster_heatmap_limited_group.py

# 5. Enrichr pathway enrichment input
python 02_exploratory_analysis/prepare_enrichr_input.py

# 6. Binary and multilabel classifiers
python 03_model_development/binary_classifier.py
python 03_model_development/multilabel_classifier.py

# 7. Superset enrichment (proteomics LASSO features → Enrichr)
python 03_model_development/superset_enrichment_analysis.py
```

### Metabolomics

```bash
# 1. Clean and normalize raw metabolomics files
python 01_data_cleaning/clean_metabolomics_data.py

# 2. Differential analysis (cross-sectional + longitudinal)
python 02_exploratory_analysis/identify_differential_analytes.py --omics-type metabolomics

# 3. Heatmaps
#    (No cross-sectional heatmaps — zero significant analytes at any timepoint)
python 02_exploratory_analysis/generate_differential_cluster_heatmap_limited_group.py \
    --mode longitudinal \
    --results-dir 04_results_and_figures/differential_analysis/metabolomics/plasma/longitudinal \
    --output-dir 04_results_and_figures/heatmaps/metabolomics/plasma/longitudinal \
    --group Complication

# 4. Binary and multilabel classifiers
python 03_model_development/binary_classifier.py \
    --plasma-dir data/cleaned/metabolomics/normalized_sliced_by_suffix/ \
    --placenta-csv data/cleaned/metabolomics/normalized_full_results/metabolomics_placenta_cleaned_with_metadata.csv \
    --output-dir 04_results_and_figures/models/binary/metabolomics/ \
    --file-prefix metabolomics

python 03_model_development/multilabel_classifier.py \
    --plasma-dir data/cleaned/metabolomics/normalized_sliced_by_suffix/ \
    --placenta-csv data/cleaned/metabolomics/normalized_full_results/metabolomics_placenta_cleaned_with_metadata.csv \
    --output-dir 04_results_and_figures/models/multilabel/metabolomics/ \
    --file-prefix metabolomics

# 5. KEGG pathway enrichment (metabolomics — uses KEGG REST API, not Enrichr)
python 03_model_development/metabolomics_enrichment_analysis.py
```

See each folder's `README.md` for full CLI documentation and output details.

---

## Key analysis notes

- **Metabolomics data are already log2-transformed** by the collaborator. Fold changes are
  computed as simple differences in log2 space (`median_comp − median_ctrl`), not as
  `log2(ratio)`.
- **Metabolomics missingness filter** (Fisher's exact test for group-imbalanced missingness)
  is omitted because the collaborator-provided data contains zero missing values.
- **Longitudinal heatmaps** show within-group change over time (Wilcoxon signed-rank vs zero),
  not differential change between groups. Each group's heatmap is independent.
- **FDR correction** is applied independently per comparison (per group × per adjacent
  timepoint step). Analytes significant across multiple independent comparisons are
  considered more robust findings.
- **Metabolomics cross-sectional analysis** yields no significant analytes at any plasma
  timepoint, reflecting genuine low between-group signal rather than a technical artefact
  (median |fold change| ≈ 0.04; SNR < 0.2 for >97% of analytes).
- **Metabolomics enrichment** uses the KEGG REST API (not Enrichr) because Enrichr requires
  gene symbols. Only named, identified metabolites are mapped; unnamed peaks (`p####`, `n####`)
  cannot be looked up without the original m/z feature table from the collaborator's instrument.

---

## Change log

| Date | Change |
|---|---|
| 2026-03-04 | Added `02_exploratory_analysis/utilities.py`; refactored differential analysis scripts to import from it. |
| 2026-03-04 | Added `02_exploratory_analysis/prepare_enrichr_input.py` for directional Enrichr gene list generation. |
| 2026-03-04 | Created top-level README and `02_exploratory_analysis/README.md`. |
| 2026-03-19 | Added superset enrichment: `03_model_development/superset_enrichment_analysis.py`. |
| 2026-03-30 | Added full metabolomics pipeline: `01_data_cleaning/clean_metabolomics_data.py`. |
| 2026-03-30 | Extended `identify_differential_analytes.py` with `--omics-type` parameter; metabolomics differential analysis now follows identical pipeline to proteomics (adjacent timepoints only, pooled Complication). |
| 2026-03-30 | Added `--file-prefix` to `binary_classifier.py` and `multilabel_classifier.py` to support metabolomics plasma filenames. |
| 2026-03-30 | Added `03_model_development/metabolomics_enrichment_analysis.py` (KEGG REST API ORA). |
| 2026-03-30 | Fixed KEGG pathway ID filter: API returns `path:map####`, not `path:hsa####`; updated `get_hsa_pathways()` accordingly. |
| 2026-03-30 | Added name-simplification fallback in KEGG compound lookup (handles `D-(+)-Maltose`, `α-Lactose`, etc.). |
| 2026-03-31 | Corrected longitudinal heatmap docstring: values are within-group `median_delta`, not `median_delta_complication − median_delta_control`. |
| 2026-03-31 | Confirmed metabolomics missingness filter can be omitted (0% missing across all plasma timepoints). |
