# DP3 Multi-Omics Analysis

Multi-omics preprocessing, differential analysis, and predictive modelling pipeline for the DP3 cohort.
Covers three omics modalities:

- **Proteomics** — plasma Olink Explore + placental tissue lysate (NPX values, log2 scale)
- **Metabolomics** — plasma + placenta untargeted LC-MS from Compound Discoverer exports (SOP v4 pipeline; MTBL_plasma / MTBL_placenta)
- **Lipidomics** — plasma + placenta untargeted LC-MS from Compound Discoverer exports (SOP v4 pipeline; LIPD_plasma / LIPD_placenta; LIPID MAPS database)

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
├── survey/                              Raw survey CSVs (EPDS, PSS, PUQE-24, diet, water quality)
│   └── cleaned/                         Cleaned survey CSVs produced by clean_survey_data.py
├── dp3 master table v2.xlsx             Master metadata (Group, Subgroup, Batch, GestAge, etc.)
├── dp3 n=133 clinical and metadata.xlsx Clinical metadata (n=133 omics cohort)
└── cleaned/
    ├── proteomics/
    │   ├── normalized_full_results/     Full cleaned matrices (plasma + placenta)
    │   └── normalized_sliced_by_suffix/ Plasma split by timepoint suffix (A–E)
    └── sop_omics_pipeline/              SOP v4 outputs (metabolomics + lipidomics)
        ├── MTBL_plasma/                 Cleaned metabolomics plasma matrix + diagnostics
        ├── MTBL_placenta/               Cleaned metabolomics placenta matrix + diagnostics
        ├── LIPD_plasma/                 Cleaned lipidomics plasma matrix + diagnostics
        └── LIPD_placenta/               Cleaned lipidomics placenta matrix + diagnostics
```

Raw Compound Discoverer exports for metabolomics and lipidomics live in the sibling
`kaylaxu/` repository (`kaylaxu/data/MTBL_plasma/`, `kaylaxu/data/LIPD_plasma/`, etc.).

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
`openpyxl`, `pycombat`, `scikit-learn`, `xgboost`, `optuna`, `gseapy` (optional, for Enrichr enrichment).

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
python 02_exploratory_analysis/identify_differential_analytes_proteomics.py

# 4. Heatmaps
python 02_exploratory_analysis/generate_differential_cluster_heatmap_limited_group.py

# 5. Enrichr pathway enrichment
python 02_exploratory_analysis/prepare_enrichr_input_proteomics.py

# 6. Binary and multilabel classifiers
python 03_model_development/binary_classifier.py
python 03_model_development/multilabel_classifier.py

# 7. Superset enrichment (proteomics LASSO features → Enrichr)
python 03_model_development/superset_enrichment_analysis.py
```

### Metabolomics + Lipidomics (SOP v4 pipeline)

```bash
# 1. Run SOP preprocessing for all four datasets
python 01_data_cleaning/sop_omics_pipeline.py
# Processes MTBL_plasma, MTBL_placenta, LIPD_plasma, LIPD_placenta
# Outputs to data/cleaned/sop_omics_pipeline/

# Run a single dataset (e.g. metabolomics plasma only)
python 01_data_cleaning/sop_omics_pipeline.py --datasets MTBL_plasma

# 2. Differential analysis on SOP outputs
python 02_exploratory_analysis/run_sop_differential.py --dataset MTBL_sop
python 02_exploratory_analysis/run_sop_differential.py --dataset LIPD_sop
# Or run both at once (omit --dataset)
python 02_exploratory_analysis/run_sop_differential.py

# 3. Binary classifiers on SOP outputs (with differential pre-filtering)
python 03_model_development/run_sop_models.py --dataset MTBL_sop
python 03_model_development/run_sop_models.py --dataset LIPD_sop

# 4. Binary classifiers on SOP outputs (no differential pre-filtering — ablation)
python 03_model_development/run_sop_nodiff.py

# 5. KEGG/HMDB pathway analysis for metabolomics
python 03_model_development/run_pathway_analysis.py
```

### Survey / Environmental Data

```bash
# 1. Clean raw survey files
python 01_data_cleaning/clean_survey_data.py

# 2. Survey score distribution analysis (Kruskal-Wallis + pairwise Mann-Whitney)
python 02_exploratory_analysis/survey_distribution_analysis.py

# 3. Water quality analysis (THM exposure vs. complication groups)
python 02_exploratory_analysis/water_quality_analysis.py

# 4. Survey-based classifiers
python 03_model_development/run_survey_models.py
```

See each folder's `README.md` for full CLI documentation and output details.

---

## Key analysis notes

- **SOP v4 pipeline** implements the April 2026 DP3 SOP from raw Compound Discoverer exports.
  Steps: missing-value standardization → ISTD normalization → median fold-change batch correction
  → feature missingness filter → sample missingness filter → log2 transformation → half-minimum
  imputation → ComBat batch correction → RSD filter → IQR filter → deduplication → annotation.
- **Fold changes** are computed in log2 space. For metabolomics, the conversion
  `log2(2^median - 1)` is applied to back-convert NPX-style log2(x+1) space before computing
  the log2 ratio, ensuring consistent FC thresholds across modalities.
- **FDR correction** is applied independently per comparison (per group × per adjacent
  timepoint step). Analytes significant across multiple independent comparisons are
  considered more robust findings.
- **Metabolomics enrichment** uses the KEGG REST API (not Enrichr) because Enrichr requires
  gene symbols. Only named, identified metabolites are mapped.
- **Survey cohort** is broader than the omics cohort (≈390 vs. 133 subjects); `clean_survey_data.py`
  uses the full `clinical data` sheet of the master table as its group source.

---

## Change log

| Date | Change |
|---|---|
| 2026-03-04 | Added `02_exploratory_analysis/utilities.py`; refactored differential analysis scripts to import from it. |
| 2026-03-04 | Added `02_exploratory_analysis/prepare_enrichr_input_proteomics.py` for directional Enrichr gene list generation. |
| 2026-03-04 | Created top-level README and `02_exploratory_analysis/README.md`. |
| 2026-03-19 | Added superset enrichment: `03_model_development/superset_enrichment_analysis.py`. |
| 2026-03-30 | Added full metabolomics pipeline (now superseded by SOP v4 for new runs). |
| 2026-03-30 | Extended differential analysis with `--omics-type` parameter. |
| 2026-03-30 | Added `03_model_development/metabolomics_enrichment_analysis.py` (KEGG REST API ORA). |
| 2026-04-xx | Added SOP v4 pipeline: `01_data_cleaning/sop_omics_pipeline.py`. Covers MTBL_plasma, MTBL_placenta, LIPD_plasma, LIPD_placenta from raw Compound Discoverer exports. |
| 2026-04-xx | Added `02_exploratory_analysis/run_sop_differential.py` for differential analysis on SOP v4 outputs. |
| 2026-04-xx | Added `03_model_development/run_sop_models.py` and `run_sop_nodiff.py` for classifiers on SOP v4 outputs. |
| 2026-04-xx | Added survey pipeline: `01_data_cleaning/clean_survey_data.py`, `02_exploratory_analysis/survey_distribution_analysis.py`, `02_exploratory_analysis/water_quality_analysis.py`, `03_model_development/run_survey_models.py`. |
| 2026-05-xx | Consolidated shared helpers (`load_data`, `get_analyte_columns`, `normalise_group_labels`, `METADATA_COLS`) into `01_data_cleaning/utilities.py` as single source of truth; imported by 02 and 03 utilities via importlib. |
| 2026-05-xx | Replaced hand-rolled XLSX parser with `openpyxl` throughout. |
| 2026-05-xx | Deleted `clean_lipids_data.py` (superseded by SOP v4 pipeline LIPD configs). |
