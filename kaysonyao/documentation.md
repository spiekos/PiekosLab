# DP3 Metabolomics – Project File Reference

**Pipeline:** SOP v2 (HMDB) · ComBat batch correction · Differential analysis → LASSO → binary classifier  
**Datasets:** MTBL_plasma, MTBL_placenta

---

## Primary Code Files (`03_model_development/`)

| File | Purpose |
|------|---------|
| `utilities.py` | Shared helpers: data loading, label normalisation, significant-analyte loading |
| `binary_classifier.py` | Core pipeline — LASSO feature selection, Optuna hyperparameter tuning, model fitting & evaluation |
| `run_sop_models.py` | Run binary classifier **with** differential pre-filter (main analysis); saves to `04_results_and_figures/models/binary/MTBL_sop/` |
| `run_sop_nodiff.py` | Run binary classifier **without** differential pre-filter (sensitivity check); saves to `…/MTBL_sop_nodiff/` |
| `superset_differential_analysis.py` | Differential analysis (Mann-Whitney U + BH FDR) across all features |
| `feature_interpretation.py` | Post-hoc: feature importance plots, SHAP, coefficient heatmaps |

---

## Input Data (`data/cleaned/sop_omics_pipeline_v2/`)

Each tissue subfolder (`MTBL_plasma`, `MTBL_placenta`) contains:

| File | Contents |
|------|---------|
| `{tissue}_cleaned_with_metadata.csv` | Full cleaned feature matrix with all sample metadata |
| `{tissue}_suffix_{A-E}.csv` | Per-timepoint slices (plasma only; A–E = timepoints 1–5) |
| `{tissue}_dedup_log.csv` | Every feature dropped during deduplication with phase and reason |
| `{tissue}_feature_filters.csv` | Features removed by QC RSD filter (step 8) with per-feature RSD |
| `{tissue}_feature_metadata.csv` | Annotation tier, compound name, m/z, RT for all retained features |
| `{tissue}_sample_filters.csv` | Samples removed and reason |
| `pipeline_log.txt` | Full preprocessing summary: filter counts, annotation tiers, batch correction notes, QC warnings |
| `diagnostics/` | PCA plots (pre/post correction, batch & group colouring), normalisation curves |

---

## Differential Analysis Outputs (`04_results_and_figures/differential_analysis/MTBL_sop/`)

```
plasma/cross_sectional/{A,B,C,D,E}/
    Control_vs_Complication_differential_results.csv   ← one row per analyte; columns: feature, log2FC, p_value, q_value
placenta/cross_sectional/
    Control_vs_Complication_differential_results.csv
```

Significant analytes (q < 0.05) per timepoint (MTBL plasma):

| Timepoint | Sig. analytes |
|-----------|--------------|
| A | 40 / 502 |
| B | 72 / 502 |
| C | 68 / 502 |
| D | 49 / 502 |
| E | 43 / 502 |

---

## Model Outputs (`04_results_and_figures/models/binary/`)

### `MTBL_sop/` — main analysis (differential pre-filter applied)

```
plasma/{A,B,C,D,E}/
    summary.json                ← n_features pre/post LASSO, best model, test metrics per complication
    lasso_selected_features.csv ← LASSO-selected feature names and coefficients
    test_results.csv            ← per-sample predictions and true labels
    tuned_hyperparams.json      ← best hyperparameters from Optuna
placenta/all/
    (same structure)
all_summaries.json              ← consolidated summary across all timepoints
```

### `MTBL_sop_nodiff/` — sensitivity run (no differential pre-filter)

Same structure. LASSO received all 502 features and retained all of them (p >> n problem). Use for comparison only.

---

## Gallery & Reports

| File | Purpose |
|------|---------|
| `pipeline_plot_gallery.html` | Interactive browser gallery — all QC plots, model outputs, and summary reports organised by dataset and category. Open in any browser; no server needed. |
| `MTBL_plasma_QC_modelling_summary.pptx` | Slide deck: POS/NEG PCA before/after correction, full feature pipeline table, deduplication breakdown |
| `documentation.md` | This file |

---

## Key Numbers (MTBL Plasma)

| Stage | Features | Note |
|-------|---------|------|
| Raw (POS + NEG merged) | 9,744 | Before any filtering |
| After QC RSD filter | 1,074 | Features with QC RSD < 30% in pooled QCs retained |
| After deduplication | 502 | 572 dropped: adduct/isotope/fragment collapse (500), named-within-compound (49), mz+RT (22), formula+RT (1) |
| After differential filter | 40–72 | Varies by timepoint; q < 0.05 threshold |
| After LASSO | 4–68 | Varies by timepoint; TP E is highly sparse (4 features) |
