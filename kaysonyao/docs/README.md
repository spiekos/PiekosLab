# DP3 multi-omics: longitudinal prediction of pregnancy complications

Preprocessing, exploratory analysis and predictive modelling for the DP3 cohort
(metabolomics, lipidomics, Olink proteomics, surveys). Run every script from the
project root (the folder that holds `data/`).

The governing SOP is the file at the project root:
`DP3_Metabolomics_Lipidomics_Preprocessing_SOP_v7 JK Edits 08122026.docx`.

## Layout

```
kaysonyao/
├── DP3_..._SOP_v7 JK Edits 08122026.docx   current SOP (only file kept at the root)
├── docs/                    this README, in-fold preprocessing plan, working notes, reorg manifest
├── 01_data_cleaning/        preprocessing (SOP pipeline, proteomics, survey, binning, split)
│   └── combat_validation/   R/Python cross-check of combat_ref.py against sva::ComBat
├── 02_exploratory_analysis/ survey and water-quality analysis
├── 03_model_development/    nested-CV base models, held-out evaluation
├── 04_results_and_figures/  all generated results and figures
├── data/                    see data/README.md
├── _legacy/                 git-ignored: last version of superseded scripts and docs
└── _to_delete/              git-ignored: staged removals, review then move to Trash
```

## Run order (current pipeline)

```bash
# 1. Metabolomics + lipidomics, all five datasets (Jenn's SOP pipeline, with ComBatRef).
#    For modelling: outcome-blind ComBat fitted on dev participants + QC pools only.
python 01_data_cleaning/sop_omics_pipeline.py --blind-combat \
    --fit-subjects data/processed/locked_split.csv --fit-split dev
#    For differential analysis only (ComBat protects Group/Subgroup/GestAgeDelivery, SOP Step 12):
#    python 01_data_cleaning/sop_omics_pipeline.py --output-root data/processed_differential

# 2. Proteomics (Olink; reference-batch ComBat)
python 01_data_cleaning/clean_proteomics_data.py

# 3. Gestational windows T1-T5, one sample per participant per window (midpoint rule)
python 01_data_cleaning/bin_echo_windows.py --scheme dp3_5t --datasets MTBL_plasma LIPD_plasma MTBL_urine proteomics_plasma

# 4. Base models (nested CV: 5 outer x 3 inner, grouped by participant, Optuna 40 trials)
python 03_model_development/run_base_models.py --windows T1 T2 T3 T4 T5 \
    --datasets MTBL_plasma LIPD_plasma MTBL_urine proteomics_plasma --out 04_results_and_figures/models_dp3_5t
```

`data/processed/locked_split.csv` is the fixed 70/30 participant split; `make_locked_split.py`
refuses to overwrite it. The 30% test set is read only by `run_holdout_evaluation.py`, once.

Survey: `clean_survey_data.py` -> `02_exploratory_analysis/survey_distribution_analysis.py` and
`water_quality_analysis.py`.

## Running the SOP pipeline from this folder

All inputs are in place for all five datasets (modification lists and urine extraction added
2026-09-30; LIPD_placenta points at the `060525` placenta export).

## Open items (pending PI)

- W4/W5 include delivery and post-delivery samples (SOP v7 Step 8b/8c not yet applied in code).
- Parametric ComBat over-compression in metabolomics/lipidomics: mean-only vs non-parametric.
- SOP v7 gaps in the code: Hex2Cer/Hex3Cer/CerPE not mapped, no per-class ISTD log,
  Step 16 neutral-lipid-only failures excluded instead of flagged for review.
- In-fold preprocessing: see docs/infold_preprocessing_plan.md.

## Environment

Python: numpy, pandas, scipy, statsmodels, scikit-learn, optuna, xgboost, matplotlib,
seaborn, openpyxl (gseapy optional). R + sva only for `combat_validation/`.
