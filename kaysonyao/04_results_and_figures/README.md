# 04 Results and Figures

Generated outputs only. Everything here can be regenerated from `data/` with the scripts.
Old results (A-E models, the Table 4 R2 runs, differential/heatmap/enrichment/presentation
folders, one-off CSVs sent to the PI) were moved to `_to_delete/` on 2026-09-29.

Currently here:

```
04_results_and_figures/
└── survey/{diet,epds,pss,puqe24,water}/   survey_distribution_analysis.py, water_quality_analysis.py
```

Where scripts write:

| Folder | Written by |
|---|---|
| `pre_post_combat_pca/<MTBL\|LIPD>/<tissue>/` (+ `diagnostics/`) | `01_data_cleaning/sop_omics_pipeline.py` |
| `trajectory_plots/<MTBL\|LIPD>/<tissue>/` | `01_data_cleaning/sop_omics_pipeline.py` |
| `models_*/<DS>/<window>/`, `models_*/model_metrics_*.csv` | `03_model_development/run_base_models.py` (`--out`) |
| `holdout_*/` | `03_model_development/run_holdout_evaluation.py` (`--out`) |
| `survey/` | survey scripts in `02_exploratory_analysis/` |
