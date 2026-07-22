# 04 Results and Figures

All analytical outputs from the DP3 pipeline are stored here, organized by pipeline step and modality.
Cleaned intermediate data (inputs to differential analysis) remain in `data/cleaned/`.

```
04_results_and_figures/
│
├── differential_analysis/
│   │
│   ├── plasma/                           ← identify_differential_analytes_proteomics.py (proteomics)
│   │   ├── sample_counts_per_group_timepoint.csv
│   │   ├── analysis_log.txt
│   │   ├── cross_sectional/
│   │   │   └── <timepoint A-E>/
│   │   │       ├── Control_vs_Complication_differential_results.csv
│   │   │       └── Control_vs_Complication_significant_analytes.csv
│   │   └── longitudinal/
│   │       └── <group>_<T_b>_minus_<T_a>_longitudinal_results.csv
│   ├── placenta/                         (proteomics)
│   │   └── cross_sectional/
│   │       ├── Control_vs_Complication_differential_results.csv
│   │       └── Control_vs_Complication_significant_analytes.csv
│   │
│   ├── MTBL_sop/                         ← run_sop_differential.py (metabolomics SOP v4)
│   │   ├── MTBL_plasma_sample_counts.csv
│   │   ├── plasma/
│   │   │   ├── cross_sectional/
│   │   │   │   └── {A,B,C,D,E}/
│   │   │   │       ├── Control_vs_Complication_differential_results.csv
│   │   │   │       └── Control_vs_Complication_significant_analytes.csv
│   │   │   ├── cross_sectional_boxplots/ Top-50 analyte boxplots per timepoint
│   │   │   ├── longitudinal/
│   │   │   │   └── <group>_<T_b>_minus_<T_a>_longitudinal_results.csv
│   │   │   ├── longitudinal_boxplots/    Top-50 analyte boxplots
│   │   │   └── metaboanalyst/
│   │   │       ├── cross_sectional/{A,B,C,D,E}/ MetaboAnalyst-formatted CSVs with m/z + RT
│   │   │       └── longitudinal/
│   │   └── placenta/
│   │       ├── cross_sectional/
│   │       │   ├── Control_vs_Complication_differential_results.csv
│   │       │   └── Control_vs_Complication_significant_analytes.csv
│   │       └── metaboanalyst/
│   │
│   └── LIPD_sop/                         ← run_sop_differential.py (lipidomics SOP v4)
│       └── plasma/                       (no placenta in current LIPD run)
│           ├── cross_sectional/{A,B,C,D,E}/
│           ├── cross_sectional_boxplots/
│           ├── longitudinal/
│           ├── longitudinal_boxplots/
│           └── metaboanalyst/
│
├── heatmaps/                             ← generate_differential_cluster_heatmap_limited_group.py
│   ├── plasma/                           (proteomics)
│   │   ├── cross_sectional/
│   │   │   └── <timepoint>/
│   │   │       └── Control_vs_Complication/
│   │   │           ├── Control_vs_Complication_heatmap.pdf
│   │   │           ├── Control_vs_Complication_heatmap.png
│   │   │           └── Control_vs_Complication_heatmap_data.csv
│   │   └── longitudinal/
│   │       ├── <group>_longitudinal_heatmap.pdf
│   │       ├── <group>_longitudinal_heatmap.png
│   │       └── <group>_longitudinal_heatmap_data.csv
│   └── placenta/                         (proteomics)
│       └── cross_sectional/
│           └── Control_vs_Complication/
│
├── enrichment/                           ← prepare_enrichr_input_proteomics.py
│   ├── plasma/
│   │   ├── cross_sectional/
│   │   │   └── <timepoint>/
│   │   │       └── Control_vs_Complication/
│   │   │           ├── higher_in_Complication.txt
│   │   │           ├── higher_in_Control.txt
│   │   │           ├── all_significant.txt
│   │   │           ├── significant_with_direction.csv
│   │   │           └── enrichment/
│   │   │               ├── higher_in_Complication_enrichment.csv
│   │   │               └── higher_in_Control_enrichment.csv
│   │   └── longitudinal/
│   │       └── <group>/<T_b>_minus_<T_a>/
│   │           ├── increasing.txt / decreasing.txt / all_significant.txt
│   │           ├── significant_with_direction.csv
│   │           └── enrichment/
│   │               ├── increasing_enrichment.csv
│   │               └── decreasing_enrichment.csv
│   └── placenta/
│       └── cross_sectional/Control_vs_Complication/
│
├── models/                               ← 03_model_development/
│   ├── binary/ multilabel/              Proteomics classifiers
│   ├── sop_models/                      SOP v4 classifiers (run_sop_models.py)
│   │   ├── MTBL_sop/
│   │   └── LIPD_sop/
│   ├── sop_nodiff/                      Ablation (run_sop_nodiff.py)
│   └── survey/                          Survey/environmental classifiers
│
└── survey/                              ← survey_distribution_analysis.py / water_quality_analysis.py
    ├── epds/ pss/ puqe24/ diet/
    │   ├── {survey}_{visit}_distribution.png
    │   ├── {survey}_stats_results.csv
    │   └── {survey}_significant_pairs.csv
    └── water/
        ├── water_avg_thm_distribution.png
        ├── water_exceed_rate_distribution.png
        └── water_stats_results.csv
```
