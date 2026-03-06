# 04_results_and_figures

All analytical outputs from the DP3 proteomics pipeline are stored here, organized by pipeline step.

```
04_results_and_figures/
│
├── data_cleaning/                   ← proteomics_diagnostics.py (01_data_cleaning/)
│   ├── plasma/
│   │   ├── pca/                     PCA pre-/post-ComBat
│   │   ├── combat_assessment/       Batch-effect correction diagnostics
│   │   ├── sample_distributions/    Per-sample NPX distribution plots
│   │   ├── sample_boxplots/         Sample boxplots
│   │   ├── density_overlay/         Density overlay plots
│   │   └── batch_comparison/        Batch comparison plots
│   └── placenta/                    (same structure)
│
├── differential_analysis/           ← identify_differential_analytes.py (02_exploratory_analysis/)
│   ├── sample_counts_per_group_timepoint.csv
│   ├── analysis_log.txt
│   ├── plasma/
│   │   ├── cross_sectional/
│   │   │   └── <timepoint A-E>/
│   │   │       ├── Control_vs_Complication_differential_results.csv
│   │   │       └── Control_vs_Complication_significant_analytes.csv
│   │   └── longitudinal/
│   │       └── <group>_<T_b>_minus_<T_a>_longitudinal_results.csv
│   └── placenta/
│       └── cross_sectional/
│           ├── Control_vs_Complication_differential_results.csv
│           └── Control_vs_Complication_significant_analytes.csv
│
├── heatmaps/                        ← generate_differential_cluster_heatmap_limited_group.py
│   ├── plasma/
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
│   └── placenta/
│       └── cross_sectional/
│           └── Control_vs_Complication/
│
└── enrichment/                      ← prepare_enrichr_input.py (02_exploratory_analysis/)
    └── plasma/
        ├── cross_sectional/
        │   └── <timepoint>/
        │       └── Control_vs_Complication/
        │           ├── higher_in_Complication.txt
        │           ├── higher_in_Control.txt
        │           ├── all_significant.txt
        │           ├── significant_with_direction.csv
        │           └── enrichment/
        │               ├── higher_in_Complication_enrichment.csv
        │               └── higher_in_Control_enrichment.csv
        └── longitudinal/
            └── <group>/
                └── <T_b>_minus_<T_a>/
                    ├── increasing.txt
                    ├── decreasing.txt
                    ├── all_significant.txt
                    ├── significant_with_direction.csv
                    └── enrichment/
                        ├── increasing_enrichment.csv
                        └── decreasing_enrichment.csv
    └── placenta/
        └── cross_sectional/
            └── Control_vs_Complication/
                ├── higher_in_Complication.txt
                ├── higher_in_Control.txt
                ├── all_significant.txt
                ├── significant_with_direction.csv
                └── enrichment/
                    ├── higher_in_Complication_enrichment.csv
                    └── higher_in_Control_enrichment.csv
```

## Notes

- Cleaned proteomics CSVs (inputs to differential analysis) remain in `data/cleaned/` as intermediate data files.
- Raw proteomics files remain in `data/proteomics/`.
- The stale pairwise enrichment results (Control_vs_FGR, Control_vs_HDP, etc.) from before the binary Control vs Complication redesign have been discarded.
