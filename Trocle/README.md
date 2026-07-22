Ashley Trocle | Rotation Summary 2026

Piekos Lab

This directory contains finalized scripts and documentation for work done related to the CST Vaginal and DP3 studies.

1. DP3 Study: Clinical/ Demographic Characterization
   
Objective: Clinical and demographic characterization of the DP3 cohort (n=347) and omics subcohort (n=133).
Analysis: participants categorized into Control, FGR, HDP, and sPTB. Statistical significance evaluated via Student’s t-tests/ANOVA and Chi-squared tests.
Outputs: Four versions of "Table 1" (Full/Omics cohorts, both 4-group and Pooled).

2. CST Vaginal Study: Preprocessing
   
Objective: Standardizing CRIB and SPEC cohorts for interomics integration. 

N-Glycan Preprocessing (Finalized)
Pipeline: 7-step process including site-specific ingestion, NA replacement, 20% missingness filtering (Fisher’s/BH correction), and CLR transformation.
Outputs: Analysis-ready CLR matrices, bias testing tables, and processing logs.

Microbiome Preprocessing (Finalized)
Pipeline: 8-step process using zero imputation on raw counts, Relative Abundance (RA) conversion, and CLR transformation.
Outputs: CLR feature matrices and feature-to-taxonomic lineage metadata tables.

Mucin Data (Pending)
Mucin preprocessing and integration will be completed at a later date.
