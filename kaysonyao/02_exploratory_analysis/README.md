# 02 Exploratory Analysis

Survey and environmental analyses. Run from the project root. The A-E-era differential,
heatmap and enrichment scripts (and this folder's `utilities.py`) were retired on 2026-09-30;
they are in `_legacy/scripts/a_e_tools/`.

| Script | Purpose |
|---|---|
| `survey_distribution_analysis.py` | Score distributions and group comparisons for EPDS, PSS, PUQE-24 and diet |
| `water_quality_analysis.py` | THM exposure (average and exceedance) by complication group |

## `survey_distribution_analysis.py`

- Kruskal-Wallis H across all 4 groups
- Two-sample KS test: Control vs FGR / HDP / sPTB at each visit
- Benjamini-Hochberg FDR per survey x visit

Input: `data/processed/survey/{epds,pss,puqe24,diet}_cleaned.csv`.
Output: `04_results_and_figures/survey/<survey>/` (`{survey}_{visit}_distribution.png`,
`{survey}_stats_results.csv`, `{survey}_significant_pairs.csv`).

```bash
python 02_exploratory_analysis/survey_distribution_analysis.py
```

## `water_quality_analysis.py`

Compares THM exposure metrics between Control and each complication group: Kruskal-Wallis plus
pairwise Mann-Whitney with BH FDR.

Input: `data/processed/survey/water_cleaned.csv`. Output: `04_results_and_figures/survey/water/`.

```bash
python 02_exploratory_analysis/water_quality_analysis.py
```
