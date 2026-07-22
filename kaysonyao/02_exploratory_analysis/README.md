# 02 Exploratory Analysis

Differential analysis and visualization pipeline for DP3 multi-omics and survey data.
All scripts are run from the **project root** (the directory containing the `data/` folder).

---

## Scripts

| Script | Description |
|---|---|
| `utilities.py` | Shared library — statistical functions and plotting for all differential analysis scripts |
| `identify_differential_analytes_proteomics.py` | Differential analysis for cleaned Olink proteomics outputs |
| `run_sop_differential.py` | Differential analysis for SOP v4 metabolomics + lipidomics outputs (MTBL_sop / LIPD_sop) |
| `generate_differential_cluster_heatmap_limited_group.py` | Z-score heatmaps from differential results |
| `prepare_enrichr_input_proteomics.py` | Directional gene lists + Enrichr API enrichment for proteomics |
| `survey_distribution_analysis.py` | Score distributions + group comparisons for EPDS, PSS, PUQE-24, diet |
| `water_quality_analysis.py` | THM exposure comparisons across clinical complication groups |

---

### `utilities.py`

Shared library. **Do not run directly** — imported by all other scripts in this folder.

Data loading helpers (`load_data`, `get_analyte_columns`, `normalise_group_labels`,
`METADATA_COLS`, `_GROUP_LABEL_MAP`) are imported from `01_data_cleaning/utilities.py` via
`importlib` to avoid duplication.

Organized into the following sections:

| Section | Contents |
|---|---|
| Shared constants | `FDR_THRESHOLD`, `MIN_N`, `LOG2_FC_THRESHOLD`, `MAX_ANALYTES`, `_TIMEPOINT_ORDER`, `_CTRL_COLOUR`, `_COMPL_COLOUR` |
| Data helpers (from 01/) | `load_data`, `normalise_group_labels`, `get_analyte_columns`, `METADATA_COLS` |
| Analysis log | `_start_analysis_log` |
| Cross-sectional | `_test_one_pair`, `run_cross_sectional` |
| Longitudinal | `_compute_deltas`, `run_longitudinal` |
| Sample counts | `write_sample_count_report` |
| MetaboAnalyst export | `write_metaboanalyst_export` |
| Heatmap helpers | `collect_significant_analytes`, `_rank_analytes_by_min_q`, `_within_group_order` |
| Heatmap (CS) | `plot_cross_sectional_heatmap`, `plot_pairwise_cross_sectional_heatmap` |
| Heatmap (long.) | `_sort_delta_columns`, `_is_adjacent_comparison`, `plot_longitudinal_heatmap` |
| Boxplots (CS) | `plot_cross_sectional_boxplots` |
| Boxplots (long.) | `plot_longitudinal_boxplots` |

---

### `identify_differential_analytes_proteomics.py`

Runs cross-sectional and/or longitudinal differential analysis on cleaned Olink proteomics CSVs.

**Cross-sectional:**
- Pools FGR, HDP, and sPTB into a single `"Complication"` label; runs one `Control vs Complication` test per dataset.
- Test: two-sided Mann-Whitney U.
- Multiple testing: Benjamini-Hochberg FDR, corrected per comparison.
- Hit criteria: q < 0.05 **and** |log2 FC| ≥ log2(1.5) ≈ 0.585.
- Minimum n = 5 non-missing observations per group per analyte.

**Longitudinal:**
- Within-group Wilcoxon signed-rank on per-participant adjacent deltas (B−A, C−B, D−C, E−D).
- Run separately for: Control, FGR, HDP, sPTB, and pooled Complication.
- Same FDR and fold-change thresholds.

**Usage:**

```bash
# Default: all plasma timepoints + placenta CS + full longitudinal
python 02_exploratory_analysis/identify_differential_analytes_proteomics.py

# Cross-sectional on a single CSV
python 02_exploratory_analysis/identify_differential_analytes_proteomics.py \
    --mode cross_sectional \
    --input data/cleaned/proteomics/normalized_sliced_by_suffix/proteomics_plasma_formatted_suffix_C.csv \
    --output-dir 04_results_and_figures/differential_analysis/plasma/cross_sectional/C
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--mode` | *(none)* | `cross_sectional`, `longitudinal`, or `both`. Omit to run the full default pipeline. |
| `--input` | — | Cleaned wide-format CSV (required for CS mode) |
| `--group-col` | `Group` | Column name containing group labels |
| `--output-dir` | `results` | Root output directory |
| `--timepoint-files` | — | Space-separated CSVs in chronological order (longitudinal mode) |
| `--timepoint-labels` | `T1 T2 …` | Labels matching each timepoint file |
| `--group` | — | Complication group for longitudinal analysis |
| `--subject-col` | `SubjectID` | Column used to pair participants across timepoints |

---

### `run_sop_differential.py`

Runs cross-sectional and longitudinal differential analysis on SOP v4 pipeline outputs.
Supports `MTBL_sop` (metabolomics) and `LIPD_sop` (lipidomics).

For `MTBL_sop`: runs placenta cross-sectional **and** all plasma timepoints (A–E) cross-sectional
and longitudinal. For `LIPD_sop`: plasma only (no placenta in current run).

Cross-sectional groups are merged to Control vs. pooled Complication.
Longitudinal is run separately for Control, FGR, HDP, sPTB, and pooled Complication.

Also produces:
- **Boxplots** (top 50 significant analytes, cross-sectional and longitudinal)
- **MetaboAnalyst exports** — formatted CSVs with m/z + RT from feature metadata for upload to MetaboAnalyst

**Inputs:** Expects cleaned CSVs from `sop_omics_pipeline.py` under
`data/cleaned/sop_omics_pipeline_v2/<TISSUE>/`.

**Usage:**

```bash
# Metabolomics (plasma + placenta)
python 02_exploratory_analysis/run_sop_differential.py --dataset MTBL_sop

# Lipidomics (plasma only)
python 02_exploratory_analysis/run_sop_differential.py --dataset LIPD_sop

# Both (default — omit --dataset)
python 02_exploratory_analysis/run_sop_differential.py
```

**Outputs** (under `04_results_and_figures/differential_analysis/<DATASET>/`):

```
MTBL_sop/
├── MTBL_plasma_sample_counts.csv
├── plasma/
│   ├── cross_sectional/{A,B,C,D,E}/
│   │   ├── Control_vs_Complication_differential_results.csv
│   │   └── Control_vs_Complication_significant_analytes.csv
│   ├── cross_sectional_boxplots/        Top-50 CS boxplots per timepoint
│   ├── longitudinal/
│   │   └── <Group>_<T_b>_minus_<T_a>_longitudinal_results.csv
│   ├── longitudinal_boxplots/           Top-50 longitudinal boxplots
│   └── metaboanalyst/
│       ├── cross_sectional/{A,B,C,D,E}/ MetaboAnalyst-formatted CSVs
│       └── longitudinal/
└── placenta/                            (MTBL_sop only)
    ├── cross_sectional/
    │   ├── Control_vs_Complication_differential_results.csv
    │   └── Control_vs_Complication_significant_analytes.csv
    └── metaboanalyst/
```

---

### `generate_differential_cluster_heatmap_limited_group.py`

Generates z-score heatmaps from differential analysis results (proteomics).

**Cross-sectional heatmap (`plot_pairwise_cross_sectional_heatmap`):**
- Rows: significant analytes (up to `MAX_ANALYTES` = 500, ranked by q-value)
- Columns: individual samples, Ward/Euclidean clustered within each group
- Values: per-analyte z-score clipped ±2.5
- Outputs: `Control_vs_Complication_heatmap.pdf`, `.png` (300 dpi), `_heatmap_data.csv`

**Longitudinal heatmap (`plot_longitudinal_heatmap`):**
- Rows: analytes significant in ≥ 1 adjacent step for the specified group
- Columns: adjacent delta comparisons (B−A, C−B, D−C, E−D)
- Values: `median_delta` z-scored row-wise, clipped ±2.0
- Significant cells marked with a dot (•)

**Usage:**

```bash
# Default: CS heatmaps for plasma (A–E) + placenta, longitudinal for all groups
python 02_exploratory_analysis/generate_differential_cluster_heatmap_limited_group.py

# Longitudinal only, one group
python 02_exploratory_analysis/generate_differential_cluster_heatmap_limited_group.py \
    --mode longitudinal \
    --results-dir 04_results_and_figures/differential_analysis/plasma/longitudinal \
    --output-dir 04_results_and_figures/heatmaps/plasma/longitudinal \
    --group Complication
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--mode` | `cross_sectional` | `cross_sectional`, `longitudinal`, or `both` |
| `--input` | — | Cleaned wide-format CSV (required for CS mode) |
| `--results-dir` | — | Directory containing differential result CSVs |
| `--output-dir` | — | Directory to save heatmap outputs |
| `--group-col` | `Group` | Column name for group labels |
| `--group` | — | Group for longitudinal heatmap |
| `--label` | `cross_sectional` | Filename prefix for CS output files |

---

### `prepare_enrichr_input_proteomics.py`

Splits significant analytes into directional gene lists and runs enrichment via the **Enrichr API**
(`gseapy`). Handles plasma CS (per timepoint), plasma longitudinal (per group × adjacent step),
and placenta CS.

**Databases queried by default:** `GO_Biological_Process_2025`, `KEGG_2026`, `Reactome_Pathways_2024`

**Usage:**

```bash
# Default: auto-discovers all CS + longitudinal + placenta results
python 02_exploratory_analysis/prepare_enrichr_input_proteomics.py

# Skip Enrichr API calls (gene lists only)
python 02_exploratory_analysis/prepare_enrichr_input_proteomics.py --skip-enrichment

# Custom databases
python 02_exploratory_analysis/prepare_enrichr_input_proteomics.py \
    --gene-sets GO_Biological_Process_2025 KEGG_2026
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--sig-csv` | — | Path to a single `*_significant_analytes.csv`; triggers single-comparison mode |
| `--g1` / `--g2` | — | Group labels (required with `--sig-csv`) |
| `--results-dir` | `04_results_and_figures/differential_analysis/plasma/cross_sectional` | Plasma CS results root |
| `--longitudinal-results-dir` | `…/plasma/longitudinal` | Longitudinal results directory |
| `--placenta-results-dir` | `…/placenta/cross_sectional` | Placenta CS results directory |
| `--output-dir` | `04_results_and_figures/enrichment` | Root output directory |
| `--gene-sets` | `GO_Biological_Process_2025 KEGG_2026 Reactome_Pathways_2024` | Enrichr databases |
| `--all-databases` | False | Query all ≈300+ Enrichr databases |
| `--skip-enrichment` | False | Write gene list files only |
| `--skip-longitudinal` | False | Skip longitudinal enrichment |
| `--skip-placenta` | False | Skip placenta enrichment |

---

### `survey_distribution_analysis.py`

Score distributions and group comparisons for EPDS, PSS, PUQE-24, and diet surveys.

- Kruskal-Wallis H: overall test across all 4 groups
- Mann-Whitney U: pairwise Control vs FGR / HDP / sPTB at each visit
- BH FDR: corrected per survey × visit

**Inputs:** `data/survey/cleaned/{epds,pss,puqe24,diet}_cleaned.csv`

**Outputs:** `04_results_and_figures/survey/<survey>/`
- `{survey}_{visit}_distribution.png` — violin + strip plots
- `{survey}_stats_results.csv` — all pairwise test results
- `{survey}_significant_pairs.csv` — BH-significant pairs only

```bash
python 02_exploratory_analysis/survey_distribution_analysis.py
```

---

### `water_quality_analysis.py`

Compares THM exposure metrics (average concentrations and exceedance rates) between Control
and each complication group. Kruskal-Wallis + pairwise Mann-Whitney with BH FDR.

**Input:** `data/survey/cleaned/water_cleaned.csv`

**Outputs:** `04_results_and_figures/survey/water/`

```bash
python 02_exploratory_analysis/water_quality_analysis.py
```

---

## Key parameters

All thresholds live in `utilities.py` and apply to both cross-sectional and longitudinal analyses.

| Parameter | Value | Description |
|---|---|---|
| `FDR_THRESHOLD` | 0.05 | Benjamini-Hochberg q-value cutoff |
| `LOG2_FC_THRESHOLD` | log2(1.5) ≈ 0.585 | Minimum absolute log2 fold change |
| `MIN_N` | 5 | Minimum non-missing observations per group per analyte |
| `MAX_ANALYTES` | 500 | Maximum rows shown in any heatmap |
| `MIN_SIG_ANALYTES` | 5 | Minimum significant analytes required to generate a heatmap |
| `_PNG_DPI` | 300 | PNG resolution for saved figures |
