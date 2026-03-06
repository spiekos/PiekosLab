# 02 Exploratory Analysis

Differential analysis and visualization pipeline for DP3 proteomics (plasma and placenta).
All scripts are run from the **project root** (the directory containing the `data/` folder).

---

## Scripts

### `utilities.py`
Shared library. Contains every constant, helper, statistical function, and plotting function
used by the other scripts in this folder. **Do not run directly** — import from it.

Organized into the following sections:

| Section | Contents |
|---|---|
| Shared constants | `FDR_THRESHOLD`, `MIN_N`, `LOG2_FC_THRESHOLD`, `MAX_ANALYTES`, `_METADATA_COLS`, `_TIMEPOINT_ORDER`, `_CTRL_COLOUR`, `_COMPL_COLOUR` |
| Data helpers | `load_data`, `normalise_group_labels`, `get_analyte_columns` |
| Analysis log | `_start_analysis_log` |
| Cross-sectional | `_test_one_pair`, `run_cross_sectional` |
| Longitudinal | `_compute_deltas`, `run_longitudinal` |
| Sample counts | `write_sample_count_report` |
| Heatmap helpers | `collect_significant_analytes`, `_rank_analytes_by_min_q`, `_within_group_order` |
| Heatmap (CS) | `plot_cross_sectional_heatmap`, `plot_pairwise_cross_sectional_heatmap` |
| Heatmap (long.) | `_sort_delta_columns`, `_is_adjacent_comparison`, `plot_longitudinal_heatmap` |

---

### `identify_differential_analytes.py`
Runs cross-sectional and/or longitudinal differential analysis on cleaned proteomics CSVs.

**Cross-sectional:**
- Pools FGR, HDP, and sPTB into a single `"Complication"` label; runs one `Control vs Complication` test per dataset.
- Test: two-sided Mann-Whitney U.
- Multiple testing: Benjamini-Hochberg FDR, corrected per comparison.
- Hit criteria: q < 0.05 **and** |log2 FC| ≥ log2(1.5) ≈ 0.585 (i.e. ≥ 1.5× linear fold change).
- Minimum n = 5 non-missing observations per group per analyte.
- Outputs: `Control_vs_Complication_differential_results.csv` (all analytes) and
  `Control_vs_Complication_significant_analytes.csv` (hits only).

**Longitudinal:**
- Within-group Wilcoxon signed-rank on per-participant adjacent deltas (B−A, C−B, D−C, E−D).
- Run separately for: `Control`, `FGR`, `HDP`, `sPTB`, and pooled `Complication` (FGR + HDP + sPTB merged).
- Same FDR and fold-change thresholds as cross-sectional.
- Output per group × adjacent step: `<group>_<T_b>_minus_<T_a>_longitudinal_results.csv`.

**Usage:**

```bash
# Default (no args): Control vs Complication (pooled) for all plasma timepoints + placenta
# (cross-sectional) and within-group longitudinal for Control/FGR/HDP/sPTB + Complication
python 02_exploratory_analysis/identify_differential_analytes.py

# Cross-sectional on a single cleaned CSV (pre-merge complications manually, or pass merged CSV)
python 02_exploratory_analysis/identify_differential_analytes.py \
    --mode cross_sectional \
    --input data/cleaned/proteomics/normalized_sliced_by_suffix/proteomics_plasma_formatted_suffix_C.csv \
    --output-dir data/diff_analysis/results/plasma/cross_sectional/C

# Longitudinal — within-group for one group (e.g. Complication, FGR, HDP, sPTB, or Control)
python 02_exploratory_analysis/identify_differential_analytes.py \
    --mode longitudinal \
    --timepoint-files data/cleaned/.../suffix_A.csv data/cleaned/.../suffix_B.csv \
    --timepoint-labels A B \
    --group Complication \
    --output-dir data/diff_analysis/results/plasma/longitudinal
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--mode` | `cross_sectional` | `cross_sectional`, `longitudinal`, or `both` |
| `--input` | — | Cleaned wide-format CSV (required for CS mode) |
| `--group-col` | `Group` | Column name containing group labels |
| `--output-dir` | `results` | Root output directory |
| `--timepoint-files` | — | Space-separated CSVs in chronological order (longitudinal) |
| `--timepoint-labels` | `T1 T2 …` | Labels matching each timepoint file (longitudinal) |
| `--group` | — | Complication group for longitudinal analysis (e.g. `FGR`, `HDP`, `sPTB`). Always compared against Control. |
| `--subject-col` | `SubjectID` | Column used to pair participants across timepoints |

---

### `generate_differential_cluster_heatmap_limited_group.py`
Generates z-score heatmaps from differential analysis results.

#### Cross-sectional heatmap (`plot_pairwise_cross_sectional_heatmap`)
- **Rows:** significant analytes only (up to `MAX_ANALYTES` = 500, ranked by q-value). Default mode runs one `Control vs Complication` (pooled) comparison per dataset.
- **Columns:** individual samples, Ward/Euclidean clustered within each group, then groups concatenated (Control first). When `g2_source_groups` is set (e.g. `["FGR", "HDP", "sPTB"]`), samples from those groups are relabelled as Complication for display.
- **Values:** per-analyte z-score across all displayed samples (mean = 0, std = 1 per row), clipped ±2.5.
- **Row clustering:** Ward linkage, Euclidean distance.
- **Column colour bar:** binary — Control (steelblue) vs Complication (tomato).
- **Outputs:** `Control_vs_Complication/Control_vs_Complication_heatmap.pdf`, `…_heatmap.png` (300 dpi), `…_heatmap_data.csv`.

#### Longitudinal heatmap (`plot_longitudinal_heatmap`)
- **Rows:** significant analytes in ≥ 1 adjacent step for the specified group, up to `MAX_ANALYTES`.
- **Columns:** adjacent delta comparisons in fixed chronological order (B−A, C−B, D−C, E−D); not clustered.
- **Values:** `median_delta`, row-wise z-scored across comparisons, clipped ±2.0.
- **Row clustering:** Ward linkage, Euclidean distance.
- **Cell annotations:** significant cells marked with a dot (•); non-significant cells dimmed.
- **Outputs:** `<group>_longitudinal_heatmap.pdf`, `…_heatmap.png` (300 dpi), `…_heatmap_data.csv`.
- Default mode generates heatmaps for Control, FGR, HDP, sPTB, and pooled Complication.

**Usage:**

```bash
# Default (no args): Control vs Complication CS heatmaps for plasma (A–E) + placenta,
# plus longitudinal heatmaps for Control/FGR/HDP/sPTB/Complication
python 02_exploratory_analysis/generate_differential_cluster_heatmap_limited_group.py

# Single CS heatmap (full cross-sectional, all groups via clustermap)
python 02_exploratory_analysis/generate_differential_cluster_heatmap_limited_group.py \
    --mode cross_sectional \
    --input data/cleaned/.../suffix_C.csv \
    --results-dir data/diff_analysis/results/plasma/cross_sectional/C \
    --output-dir data/diff_analysis/results/plasma/cross_sectional/C \
    --label cross_sectional

# Longitudinal heatmap for one group (e.g. Complication, FGR, HDP, sPTB, or Control)
python 02_exploratory_analysis/generate_differential_cluster_heatmap_limited_group.py \
    --mode longitudinal \
    --results-dir data/diff_analysis/results/plasma/longitudinal \
    --output-dir data/diff_analysis/results/plasma/longitudinal \
    --group Complication
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--mode` | `cross_sectional` | `cross_sectional`, `longitudinal`, or `both` |
| `--input` | — | Cleaned wide-format CSV (required for CS mode) |
| `--results-dir` | — | Directory containing differential result CSVs (required) |
| `--output-dir` | — | Directory to save heatmap outputs (required) |
| `--group-col` | `Group` | Column name for group labels |
| `--group` | — | Group for longitudinal heatmap (e.g. `Control`, `FGR`, `HDP`, `sPTB`, or `Complication`). |
| `--label` | `cross_sectional` | Filename prefix for CS heatmap output files |

---

### `prepare_enrichr_input.py`
Splits significant analytes into directional gene lists and runs pathway enrichment
via the **Enrichr API** using `gseapy` — equivalent to the [Enrichr website](https://maayanlab.cloud/Enrichr/) but fully automated. Handles plasma cross-sectional (per timepoint), plasma longitudinal (per group × adjacent step), and placenta cross-sectional results.

**Cross-sectional direction logic** (`fold_change = median_Complication − median_Control`):
- `fold_change > 0` → protein **higher in Complication**
- `fold_change < 0` → protein **higher in Control**

**Longitudinal direction logic** (`median_delta = value_T_later − value_T_earlier`):
- `median_delta > 0` → protein **increasing** at this timepoint step
- `median_delta < 0` → protein **decreasing** at this timepoint step

**Databases queried by default:** `GO_Biological_Process_2025`, `KEGG_2026`, `Reactome_Pathways_2024`

**Plasma cross-sectional outputs** (saved to `04_results_and_figures/enrichment/plasma/cross_sectional/<timepoint>/Control_vs_Complication/`):

| File | Contents |
|---|---|
| `higher_in_Complication.txt` | Upregulated in Complication; can be pasted into Enrichr directly |
| `higher_in_Control.txt` | Upregulated in Control |
| `all_significant.txt` | All significant analytes regardless of direction |
| `significant_with_direction.csv` | Full table with `direction` column |
| `enrichment/higher_in_Complication_enrichment.csv` | Enrichr results, all databases, sorted by adj. p-value |
| `enrichment/higher_in_Control_enrichment.csv` | Enrichr results for Control-upregulated proteins |

**Placenta cross-sectional outputs** (saved to `04_results_and_figures/enrichment/placenta/cross_sectional/Control_vs_Complication/`):

Same file layout as plasma cross-sectional. No timepoint subdirectories (placenta has a single cross-sectional comparison).

**Longitudinal outputs** (saved to `04_results_and_figures/enrichment/plasma/longitudinal/<group>/<T_b>_minus_<T_a>/`):

| File | Contents |
|---|---|
| `increasing.txt` | Analytes rising at this timepoint step |
| `decreasing.txt` | Analytes falling at this timepoint step |
| `all_significant.txt` | All significant analytes |
| `significant_with_direction.csv` | Full table with `direction` column |
| `enrichment/increasing_enrichment.csv` | Enrichr results for increasing proteins |
| `enrichment/decreasing_enrichment.csv` | Enrichr results for decreasing proteins |

**Key enrichment output columns:** `Gene_set` (database), `Term`, `Overlap`, `P_value`, `Adj_P_value`, `Odds_Ratio`, `Combined_Score`, `Genes`

**Dependency:** requires `gseapy` (`pip install gseapy`). If not installed, the script still writes the gene list text files but skips the API calls.

**Usage:**

```bash
# Default (no args): auto-discovers plasma CS (A–E) + longitudinal + placenta CS, runs all enrichment
python 02_exploratory_analysis/prepare_enrichr_input.py

# Query every available Enrichr database (~300+), mirroring the website behaviour
python 02_exploratory_analysis/prepare_enrichr_input.py --all-databases

# Single CS comparison with all databases
python 02_exploratory_analysis/prepare_enrichr_input.py \
    --sig-csv 04_results_and_figures/differential_analysis/plasma/cross_sectional/C/Control_vs_Complication_significant_analytes.csv \
    --g1 Control \
    --g2 Complication \
    --all-databases

# Gene lists only (no Enrichr API calls)
python 02_exploratory_analysis/prepare_enrichr_input.py --skip-enrichment

# Skip longitudinal enrichment, run CS only (plasma + placenta)
python 02_exploratory_analysis/prepare_enrichr_input.py --skip-longitudinal

# Skip placenta enrichment
python 02_exploratory_analysis/prepare_enrichr_input.py --skip-placenta

# Custom databases
python 02_exploratory_analysis/prepare_enrichr_input.py \
    --gene-sets GO_Biological_Process_2025 GO_Molecular_Function_2023 KEGG_2026
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--sig-csv` | — | Path to a single `*_significant_analytes.csv`; triggers single CS comparison mode |
| `--g1` | — | Group 1 label (required with `--sig-csv`) |
| `--g2` | — | Group 2 label (required with `--sig-csv`) |
| `--results-dir` | `04_results_and_figures/differential_analysis/plasma/cross_sectional` | Root plasma CS results directory for auto-discovery |
| `--longitudinal-results-dir` | `04_results_and_figures/differential_analysis/plasma/longitudinal` | Longitudinal results directory for auto-discovery |
| `--placenta-results-dir` | `04_results_and_figures/differential_analysis/placenta/cross_sectional` | Placenta CS results directory for auto-discovery |
| `--output-dir` | `04_results_and_figures/enrichment` | Root output directory |
| `--gene-sets` | `GO_Biological_Process_2025 KEGG_2026 Reactome_Pathways_2024` | Space-separated Enrichr databases; ignored if `--all-databases` is set |
| `--all-databases` | `False` | Query every available Enrichr database (~300+); overrides `--gene-sets` |
| `--skip-enrichment` | `False` | Write gene list files only; skip all Enrichr API calls |
| `--skip-longitudinal` | `False` | Skip longitudinal enrichment |
| `--skip-placenta` | `False` | Skip placenta enrichment |

---

## Output directory structure

All outputs are written under `04_results_and_figures/`, organized by pipeline step.

```
04_results_and_figures/
│
├── differential_analysis/          ← identify_differential_analytes.py
│   ├── sample_counts_per_group_timepoint.csv
│   ├── plasma/
│   │   ├── cross_sectional/
│   │   │   ├── A/
│   │   │   │   ├── Control_vs_Complication_differential_results.csv
│   │   │   │   └── Control_vs_Complication_significant_analytes.csv
│   │   │   ├── B/ … E/  (same structure)
│   │   └── longitudinal/
│   │       ├── Control_B_minus_A_longitudinal_results.csv
│   │       ├── FGR_B_minus_A_longitudinal_results.csv
│   │       ├── HDP_B_minus_A_longitudinal_results.csv
│   │       ├── sPTB_B_minus_A_longitudinal_results.csv
│   │       ├── Complication_B_minus_A_longitudinal_results.csv
│   │       └── … (one file per group × adjacent timepoint pair)
│   └── placenta/
│       └── cross_sectional/
│           ├── Control_vs_Complication_differential_results.csv
│           └── Control_vs_Complication_significant_analytes.csv
│
├── heatmaps/                        ← generate_differential_cluster_heatmap_limited_group.py
│   ├── plasma/
│   │   ├── cross_sectional/
│   │   │   ├── A/
│   │   │   │   └── Control_vs_Complication/
│   │   │   │       ├── Control_vs_Complication_heatmap.pdf
│   │   │   │       ├── Control_vs_Complication_heatmap.png
│   │   │   │       └── Control_vs_Complication_heatmap_data.csv
│   │   │   ├── B/ … E/  (same structure)
│   │   └── longitudinal/
│   │       ├── Control_longitudinal_heatmap.pdf
│   │       ├── FGR_longitudinal_heatmap.pdf
│   │       ├── HDP_longitudinal_heatmap.pdf
│   │       ├── sPTB_longitudinal_heatmap.pdf
│   │       └── Complication_longitudinal_heatmap.pdf
│   └── placenta/
│       └── cross_sectional/
│           └── Control_vs_Complication/
│               ├── Control_vs_Complication_heatmap.pdf
│               ├── Control_vs_Complication_heatmap.png
│               └── Control_vs_Complication_heatmap_data.csv
│
└── enrichment/                      ← prepare_enrichr_input.py
    ├── plasma/
    │   ├── cross_sectional/
    │   │   └── <timepoint>/
    │   │       └── Control_vs_Complication/
    │   │           ├── higher_in_Complication.txt
    │   │           ├── higher_in_Control.txt
    │   │           ├── all_significant.txt
    │   │           ├── significant_with_direction.csv
    │   │           └── enrichment/
    │   │               ├── higher_in_Complication_enrichment.csv
    │   │               └── higher_in_Control_enrichment.csv
    │   └── longitudinal/
    │       └── <group>/
    │           └── <T_b>_minus_<T_a>/
    │               ├── increasing.txt
    │               ├── decreasing.txt
    │               ├── all_significant.txt
    │               ├── significant_with_direction.csv
    │               └── enrichment/
    │                   ├── increasing_enrichment.csv
    │                   └── decreasing_enrichment.csv
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

---

## Key parameters

All thresholds live in `utilities.py` and apply to both cross-sectional and longitudinal analyses.

| Parameter | Value | Description |
|---|---|---|
| `FDR_THRESHOLD` | 0.05 | Benjamini-Hochberg q-value cutoff |
| `LOG2_FC_THRESHOLD` | log2(1.5) ≈ 0.585 | Minimum absolute log2 fold change (= 1.5× linear FC) |
| `MIN_N` | 5 | Minimum non-missing observations per group per analyte |
| `MAX_ANALYTES` | 500 | Maximum rows shown in any heatmap |
| `MIN_SIG_ANALYTES` | 5 | Minimum significant analytes required to generate a heatmap |
| `_PNG_DPI` | 300 | PNG resolution for saved heatmaps |

