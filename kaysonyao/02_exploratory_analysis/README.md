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
| Shared constants | `FDR_THRESHOLD`, `MIN_N`, `LOG2_FC_THRESHOLD`, `MAX_ANALYTES`, `_METADATA_COLS`, `_TIMEPOINT_ORDER` |
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
- All pairwise group comparisons derived from the `Group` column.
- Test: two-sided Mann-Whitney U.
- Multiple testing: Benjamini-Hochberg FDR, corrected independently per comparison.
- Hit criteria: q < 0.05 **and** |log2 FC| ≥ log2(1.5) ≈ 0.585 (i.e. ≥ 1.5× linear fold change).
- Minimum n = 5 non-missing observations per group per analyte.
- Outputs per comparison: `<g1>_vs_<g2>_differential_results.csv` (all analytes) and
  `<g1>_vs_<g2>_significant_analytes.csv` (hits only).

**Longitudinal:**
- Adjacent timepoint deltas only (B−A, C−B, D−C, E−D) for a specified group.
- Per-participant delta = value_T_later − value_T_earlier (paired on `SubjectID`).
- Test: two-sided Wilcoxon signed-rank (`zero_method='wilcox'`).
- Same FDR and fold-change thresholds as cross-sectional.
- Output per delta: `<group>_<T_later>_minus_<T_earlier>_longitudinal_results.csv`.

**Usage:**

```bash
# Default (no args): runs all plasma timepoints + placenta, both cross-sectional
# and longitudinal, using hard-coded paths relative to project root
python 02_exploratory_analysis/identify_differential_analytes.py

# Cross-sectional on a single cleaned CSV
python 02_exploratory_analysis/identify_differential_analytes.py \
    --mode cross_sectional \
    --input data/cleaned/proteomics/normalized_sliced_by_suffix/proteomics_plasma_formatted_suffix_C.csv \
    --output-dir data/diff_analysis/results/plasma/cross_sectional/C

# Longitudinal for one group
python 02_exploratory_analysis/identify_differential_analytes.py \
    --mode longitudinal \
    --timepoint-files data/cleaned/.../suffix_A.csv data/cleaned/.../suffix_B.csv \
    --timepoint-labels A B \
    --group Control \
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
| `--group` | — | Target group for longitudinal analysis (e.g. `Control`) |
| `--subject-col` | `SubjectID` | Column used to pair participants across timepoints |

---

### `generate_differential_cluster_heatmap_limited_group.py`
Generates z-score heatmaps from differential analysis results.

#### Cross-sectional heatmap (`plot_pairwise_cross_sectional_heatmap`)
- **Rows:** all analytes up to `MAX_ANALYTES` (500), significant ones prioritised and marked with ★.
- **Columns:** individual samples, Ward/Euclidean clustered within each group, then groups concatenated alphabetically.
- **Values:** per-analyte z-score across all displayed samples (mean = 0, std = 1 per row), clipped ±2.5.
- **Row clustering:** Ward linkage, Euclidean distance.
- **Column colour bar:** one colour per group.
- **Outputs:** `<g1>_vs_<g2>/<g1>_vs_<g2>_heatmap.pdf`, `…_heatmap.png` (300 dpi), `…_heatmap_data.csv`.

> **Note on the mosaic appearance:** Including non-significant analytes (those without ★)
> causes a salt-and-pepper mosaic because per-row z-scoring amplifies noise in
> non-differential proteins to the same ±2.5 colour range as true hits. This is a
> known visual artefact of the "show all" design; see PI discussion in progress.

#### Longitudinal heatmap (`plot_longitudinal_heatmap`)
- **Rows:** analytes significant in ≥ 1 adjacent delta comparison, up to `MAX_ANALYTES`.
- **Columns:** adjacent delta comparisons in fixed chronological order (B−A, C−B, D−C, E−D); not clustered.
- **Values:** median delta per analyte per comparison, row-wise z-scored across comparisons, clipped ±2.0.
- **Row clustering:** Ward linkage, Euclidean distance.
- **Cell annotations:** significant cells (q < 0.05) marked with a dot (•); non-significant cells dimmed with a white overlay.
- **Outputs:** `<group>_longitudinal_heatmap.pdf`, `…_heatmap.png` (300 dpi), `…_heatmap_data.csv`.

**Usage:**

```bash
# Default (no args): generates all plasma pairwise CS heatmaps (A–E) +
# placenta CS heatmaps + longitudinal heatmaps for all 4 groups
python 02_exploratory_analysis/generate_differential_cluster_heatmap_limited_group.py

# Single pairwise CS heatmap
python 02_exploratory_analysis/generate_differential_cluster_heatmap_limited_group.py \
    --mode cross_sectional \
    --input data/cleaned/.../suffix_C.csv \
    --results-dir data/diff_analysis/results/plasma/cross_sectional/C \
    --output-dir data/diff_analysis/results/plasma/cross_sectional/C \
    --label Control_vs_HDP

# Longitudinal heatmap for one group
python 02_exploratory_analysis/generate_differential_cluster_heatmap_limited_group.py \
    --mode longitudinal \
    --results-dir data/diff_analysis/results/plasma/longitudinal \
    --output-dir data/diff_analysis/results/plasma/longitudinal \
    --group Control
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--mode` | `cross_sectional` | `cross_sectional`, `longitudinal`, or `both` |
| `--input` | — | Cleaned wide-format CSV (required for CS mode) |
| `--results-dir` | — | Directory containing differential result CSVs (required) |
| `--output-dir` | — | Directory to save heatmap outputs (required) |
| `--group-col` | `Group` | Column name for group labels |
| `--group` | — | Target group for longitudinal heatmap (required for longitudinal mode) |
| `--label` | `cross_sectional` | Filename prefix for CS heatmap output files |

---

### `prepare_enrichr_input.py`
Splits significant analytes from each pairwise comparison into directional gene lists
ready to paste into [Enrichr](https://maayanlab.cloud/Enrichr/) for pathway enrichment.

**Direction logic:**
`fold_change = median_group2 − median_group1` (NPX values are in log2 scale), so:
- `fold_change > 0` → protein is **higher in group2** relative to group1
- `fold_change < 0` → protein is **higher in group1** relative to group2

**Outputs per comparison** (saved to `data/enrichr_input/<timepoint>/<g1>_vs_<g2>/`):

| File | Contents |
|---|---|
| `higher_in_<g2>.txt` | Upregulated in group2; paste directly into Enrichr |
| `higher_in_<g1>.txt` | Upregulated in group1; paste directly into Enrichr |
| `all_significant.txt` | All significant analytes regardless of direction |
| `significant_with_direction.csv` | Full table with `direction` column for reference |

**Usage:**

```bash
# Default (no args): auto-discovers all plasma cross-sectional significant_analytes.csv
# files across timepoints A–E and writes lists to data/enrichr_input/
python 02_exploratory_analysis/prepare_enrichr_input.py

# Single comparison
python 02_exploratory_analysis/prepare_enrichr_input.py \
    --sig-csv data/diff_analysis/results/plasma/cross_sectional/C/Control_vs_HDP_significant_analytes.csv \
    --g1 Control \
    --g2 HDP \
    --output-dir data/enrichr_input/C/Control_vs_HDP
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--sig-csv` | — | Path to a single `*_significant_analytes.csv`; triggers single-comparison mode |
| `--g1` | — | Group 1 label (required with `--sig-csv`) |
| `--g2` | — | Group 2 label (required with `--sig-csv`) |
| `--results-dir` | `data/diff_analysis/results/plasma/cross_sectional` | Root results directory for auto-discovery |
| `--output-dir` | `data/enrichr_input` | Root output directory |

---

## Output directory structure

```
data/diff_analysis/results/
├── plasma/
│   ├── cross_sectional/
│   │   ├── A/
│   │   │   ├── Control_vs_HDP_differential_results.csv
│   │   │   ├── Control_vs_HDP_significant_analytes.csv
│   │   │   ├── … (one pair per comparison)
│   │   │   └── Control_vs_HDP/
│   │   │       ├── Control_vs_HDP_heatmap.pdf
│   │   │       ├── Control_vs_HDP_heatmap.png
│   │   │       └── Control_vs_HDP_heatmap_data.csv
│   │   ├── B/ … E/  (same structure)
│   └── longitudinal/
│       ├── Control_B_minus_A_longitudinal_results.csv
│       ├── … (one file per group × adjacent timepoint pair)
│       ├── Control_longitudinal_heatmap.pdf
│       └── Control_longitudinal_heatmap_data.csv
├── placenta/
│   └── cross_sectional/  (same structure as plasma)
└── sample_counts_per_group_timepoint.csv

data/enrichr_input/
└── <timepoint>/
    └── <g1>_vs_<g2>/
        ├── higher_in_<g2>.txt
        ├── higher_in_<g1>.txt
        ├── all_significant.txt
        └── significant_with_direction.csv
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

---

## Change log

| Date | Change |
|---|---|
| 2026-03-04 | Added `utilities.py`; moved all functions/constants from `identify_differential_analytes.py` and `generate_differential_cluster_heatmap_limited_group.py` into it. Both scripts now import from `utilities`. |
| 2026-03-04 | Added `prepare_enrichr_input.py` to split significant analytes into directional Enrichr-ready gene lists. |
| 2026-03-04 | Heatmap script changed to show **all analytes** (up to 500) with significant ones marked ★, rather than filtering to significant-only. Pending PI review of mosaic appearance. |
