## 02_exploratory_analysis

This directory contains the scripts responsible for performing exploratory data analysis (EDA) on the filtered Fitbit data, placental histopathology features, and delivery variables. It also manages the generation of statistical logs, correlation matrices, and data visualization reports.

---

## 📁 Directory Structure

```text
02_exploratory_analysis/
├── analyze_fitbit.py
├── correlation.py
├── histogram.py
└── outputs/               # Subdirectory containing all generated results
```

## ⚙️ Script Overview & Functionality

### 1. `analyze_fitbit.py`

* **Purpose:** Profiles and computes comprehensive missingness and summary statistics for the Fitbit dataset.
* **Data Pipeline:** 1. Loads the master Fitbit dataset sheet.
  2. Filters rows strictly to include "Fitbit data" events occurring **during the pregnancy window** .
  3. Computes data quality metrics and demographic summaries.
* **Outputs Generated:** Saves results to `outputs/fitbit_data_analysis.txt`, which includes:
  * Total count of missing values across the entire dataset.
  * A patient-level table tracking missing days (a day is only flagged as missing if *all* metrics for that day are null).
  * A matrix tracking the maximum consecutive number of missing days per feature, per patient.
  * Total count of unique dates recorded across the cohort.
  * Metric-level summary statistics (Median and Interquartile Range [IQR]).

### 2. `correlation.py`

* **Purpose:** Evaluates the statistical relationships between placental pathology and clinical delivery outcomes.
* **Data Pipeline:**
  1. Loads two data sheets: placental histopathology variables and delivery variables of interest.
  2. Computes the **Spearman correlation coefficient** (**$\rho$**) and corresponding **$p$**-values between every cross-set pair (ignoring within-set pairs).
  3. Applies a **False Discovery Rate (FDR)** correction to account for multiple testing.
  4. Segregates results based on statistical significance and the direction of the association.
* **Outputs Generated:** Creates four separate files in the `outputs/` directory:
  * `full_correlation_table.txt`: The complete statistical matrix for all tested pairs.
  * `filtered_correlation_table.txt`: Subset containing only pairs that successfully passed the FDR threshold.
  * `positively_associated_delivery_vars.txt`: Significant pairs where **$\rho > 0$** and FDR is satisfied.
  * `negatively_associated_delivery_vars.txt`: Significant pairs where **$\rho < 0$** and FDR is satisfied.

### 3. `histogram.py`

* **Purpose:** Visualizes patient data density and longitudinal tracking over the course of pregnancy.
* **Data Pipeline:**
  1. Loads the master Fitbit dataset sheet.
  2. Filters cohort to include only patients who have at least one non-null metric entry.
  3. Generates comparative distributions plotted against time, complete with vertical dotted markers highlighting **pregnancy trimesters** and the **typical delivery date**.
* **Outputs Generated:** Compiles a dual-plot PDF report saved to `outputs/pregnancy_plots_report.pdf`:
  * **Plot 1 (All Valid Data):** Plots active patient counts per day across all available valid data updates.
  * **Plot 2 (Pregnancy-Only Data):** Identical plot layout, but strictly truncated to data captured within the formal pregnancy timeline.

## 🔄 Dependency Graph

```
SCRIPTS                                     LOCAL OUTPUTS
├── analyze_fitbit.py   ──────────────>     outputs/fitbit_data_analysis.txt
│
├── histogram.py        ──────────────>     outputs/pregnancy_plots_report.pdf
│
└── correlation.py      ──────────────>     outputs/full_correlation_table.txt
                        ──────────────>     outputs/filtered_correlation_table.txt
                        ──────────────>     outputs/positively_associated_delivery_vars.txt
                        ──────────────>     outputs/negatively_associated_delivery_vars.txt
```
