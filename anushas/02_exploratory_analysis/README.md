## 02_exploratory_analysis

This directory contains the scripts responsible for performing exploratory data analysis (EDA) on the filtered Fitbit data, placental histopathology features, and delivery variables. It manages data aggregation across pregnancy trimesters, missingness tracking, data quality filtering, and multi-hypothesis statistical testing.

---

## Directory Structure

```text
02_exploratory_analysis/
├── analyze_fitbit.py
├── correlation.py
├── histogram.py
└── outputs/               # Subdirectory containing all generated results
```

## Script Overview & Functionality

### 1. `analyze_fitbit.py`

* **Purpose:** Profiles missingness patterns, checks longitudinal cohort compliance, computes baseline summary statistics, and compiles an integrated trimester-level master dataset.
* **Data Pipeline:**
  1. Loads the processed Fitbit dataset, cleaned placental dataset, and the raw variables-of-interest table.
  2. Filters rows to isolate events labeled `"Fitbit Data"` captured exclusively within the formal pregnancy window. If a patient's gestational age at delivery is missing, the window defaults to 40 weeks.
  3. Sorts and splits the valid tracking timelines into four discrete trimesters:  *First Trimester*, *Early Second Trimester*, *Late Second and Early Third Trimester*, and *Late Third Trimester*.
  4. Generates data compliance matrices determining whether each patient contributed non-missing data for at least 80% of their valid pregnancy tracking days across specific feature categories.
  5. Collapses multi-row tracking streams into single-row patient medians across each trimester window, merging the results side-by-side with clinical delivery records and placental pathology variables.
* **Outputs Generated:**
  * `outputs/fitbit_data_analysis.txt`: Comprehensive statistics log outlining total data gaps, missing days per ID, max consecutive missing day streaks per feature, metric summaries (Median and IQR), unique patients active per trimester, and detailed 80%+ data density compliance metrics.
  * `outputs/master_fitbit_clinical_correlation_data.csv`: The integrated cross-set master dataset compiled for follow-up statistical pipelines.

### 2. `correlation.py`

* **Purpose:** Evaluates statistical relationships across clinical datasets using multi-hypothesis testing to identify meaningful biological associations.
* **Data Pipeline:**
  1. **Test 1 (Placental vs. Delivery):** Merges independent placental histopathology variables with dependent clinical delivery variables by matching patient identifiers.
  2. **Test 2 (Fitbit vs. All Outcomes):** Sources data from the newly generated master Fitbit table, isolating metrics containing `"Trimester"` markers to compare individual trimester-level habits against all recorded clinical and placental endpoints.
  3. **Core Testing Engine:** Runs a vectorized Spearman rank correlation protocol (**$\rho$**) across every valid cross-set pair. Pairs are excluded if variables are non-numeric, constant, or contain fewer than 10 overlapping patient datapoints.
  4. **Multiple Testing Correction:** Applies the Benjamini-Hochberg False Discovery Rate (FDR) procedure to control for multiple comparisons, flagging relationships that satisfy the significance criteria (**$FDR \le 0.05$**).
* **Outputs Generated:** To prevent file collision across separate evaluations, output names use descriptive testing prefixes (`placenta_` and `fitbit_`):
  * `outputs/[prefix]full_correlation_table.txt`: Complete statistical matrix sorted by independent variables containing **$\rho$**, **$p$**-values, sample sizes (**$N$**), and computed FDR **$q$**-values formatted as a scannable Markdown table.
  * `outputs/[prefix]filtered_correlation_table.txt`: Truncated statistical matrix showing only pairs that successfully passed the FDR significance threshold.
  * `outputs/[prefix]positively_associated_vars.txt`: Flat list containing unique target dependent variables exhibiting significant positive relationships (**$\rho > 0$**, **$FDR \le 0.05$**).
  * `outputs/[prefix]negatively_associated_vars.txt`: Flat list containing unique target dependent variables exhibiting significant negative relationships (**$\rho < 0$**, **$FDR \le 0.05$**).

### 3. `histogram.py`

* **Purpose:** Visualizes patient data density and longitudinal tracking over the course of pregnancy.
* **Data Pipeline:**
  1. Loads the master Fitbit dataset sheet.
  2. Filters cohort to include only patients who have at least one non-null metric entry.
  3. Generates comparative distributions plotted against time, complete with vertical dotted markers highlighting **pregnancy trimesters** and the **typical delivery date**.
* **Outputs Generated:** Compiles a dual-plot PDF report saved to `outputs/pregnancy_plots_report.pdf`:
  * **Plot 1 (All Valid Data):** Plots active patient counts per day across all available valid data updates.
  * **Plot 2 (Pregnancy-Only Data):** Identical plot layout, but strictly truncated to data captured within the formal pregnancy timeline.


## Dependency Graph & Pipeline Sequence

For the exploratory data analysis pipeline to evaluate cleanly, scripts should be run in chronological sequence:

```
01_data_cleaning/ processed data outputs
       │
       ▼
02_exploratory_analysis/
 ├── analyze_fitbit.py   ──► Generates master_fitbit_clinical_correlation_data.csv & log
 │     │
 │     ▼
 ├── correlation.py      ──► Consumes master CSV; executes placenta_ and fitbit_ test logs
 │
 └── histogram.py        ──► Generates pregnancy_plots_report.pdf
```
