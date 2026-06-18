# README: Processed Data Directory

This directory contains the cleaned, merged, and filtered datasets used for downstream analysis. These files are programmatically generated from raw data sources by standardizing schemas, applying quality thresholds, and encoding categorical features.

## 📁 Directory Structure

* `processed_fitbit_data.csv` (or `.xlsx`) - Cleaned and filtered patient biometric data.
* `processed_placental_histopathology.csv` (or `.xlsx`) - Cleaned, encoded, and filtered placental tissue data.

---

## 🛠️ Data Processing Pipelines

### 1. Fitbit Data Pipeline (`processed_fitbit_data`)

This dataset standardizes patient activity tracker metrics and filters out records with significant data gaps.

* **Ingestion & Standardization:** Reads from two separate raw Fitbit data files and renames columns to establish a consistent naming convention.
* **Merging & Formatting:** Merges the two data sources and reorders the columns into the desired analytical sequence.
* **Quality Filtering (Missing Data):**
  * Extracts a reference table from an auxiliary file detailing the *maximum number of consecutive missing days* per feature, per patient.
  * **Drop Rule:** Automatically removes any patient who has **two or more consecutive days of missing data** for any single feature.
* **Output:** Writes the final, high-fidelity cleaned sheet to the directory.

### 2. Placental Histopathology Data Pipeline (`processed_placental_histopathology`)

This dataset standardizes clinical pathology records, removes uninformative features, and prepares the data for statistical modeling.

* **Ingestion & Standardization:** Reads from two sheets of raw placental histopathology records and standardizes conflicting column names.
* **Merging & Cohort Selection:** Merges the sheets and **deletes all patients with no recorded slides**.
* **Feature Trimming:**
  * Drops explicitly unnecessary administrative columns.
  * Drops any feature/column where **zero patients tested positive** (zero variance).
* **Categorical Encoding:** Encodes the remaining qualitative pathology features into numerical formats (`0`, `1`, `2`, `3`, etc.) representing severity or classifications.
* **Output:** Writes the structured, encoded dataset to the directory.

---

## ⚠️ Notes for Users

> **Do not edit these files manually.** Any manual changes will be overwritten the next time the data processing scripts are executed. If you need to modify the cleaning criteria (e.g., changing the missing day threshold for Fitbit data), update the upstream processing script instead.
>
