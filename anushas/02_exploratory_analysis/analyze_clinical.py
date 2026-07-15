import pandas as pd
import numpy as np


# load and return the clinical dataset
def load_sheet():
    sheet = pd.read_csv("01_data_cleaning/processed_data/processed_clinical_data.csv")
    return sheet


# returns a diagnostic summary of data missingness across all patients for the following information:
# maternal age, fetal sex, prepregnancy bmi, race, ethnicity, smoking status
def summarize_missing_info(sheet):
    features = ["maternal_age", "infant_sex", "prepregnancy_bmi_self_or_record", "race", "ethnicity", "smoking"]

    summary_data = []
    missing_ids = {}
    total_patients = len(sheet)

    local_sheet = sheet.copy()

    for feature in features:
        if feature in local_sheet.columns:
            local_sheet[feature] = local_sheet[feature].replace(
                ["nan", "na", "none", "", "null", "n/a"], np.nan
            )
        
        if feature == "race" and "race_is_missing" in local_sheet.columns:
            local_sheet["race_is_missing"] = pd.to_numeric(local_sheet["race_is_missing"], errors="coerce")
            mask = local_sheet["race_is_missing"] == 1
        elif feature == "ethnicity" and "eth_is_missing" in local_sheet.columns:
            local_sheet["eth_is_missing"] = pd.to_numeric(local_sheet["eth_is_missing"], errors="coerce")
            mask = local_sheet["eth_is_missing"] == 1
        elif feature == "smoking" and "smoking_is_missing" in local_sheet.columns:
            local_sheet["smoking_is_missing"] = pd.to_numeric(local_sheet["smoking_is_missing"], errors="coerce")
            mask = local_sheet["smoking_is_missing"] == 1
        else:
            mask = local_sheet[feature].isnull()

        null_count = mask.sum()
        pct_missing = (null_count / total_patients) * 100

        missing_ids[feature] = local_sheet.loc[mask, "id"].tolist()

        summary_data.append({
            "Feature / Metric": feature,
            "Missing Count (NaNs)": int(null_count),
            "Percent Missing": f"{pct_missing:.2f}%"
        })

    return pd.DataFrame(summary_data), missing_ids


# calculates summary statistics for clinical demographic features
# returns two tables containing these statistics:
# one calculates median/IQR for continuous features, and the other calculates count/% for categorical features
def calc_demographic_stats(df):
    continuous_features = ["maternal_age", "prepregnancy_bmi_self_or_record"]
    categorical_features = ["infant_sex", "race", "ethnicity", "smoking_encoded"]
    
    continuous_stats = []
    categorical_stats = []
    total_n = len(df)

    # continuous features: calculate median, IQR
    for col in continuous_features:
        if col in df.columns:
            clean_col = df[col].dropna()
            median = clean_col.median()
            q1 = clean_col.quantile(0.25)
            q3 = clean_col.quantile(0.75)
            iqr = q3 - q1
            continuous_stats.append({
                "Feature": col,
                "Median": f"{median:.2f}",
                "IQR": f"{iqr:.2f}"
            })

    # categorical features: calculate count, %
    for col in categorical_features:
        if col in df.columns:
            counts = df[col].value_counts(dropna=False)
            for val, count in counts.items():
                pct = (count / total_n) * 100
                categorical_stats.append({
                    "Feature": col,
                    "Category": val,
                    "Count": f"{count}",
                    "%": f"{pct:.1f}%"
                })

    return pd.DataFrame(continuous_stats), pd.DataFrame(categorical_stats)


def print_log(missing_report, missing_ids, cont_summary_table, cat_summary_table):
    log_path = "02_exploratory_analysis/outputs/clinical_data_analysis.txt"

    with open(log_path, "w") as f:
        f.write("Following are various statistics about the clinical dataset.\n\n")

        f.write("Patient missingness summary report:\n")
        f.write('(Note that there are truly 5 missing values for the feature "prepregnancy_bmi_self_or_record".\n')
        f.write("However, median imputation has been performed on this feature, so 0 missing values are currently displayed.)\n")
        f.write(missing_report.to_string(index = False))
        f.write("\n\n")

        f.write("Patient IDs of patients with missing data per feature:\n")
        for feature, ids in missing_ids.items():
            f.write(f"{feature}: {', '.join(map(str, ids)) if ids else 'None'}\n")
        f.write("\n")

        f.write("Summary statistics for clinical demographic features:\n")
        f.write("Continuous Features (Median, IQR):\n")
        f.write(f"{'Feature':<35} {'Median':<15} {'IQR':<15}\n")
        for _, row in cont_summary_table.iterrows():
            f.write(f"{row['Feature']:<35} {row['Median']:<15} {row['IQR']:<15}\n")
        f.write("\n\n")
        
        f.write("Categorical Features (Count, %):\n")
        f.write(f"{'Feature':<25} {'Category':<20} {'Count':<15} {'%':<15}\n")
        for _, row in cat_summary_table.iterrows():
            f.write(f"{row['Feature']:<25} {str(row['Category']):<20} {row['Count']:<15} {row['%']:<15}\n")
        f.write("\n\n")


def main():
    clinical_sheet = load_sheet()

    missing_report, missing_ids = summarize_missing_info(clinical_sheet)
    cont_summary_table, cat_summary_table = calc_demographic_stats(clinical_sheet)

    print_log(missing_report, missing_ids, cont_summary_table, cat_summary_table)


if __name__ == "__main__":
    main()