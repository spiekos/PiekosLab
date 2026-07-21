import pandas as pd
import numpy as np


# load and return the clinical dataset
def load_sheet():
    clinical_sheet = pd.read_csv("01_data_cleaning/processed_data/processed_clinical_data.csv")
    master_clinical = pd.read_csv("00_raw_data/dp3 master table v2.xlsx - clinical data.csv")
    placental_sheet = pd.read_csv("01_data_cleaning/processed_data/processed_placental_data.csv")
    master_main = pd.read_csv("00_raw_data/dp3 master table v2.xlsx - Sheet1.csv")
    return clinical_sheet, master_clinical, placental_sheet, master_main


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
        elif feature == "maternal_age":
            mask = local_sheet[feature].isnull() | (local_sheet[feature] == 0)
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


# creates and returns a table tracking total counts for each race/ethnicity intersection
def get_race_counts(sheet):
    patient_df = sheet.copy()
    total_patients = len(patient_df)

    race_cols = sorted([col for col in patient_df.columns if col.startswith("race_") and col != "race_is_missing"])
    eth_col = "hispanic/latino"

    lines = []

    lines.append(f"{'Demographic Group Subtype':<50} | {'Count':<6} | {'Percentage':<8}")
    lines.append("-" * 75)

    for r_col in race_cols:
        if r_col in patient_df.columns and eth_col in patient_df.columns:
            # clean name for presentation
            display_race = r_col.replace("race_", "").title()

            # calculate hispanic intersection
            hisp_mask = (patient_df[r_col] == 1) & (patient_df[eth_col] == 1)
            hisp_count = hisp_mask.sum()
            hisp_pct = (hisp_count / total_patients) * 100 if total_patients > 0 else 0
            if hisp_count > 0:
                lines.append(f"{f'{display_race} / Hispanic':<50} | {hisp_count:<6} | {hisp_pct:>6.1f}%")

            # calculate non-hispanic intersection
            non_hisp_mask = (patient_df[r_col] == 1) & (patient_df[eth_col] == 0)
            non_hisp_count = non_hisp_mask.sum()
            non_hisp_pct = (non_hisp_count / total_patients) * 100 if total_patients > 0 else 0
            if non_hisp_count > 0:
                lines.append(f"{f'{display_race} / Non-Hispanic':<50} | {non_hisp_count:<6} | {non_hisp_pct:>6.1f}%")
                        
    lines.append("\n")
        
    return lines, total_patients


# calculates summary statistics for clinical demographic features
# returns two tables containing these statistics:
# one calculates median/IQR for continuous features, and the other calculates count/% for categorical features
def calc_demographic_stats(df):
    continuous_features = ["maternal_age", "prepregnancy_bmi_self_or_record"]
    categorical_features = ["infant_sex", "race", "ethnicity", "smoking_encoded"]
    
    continuous_stats = []
    categorical_stats = []
    total_n = len(df)

    race_mapping = {
        "african american": "black",
        "korean": "asian"
    }
    ethnicity_mapping = {
        "non hispanic": "not hispanic or latino"
    }

    df["race"] = df["race"].replace(race_mapping)
    df["ethnicity"] = df["ethnicity"].replace(ethnicity_mapping)

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


# generates a table ("Table 1") that introduces our cohort by providing statistics for various demographic features
# categorical variables: count (%). continuous variables: median (IQR).
def generate_table_one(clinical_sheet, master_clinical, master_main):
    df = master_clinical.merge(
        master_main, left_on="ID", right_on="ID", how="inner"
    )
    df = df.merge(
        clinical_sheet, left_on="ID", right_on="id", how="inner"
    )

    # map exact columns to standard feature names
    mapping = {
        "PARA_PRE_TERM": "preterm birth history",
        "HOSP_INSURANCE_GRPING": "insurance",
        "ADI_NATIONAL_RANK": "adi ranking",
        "risk factor y/n": "presence of risk factor",
        "race": "race",
        "ethnicity": "ethnicity",
        "maternal_age": "maternal age",
        "prepregnancy_bmi_self_or_record": "pregravid bmi",
        "smoking_encoded": "smoking",
        "parity": "parity",
        "gravida": "gravida",
        "diabetes": "diabetes",
        "chtn": "hypertension"
    }
    df = df.rename(columns=mapping)

    table1_vars = [
        "preterm birth history", "insurance", "adi ranking", "presence of risk factor", "race", "ethnicity", "maternal age", "pregravid bmi",
        "smoking", "parity", "gravida", "diabetes", "hypertension"
    ]

    return build_summary(df, table1_vars)


# generates a table that introduces our cohort by providing statistics for various pregnancy outcome features
# categorical variables: count (%). continuous variables: median (IQR).
def generate_outcomes_table(clinical_sheet, placental_sheet):
    df = clinical_sheet.merge(placental_sheet, on="id", how="inner")

    # map exact columns to standard feature names
    mapping = {
        "route_of_delivery_1-vag,_2-cs": "mode of delivery",
        "gest_age_del": "gest age del",
        "birthweight": "birth weight",
        "apgar_1": "APGAR 1",
        "apgar_5": "APGAR 5",
        "nicu_days": "NICU days",
        "infant_sex": "fetal sex",
        "O_GDM": "gest diabetes",
        "spontaneous_preterm_birth": "spontaneous preterm birth"
    }
    df = df.rename(columns=mapping)

    # parse gestational hypertension and preeclampsia from "group" and "subgroup" columns
    group_cols = [c for c in ["group", "subgroup"] if c in df.columns]
    if group_cols:
        # combine the text in the "group" and "subgroup" columns for easy parsing
        combined_text = df[group_cols].astype(str).agg(" ".join, axis=1)

        # gestational hypertension: "hdp" or anything containing "ghtn" (case-insensitive)
        df["gest hypertension"] = combined_text.str.contains(
            r"hdp|ghtn", case=False, regex=True, na=False
        )

        # preeclampsia: anything containing "pe" (case-insensitive)
        df["preeclampsia"] = combined_text.str.contains(
            r"pe", case=False, regex=True, na=False
        )

    outcomes_vars = [
        "mode of delivery",
        "gest hypertension",
        "gest diabetes",
        "preeclampsia",
        "spontaneous preterm birth",
        "gest age del",
        "birth weight",
        "apgar 1",
        "apgar 5",
        "nicu days",
        "fetal sex",
    ]

    return build_summary(df, outcomes_vars)


# helper function to calculate counts (%) for categorical variables and median (IQR) for continuous variables
def build_summary(df, variables):
    summary_rows = []

    for var in variables:
        if var in df.columns:
            if df[var].dtype == bool:
                df[var] = df[var].map({True: "Yes", False: "No"})

            is_categorical = (
                df[var].dtype == object
                or df[var].dtype == bool
                or df[var].nunique() < 10
            )

            # capitalize the variable name
            formatted_var_name = str(var).replace("_", " ").title()
            if formatted_var_name.lower() == "pregravid bmi":
                formatted_var_name = "Pregravid BMI"

            if is_categorical:
                summary_rows.append(
                    {"Variable / Category": formatted_var_name, "Statistic": ""}
                )
                counts = df[var].value_counts(dropna=False)
                percents = (
                    df[var].value_counts(normalize=True, dropna=False) * 100
                )

                for cat, count in counts.items():
                    pct = percents[cat]
                    formatted_cat = str(cat).capitalize()

                    summary_rows.append(
                        {
                            "Variable / Category": f"    {formatted_cat}",
                            "Statistic": f"{count} ({pct:.1f}%)",
                        }
                    )

            else:
                valid_data = pd.to_numeric(df[var], errors="coerce").dropna()
                if not valid_data.empty:
                    median = valid_data.median()
                    q25 = valid_data.quantile(0.25)
                    q75 = valid_data.quantile(0.75)
                    iqr = q75 - q25

                    summary_rows.append(
                        {
                            "Variable / Category": formatted_var_name,
                            "Statistic": f"{median:.1f} ({iqr:.1f})",
                        }
                    )

    return pd.DataFrame(summary_rows)


# writes table 1 and outcomes tables to a single text file
def export_formatted_tables_to_file(
    table1_df,
    outcomes_df,
    output_filename="04_results_and_figures/data_analysis/clinical/clinical_summary_tables.txt"
):
    def format_df_to_string(df):
        lines = []
        header_col1 = "Variable / Category".ljust(45)
        header_col2 = "Statistic"
        lines.append(f"{header_col1} {header_col2}")
        lines.append("-" * 60)

        for _, row in df.iterrows():
            col1_val = str(row["Variable / Category"]).ljust(45)
            col2_val = str(row["Statistic"])
            lines.append(f"{col1_val} {col2_val}")
        return "\n".join(lines)

    with open(output_filename, "w") as f:
        f.write("Table 1: Demographic Characteristics\n")
        f.write("-" * 60 + "\n")
        f.write(format_df_to_string(table1_df))

        f.write("\n\n\n")

        f.write("Outcomes Table\n")
        f.write("-" * 60 + "\n")
        f.write(format_df_to_string(outcomes_df))


def print_log(missing_report, missing_ids, race_table, total_patients, cont_summary_table, cat_summary_table):
    log_path = "04_results_and_figures/data_analysis/clinical/clinical_data_analysis.txt"

    with open(log_path, "w") as f:
        f.write("Clinical Sheet Demographics Summary Report\n")
        f.write(f"Total Patient Records Analyzed: {total_patients}\n\n\n")

        f.write("Patient missingness summary report:\n")
        f.write('(Note that there are truly 5 missing values for the feature "prepregnancy_bmi_self_or_record".\n')
        f.write("However, median imputation has been performed on this feature, so 0 missing values are displayed\n")
        f.write("to accurately reflect the current status of the dataset.)\n")
        f.write(missing_report.to_string(index = False))
        f.write("\n\n\n")

        f.write("Patient IDs of patients with missing data per feature:\n")
        for feature, ids in missing_ids.items():
            f.write(f"{feature}: {', '.join(map(str, ids)) if ids else 'None'}\n")
        f.write("\n\n")

        f.write("Counts for race/ethnicity intersections:\n")
        f.write("\n".join(race_table))
        f.write("\n")

        f.write("Summary statistics for clinical demographic features:\n")
        f.write("\n--- Continuous Features (Median, IQR) ---\n")
        f.write(f"{'Feature':<40} | {'Median':<20} | {'IQR':<20}\n")
        for _, row in cont_summary_table.iterrows():
            f.write(f"{row['Feature']:<40} | {row['Median']:<20} | {row['IQR']:<20}\n")

        f.write("\n--- Categorical/Binary Features (Count, %) ---\n")
        f.write(f"{'Feature':<25} | {'Category':<25} | {'Count':<10} | {'%':<10}\n")
        f.write("-" * 75 + "\n")
        for _, row in cat_summary_table.iterrows():
            f.write(f"{row['Feature']:<25} | {str(row['Category']):<25} | {row['Count']:<10} | {row['%']:<10}\n")


def main():
    clinical_sheet, master_clinical, placental_sheet, master_main = load_sheet()

    missing_report, missing_ids = summarize_missing_info(clinical_sheet)
    race_table, total_patients = get_race_counts(clinical_sheet)
    cont_summary_table, cat_summary_table = calc_demographic_stats(clinical_sheet)

    table_1 = generate_table_one(clinical_sheet, master_clinical, master_main)
    outcomes_table = generate_outcomes_table(clinical_sheet, placental_sheet)
    export_formatted_tables_to_file(table_1, outcomes_table)

    print_log(missing_report, missing_ids, race_table, total_patients, cont_summary_table, cat_summary_table)


if __name__ == "__main__":
    main()