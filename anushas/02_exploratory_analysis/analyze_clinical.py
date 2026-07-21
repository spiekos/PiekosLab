import pandas as pd
import numpy as np


# load, clean, and return the datasets
def load_sheet():
    clinical_sheet = pd.read_csv("01_data_cleaning/processed_data/processed_clinical_data.csv")
    
    # read the first two rows to handle the split header layout
    raw_header_1 = pd.read_csv("00_raw_data/dp3 master table v2.xlsx - clinical data.csv", nrows=1, header=None).values[0]
    raw_header_2 = pd.read_csv("00_raw_data/dp3 master table v2.xlsx - clinical data.csv", skiprows=1, nrows=1, header=None).values[0]
    
    # combine them: use row 2 headers for columns where row 1 is blank/unnamed or where the secondary block starts
    combined_headers = []
    for h1, h2 in zip(raw_header_1, raw_header_2):
        h1_str = str(h1).strip()
        h2_str = str(h2).strip()
        
        if h2_str and h2_str.lower() != "nan" and ("unnamed" in h1_str.lower() or h1_str == ""):
            combined_headers.append(h2_str)
        elif h1_str and h1_str.lower() != "nan":
            combined_headers.append(h1_str)
        else:
            combined_headers.append(h2_str if h2_str and h2_str.lower() != "nan" else h1_str)

    # load the actual data skipping the header rows
    master_clinical = pd.read_csv("00_raw_data/dp3 master table v2.xlsx - clinical data.csv", skiprows=2, header=None)
    master_clinical.columns = [str(c).strip() for c in combined_headers[:len(master_clinical.columns)]]

    placental_sheet = pd.read_csv("01_data_cleaning/processed_data/processed_placental_data.csv")
    master_main = pd.read_csv("00_raw_data/dp3 master table v2.xlsx - Sheet1.csv")
    
    # normalize ID column and lowercase all headers across all sheets
    for df in [clinical_sheet, master_clinical, placental_sheet, master_main]:
        if df is not None:
            for col in df.columns:
                if str(col).strip().lower() == "id":
                    df.rename(columns={col: "id"}, inplace=True)
            df.columns = [str(c).strip().lower() for c in df.columns]
                    
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


# combines _x and _y duplicate columns resulting from merges into single columns
def coalesce_merged_columns(df):
    df = df.copy()
    x_cols = [c for c in df.columns if c.endswith("_x")]

    for x_col in x_cols:
        base_name = x_col[:-2]
        y_col = f"{base_name}_y"

        if y_col in df.columns:
            # combine _x and _y
            combined = df[x_col].fillna(df[y_col])

            # fix mixed types: try converting back to numeric first
            numeric_combined = pd.to_numeric(combined, errors="coerce")
            df[base_name] = numeric_combined

            df.drop(columns=[x_col, y_col], inplace=True)
        else:
            df.rename(columns={x_col: base_name}, inplace=True)

    # clean up any remaining _y columns
    y_cols = [c for c in df.columns if c.endswith("_y")]
    for y_col in y_cols:
        base_name = y_col[:-2]
        if base_name not in df.columns:
            df.rename(columns={y_col: base_name}, inplace=True)
        else:
            df.drop(columns=[y_col], inplace=True)

    return df


def standardize_race_ethnicity(df):
    if "race" in df.columns:
        def map_race(val):
            if pd.isna(val):
                return "Na"
            v = str(val).strip().lower()
            if v in ["africian american", "black"]:
                return "Black"
            elif v == "white":
                return "White"
            elif v in ["asian", "korean"]:
                return "Asian"
            elif v in ["american indian or alaska native", "american indian/alaska native"]:
                return "American Indian/Alaska Native"
            elif v == "declined":
                return "Declined"
            elif v in ["unknown", "na", "nan"]:
                return "Na"
            return "Na"  # Fallback for any unexpected entries
        
        df["race"] = df["race"].apply(map_race)

    if "ethnicity" in df.columns:
        def map_ethnicity(val):
            if pd.isna(val):
                return "Na"
            v = str(val).strip().lower()
            if v in ["hispanic or latino", "hispanic"]:
                return "Hispanic/Latino"
            elif v in ["not hispanic or latino", "non hispanic"]:
                return "Not Hispanic/Latino"
            elif v == "declined":
                return "Declined"
            elif v in ["unspecified", "unknown", "na", "nan"]:
                return "Na"
            return "Na"  # Fallback for any unexpected entries
            
        df["ethnicity"] = df["ethnicity"].apply(map_ethnicity)
        
    return df


# generates a table ("Table 1") that introduces our cohort by providing statistics for various demographic features
# categorical variables: count (%). continuous variables: median (IQR).
def generate_table_one(clinical_sheet, master_clinical, master_main):
    mc = master_clinical.copy()
    print("MC COLUMNS:\n", mc.columns)
    print("para_pre_term" in mc.columns)
    mm = master_main.copy()
    cs = clinical_sheet.copy()

    # drop duplicate columns from mm and cs that already exist in mc 
    # (except 'id') to prevent any column collision suffixes from happening
    mm = mm[[c for c in mm.columns if c == "id" or c not in mc.columns]]
    cs = cs[[c for c in cs.columns if c == "id" or c not in mc.columns and c not in mm.columns]]

    df = mc.merge(mm, on="id", how="left")
    df = df.merge(cs, on="id", how="left")

    df = coalesce_merged_columns(df)
    df = standardize_race_ethnicity(df)

    table1_vars = [
        "para_pre_term",
        "hosp_insurance_grping",
        "adi_national_rank",
        "risk factor y/n",
        "race",
        "ethnicity",
        "maternal_age",
        "prepregnancy_bmi_self_or_record",
        "smoking_encoded",
        "parity",
        "gravida",
        "diabetes",
        "chtn"
    ]
    



    print("DataFrame columns after merge in generate_table_one:")
    print(df.columns.tolist())
    for var in table1_vars:
        if var not in df.columns:
            print(f"MISSING IN DF: {var}")
        else:
            print(f"FOUND IN DF: {var} (Non-null count: {df[var].notna().sum()}/{len(df)})")
    print(df.columns)




    return build_summary(df, table1_vars)


# generates a table that introduces our cohort by providing statistics for various pregnancy outcome features
# categorical variables: count (%). continuous variables: median (IQR).
def generate_outcomes_table(clinical_sheet, master_clinical, placental_sheet):
    df = clinical_sheet.merge(master_clinical, on="id", how="left")
    df = df.merge(placental_sheet, on="id", how="left")

    df = coalesce_merged_columns(df)

    # pull "O_GDM" column from master_clinical
    gdm_cols = [c for c in ["id", "o_gdm", "O_GDM"] if c in master_clinical.columns]
    if len(gdm_cols) >= 1:
        # make sure 'id' is in gdm_cols to merge on
        if "id" not in gdm_cols:
            gdm_cols.insert(0, "id")
        
        # drop it from df first if it's already there to avoid duplicate collision suffixes
        clean_gdm = master_clinical[gdm_cols].drop_duplicates(subset=["id"])
        if "o_gdm" in df.columns:
            df = df.drop(columns=["o_gdm"])
        if "O_GDM" in df.columns:
            df = df.drop(columns=["O_GDM"])
            
        df = df.merge(clean_gdm, on="id", how="left")

    # force rename or map gestational outcomes if they exist under alternative names
    print("DF COLUMNS: ", df.columns.tolist())
    col_mappings = {}
    for col in df.columns:
        col_lower = str(col).strip().lower()
        if "gdm" in col_lower or "gestational_diabetes" in col_lower or col_lower == "o_gdm":
            col_mappings[col] = "gest_diabetes"
        elif "hpt" in col_lower or "hypertension" in col_lower or "ghtn" in col_lower or col_lower == "o_ghtn":
            col_mappings[col] = "gest_hypertension"
            
    df = df.rename(columns=col_mappings)

    # parse gestational hypertension and preeclampsia from "group" and "subgroup" columns
    group_cols = [c for c in ["group", "subgroup"] if c in df.columns]
    if group_cols:
        # combine the text in the "group" and "subgroup" columns for easy parsing
        combined_text = df[group_cols].apply(
            lambda row: " ".join([str(val) for val in row if pd.notna(val) and val != ""]),
            axis=1,
        )
        # gestational hypertension: "hdp" or anything containing "ghtn" (case-insensitive)
        df["gest hypertension"] = combined_text.str.contains(r"hdp|ghtn|gestational.*hypertension", case=False, regex=True, na=False)

        # preeclampsia: anything containing "pe" (case-insensitive)
        df["preeclampsia"] = combined_text.str.contains(r"pe\b|preeclampsia", case=False, regex=True, na=False)
    else:
        # fallback if group/subgroup columns are missing
        if "preeclampsia" not in df.columns:
            df["preeclampsia"] = False
        if "gest_hypertension" not in df.columns:
            df["gest_hypertension"] = False

    # ensure outcomes columns are explicitly present in df before summary
    if "gest_hypertension" not in df.columns and "gest hypertension" in df.columns:
        df["gest_hypertension"] = df["gest hypertension"]

    # Robustly convert o_gdm to numeric, treating anything non-1 as 0 or missing
    if "o_gdm" in df.columns:
        df["o_gdm"] = pd.to_numeric(df["o_gdm"], errors="coerce")
        # Map explicitly: 1 becomes "Yes", 0 becomes "No", and NaNs remain NaN (or missing)
        df["o_gdm"] = df["o_gdm"].map({1.0: "Yes", 0.0: "No", 1: "Yes", 0: "No"})
    else:
        # search for any alternative gdm column if o_gdm wasn't merged properly
        gdm_match = [c for c in df.columns if "gdm" in c.lower()]
        if gdm_match:
            df["o_gdm"] = df[gdm_match[0]]
        else:
            df["o_gdm"] = np.nan

    if "o_gdm" in df.columns:
        print("--- O_GDM UNIQUE VALUES ---")
        print(df["o_gdm"].value_counts(dropna=False))

    # convert outcome columns to clean yes/no strings
    for col in ["gest_hypertension", "preeclampsia", "o_gdm"]:
        if col in df.columns:
            df[col] = df[col].map({True: "Yes", False: "No", 1: "Yes", 0: "No", 1.0: "Yes", 0.0: "No", "1": "Yes", "0": "No"})

    outcomes_vars = [
        "route_of_delivery_1-vag,_2-cs",
        "gest_hypertension",
        "o_gdm",
        "preeclampsia",
        "spontaneous_preterm_birth",
        "gest_age_del",
        "birthweight",
        "apgar_1",
        "apgar_5",
        "nicu_days",
        "infant_sex"
    ]

    return build_summary(df, outcomes_vars)


# helper function to calculate counts (%) for categorical variables and median (IQR) for continuous variables
def build_summary(df, variables):
    summary_rows = []

    for var in variables:
        if var in df.columns:
            # map binary 0/1 indicator columns to yes/no
            if set(df[var].dropna().unique()).issubset({0, 1, 0.0, 1.0, "0", "1"}):
                df[var] = df[var].map({1: "Yes", 0: "No", 1.0: "Yes", 0.0: "No", "1": "Yes", "0": "No"})

            # format display name
            formatted_var_name = str(var).replace("_", " ").title()
            if formatted_var_name.lower() == "pregravid bmi":
                formatted_var_name = "Pregravid BMI"
            
            if df[var].dtype == bool:
                df[var] = df[var].map({True: "Yes", False: "No"})

            is_categorical = (
                df[var].dtype == object
                or df[var].dtype == bool
                or df[var].nunique() < 10
            )

            if is_categorical:
                summary_rows.append(
                    {"Variable / Category": formatted_var_name, "Statistic": ""}
                )
                counts = df[var].value_counts(dropna=False).sort_index()
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



    print("--- ACTUAL MASTER_CLINICAL COLUMNS ---")
    print(master_clinical.columns.tolist())


    missing_report, missing_ids = summarize_missing_info(clinical_sheet)
    race_table, total_patients = get_race_counts(clinical_sheet)
    cont_summary_table, cat_summary_table = calc_demographic_stats(clinical_sheet)
    









    import difflib

    # 1. Define target variables we are looking for
    targets = {
        "Table 1": [
            "para_pre_term",
            "hosp_insurance_grping",
            "adi_national_rank",
            "risk_factor_y/n",
            "race",
            "ethnicity",
            "maternal_age",
            "prepregnancy_bmi_self_or_record",
            "smoking_encoded",
            "parity",
            "gravida",
            "diabetes",
            "chtn",
        ],
        "Outcomes Table": [
            "route_of_delivery_1-vag,_2-cs",
            "gest_hypertension",
            "o_gdm",
            "preeclampsia",
            "spontaneous_preterm_birth",
            "gest_age_del",
            "birthweight",
            "apgar_1",
            "apgar_5",
            "nicu_days",
            "infant_sex",
            "group",
            "subgroup",
        ],
    }

    # 2. Gather available sheets
    sheets = {
        "master_clinical": master_clinical,
        "master_main": master_main,
        "clinical_sheet": clinical_sheet,
        "placental_sheet": placental_sheet,
    }


    def find_column_matches(sheet_dict, target_dict):
        print("=" * 70)
        print("COLUMN DIAGNOSTIC AUDIT")
        print("=" * 70)

        for table_name, target_vars in target_dict.items():
            print(f"\n--- Searching for {table_name} Variables ---")

            for target in target_vars:
                found_locations = []
                similar_suggestions = []

                for sheet_name, df in sheet_dict.items():
                    if df is None or df.empty:
                        continue

                    # Standardize column headers temporarily for matching
                    clean_cols = [
                        str(c).strip().lower().replace(" ", "_") for c in df.columns
                    ]
                    col_map = dict(zip(clean_cols, df.columns))

                    # Direct match check
                    clean_target = target.strip().lower().replace(" ", "_")
                    if clean_target in col_map:
                        found_locations.append(
                            f"'{col_map[clean_target]}' in [{sheet_name}]"
                        )

                    # Fuzzy match search for alternative candidates
                    matches = difflib.get_close_matches(
                        clean_target, clean_cols, n=3, cutoff=0.4
                    )
                    for m in matches:
                        actual_col = col_map[m]
                        if actual_col not in similar_suggestions:
                            similar_suggestions.append(
                                f"'{actual_col}' in [{sheet_name}]"
                            )

                if found_locations:
                    print(f"✅ FOUND '{target}':")
                    for loc in found_locations:
                        print(f"    - {loc}")
                else:
                    print(f"❌ MISSING '{target}'")
                    if similar_suggestions:
                        print(f"   💡 Potential alternatives found:")
                        for sug in similar_suggestions:
                            print(f"       -> {sug}")
                    else:
                        print(
                            f"       -> No close column name matches in any input sheet."
                        )


    # Run diagnostic
    find_column_matches(sheets, targets)











    table_1 = generate_table_one(clinical_sheet, master_clinical, master_main)
    outcomes_table = generate_outcomes_table(clinical_sheet, master_clinical, placental_sheet)
    export_formatted_tables_to_file(table_1, outcomes_table)

    print_log(missing_report, missing_ids, race_table, total_patients, cont_summary_table, cat_summary_table)


if __name__ == "__main__":
    main()