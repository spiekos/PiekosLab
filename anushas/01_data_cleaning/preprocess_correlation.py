import pandas as pd
import numpy as np


# load and return the fitbit dataset, the placental dataset, and the clinical dataset
def load_sheets():
    sheet1 = pd.read_csv("01_data_cleaning/processed_data/processed_fitbit_data.csv", low_memory = False)
    sheet2 = pd.read_csv("01_data_cleaning/processed_data/processed_placental_data.csv")
    sheet3 = pd.read_csv("01_data_cleaning/processed_data/processed_clinical_data.csv")
    return sheet1, sheet2, sheet3


# filters the fitbit data sheet to only include "fitbit data" events and only include events during pregnancy
def filter_sheet(sheet):
    # filter out all events from the sheet except the "fitbit data" ones
    sheet_filtered = sheet[sheet["event_name"] == "fitbit data"].copy()

    # only include events during pregnancy
    # if gestational age at delivery is not in the dataset, assume that delivery occurred at 40 weeks
    sheet_filtered["current_weeks"] = sheet_filtered["timepoint"] / 7
    sheet_filtered = sheet_filtered[(sheet_filtered["current_weeks"] <= sheet_filtered["gest_age_del"]) | 
                                    (sheet_filtered["gest_age_del"].isna() & sheet_filtered["current_weeks"] <= 40)]
    return sheet_filtered


# splits the filtered dataset into four smaller datasets, based on which trimester of the pregnancy each datapoint is in
# the four datasets are: 1st trimester, early 2nd trimester, late 2nd/early 3rd trimester, late 3rd trimester
# returns a list containing these four datasets
# note that the input fitbit dataset has already been filtered and only includes events during pregnancy
def bucket_data(sheet):
    local_sheet = sheet.copy()

    local_sheet["current_weeks"] = pd.to_numeric(local_sheet["current_weeks"], errors="coerce")

    bins = [float("-inf"), 14, 22, 32, float("inf")]
    labels = ["first", "early_second", "late_second_early_third", "late_third"]

    local_sheet["group"] = pd.cut(local_sheet["current_weeks"], bins = bins, labels = labels, right = False)

    outputs = []
    
    for label in labels:
        group_sheet = local_sheet[local_sheet["group"] == label].drop(columns = ["group"])
        outputs.append(group_sheet)

    return outputs


# collapses multi-row trimester data into single-row patient averages for each metric
# merges this with delivery and placental data
# filters to only include patients who have placental reports
# returns a table containing fitbit data (averaged by metric by trimester) and delivery + placental metrics by patient
# table contains all timeframes; column names reflect which timeframe the data was collected in
def prepare_correlation_data(sheet_bucketed, feature_cols, timeframe_names, clinical_raw, placental_raw):
    trimester_dfs = []

    # collapse each trimester bucket into metric averages by patient
    for df, label in zip(sheet_bucketed, timeframe_names):
        if df.empty:
            continue

        # take the mean of each metric by patient in this timeframe
        # taking the mean manually ensures that we are only counting possible days for which data could have been collected
        plausible_mask = df[feature_cols].notna() & (df[feature_cols] >= 0) 

        df_plausible = df.copy()
        df_plausible[feature_cols] = df_plausible[feature_cols].where(plausible_mask)

        pt_sums = df_plausible.groupby("record_id")[feature_cols].sum()
        pt_counts = df_plausible.groupby("record_id")[feature_cols].count()

        # using replace(0, np.nan) prevents division by zero errors for patients with no data
        pt_trimester_avg = pt_sums / pt_counts.replace(0, np.nan)

        # rename the columns
        pt_trimester_avg.columns = [f"{label}_{col}" for col in pt_trimester_avg.columns]
        pt_trimester_avg = pt_trimester_avg.copy() # de-fragment the DataFrame
        trimester_dfs.append(pt_trimester_avg)

    if not trimester_dfs:
        print("Warning: No trimester data found to collapse.")
        return pd.DataFrame()
    
    # join all trimesters side-by-side
    fitbit_pivoted = pd.concat(trimester_dfs, axis = 1).reset_index()

    # isolate target clinical variables
    clinical_targets = [
        "maternal_age", "height_(cm)", "weight_(kg)", "delivery_bmi", "prepregnancy_weight_self_or_record", "prepregnancy_bmi_self_or_record", 
        "gravida", "parity", "diabetes", "chtn", "route_of_delivery_1-vag,_2-cs", "gest_age_del", "birthweight", "apgar_1", "apgar_5", "nicu_days"
    ]
    # keep only the columns that exist in the spreadsheet
    existing_clinical = [col for col in clinical_targets if col in clinical_raw.columns]
    # aggregate all clinical data for each patient into a dataframe
    clinical_clean = clinical_raw.rename(columns = {"id": "record_id"})
    clinical_clean = clinical_clean.groupby("record_id")[existing_clinical].first().reset_index()

    # isolate target placental variables
    placental_targets = [
        "placental_infarction", "distal_villous_hypoplasia_focal/diffuse", "accelerated_villous_maturation", "increased_syncytial_knots", 
        "decidual_arteriopathy_membrane_role/basal_plate/both", "segmental_avascular_villi_small/intermediate/large", "delayed_villous_maturation", 
        "maternal_inflammatory_response_stage/grade", "villitis_of_unknown_etiology,_high/low_grade,_focal/diffuse", "increased_perivillous_fibrin_deposition", 
        "chorangiosis"
    ]
    # keep only the columns that exist in the spreadsheet
    existing_placental = [col for col in placental_targets if col in placental_raw.columns]
    # aggregate all placental data for each patient into a dataframe
    placental_clean = placental_raw.rename(columns = {"id": "record_id"})
    placental_clean = placental_clean.groupby("record_id")[existing_placental].first().reset_index()

    # merge fitbit, clinical, and placental data
    master_corr_df = pd.merge(fitbit_pivoted, clinical_clean, on = "record_id", how = "inner")
    master_corr_df = pd.merge(master_corr_df, placental_clean, on = "record_id", how = "inner")

    # determine which placental columns survived the merge process
    final_placental_cols = [col for col in existing_placental if col in master_corr_df.columns]
    
    if final_placental_cols:
        # drop rows for which all placental metrics are missing
        master_corr_df = master_corr_df.dropna(subset = final_placental_cols, how = "all")
    
    return master_corr_df


def main():
    fitbit_sheet, placental_sheet, clinical_sheet = load_sheets()

    fitbit_sheet = filter_sheet(fitbit_sheet)

    feature_cols = [col for col in fitbit_sheet.columns if col.startswith(("activities", "sleep", "heart_rate"))]
    numeric_cols = feature_cols + ["gestational_age_by_reported_lmp", "gest_age_del"]

    # forcefully convert all feature columns + gest age columns into numeric types
    fitbit_sheet[numeric_cols] = fitbit_sheet[numeric_cols].apply(pd.to_numeric, errors = "coerce")

    sheet_bucketed = bucket_data(fitbit_sheet)

    timeframe_names = ["First Trimester", "Early Second Trimester", "Late Second and Early Third Trimester", "Late Third Trimester"]

    correlation_ready_df = prepare_correlation_data(sheet_bucketed, feature_cols, timeframe_names, clinical_sheet, placental_sheet)
    if not correlation_ready_df.empty:
        output_csv_path = "01_data_cleaning/processed_data/master_fitbit_clinical_correlation_data.csv"
        correlation_ready_df.to_csv(output_csv_path, index = False)


if __name__ == "__main__":
    main()