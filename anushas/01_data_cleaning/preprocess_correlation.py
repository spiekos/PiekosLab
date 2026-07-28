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
        group_sheet = local_sheet[local_sheet["group"] == label].drop(columns = ["group"]).copy()
        
        metric_cols = [c for c in group_sheet.columns if c.startswith(("activities", "sleep", "heart_rate"))]
        rename_map = {col: f"{label}_{col}" for col in metric_cols}
        group_sheet = group_sheet.rename(columns=rename_map)

        outputs.append(group_sheet)

    return outputs


# applies coverage requirements per patient x bin x metric:
# calculates total possible days from enrollment to delivery (or uses bin span).
# checks if valid/plausible days meet >= min_days_absolute AND >= pct_cutoff of possible days.
# computes the mean of each metric over valid days.
# carries forward the valid day count as a new column.
# nulls out data for metrics that fail the threshold.
def apply_bucketed_coverage(df, prefix):
    id_col = "id"
    min_days_absolute = 5
    pct_cutoff = 0.50

    df_filtered = df.copy()

    # calculate whether each day in the dataframe is valid
    # a day is valid if it occurs after LMP gestational age and before delivery gestational age
    lmp_days = pd.to_numeric(df_filtered["gestational_age_by_reported_lmp"], errors="coerce") * 7
    delivery_days = pd.to_numeric(df_filtered["gest_age_del"], errors="coerce") * 7
    timepoint = pd.to_numeric(df_filtered["timepoint"], errors="coerce")
    df_filtered["is_valid_day"] = (timepoint >= lmp_days) & (timepoint <= delivery_days)
    
    results = []

    print("df_filtered.columns:",df_filtered.columns)

    # identify metric columns belonging to this specific prefix
    metric_cols = [
        c for c in df_filtered.columns 
        if any(c.startswith(f"{prefix}_{m}") for m in ["activities", "heart_rate", "sleep"])
    ]

    print("metric_cols:", metric_cols)

    for patient_id, group in df_filtered.groupby(id_col):
        patient_row = {id_col: patient_id}

        total_possible_days = len(group)
        group_valid_mask = group["is_valid_day"]
        valid_days_count = group_valid_mask.sum()
        
        # check criteria: meets min_days_absolute and pct_cutoff criteria
        meets_criteria = (valid_days_count >= min_days_absolute) and (valid_days_count >= (total_possible_days * pct_cutoff))

        patient_row[f"{prefix}_valid_day_count"] = valid_days_count

        for metric in metric_cols:
            if meets_criteria:
                valid_values = pd.to_numeric(group.loc[group_valid_mask, metric], errors="coerce")
                patient_row[metric] = valid_values.mean() if len(valid_values) > 0 else None
            else:
                patient_row[metric] = None

        results.append(patient_row)

    return pd.DataFrame(results)


# merges pre-calculated, coverage-filtered trimester data side-by-side by patient
# merges this with delivery and placental data
# filters to only include patients who have placental reports
# returns a table containing fitbit data (averaged by metric by trimester) and delivery + placental metrics by patient
# table contains all timeframes; column names reflect which timeframe the data was collected in
def prepare_correlation_data(sheet_bucketed, timeframe_names, clinical_raw, placental_raw):
    trimester_dfs = []

    # collapse each trimester bucket into metric averages by patient
    for df, _ in zip(sheet_bucketed, timeframe_names):
        if df.empty:
            continue

        trimester_dfs.append(df.copy())

    if not trimester_dfs:
        print("Warning: No trimester data found to merge.")
        return pd.DataFrame()
    
    # merge all the trimester dataframes on ID
    fitbit_pivoted = trimester_dfs[0]
    for next_df in trimester_dfs[1:]:
        fitbit_pivoted = pd.merge(fitbit_pivoted, next_df, on="id", how="outer")

    clinical_targets = [
        "maternal_age", "height_(cm)", "weight_(kg)", "delivery_bmi", "prepregnancy_weight_self_or_record", "prepregnancy_bmi_self_or_record", 
        "gravida", "parity", "diabetes", "chtn", "route_of_delivery_1-vag,_2-cs", "gest_age_del", "birthweight", "apgar_1", "apgar_5", "nicu_days"
    ]
    existing_clinical = [col for col in clinical_targets if col in clinical_raw.columns]
    clinical_clean = clinical_raw.groupby("id")[existing_clinical].first().reset_index()

    placental_targets = [
        "placental_infarction", "distal_villous_hypoplasia_focal/diffuse", "accelerated_villous_maturation", "increased_syncytial_knots", 
        "decidual_arteriopathy_membrane_role/basal_plate/both", "segmental_avascular_villi_small/intermediate/large", "delayed_villous_maturation", 
        "maternal_inflammatory_response_stage/grade", "villitis_of_unknown_etiology,_high/low_grade,_focal/diffuse", "increased_perivillous_fibrin_deposition", 
        "chorangiosis"
    ]
    existing_placental = [col for col in placental_targets if col in placental_raw.columns]
    placental_clean = placental_raw.groupby("id")[existing_placental].first().reset_index()

    master_corr_df = pd.merge(fitbit_pivoted, clinical_clean, on="id", how="inner")
    master_corr_df = pd.merge(master_corr_df, placental_clean, on="id", how="inner")

    # filter out rows missing all placental metrics
    final_placental_cols = [col for col in existing_placental if col in master_corr_df.columns]
    if final_placental_cols:
        master_corr_df = master_corr_df.dropna(subset=final_placental_cols, how="all")
    
    return master_corr_df


def main():
    fitbit_sheet, placental_sheet, clinical_sheet = load_sheets()

    fitbit_sheet = filter_sheet(fitbit_sheet)

    feature_cols = [col for col in fitbit_sheet.columns if col.startswith(("activities", "sleep", "heart_rate"))]
    numeric_cols = feature_cols + ["gestational_age_by_reported_lmp", "gest_age_del"]

    # forcefully convert all feature columns + gest age columns into numeric types
    fitbit_sheet[numeric_cols] = fitbit_sheet[numeric_cols].apply(pd.to_numeric, errors = "coerce")

    sheets_bucketed = bucket_data(fitbit_sheet)

    for sheet in sheets_bucketed:
        print(sheet.head())
    for sheet in sheets_bucketed:
        print(sheet.columns)

    labels = ["first", "early_second", "late_second_early_third", "late_third"]
    sheets_bucketed = [apply_bucketed_coverage(df, label) for df, label in zip(sheets_bucketed, labels)]

    for sheet in sheets_bucketed:
        print(sheet.head())
    for sheet in sheets_bucketed:
        print(sheet.columns)

    timeframe_names = ["First Trimester", "Early Second Trimester", "Late Second and Early Third Trimester", "Late Third Trimester"]
    correlation_ready_df = prepare_correlation_data(sheets_bucketed, timeframe_names, clinical_sheet, placental_sheet)
    if not correlation_ready_df.empty:
        output_csv_path = "01_data_cleaning/processed_data/master_fitbit_clinical_correlation_data.csv"
        correlation_ready_df.to_csv(output_csv_path, index = False)


if __name__ == "__main__":
    main()