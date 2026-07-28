import io
import pandas as pd
import numpy as np


# loads both fitbit data sheets and returns both sheets
def load_sheets():
    sheet1 = pd.read_csv("00_raw_data/DP3_playset.csv", index_col = 0)
    sheet2 = pd.read_csv("00_raw_data/DP3-FitbitFullReport_DATA_LABELS_2025-02-18_1356.csv")
    return sheet1, sheet2


# standardizes all values to lowercase (except the values in the "id" column)
# replaces spaces with underscores
# replaces hyphens with underscores for column names of Fitbit metrics
def standardize_sheet(df):
    if df is not None and not df.empty:
        if df.index.max() > 0:
            df = df.iloc[1:].copy()

        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

        # replace hyphens with underscores only for column names of Fitbit metrics
        target_prefixes = ("activities", "sleep", "heart_rate", "heart rate")
        df.columns = [
            (
                col.replace("_-_", "_")
                if col.startswith(target_prefixes)
                else col
            )
            for col in df.columns
        ]

        ids = ["record id", "record_id", "record.id"]

        for col in df.columns:
            if col not in ids:
                df[col] = df[col].map(lambda x: str(x).strip().lower() if pd.notnull(x) else np.nan)

    return df


# rename columns of both sheets so that column names are consistent across sheets
def rename_columns(sheet1, sheet2):
    sheet1 = sheet1.rename(columns = {
        "record.id": "id",
        "event.name": "event_name",
        "gestational.age.by.reported.lmp": "gestational_age_by_reported_lmp",
        "activities...summary...steps": "activities_summary_steps",
        "activities...summary...totaldistances": "activities_summary_totaldistances",
        "activities...summary...veryactiveminutes": "activities_summary_veryactiveminutes",
        "sleep...summary...total.minutes.asleep": "sleep_summary_total_minutes_asleep"
    })

    sheet2 = sheet2.rename(columns = {
        "record_id": "id"
    })

    return sheet1, sheet2


# merges the two input sheets on ID and date
# ensures that no columns are duplicated and all column names are consistent
def merge_sheets(sheet1, sheet2):
    sheet1, sheet2 = rename_columns(sheet1, sheet2)

    # merge both sheets
    merged = pd.merge(sheet1, sheet2, on = ["id", "date"], how = "inner")

    # merge all columns ending in _x and _y
    # identify all base column names that ended up with _x and _y
    x_cols = [c for c in merged.columns if c.endswith("_x")]

    for x_col in x_cols:
        base_name = x_col[:-2]  # remove '_x'
        y_col = f"{base_name}_y"

        if y_col in merged.columns:
            # fill missing values in _x using _y
            merged[base_name] = merged[x_col].fillna(merged[y_col])
            # drop the original suffix columns
            merged.drop(columns=[x_col, y_col], inplace=True)

    return merged


# deletes certain Fitbit metric columns from the sheet if they are deprecated, uninformative, redundant, etc.
def delete_columns(sheet):
    columns_to_delete = [
        "activities_goals_steps", "activities_goals_distance", "activities_goals_activeminutes", "activities_goals_caloriesout", 
        "activities_summary_activescore", "activities_summary_caloriesbmr", "activities_summary_marginalcalories", 
        "heart_rate_zone:_out_of_range_min", "heart_rate_zone:_fat_burn_min", "heart_rate_zone:_cardio_min", "heart_rate_zone:_peak_min",
        "heart_rate_zone:_out_of_range_max", "heart_rate_zone:_fat_burn_max", "heart_rate_zone:_cardio_max", "heart_rate_zone:_peak_max",
        "heart_rate_zone:_out_of_range_caloriesout", "heart_rate_zone:_fat_burn_caloriesout", "heart_rate_zone:_cardio_caloriesout", 
        "heart_rate_zone:_peak_caloriesout", "heart_rate_zone:_out_of_range_minutes"
    ]

    sheet = sheet.drop(columns=columns_to_delete, errors="ignore")

    return sheet


# sort columns such that the activity/sleep/heart rate/etc. columns are all at the end
def sort_columns(sheet):
    front_columns = [
        "id", "event_name", "gestational_age_by_reported_lmp", "gest_age_del", "date", "group", "group_bin", "timepoint", "repeat_instrument",
        "repeat_instance", "fitbit_activity_data_uploaded", "fitbit_heart_rate_data_uploaded", "fitbit_sleep_data_uploaded", "do_we_have_all_the_data?",
        "complete?", "was_a_complication_diagnosed_during_the_current_pregnancy?"
    ]

    other_columns = [col for col in sheet.columns if col not in front_columns]

    sheet = sheet[front_columns + other_columns]
    
    return sheet


# helper function of null_implausible_values()
# excludes the whole day if the following conditions aren't met:
# >=10 hours/day wear time (or HR-wear minutes / 1440 >= 0.3-0.4)
def wear_time_day_validity(df):
    df_filtered = df.copy()

    wear_time_col=None
    hr_wear_minutes_col=None

    # Automatically identify potential column names if not explicitly provided
    if wear_time_col is None:
        matches = [c for c in df.columns if "wear" in c.lower() and "time" in c.lower() and "min" not in c.lower() and "hr" not in c.lower()]
        wear_time_col = matches[0] if matches else None
        
    if hr_wear_minutes_col is None:
        matches_hr = [c for c in df.columns if "hr" in c.lower() and "wear" in c.lower()]
        hr_wear_minutes_col = matches_hr[0] if matches_hr else None

    print(wear_time_col)
    print(hr_wear_minutes_col)
    
    # Check primary wear time column (assuming hours or minutes; standard is typically hours or minutes >= 10 hours)
    if wear_time_col and wear_time_col in df_filtered.columns:
        wt = pd.to_numeric(df_filtered[wear_time_col], errors="coerce")
        # If wear time is recorded in hours (< 24), threshold is >= 10. If in minutes, threshold is >= 600.
        # Here we assume standard hours based on ">=10 h/day" rule, but handle cleanly:
        valid_mask = (wt >= 10) | (wt >= 600) # handles both hour and minute formats safely
        
        # If AoU HR-wear minutes proxy is available, apply alternative proxy condition (HR-wear minutes / 1440 >= 0.3)
        if hr_wear_minutes_col and hr_wear_minutes_col in df_filtered.columns:
            hr_min = pd.to_numeric(df_filtered[hr_wear_minutes_col], errors="coerce")
            proxy_mask = (hr_min / 1440.0) >= 0.3
            valid_mask = valid_mask | proxy_mask
            
        # Exclude the whole day (drop row) if the condition is not met
        df_filtered = df_filtered[valid_mask].copy()
    elif hr_wear_minutes_col and hr_wear_minutes_col in df_filtered.columns:
        hr_min = pd.to_numeric(df_filtered[hr_wear_minutes_col], errors="coerce")
        valid_mask = (hr_min / 1440.0) >= 0.3
        df_filtered = df_filtered[valid_mask].copy()

    return df_filtered


# helper function of null_implausible_values()
# excludes the whole day if steps < 100 AND sedentary ~= 1440 AND resting HR missing
def non_wear_signature(df):
    df_filtered = df.copy()

    steps_col = "activities_summary_steps"
    sedentary_col = "activities_summary_sedentaryminutes"
    resting_hr_col = "heart_rate_resting_heart_rate"
    
    # build conditions:
    # steps < 100 AND sedentary close to 1440 (e.g., >= 1400 or close to full day) AND resting HR missing (null)
    is_non_wear = pd.Series(False, index=df_filtered.index)
    
    if steps_col and steps_col in df_filtered.columns:
        steps = pd.to_numeric(df_filtered[steps_col], errors="coerce")
        cond_steps = steps < 100
    else:
        cond_steps = False
        
    if sedentary_col and sedentary_col in df_filtered.columns:
        sed = pd.to_numeric(df_filtered[sedentary_col], errors="coerce")
        # sedentary minutes roughly covering the whole day (~1440 minutes)
        cond_sed = sed >= 1400
    else:
        cond_sed = False
        
    if resting_hr_col and resting_hr_col in df_filtered.columns:
        cond_hr_missing = df_filtered[resting_hr_col].isnull()
    else:
        cond_hr_missing = True
        
    # combine conditions: steps < 100 AND sedentary ~1440 AND resting HR missing
    is_non_wear = cond_steps & cond_sed & cond_hr_missing
    
    # exclude the day (drop the row) if it matches the non-wear signature
    df_filtered = df_filtered[~is_non_wear].copy()
    
    return df_filtered


# helper function of null_implausible_values()
# excludes participant from sleep analyses if >30% of a participant's monitored days have <240 min asleep
def filter_person_level_sleep(df):
    id_col = "id"
    sleep_duration_col = "sleep_summary_total_minutes_asleep"
    sleep_cols = [c for c in df.columns if "sleep" in c.lower()]

    df_filtered = df.copy()
    invalid_ids = []
    
    for patient_id, group in df_filtered.groupby(id_col):
        sleep_vals = pd.to_numeric(group[sleep_duration_col], errors="coerce").dropna()
        
        if len(sleep_vals) == 0:
            continue
            
        short_sleep_days = (sleep_vals < 240).sum()
        total_monitored_days = len(sleep_vals)
        fraction_short = short_sleep_days / total_monitored_days
        
        if fraction_short > 0.30:
            invalid_ids.append(patient_id)
            
    # null out only the sleep columns for these participants
    mask = df_filtered[id_col].isin(invalid_ids)
    df_filtered.loc[mask, sleep_cols] = None
    
    return df_filtered


# null out and/or exclude all implausible values in the dataset
# CUTOFFS SUBJECT TO CHANGE. keep in mind that these may not be the final cutoff values; they may be adjusted in the future
def null_implausible_values(df):
    # day and person validity
    df_cleaned = df # DELETE
    # df_cleaned = wear_time_day_validity(df) # FUNCTION NOT PROOFREAD YET
    df_cleaned = non_wear_signature(df_cleaned)
    df_cleaned = filter_person_level_sleep(df_cleaned)







    return df_cleaned
    








# for patients with no recording of 7+ days in a row for a certain feature, null out their data for that feature only
# inputs:
# sheet: the fitbit data sheet, which contains patient information and all collected metrics
# outputs:
# sheet_clean: the input sheet cleaned, i.e. after the appropriate data has been nulled
# exclusion_counts: a table containing the number of patients whose data has been nulled, per metric
def drop_patients(sheet):
    metric_cols = [col for col in sheet.columns if col.startswith(("activities", "sleep", "heart_rate"))]

    sheet_clean = sheet.copy()
    
    exclusion_counts = {metric: 0 for metric in metric_cols}
    
    sheet_clean["date"] = pd.to_datetime(sheet_clean["date"], errors = "coerce")
    sheet_clean = sheet_clean.dropna(subset = ["date", "id"])

    # dictionary: {pt_id: [list of metrics to nullify for this patient]}
    nullification_map = {}

    for pt_id, pt_df in sheet_clean.groupby("id"):
        if pt_df["date"].isna().all():
            continue

        pt_df = pt_df.sort_values(by = "date")
        pt_indexed = pt_df.set_index("date")
        
        for metric in metric_cols:
            if metric not in pt_indexed.columns:
                continue

            # create a daily calendar for this patient
            valid_metric_series = pt_indexed[metric].dropna()
            if valid_metric_series.empty:
                continue
            metric_start = valid_metric_series.index.min()
            metric_end = valid_metric_series.index.max()
            full_range = pd.date_range(start = metric_start, end = metric_end, freq = "D")
            pt_daily = pt_indexed[metric].reindex(full_range)
            
            # count consecutive missing days
            is_null = pt_daily.isnull()
            # the first cumsum() creates a unique group id that only increments when we hit valid data
            # the groupby() groups consecutive missing days together under the same id number
            # the last cumsum() calculates the lengths of missing data streaks, for this patient and this metric
            consecutive_missing = is_null.groupby((~is_null).cumsum()).cumsum()
            max_gap = consecutive_missing.max()

            # if the patient passed the 7-day limit, null out this metric for this patient
            if max_gap >= 7:
                if pt_id not in nullification_map:
                    nullification_map[pt_id] = []
                nullification_map[pt_id].append(metric)

                exclusion_counts[metric] += 1

    # nullify all appropriate data
    if nullification_map:
        for pt_id, metrics_to_null in nullification_map.items():
            sheet_clean.loc[sheet_clean["id"] == pt_id, metrics_to_null] = np.nan

    return sheet_clean, exclusion_counts


# print to a log file explaining why you dropped each patient from the dataset
# i.e. creates a table containing each patient that was dropped and the maximum consecutive number of days their data was missing
def print_log(exclusion_counts):
    log_path = "04_results_and_figures/data_analysis/fitbit/dropped_patients_fitbit_log.txt"
    with open(log_path, "w") as f:
        f.write("This file contains a table consisting of the number of patients whose data was nulled out, organized\n")
        f.write("per metric. Patients had their data nulled out for a certain metric if they had more than 7 consecutive\n")
        f.write("days of data missing for that metric.\n\n\n")

        f.write(f"{'Metric':<45} | {'Patients Excluded'}\n")
        for metric, count in exclusion_counts.items():
            f.write(f"{str(metric).strip():<45} | {count}\n")


def main():
    sheet1, sheet2 = load_sheets()
    sheet1 = standardize_sheet(sheet1)
    sheet2 = standardize_sheet(sheet2)
    merged = merge_sheets(sheet1, sheet2)
    merged = delete_columns(merged)
    merged = sort_columns(merged)
    merged = null_implausible_values(merged)

    merged_clean, exclusion_counts = drop_patients(merged)

    print_log(exclusion_counts)

    # write sheet to an output file
    merged_clean.to_csv("01_data_cleaning/processed_data/processed_fitbit_data.csv", index = False)


if __name__ == "__main__":
    main()