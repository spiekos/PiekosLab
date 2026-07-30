import io
import pandas as pd
import numpy as np


# loads both fitbit data sheets and returns both sheets
def load_sheets():
    sheet1 = pd.read_csv("00_raw_data/DP3_playset.csv", index_col = 0)
    sheet2 = pd.read_csv("00_raw_data/DP3-FitbitFullReport_DATA_LABELS_2025-02-18_1356.csv")
    sheet3 = pd.read_csv("00_raw_data/implausible_value_bounds.csv")
    return sheet1, sheet2, sheet3


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
# only includes "Fitbit Data" events
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

    merged = merged[merged["event_name"] == "fitbit data"]

    return merged


# deletes certain Fitbit metric columns from the sheet if they are deprecated, uninformative, redundant, etc.
# note that we are supposed to delete the column "activities_summary_caloriesbmr", but we are leaving it in for now (as it's used in the function
# check_consistency() later on). this column will be deleted after check_consistency() runs
def delete_columns(sheet):
    columns_to_delete = [
        "activities_goals_steps", "activities_goals_distance", "activities_goals_activeminutes", "activities_goals_caloriesout", 
        "activities_summary_activescore", "activities_summary_marginalcalories", "heart_rate_zone:_out_of_range_min", 
        "heart_rate_zone:_fat_burn_min", "heart_rate_zone:_cardio_min", "heart_rate_zone:_peak_min", "heart_rate_zone:_out_of_range_max", 
        "heart_rate_zone:_fat_burn_max", "heart_rate_zone:_cardio_max", "heart_rate_zone:_peak_max", "heart_rate_zone:_out_of_range_caloriesout", 
        "heart_rate_zone:_fat_burn_caloriesout", "heart_rate_zone:_cardio_caloriesout", "heart_rate_zone:_peak_caloriesout", 
        "heart_rate_zone:_out_of_range_minutes"
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
# excludes the day for that metric + patient (i.e. just that cell) if the following conditions aren't met:
# that metric's own stream flag = yes (i.e. 'Fitbit Activity/Heart Rate/Sleep Data Uploaded')
# that metric's field is populated (non-blank, Resting Heart Rate isn't 'No Value')
def wear_time_day_validity(df, metric_cols):
    df_copy = df.copy()

    for metric in metric_cols:
        if metric.startswith("activities"):
            flag_col = "fitbit_activity_data_uploaded"
        elif metric.startswith("heart_rate"):
            flag_col = "fitbit_heart_rate_data_uploaded"
        elif metric.startswith("sleep"):
            flag_col = "fitbit_sleep_data_uploaded"

        is_blank_or_no_value = df_copy[metric].isnull() | (df_copy[metric] == "No Value")
        is_stream_yes = df_copy[flag_col].astype(str).str.strip().str.lower() == "yes"

        exclude_day = is_blank_or_no_value | ~is_stream_yes

        df_copy.loc[exclude_day, metric] = np.nan
    
    return df_copy


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


# filter for day and person validity
# null out and/or exclude all improbable values in the dataset
# CUTOFFS IN THE BOUNDS SHEET ARE SUBJECT TO CHANGE
def null_implausible_values(df, metric_cols, bounds):
    # day and person validity
    df_cleaned = wear_time_day_validity(df, metric_cols)
    df_cleaned = non_wear_signature(df_cleaned)
    df_cleaned = filter_person_level_sleep(df_cleaned)

    # filter for improbable values
    for metric in bounds["Source column(s)"]:
        df_cleaned[metric] = pd.to_numeric(df_cleaned[metric], errors="coerce")

        metric_row = bounds[bounds["Source column(s)"] == metric]
        keep_min = metric_row["Keep min"].values[0]
        keep_max = metric_row["Keep max"].values[0]

        min_mask = df_cleaned[metric] < keep_min
        max_mask = df_cleaned[metric] > keep_max
        bounds_mask = min_mask | max_mask

    # nullify all values out of bounds for this metric
    df_cleaned.loc[bounds_mask, metric] = np.nan

    # address inclusive bounds for sleep summaries (0 values)
    sleep_cols = ["sleep_summary_total_minutes_asleep", "sleep_summary_total_time_in_bed"]
    for col in sleep_cols:
        zero_mask = df_cleaned[col] == 0
        df_cleaned.loc[zero_mask, col] = np.nan

    return df_cleaned


# performs consistency checks on the Fitbit dataset to ensure that all values are reasonable
# flags or nulls values inconsistent values
# checks performed:
# 1. Minutes budget: veryActive + fairlyActive + lightlyActive + sedentary <= 1440
# 2. Sleep containment: total minutes asleep <= total time in bed
# 3. Stage sum: deep + light + rem ~= total minutes asleep
# 4. Total EE vs basal: caloriesOut >= caloriesBMR
# 5. Distance-steps coherence: distance ~= steps * stride
# outputs:
# df: the cleaned and checked dataset, with appropriate values nulled out
# flagged_records: a list of all flagged records
def check_consistency(df):
    df_checked = df.copy()
    flagged_records = []

    id_col = "id"
    date_col = "date"

    # rule 1: minutes budget: veryActive + fairlyActive + lightlyActive + sedentary <= 1440
    minutes_cols = [
        "activities_summary_veryactiveminutes",
        "activities_summary_fairlyactiveminutes",
        "activities_summary_lightlyactiveminutes",
        "activities_summary_sedentaryminutes",
    ]
    for col in minutes_cols:
        df_checked[col] = pd.to_numeric(df_checked[col], errors="coerce")

    total_minutes = df_checked[minutes_cols].sum(axis=1, min_count=1)
    budget_mask = total_minutes > 1440

    if budget_mask.sum() > 0:
        subset = df_checked.loc[budget_mask].copy()
        for idx, row in subset.iterrows():
            flagged_records.append(
                {
                    "index": idx,
                    "id": row[id_col],
                    "date": row[date_col],
                    "Rule_Violated": "Minutes budget",
                    "Description": f"Total active+sedentary minutes ({total_minutes.loc[idx]}) > 1440",
                }
            )

    # rule 2: sleep containment: total minutes asleep <= total time in bed
    asleep_col = "sleep_summary_total_minutes_asleep"
    bed_col = "sleep_summary_total_time_in_bed"

    df_checked[asleep_col] = pd.to_numeric(df_checked[asleep_col], errors="coerce")
    df_checked[bed_col] = pd.to_numeric(df_checked[bed_col], errors="coerce")

    containment_mask = df_checked[asleep_col] > df_checked[bed_col]

    if containment_mask.sum() > 0:
        subset = df_checked.loc[containment_mask].copy()
        for idx, row in subset.iterrows():
            flagged_records.append(
                {
                    "index": idx,
                    "id": row[id_col],
                    "date": row[date_col],
                    "Rule_Violated": "Sleep containment",
                    "Description": f"Asleep ({row[asleep_col]}) > In bed ({row[bed_col]})",
                }
            )
        df_checked.loc[containment_mask, asleep_col] = np.nan
        df_checked.loc[containment_mask, bed_col] = np.nan

    # rule 3: stage sum: deep + light + rem ~= total minutes asleep
    stage_cols = [
        "sleep_summary_stages_deep",
        "sleep_summary_stages_light",
        "sleep_summary_stages_rem",
    ]

    for col in stage_cols:
        df_checked[col] = pd.to_numeric(df_checked[col], errors="coerce")

    stage_total = df_checked[stage_cols].sum(axis=1, min_count=1)
    stage_mismatch_mask = (
        ((stage_total - df_checked[asleep_col]).abs() > 30)
        & stage_total.notna()
        & df_checked[asleep_col].notna()
    )

    if stage_mismatch_mask.sum() > 0:
        subset = df_checked.loc[stage_mismatch_mask].copy()
        for idx, row in subset.iterrows():
            flagged_records.append(
                {
                    "index": idx,
                    "id": row[id_col],
                    "date": row[date_col],
                    "Rule_Violated": "Stage sum mismatch",
                    "Description": f"Stage total ({stage_total.loc[idx]}) differs from asleep ({row[asleep_col]}) by > 30 mins",
                }
            )
        for col in stage_cols:
            df_checked.loc[stage_mismatch_mask, col] = np.nan

    # rule 4: total EE vs basal: caloriesOut >= caloriesBMR
    ee_col = "activities_summary_caloriesout"
    bmr_col = "activities_summary_caloriesbmr"
    
    df_checked[ee_col] = pd.to_numeric(df_checked[ee_col], errors="coerce")
    df_checked[bmr_col] = pd.to_numeric(df_checked[bmr_col], errors="coerce")

    ee_mask = df_checked[ee_col] < df_checked[bmr_col]
    
    if ee_mask.sum() > 0:
        subset = df_checked.loc[ee_mask].copy()
        for idx, row in subset.iterrows():
            flagged_records.append(
                {
                    "index": idx,
                    "id": row[id_col],
                    "date": row[date_col],
                    "Rule_Violated": "Total EE vs basal",
                    "Description": f"CaloriesOut ({row[ee_col]}) < CaloriesBMR ({row[bmr_col]})",
                }
            )

    # rule 5: distance-steps coherence: distance ~= steps * stride
    dist_col = "activities_summary_totaldistances"
    steps_col = "activities_summary_steps"

    df_checked[dist_col] = pd.to_numeric(df_checked[dist_col], errors="coerce")
    df_checked[steps_col] = pd.to_numeric(df_checked[steps_col], errors="coerce")

    implied_stride = df_checked[dist_col] / df_checked[steps_col]
    stride_mask = (
        (df_checked[steps_col] > 0)
        & ((implied_stride < 0.0007) | (implied_stride > 0.0013))
    )

    if stride_mask.sum() > 0:
        subset = df_checked.loc[stride_mask].copy()
        for idx, row in subset.iterrows():
            flagged_records.append(
                {
                    "index": idx,
                    "id": row[id_col],
                    "date": row[date_col],
                    "Rule_Violated": "Distance-steps coherence",
                    "Description": f"Implied stride ({implied_stride.loc[idx]:.5f} km/step) out of bounds",
                }
            )


    report_df = pd.DataFrame(flagged_records)
    return df_checked, report_df


# for patients with no recording of 7+ days in a row for a certain feature, flag their data for that feature only
# inputs:
# sheet: the fitbit data sheet, which contains patient information and all collected metrics
# outputs:
# flagged_counts: a table containing the number of patients whose data has been flagged, per metric
# flag_map: a dictionary mapping patient IDs to lists of metrics that have been flagged for that patient
def flag_patients(sheet, metric_cols):
    sheet_flagged = sheet.copy()
    
    flag_counts = {metric: 0 for metric in metric_cols}
    
    sheet_flagged["date"] = pd.to_datetime(sheet_flagged["date"], errors = "coerce")
    sheet_flagged = sheet_flagged.dropna(subset = ["date", "id"])

    # dictionary: {pt_id: [list of metrics to flag for this patient]}
    flag_map = {}

    for pt_id, pt_df in sheet_flagged.groupby("id"):
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

            # if the patient passed the 7-day limit, flag this metric for this patient
            if max_gap >= 7:
                if pt_id not in flag_map:
                    flag_map[pt_id] = []
                flag_map[pt_id].append(metric)
                flag_counts[metric] += 1

    return flag_counts, flag_map


# print to a log file explaining which patients were flagged during consistency checks and why
# print to another log file explaining why you flagged each patient in the dataset
# i.e. creates a table containing the total number of flagged patients per metric
# also creates a table containing each patient that was flagged and the metrics for which the maximum consecutive missing days > 7
def print_log(flags, flag_counts, flag_map):
    log_path1 = "04_results_and_figures/data_analysis/fitbit/consistency_check_flags_fitbit_log.txt"
    log_path2 = "04_results_and_figures/data_analysis/fitbit/flagged_patients_fitbit_log.txt"
    
    with open(log_path1, "w") as f:
        f.write("This file contains a list of all patients who were flagged during the consistency checks, along with the reasons for the flags.\n\n")

        # define column widths
        w_idx = 10
        w_id = 20
        w_date = 15
        w_rule = 28
        w_desc = 50

        header = (
            f"{'index'.ljust(w_idx)}"
            f"{'id'.ljust(w_id)}"
            f"{'date'.ljust(w_date)}"
            f"{'Rule_Violated'.ljust(w_rule)}"
            f"{'Description'.ljust(w_desc)}"
        )

        separator = "-" * len(header)

        rows_formatted = []
        for _, row in flags.iterrows():
            line = (
                f"{str(row["index"]).ljust(w_idx)}"
                f"{str(row["id"]).ljust(w_id)}"
                f"{str(row["date"]).ljust(w_date)}"
                f"{str(row["Rule_Violated"]).ljust(w_rule)}"
                f"{str(row["Description"]).ljust(w_desc)}"
            )
            rows_formatted.append(line)

        table_string = "\n".join([header, separator] + rows_formatted)

        if not flags.empty:
            f.write(
                "=" * len(header)
                + "\n"
                + " FITBIT DATA CONSISTENCY VIOLATION LOG\n"
                + "=" * len(header)
                + "\n\n"
                + f"Total violations found: {len(flags)}\n\n"
                + table_string
                + "\n\n"
                + "=" * len(header)
                + "\n"
            )
        else:
            f.write("FITBIT DATA CONSISTENCY CHECK: No rule violations found. All records passed.")

    counts_df = pd.DataFrame(list(flag_counts.items()), columns=["Metric", "Total_Flagged_Patients"])

    with open(log_path2, "w") as f:
        # format flag map table
        w_id = 25
        w_metrics_list = 60
        header_2 = f"{'Patient_ID'.ljust(w_id)}{'Flagged_Metrics'.ljust(w_metrics_list)}"
        sep_2 = "-" * len(header_2)

        rows_2 = []
        if not flag_map.empty:
            for _, row in flag_map.iterrows():
                rows_2.append(
                    f"{str(row['Patient_ID']).ljust(w_id)}{str(row['Flagged_Metrics']).ljust(w_metrics_list)}"
                )
            table_2 = "\n".join([header_2, sep_2] + rows_2)
        else:
            table_2 = "No patient metrics exceeded the gap threshold."

        # print to log file
        f.write(
            "=" * 80
            + "\n"
            + " PATIENT GAP THRESHOLD FLAG REPORT\n"
            + "=" * 80
            + "\n\n"
            + "--- Summary: Flag Counts per Metric ---\n"
            + flag_counts.to_string(index=False)
            + "\n\n"
            + "=" * 80
            + "\n\n"
            + "--- Details: Flagged Metrics per Patient ---\n"
            + table_2
            + "\n\n"
            + "=" * 80
            + "\n"
        )


def main():
    sheet1, sheet2, improbable_value_bounds = load_sheets()
    sheet1 = standardize_sheet(sheet1)
    sheet2 = standardize_sheet(sheet2)
    merged = merge_sheets(sheet1, sheet2)
    merged = delete_columns(merged)
    merged = sort_columns(merged)

    metric_cols = [col for col in merged.columns if col.startswith(("activities", "sleep", "heart_rate"))]
    metric_cols.remove("activities_summary_caloriesbmr") # we don't need to analyze this column

    merged = null_implausible_values(merged, metric_cols, improbable_value_bounds)
    merged, flags = check_consistency(merged)

    # this column should have been dropped earlier, as it's insignificant and won't be needed in the future. however, it's used in the function
    # check_consistency(), so we drop it now instead of with the other columns that were dropped earlier.
    merged = merged.drop(columns=["activities_summary_caloriesbmr"], errors="ignore")

    flag_counts, flag_map = flag_patients(merged, metric_cols)

    # convert flag_counts and flag_map to DataFrames so they can be easily printed to the log file
    counts_df = pd.DataFrame(list(flag_counts.items()), columns=["Metric", "Total_Flagged_Patients"])
    map_rows = []
    for pt_id, metrics in flag_map.items():
        map_rows.append(
            {
                "Patient_ID": pt_id,
                "Flagged_Metrics": ", ".join(metrics)
                if isinstance(metrics, list)
                else str(metrics),
            }
        )
    map_df = pd.DataFrame(map_rows)

    print_log(flags, counts_df, map_df)

    # write sheet to an output file
    merged.to_csv("01_data_cleaning/processed_data/processed_fitbit_data.csv", index = False)


if __name__ == "__main__":
    main()