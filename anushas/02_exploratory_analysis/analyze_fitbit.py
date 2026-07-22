import pandas as pd
import numpy as np


# load and return the fitbit dataset
def load_sheet():
    sheet = pd.read_csv("01_data_cleaning/processed_data/processed_fitbit_data.csv", low_memory = False)
    return sheet


# filters the sheet to only include "fitbit data" events and only include events during pregnancy
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
# note that the input dataset has already been filtered and only includes events during pregnancy
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


# returns the total number of unique patients, after data has been filtered
def get_total_patients(sheet):
    # filter for rows for which record id starts with "dp3-"
    # this ensures we only count actual patients
    dp3_patients = sheet[sheet["record_id"].astype(str).str.startswith("dp3-")]
    return dp3_patients["record_id"].nunique()


# returns the total number of missing (aka "na") values in the dataset across all columns. excludes the "na" values corresponding to the general information 
# rows for each patient, as these do not represent missing values in the data.
def count_total_missing(sheet):
    # sum all nan values across the whole sheet
    return sheet.isna().sum().sum()


# returns a table containing the number of days missing per patient
# a day is only considered missing if every value for that day is nan
def get_missing_per_patient(sheet, feature_cols):
    local_sheet = sheet.copy()
    # check if all values in each row are nan
    local_sheet["is_missing"] = local_sheet[feature_cols].isna().all(axis = 1)

    # sum the true values by patient id
    result = (
        local_sheet.groupby("record_id")["is_missing"]
        .sum()
        .reset_index()
        .rename(columns = {"record_id": "id", "is_missing": "missing_days"})
    )

    return result


# returns a table containing the maximum consecutive number of days missing per feature per patient
def get_max_consecutive_missing(sheet, feature_cols):
    feature_results = {}

    # ensure the dataframe is sorted chronologically per patient
    sorted_sheet = sheet.sort_values(by = ["record_id", "date"]).reset_index(drop = True)

    for col in feature_cols:
        is_missing = sorted_sheet[col].isna()

        # create a block_id that changes only when a non-nan value is seen
        # i.e. if a non-nan value is seen, the is_missing value is false. therefore, the ~is_missing value is true. adding this to a sum would add 1 to the sum.
        # this means consecutive missing values will all share the same block_id number
        block_id = (~is_missing).cumsum()

        # group is_missing by (patient id, block id). since block_id only changes when a non-nan value is seen, these groups will each contain streaks of nan 
        # values, organized in chronological order, for each patient.
        # then sum the true values (by groups) in is_missing. this gives the lengths of every missing streak for that patient
        streak_lengths = is_missing.groupby([sorted_sheet["record_id"], block_id]).sum()

        # find the maximum streak length for each patient
        max_streak = streak_lengths.groupby("record_id").max()

        feature_results[col] = max_streak

    # combine results for all features into a table
    final_table = pd.DataFrame(feature_results).reset_index()
    final_table = final_table.rename(columns = {"record_id": "id"})

    return final_table


# returns total number of unique dates recorded across all patients
def count_unique_dates(sheet):
    return sheet["date"].nunique()


# returns median + interquartile range for each relevant metric:
# gestational age at start of study, gestational age at delivery, steps, total distance, very active minutes, total minutes asleep
def calc_summary_stats(sheet, feature_cols):
    new_cols = feature_cols + ["gestational_age_by_reported_lmp", "gest_age_del"]

    def iqr(x):
        return x.quantile(0.75) - x.quantile(0.25)
    
    # force-convert to numeric
    summary_df = sheet[new_cols].copy()
    for col in new_cols:
        summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")
    
    # aggregate across the entire dataset for median and iqr
    summary = summary_df.agg(["median", iqr]).T

    summary = summary.reset_index()
    summary.columns = ["Feature", "Median", "IQR"]

    return summary


# returns a table containing the number of patients with fitbit data for at least one metric during each timeframe
# the timeframes are: 1st trimester, early 2nd trimester, late 2nd/early 3rd trimester, late 3rd trimester
# @param sheets: a list containing four datasets, each containing the data for one of the above timeframes
def get_patients_per_timeframe(sheets, feature_cols, timeframe_names):
    summary_data = []

    for df, timeframe in zip(sheets, timeframe_names):
        # drop rows for which all metric columns are missing
        df_cleaned = df.dropna(subset = feature_cols, how = "all")

        patient_count = df_cleaned["record_id"].nunique()

        summary_data.append({
            "Timeframe": timeframe,
            "Patient Count": patient_count
        })

    return pd.DataFrame(summary_data)


# returns a true/false matrix (patients x metrics) showing whether each patient has non-missing data for at least 80% of their valid pregnancy tracking days
# also returns a table containing the number of metrics with 80+% of valid data, per patient
# also returns a table containing the number of patients with 80+% of valid data, per metric
def get_metric_representation_matrices(sheet, feature_cols):
    df = sheet.sort_values(by = ["record_id", "timepoint"]).copy()

    # extract start and end dates per patient
    # start date marks the day the patient enrolled/started data collection
    # end data marks delivery
    enrollment_days = df.groupby("record_id")["timepoint"].min()
    max_recorded_days = df.groupby("record_id")["timepoint"].max()

    delivery_weeks = df.groupby("record_id")["gest_age_del"].first()
    delivery_days = delivery_weeks * 7

    # apply fallback logic for the end of the tracking window:
    # use gestational age at delivery if present; otherwise, use latest recorded day
    end_days = delivery_days.fillna(max_recorded_days)

    recorded_data_lengths = (end_days - enrollment_days) + 1
    recorded_data_lengths = recorded_data_lengths.clip(lower = 1) # ensure values are positive

    # count how many days of valid data exist per metric per patient
    active_days_per_metric = df.groupby("record_id")[feature_cols].agg(lambda x: x.notna().sum())

    # contains percent of data that is valid per metric per patient
    representation_matrix = active_days_per_metric.div(active_days_per_metric, axis = 0)
    
    final_table = representation_matrix >= 0.80
    final_table = final_table.reset_index().rename(columns = {"record_id": "patient_id"})

    # contains the number of metrics with 80+% of valid data, per patient
    pt_summary = pd.DataFrame({
        "Patient ID": final_table["patient_id"],
        "Compliant Metrics Count": final_table[feature_cols].sum(axis = 1)
    })
    pt_summary = pt_summary.sort_values(by = "Compliant Metrics Count", ascending = False).reset_index(drop = True)

    # contains the number of patients with 80+% of valid data, per metric
    patient_counts_per_metric = final_table[feature_cols].sum(axis = 0)
    metric_summary = pd.DataFrame({
        "Fitbit Metric": patient_counts_per_metric.index,
        "Patients With >= 80% Density": patient_counts_per_metric.values
    })
    metric_summary = metric_summary.sort_values(by = "Patients With >= 80% Density", ascending = False).reset_index(drop = True)

    return final_table, pt_summary, metric_summary


# print all calculated data into a log file
def print_log(total_patients, total_missing, per_patient, max_con_missing, unique_dates, summary_stats, 
              patients_per_timeframe, metric_matrix, pt_summary, metric_summary, num_metrics):
    log_path = "04_results_and_figures/data_analysis/fitbit/fitbit_data_analysis.txt"

    with open(log_path, "w") as f:
        f.write("Following are various statistics about the Fitbit dataset. \n")
        f.write("Note that this data has been filtered to only include datapoints during pregnancy.\n\n\n")

        f.write(f"Total number of patients: {total_patients}")
        f.write("\n\n\n")

        f.write(f"Total number of missing values: {total_missing}")
        f.write("\n\n\n")

        f.write("Number of missing days per patient:\n")
        f.write(per_patient.to_string(index = False))
        f.write("\n\n\n")

        f.write("Maximum consecutive number of days missing per feature per patient:\n")
        f.write(max_con_missing.to_string(index = False))
        f.write("\n\n\n")

        f.write(f"Total number of unique dates recorded across all patients: {unique_dates}")
        f.write("\n\n\n")

        f.write("Summary statistics by metric:\n")
        f.write(summary_stats.to_string(index = False))
        f.write("\n\n\n")

        f.write("Number of unique patients per timeframe:\n")
        f.write(patients_per_timeframe.to_string(index = False))
        f.write("\n\n\n")

        f.write("Which patients contributed valid data for at least 80% of their pregnancy, per feature:\n")
        f.write(metric_matrix.to_string(index = False))
        f.write("\n\n\n")

        f.write(f"Number of the {num_metrics} metrics for which each patient achieved >= 80% data density:\n")
        f.write(pt_summary.to_string(index = False))
        f.write("\n\n\n")

        f.write(f"Number of the {total_patients} patients that achieved >= 80% data density, per metric:\n")
        f.write(metric_summary.to_string(index = False))
        f.write("\n\n\n")


def main():
    fitbit_sheet = load_sheet()

    sheet_filtered = filter_sheet(fitbit_sheet)

    feature_cols = [col for col in sheet_filtered.columns if col.startswith(("activities", "sleep", "heart_rate"))]
    numeric_cols = feature_cols + ["gestational_age_by_reported_lmp", "gest_age_del"]

    # forcefully convert all feature columns + gest age columns into numeric types
    sheet_filtered[numeric_cols] = sheet_filtered[numeric_cols].apply(pd.to_numeric, errors = "coerce")
    
    sheet_bucketed = bucket_data(sheet_filtered)

    timeframe_names = ["First Trimester", "Early Second Trimester", "Late Second and Early Third Trimester", "Late Third Trimester"]

    total_patients = get_total_patients(sheet_filtered)
    total_missing = count_total_missing(sheet_filtered)
    per_patient = get_missing_per_patient(sheet_filtered, feature_cols)
    max_con_missing = get_max_consecutive_missing(sheet_filtered, feature_cols)
    unique_dates = count_unique_dates(sheet_filtered)
    summary_stats = calc_summary_stats(sheet_filtered, feature_cols)
    patients_per_timeframe = get_patients_per_timeframe(sheet_bucketed, feature_cols, timeframe_names)
    metric_matrix, pt_summary, metric_summary = get_metric_representation_matrices(sheet_filtered, feature_cols)

    print_log(total_patients, total_missing, per_patient, max_con_missing, unique_dates, summary_stats, 
              patients_per_timeframe, metric_matrix, pt_summary, metric_summary, len(feature_cols))


if __name__ == "__main__":
    main()