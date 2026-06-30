import pandas as pd

# load and return the dataset
def load_sheet():
    sheet = pd.read_csv("01_data_cleaning/processed_data/processed_fitbit_data.csv")
    return sheet

# filters the sheet to only include "Fitbit Data" events and only include events during pregnancy
def filter_sheet(sheet):
    # filter out all events from the sheet except the "Fitbit Data" ones
    sheet_filtered = sheet[sheet["Event Name"] == "Fitbit Data"].copy()

    # only include events during pregnancy
    # if gestational age at delivery is not in the dataset, assume that delivery occurred at 40 weeks
    sheet_filtered["current_weeks"] = sheet_filtered["timepoint"] / 7
    sheet_filtered = sheet_filtered[(sheet_filtered["current_weeks"] <= sheet_filtered["gest age del"]) | 
                                    (sheet_filtered["gest age del"].isna() & sheet_filtered["current_weeks"] <= 40)]
    return sheet_filtered

# splits the filtered dataset into four smaller datasets, based on which trimester of the pregnancy each datapoint is in
# the four datasets are: 1st trimester, early 2nd trimester, late 2nd/early 3rd trimester, late 3rd trimester
# returns a list containing these four datasets
# note that the input dataset has already been filtered and only includes events during pregnancy
def bucket_data(sheet):
    local_sheet = sheet.copy()

    bins = [float("-inf"), 14, 22, 32, float("inf")]
    labels = ["first", "early_second", "late_second_early_third", "late_third"]

    local_sheet["group"] = pd.cut(local_sheet["current_weeks"], bins = bins, labels = labels, right = False)

    outputs = []
    
    for label in labels:
        group_sheet = local_sheet[local_sheet["group"] == label].drop(columns = ["group"])
        outputs.append(group_sheet)

    return outputs

# returns the total number of missing (aka "NA") values in the dataset across all columns. excludes the "NA" values corresponding to the general information 
# rows for each patient, as these do not represent missing values in the data.
def count_total_missing(sheet):
    # sum all NaN values across the whole sheet
    return sheet.isna().sum().sum()

# returns a table containing the number of days missing per patient
# a day is only considered missing if every value for that day is NaN
def get_missing_per_patient(sheet, feature_cols):
    local_sheet = sheet.copy()
    # check if all values in each row are NaN
    local_sheet["is_missing"] = local_sheet[feature_cols].isna().all(axis = 1)

    # sum the True values by patient ID
    result = (
        local_sheet.groupby("Record ID")["is_missing"]
        .sum()
        .reset_index()
        .rename(columns = {"Record ID": "ID", "is_missing": "Missing Days"})
    )

    return result

# returns a table containing the maximum consecutive number of days missing per feature per patient
def get_max_consecutive_missing(sheet, feature_cols):
    feature_results = {}

    # ensure the dataframe is sorted chronologically per patient
    sorted_sheet = sheet.sort_values(by = ["Record ID", "Date"]).reset_index(drop = True)

    for col in feature_cols:
        is_missing = sorted_sheet[col].isna()

        # create a block_id that changes only when a non-NaN value is seen
        # i.e. if a non-NaN value is seen, the is_missing value is false. therefore, the ~is_missing value is true. adding this to a sum would add 1 to the sum.
        # this means consecutive missing values will all share the same block_id number
        block_id = (~is_missing).cumsum()

        # group is_missing by (patient id, block id). since block_id only changes when a non-NaN value is seen, these groups will each contain streaks of NaN 
        # values, organized in chronological order, for each patient.
        # then sum the true values (by groups) in is_missing. this gives the lengths of every missing streak for that patient
        streak_lengths = is_missing.groupby([sorted_sheet["Record ID"], block_id]).sum()

        # find the maximum streak length for each patient
        max_streak = streak_lengths.groupby("Record ID").max()

        feature_results[col] = max_streak

    # combine results for all features into a table
    final_table = pd.DataFrame(feature_results).reset_index()
    final_table = final_table.rename(columns = {"Record ID": "id"})

    return final_table

# returns total number of unique dates recorded across all patients
def count_unique_dates(sheet):
    return sheet["Date"].nunique()

# returns median + interquartile range for each relevant metric:
# gestational age at start of study, gestational age at delivery, steps, total distance, very active minutes, total minutes asleep
def calc_summary_stats(sheet, feature_cols):
    new_cols = feature_cols + ["Gestational age by reported LMP", "gest age del"]

    def iqr(x):
        return x.quantile(0.75) - x.quantile(0.25)
    
    # aggregate across the entire dataset for median and IQR
    summary = sheet[new_cols].agg(["median", iqr]).T

    summary = summary.reset_index()
    summary.columns = ["Feature", "Median", "IQR"]

    return summary

# returns a table containing the number of patients with Fitbit data for at least one metric during each timeframe
# the timeframes are: 1st trimester, early 2nd trimester, late 2nd/early 3rd trimester, late 3rd trimester
# @param sheets: a list containing four datasets, each containing the data for one of the above timeframes
def get_patients_per_timeframe(sheets, feature_cols):
    timeframe_names = ["First Trimester", "Early Second Trimester", "Late Second and Early Third Trimester", "Late Third Trimester"]

    summary_data = []

    for df, timeframe in zip(sheets, timeframe_names):
        # drop rows for which all metric columns are missing
        df_cleaned = df.dropna(subset = feature_cols, how = "all")

        patient_count = df_cleaned["Record ID"].nunique()

        summary_data.append({
            "Timeframe": timeframe,
            "Patient Count": patient_count
        })

    return pd.DataFrame(summary_data)

# print all calculated data into a log file
def print_log(total_missing, per_patient, max_con_missing, unique_dates, summary_stats, patients_per_timeframe):
    log_path = "02_exploratory_analysis/outputs/fitbit_data_analysis.txt"

    with open(log_path, "w") as f:
        f.write("Following are various statistics about the Fitbit dataset. \n")
        f.write("Note that this data has been filtered to only include datapoints during pregnancy.\n\n")

        f.write(f"Total number of missing values: {total_missing}")
        f.write("\n\n")

        f.write("Number of missing days per patient:\n")
        f.write(per_patient.to_string(index = False))
        f.write("\n\n")

        f.write("Maximum consecutive number of days missing per feature per patient:\n")
        f.write(max_con_missing.to_string(index = False))
        f.write("\n\n")

        f.write(f"Total number of unique dates recorded across all patients: {unique_dates}")
        f.write("\n\n")

        f.write("Summary statistics by metric:\n")
        f.write(summary_stats.to_string(index = False))
        f.write("\n\n")

        f.write("Number of unique patients per timeframe:\n")
        f.write(patients_per_timeframe.to_string(index = False))
        f.write("\n\n")

def main():
    sheet = load_sheet()

    sheet_filtered = filter_sheet(sheet)

    feature_cols = [col for col in sheet_filtered.columns if col.startswith(("Activities", "Sleep", "Heart Rate"))]

    # forcefully convert all feature columns into numeric types
    for col in feature_cols:
        sheet_filtered[col] = pd.to_numeric(sheet_filtered[col], errors = "coerce")
    sheet_filtered["Gestational age by reported LMP"] = pd.to_numeric(sheet_filtered["Gestational age by reported LMP"], errors = "coerce")
    sheet_filtered["gest age del"] = pd.to_numeric(sheet_filtered["gest age del"], errors = "coerce")

    sheet_bucketed = bucket_data(sheet_filtered)

    total_missing = count_total_missing(sheet_filtered)
    per_patient = get_missing_per_patient(sheet_filtered, feature_cols)
    max_con_missing = get_max_consecutive_missing(sheet_filtered, feature_cols)
    unique_dates = count_unique_dates(sheet_filtered)
    summary_stats = calc_summary_stats(sheet_filtered, feature_cols)
    patients_per_timeframe = get_patients_per_timeframe(sheet_bucketed, feature_cols)

    print_log(total_missing, per_patient, max_con_missing, unique_dates, summary_stats, patients_per_timeframe)

if __name__ == "__main__":
    main()