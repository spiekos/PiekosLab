import pandas as pd

# load and return the dataset
def load_sheet():
    sheet = pd.read_csv("02_exploratory_analysis/analyze_fitbit_data/DP3_playset_PE.csv")
    return sheet

# filters the sheet to only include "Fitbit Data" events and only include events during pregnancy
def filter_sheet(sheet):
    # filter out all events from the sheet except the "Fitbit Data" ones
    sheet_filtered = sheet[sheet["Event.Name"] == "Fitbit Data"].copy()

    # only include events during pregnancy
    sheet_filtered["current_weeks"] = sheet_filtered["timepoint"] / 7
    sheet_filtered = sheet_filtered[sheet_filtered["current_weeks"] <= sheet_filtered["gest age del"]]

    return sheet_filtered

# returns the total number of missing (aka "NA") values in the dataset across all columns. excludes the "NA" values corresponding to the general information 
# rows for each patient, as these do not represent missing values in the data.
def count_total_missing(sheet):
    # sum all NaN values across the whole sheet
    return sheet.isna().sum().sum()

# returns a table containing the number of days missing per patient
# a day is only considered missing if every value for that day is NaN
def missing_per_patient(sheet, feature_cols):
    # check if all values in each row are NaN
    sheet["is_missing"] = sheet[feature_cols].isna().all(axis = 1)

    # sum the True values by patient ID
    result = (
        sheet.groupby("Record.ID")["is_missing"]
        .sum()
        .reset_index()
        .rename(columns = {"Record.ID": "ID", "is_missing": "Missing Days"})
    )

    return result

# returns a table containing the maximum consecutive number of days missing per feature per patient
def max_consecutive_missing(sheet, feature_cols):
    feature_results = {}

    for col in feature_cols:
        is_missing = sheet[col].isna()

        # create a block_id that changes only when a non-NaN value is seen
        # i.e. if a non-NaN value is seen, the is_missing value is false. therefore, the ~is_missing value is true. adding this to a sum would add 1 to the sum.
        # this means consecutive missing values will all share the same block_id number
        block_id = (~is_missing).cumsum()

        # group is_missing by (patient id, block id). since block_id only changes when a non-NaN value is seen, these groups will each contain streaks of NaN 
        # values, organized in chronological order, for each patient.
        # then sum the true values (by groups) in is_missing. this gives the lengths of every missing streak for that patient
        streak_lengths = is_missing.groupby([sheet["Record.ID"], block_id]).sum()

        # find the maximum streak length for each patient
        max_streak = streak_lengths.groupby("Record.ID").max()

        feature_results[col] = max_streak

    # combine results for all features into a table
    final_table = pd.DataFrame(feature_results).reset_index()
    final_table = final_table.rename(columns = {"Record.ID": "id"})

    return final_table

# returns total number of unique dates recorded across all patients
def count_unique_dates(sheet):
    total = sheet["Date"].nunique()
    return total

# returns median + interquartile range for each relevant metric:
# gestational age at start of study, gestational age at delivery, steps, total distance, very active minutes, total minutes asleep
def calc_summary_stats(sheet, feature_cols):
    new_cols = feature_cols + ["Gestational.age.by.reported.LMP", "gest age del"]

    def iqr(x):
        return x.quantile(0.75) - x.quantile(0.25)
    
    # aggregate across the entire dataset for median and IQR
    summary = sheet[new_cols].agg(["median", iqr]).T

    summary = summary.reset_index()
    summary.columns = ["Feature", "Median", "IQR"]

    return summary

# print all calculated data into a log file
def print_log(total_missing, per_patient, max_con_missing, unique_dates, summary_stats):
    log_path = "02_exploratory_analysis/analyze_fitbit_data/log.txt"

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

def main():
    feature_cols = [
        "Activities...Summary...steps", 
        "Activities...Summary...totalDistances", 
        "Activities...Summary...veryActiveMinutes", 
        "Sleep...Summary...total.minutes.asleep"
    ]

    sheet = load_sheet()

    sheet_filtered = filter_sheet(sheet)

    total_missing = count_total_missing(sheet_filtered)
    per_patient = missing_per_patient(sheet_filtered, feature_cols)
    max_con_missing = max_consecutive_missing(sheet_filtered, feature_cols)
    unique_dates = count_unique_dates(sheet_filtered)
    summary_stats = calc_summary_stats(sheet_filtered, feature_cols)

    print_log(total_missing, per_patient, max_con_missing, unique_dates, summary_stats)

if __name__ == "__main__":
    main()