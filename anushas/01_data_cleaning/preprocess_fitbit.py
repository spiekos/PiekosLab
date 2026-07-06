import io
import pandas as pd
import numpy as np


# loads both fitbit data sheets and returns both sheets
def load_sheets():
    sheet1 = pd.read_csv("00_raw_data/DP3_playset.csv", index_col = 0)
    sheet2 = pd.read_csv("00_raw_data/DP3-FitbitFullReport_DATA_LABELS_2025-02-18_1356.csv")
    return sheet1, sheet2


# rename columns of both sheets so that column names are consistent across sheets
def rename_columns(sheet1, sheet2):
    sheet1 = sheet1.rename(columns = {
        "Record.ID": "Record ID",
        "Event.Name": "Event Name",
        "Gestational.age.by.reported.LMP": "Gestational age by reported LMP",
        "Activities...Summary...steps": "Activities - Summary - Steps",
        "Activities...Summary...totalDistances": "Activities - Summary - totalDistances",
        "Activities...Summary...veryActiveMinutes": "Activities - Summary - veryActiveMinutes",
        "Sleep...Summary...total.minutes.asleep": "Sleep - Summary - total minutes asleep"
    })

    return sheet1, sheet2


# merges the two input sheets on record ID and date
# ensures that no columns are duplicated and all column names are consistent
def merge_sheets(sheet1, sheet2):
    sheet1, sheet2 = rename_columns(sheet1, sheet2)

    # merge both sheets
    merged = pd.merge(sheet1, sheet2, on = ["Record ID", "Date"], how = "outer")

    # combine the two gestational age columns
    merged['Gestational age by reported LMP'] = merged['Gestational age by reported LMP_x'].combine_first(merged['Gestational age by reported LMP_y'])
    merged = merged.drop(columns=['Gestational age by reported LMP_x', 'Gestational age by reported LMP_y'])

    # combine the two event name columns
    merged['Event Name'] = merged['Event Name_x'].combine_first(merged['Event Name_y'])
    merged = merged.drop(columns=['Event Name_x', 'Event Name_y'])

    return merged


# sort columns such that the activity/sleep/heart rate/etc. columns are all at the end
def sort_columns(sheet):
    front_columns = [
        "Record ID", "Event Name", "Gestational age by reported LMP", "gest age del", "Date", "group", "group_bin", "timepoint", "Repeat Instrument", 
        "Repeat Instance", "Fitbit Activity Data Uploaded", "Fitbit Heart Rate Data Uploaded", "Fitbit Sleep Data Uploaded", "Do we have all the data?", 
        "Complete?", "Was a complication diagnosed during the current pregnancy?"
    ]

    other_columns = [col for col in sheet.columns if col not in front_columns]

    sheet = sheet[front_columns + other_columns]
    
    return sheet


# for patients with no recording of 7+ days in a row for a certain feature, drop their data for that feature only
# nulls out the appropriate data
# inputs:
# sheet: the fitbit data sheet, which contains patient information and all collected metrics
# table: the dataframe containing a table of max number of consecutive days missing per feature per patient
# outputs:
# sheet_clean: the input sheet cleaned, i.e. after the appropriate data has been nulled
# exclusion_counts: a table containing the number of patients whose data has been nulled, per metric
def drop_patients(sheet):
    metric_cols = [col for col in sheet.columns if col.startswith(("Activities", "Sleep", "Heart Rate"))]

    sheet_clean = sheet.copy()
    
    exclusion_counts = {metric: 0 for metric in metric_cols}
    
    sheet_clean["Date"] = pd.to_datetime(sheet_clean["Date"], errors = "coerce")
    sheet_clean = sheet_clean.dropna(subset = ["Date", "Record ID"])

    # dictionary: {pt_id: [list of metrics to nullify for this patient]}
    nullification_map = {}

    for pt_id, pt_df in sheet_clean.groupby("Record ID"):
        if pt_df["Date"].isna().all():
            continue

        pt_df = pt_df.sort_values(by = "Date")
        pt_indexed = pt_df.set_index("Date")
        
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
            # the first cumsum() creates a unique group ID that only increments when we hit valid data
            # the groupby() groups consecutive missing days together under the same ID number
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
            sheet_clean.loc[sheet_clean["Record ID"] == pt_id, metrics_to_null] = np.nan

    return sheet_clean, exclusion_counts


# print to a log file explaining why you dropped each patient from the dataset
# i.e. creates a table containing each patient that was dropped and the maximum consecutive number of days their data was missing
def print_log(exclusion_counts):
    log_path = "02_exploratory_analysis/outputs/dropped_patients_fitbit_log.txt"
    with open(log_path, "w") as f:
        f.write("This file contains a table consisting of the number of patients whose data was nulled out, organized\n")
        f.write("per metric. Patients had their data nulled out for a certain metric if they had more than 7 consecutive\n")
        f.write("days of data missing for that metric.\n\n")

        f.write(f"{'Metric':<45} | {'Patients Excluded'}\n")
        for metric, count in exclusion_counts.items():
            f.write(f"{str(metric).strip():<45} | {count}\n")


def main():
    sheet1, sheet2 = load_sheets()
    merged = merge_sheets(sheet1, sheet2)
    merged = sort_columns(merged)

    merged_clean, exclusion_counts = drop_patients(merged)

    print_log(exclusion_counts)

    # write sheet to an output file
    merged_clean.to_csv("01_data_cleaning/processed_data/processed_fitbit_data.csv", index = False)


if __name__ == "__main__":
    main()