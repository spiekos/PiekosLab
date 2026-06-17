import io
import pandas as pd

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

# reads from the fitbit analysis log file
# extracts the table containing data on max number of consecutive days missing per feature per patient, and returns this table
def extract_days_missing_table():
    fitbit_analysis_path = "02_exploratory_analysis/outputs/fitbit_data_analysis.txt"
    target_title = "Maximum consecutive number of days missing per feature per patient:"

    table_lines = []
    inside_target_table = False
    
    with open(fitbit_analysis_path, 'r') as f:
        for line in f:
            cleaned_line = line.strip()

            # if we find the title, start capturing lines
            if target_title in cleaned_line:
                inside_target_table = True
                continue # skip the title line

            # if we are inside the table and hit a blank line, we're done reading
            if inside_target_table and not cleaned_line:
                break

            # collect the table's rows
            if inside_target_table:
                table_lines.append(line)

    # convert into a dataframe
    table_data = "".join(table_lines)

    # read the csv, treating any sequence of spaces as a separator
    df_target = pd.read_csv(io.StringIO(table_data), sep = r'\s+')

    df_target.columns = df_target.columns.str.strip()

    return df_target

# drop all patients with no recording of 2+ days in a row for any feature
# sheet: the fitbit data sheet, which contains patient information and all collected metrics
# table: the dataframe containing a table of max number of consecutive days missing per feature per patient
def drop_patients(sheet, table):
    metric_cols = [col for col in table.columns if col != "id"]

    # find rows for which any metric value is >= 2
    drop_mask = (table[metric_cols] >= 2).any(axis = 1)

    # find the unique IDs of the patients who failed this test
    patients_to_drop = table.loc[drop_mask, "id"].unique()

    # drop the appropriate patients
    sheet_clean = sheet[~sheet["Record ID"].isin(patients_to_drop)]

    return sheet_clean, patients_to_drop

def main():
    sheet1, sheet2 = load_sheets()
    merged = merge_sheets(sheet1, sheet2)
    merged = sort_columns(merged)

    days_missing_table = extract_days_missing_table()

    merged_clean, patients_to_drop = drop_patients(merged, days_missing_table)

    # write sheet to an output file
    merged_clean.to_csv("01_data_cleaning/processed_data/processed_fitbit_data.csv", index = False)

if __name__ == "__main__":
    main()