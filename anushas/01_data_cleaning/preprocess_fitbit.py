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






def main():
    sheet1, sheet2 = load_sheets()
    merged = merge_sheets(sheet1, sheet2)
    merged = sort_columns(merged)

    # write sheet to an output file
    merged.to_csv("01_data_cleaning/processed_data/processed_fitbit_data.csv", index = False)





if __name__ == "__main__":
    main()