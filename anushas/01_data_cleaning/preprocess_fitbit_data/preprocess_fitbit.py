import pandas as pd

# load and return the dataset
def load_sheet():
    sheet = pd.read_csv("01_data_cleaning/preprocess_fitbit_data/DP3_playset_PE.csv")
    return sheet

# returns the total number of missing (aka "NA") values in the dataset across all columns. excludes the "NA" values corresponding to the general information 
# rows for each patient, as these do not represent missing values in the data.
def count_total_missing(sheet):
    # filter out "General" events from the sheet
    sheet_filtered = sheet[sheet["Event.Name"] != "General"]

    return sheet_filtered.isna().sum().sum()

# returns a table containing the number of days missing per patient
def missing_per_patient(sheet):
    target_cols = [
        "Activities...Summary...steps", 
        "Activities...Summary...totalDistances", 
        "Activities...Summary...veryActiveMinutes", 
        "Sleep...Summary...total.minutes.asleep"
    ]

    # filter out "General" events from the sheet
    sheet_filtered = sheet[sheet["Event.Name"] != "General"].copy()
    
    # check if any value in each row is NaN
    sheet_filtered["is_missing"] = sheet_filtered[target_cols].isna().any(axis = 1)

    # sum the True values by patient ID
    result = (
        sheet_filtered.groupby("Record.ID")["is_missing"]
        .sum()
        .reset_index()
        .rename(columns = {"Record.ID": "id", "is_missing": "missing days"})
    )

    return result




# print all calculated data into a log file
def print_log(total_missing, per_patient):
    log_path = "01_data_cleaning/preprocess_fitbit_data/log.txt"

    with open(log_path, "w") as f:
        f.write(f"total number of missing values: {total_missing}\n\n")

        f.write("number of missing days per patient:\n")
        f.write(per_patient.to_string(index = False))

def main():
    sheet = load_sheet()
    total_missing = count_total_missing(sheet)
    per_patient = missing_per_patient(sheet)

    print_log(total_missing, per_patient)

if __name__ == "__main__":
    main()