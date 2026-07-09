import pandas as pd


# load and return the clinical dataset
def load_sheet():
    sheet = pd.read_csv("00_raw_data/dp3 master table v2.xlsx - variables of interest.csv")
    return sheet


# creates binary indicator columns (1/0) for categorical features where text strings indicate missing data
def add_missingness_indicators(sheet):
    sheet_copy = sheet.copy()

    custom_null_flags = {"DECLINED", "UNKNOWN", "NA"}

    is_std_null = sheet_copy["race"].isnull()
    is_cust_null = sheet_copy["race"].astype(str).str.strip().str.upper().isin(custom_null_flags)

    race_missing_values = (is_std_null | is_cust_null).astype(int)

    # insert the "race_is_missing" column immediately after the "race" column
    race_index = sheet_copy.columns.get_loc("race")
    target_index = race_index + 1
    sheet_copy.insert(loc = target_index, column = "race_is_missing", value = race_missing_values)
    
    return sheet_copy


def main():
    clinical_sheet = load_sheet()

    clinical_sheet_cleaned = add_missingness_indicators(clinical_sheet)

    if not clinical_sheet_cleaned.empty:
        clinical_csv_path = "01_data_cleaning/processed_data/clinical_sheet_cleaned.csv"
        clinical_sheet_cleaned.to_csv(clinical_csv_path, index = False)


if __name__ == "__main__":
    main()