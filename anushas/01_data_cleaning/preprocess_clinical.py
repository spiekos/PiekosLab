import pandas as pd
import numpy as np


# load and return the clinical dataset
def load_sheet():
    sheet = pd.read_csv("00_raw_data/dp3 master table v2.xlsx - variables of interest.csv")
    return sheet


# fixes spelling errors in the dataframe
def fix_typos(sheet):
    sheet_copy = sheet.copy()

    if "race" in sheet_copy.columns:
        race_upper = sheet_copy["race"].astype(str).str.strip().str.upper()

        typo_mask = race_upper == "AFRICIAN AMERICAN"
        sheet_copy.loc[typo_mask, "race"] = "AFRICAN AMERICAN"

    return sheet_copy


# creates binary indicator columns (1/0) for categorical features where certain
# currently, a binary indicator column is only being added for the "race" column
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


# one-hot encodes the "race" and "ethnicity" columns of the clinical sheet, prefixes the new features, and places them directly after the "race_is_missing" column
def one_hot_encode_demographics(sheet):
    sheet_copy = sheet.copy()

    # pull out row 2 (this row doesn't contain any data)
    row_2 = sheet_copy.iloc[[0]].copy()
    rest_of_df = sheet_copy.iloc[1:].copy()

    # reset index to prevent alignment bugs
    original_index = rest_of_df.index
    rest_of_df = rest_of_df.reset_index(drop = True)

    race_clean = rest_of_df["race"].astype(str).str.strip().str.upper()

    print("\n--- 🔍 DIAGNOSTIC: UNIQUE RAW RACE VALUES ---")
    print(race_clean.unique())

    rest_of_df["race_WHITE"] = 0
    rest_of_df["race_BLACK"] = 0
    rest_of_df["race_ASIAN"] = 0

    # map White
    white_mask = race_clean.str.contains("WHITE", na = False)
    rest_of_df.loc[white_mask, "race_WHITE"] = 1

    # group Black/African American
    black_mask = race_clean.str.contains("BLACK|AFRICAN AMERICAN|AFRICAN", na = False)
    rest_of_df.loc[black_mask, "race_BLACK"] = 1

    # group Asian (including Korean)
    asian_mask = race_clean.str.contains("ASIAN|KOREAN", na = False)
    rest_of_df.loc[asian_mask, "race_ASIAN"] = 1

    print("\n--- 🔍 DIAGNOSTIC: MASK ASSIGNMENT CHECKS ---")
    print(f"Total rows processed: {len(rest_of_df)}")
    print(f"Rows marked race_WHITE == 1: {rest_of_df['race_WHITE'].sum()}")
    print(f"Rows marked race_BLACK == 1: {rest_of_df['race_BLACK'].sum()}")
    print(f"Rows marked race_ASIAN == 1: {rest_of_df['race_ASIAN'].sum()}")
    
    # Show us rows that have 'AFRICAN' but race_BLACK is somehow still 0
    failed_african_rows = rest_of_df[race_clean.str.contains("AFRICAN", na=False) & (rest_of_df["race_BLACK"] == 0)]
    if not failed_african_rows.empty:
        print("\n⚠️ ALERT! Found rows containing 'AFRICAN' that failed to map to race_BLACK:")
        print(failed_african_rows["race"].unique())

    # combine "race_AFRICAN AMERICAN" column and "race_BLACK" column
    # check if the African American column exists
    existing_aa_col = None
    for col in rest_of_df.columns:
        if col.strip().upper() in ["RACE_AFRICAN AMERICAN", "RACE_AFRICAN_AMERICAN", "AFRICAN AMERICAN"]:
            existing_aa_col = col
            break

    if existing_aa_col is not None:
        # copy postiive values to the "race_BLACK" column
        aa_mask = rest_of_df[existing_aa_col].astype(str).str.strip() == "1"
        rest_of_df.loc[aa_mask, "race_BLACK"] = 1

        # drop old column frop dataframe and row_2
        rest_of_df = rest_of_df.drop(columns = [existing_aa_col])
        if existing_aa_col in row_2.columns:
            row_2 = row_2.drop(columns = [existing_aa_col])

    # filter out categories we handled manually or that count as missing data
    ignore_keywords = [
        "WHITE",
        "BLACK",
        "AFRICAN AMERICAN",
        "AFRICAN",
        "ASIAN",
        "KOREAN",
        "DECLINED",
        "UNKNOWN",
        "NA",
        "NAN",
        "UNSPECIFIED",
    ]
    remaining_mask = ~race_clean.apply(lambda x: any(kw in x for kw in ignore_keywords))

    # one-hot encode any remaining values
    if remaining_mask.any():
        race_dummies = pd.get_dummies(
            race_clean.loc[remaining_mask], prefix = "race", dtype = int
        ).reindex(rest_of_df.index, fill_value = 0)

        race_dummies.columns = race_dummies.columns.str.strip()

        # drop columns we mapped manually if they somehow slipped through as substrings
        cols_to_drop = [col for col in race_dummies.columns if any(r in col.upper() for r in ["WHITE", "BLACK", "AFRICAN", "ASIAN", "KOREAN"])]
        race_dummies = race_dummies.drop(columns = cols_to_drop, errors = "ignore")
        
        # combine back with the main dataframe slice
        rest_of_df = pd.concat([rest_of_df, race_dummies], axis = 1).fillna(0)

    # encode the Hispanic/Latino column
    eth_clean = rest_of_df["ethnicity"].astype(str).str.strip().str.upper()
    rest_of_df["HISPANIC/LATINO"] = np.nan  # initialize with NaN to track uncaught rows

    # map conditions, ensuring that each column you map has not been already mapped
    is_missing_eth = eth_clean.isin(["DECLINED", "NA", "UNKNOWN", "UNSPECIFIED", "NAN"])
    rest_of_df.loc[is_missing_eth, "HISPANIC/LATINO"] = 0

    is_not_hispanic = eth_clean.str.contains("NON|NOT", na=False) & (rest_of_df["HISPANIC/LATINO"].isna())
    rest_of_df.loc[is_not_hispanic, "HISPANIC/LATINO"] = 0

    is_hispanic = eth_clean.str.contains("HISPANIC|LATINO", na=False) & (rest_of_df["HISPANIC/LATINO"].isna())
    rest_of_df.loc[is_hispanic, "HISPANIC/LATINO"] = 1

    rest_of_df["HISPANIC/LATINO"] = rest_of_df["HISPANIC/LATINO"].astype(int)

    rest_of_df.index = original_index

    # reconstruct the complete dataframe
    # add empty columns to row_2 with 0 values so it matches shape during merge
    new_cols = [col for col in rest_of_df.columns if col not in row_2.columns]
    for col in new_cols:
        row_2[col] = 0
        rest_of_df[col] = rest_of_df[col].astype(int)
    final_df = pd.concat([row_2, rest_of_df], axis = 0).reset_index(drop = True)

    # determine where to insert the new columns (after the "race_is_missing" column)
    if "race_is_missing" in final_df.columns:
        insert_idx = final_df.columns.get_loc("race_is_missing") + 1
    else:
        insert_idx = final_df.columns.get_loc("race") + 1

    # ensure "HISPANIC/LATINO" is at the end of the list
    encoded_cols = [c for c in new_cols if c != "HISPANIC/LATINO"] + [
        "HISPANIC/LATINO"
    ]

    # insert one-hot encoded columns into the dataframe
    for col in reversed(encoded_cols):
        col_series = final_df.pop(col)
        final_df.insert(loc = insert_idx, column = col, value = col_series)

    return final_df


def main():
    sheet = load_sheet()

    sheet_cleaned = fix_typos(sheet)
    sheet_cleaned = add_missingness_indicators(sheet_cleaned)
    sheet_encoded = one_hot_encode_demographics(sheet_cleaned)

    if not sheet_encoded.empty:
        clinical_csv_path = "01_data_cleaning/processed_data/clinical_sheet_encoded.csv"
        sheet_encoded.to_csv(clinical_csv_path, index = False)


if __name__ == "__main__":
    main()