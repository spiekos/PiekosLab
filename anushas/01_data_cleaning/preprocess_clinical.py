import pandas as pd
import numpy as np


# load and return the clinical dataset
def load_sheet():
    sheet = pd.read_csv("00_raw_data/dp3 master table v2.xlsx - variables of interest.csv")
    return sheet


# standardizes all values to lowercase (except the values in the "id" column)
# replaces spaces with underscores
def standardize_sheet(df):
    if df is not None and not df.empty:
        if df.index.max() > 0:
            df = df.iloc[1:].copy()

        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

        cols_to_lower = [col for col in df.columns if col != "id"]
        df[cols_to_lower] = df[cols_to_lower].map(lambda x: str(x).strip().lower() if pd.notnull(x) else np.nan)

    return df


# fixes spelling errors in the dataframe
def fix_typos(sheet):
    sheet_copy = sheet.copy()

    if "race" in sheet_copy.columns:
        race_clean = sheet_copy["race"].astype(str).str.strip()

        typo_mask = race_clean == "africian american"
        sheet_copy.loc[typo_mask, "race"] = "african american"

    return sheet_copy


# include only patients with "delivered" in their "status" column
def filter_by_status(sheet):
    sheet_copy = sheet.copy()

    if "status" in sheet_copy.columns:
        status_clean = sheet_copy["status"].astype(str).str.strip()
        keep_mask = status_clean == "delivered"
        sheet_copy = sheet_copy[keep_mask].reset_index(drop=True)
                
    return sheet_copy


# converts all values representing nulls to true numpy nan values
def unmask_missing_data(df):    
    missing_placeholders = [
        "n/a", "na", "unknown", "declined", "unspecified", 
        "not applicable", "missing", "nan", "none", "null"
    ]
    
    for col in df.columns:
        if df[col].dtype == "object":
            pattern = r'^\s*(' + '|'.join(missing_placeholders) + r')\s*$'
            df[col] = df[col].replace(pattern, np.nan, regex=True)
            
    return df


# creates binary indicator columns (1/0) for categorical features
# currently, a binary indicator column is being added for the "race", "ethnicity", and "smoking" columns
def add_missingness_indicators(sheet):
    sheet_copy = sheet.copy()

    # add missingness indicators for race
    race_missing_values = sheet_copy["race"].isnull().astype(int)

    # insert the "race_is_missing" column immediately after the "race" column
    race_index = sheet_copy.columns.get_loc("race")
    target_race_index = race_index + 1
    sheet_copy.insert(loc = target_race_index, column = "race_is_missing", value = race_missing_values)

    # add missingness indicators for ethnicity
    eth_missing_values = sheet_copy["ethnicity"].isnull().astype(int)

    # insert the "eth_is_missing" column immediately after the "ethnicity" column
    eth_index = sheet_copy.columns.get_loc("ethnicity")
    target_eth_index = eth_index + 1
    sheet_copy.insert(loc = target_eth_index, column = "eth_is_missing", value = eth_missing_values)

    # add missingness indicators for smoking
    missing_placeholders = ["nan", "na", "n/a", "unknown", "declined", "missing"]
    smoking_series = sheet["smoking"].astype(str)
    missing_smoking_mask = smoking_series.isin(missing_placeholders) | sheet["smoking"].isnull()

    # insert the "smoking_is_missing" column immediately after the "smoking" column
    smoking_index = sheet_copy.columns.get_loc("smoking")
    target_smoking_index = smoking_index + 1
    sheet_copy.insert(loc = target_smoking_index, column = "smoking_is_missing", value = missing_smoking_mask)

    return sheet_copy


# encodes smoking column: "never" and "quit" -> 0, "yes" -> 1
def encode_smoking_status(df):
    df_copy = df.copy()
    
    if "smoking" in df_copy.columns:
        s = df_copy["smoking"].astype(str).str.strip()
        mapping = {
            'never': 0, 
            'quit': 0, 
            'yes': 1
        }
        encoded_series = s.map(mapping)
        encoded_series = encoded_series.fillna(0).astype(int)

        smoking_idx = df_copy.columns.get_loc("smoking")
        df_copy.insert(smoking_idx + 1, "smoking_encoded", encoded_series)

        # df_copy = df_copy.drop(columns = ["smoking"], errors = "ignore")
                        
    return df_copy


# imputes missing values in "prepregnancy_bmi_self_or_record" with the median of the existing non-missing values
def impute_bmi_median(df):
    df_clean = df.copy()

    col_name = "prepregnancy_bmi_self_or_record"

    if col_name in df_clean.columns:
        # ensure the column is numeric; convert errors to NaN
        df_clean[col_name] = pd.to_numeric(df_clean[col_name], errors="coerce")
        
        bmi_median = df_clean[col_name].median()
        
        df_clean[col_name] = df_clean[col_name].fillna(bmi_median)
            
    return df_clean


# one-hot encodes the "race" and "ethnicity" columns of the clinical sheet, prefixes the new features, and places them directly after the "race_is_missing" column
def one_hot_encode_demographics(sheet):
    final_df = sheet.copy()

    final_df["race_white"] = 0
    final_df["race_black"] = 0
    final_df["race_asian"] = 0

    race_clean = final_df["race"].astype(str).str.strip()

    # map white
    white_mask = race_clean.str.contains("white", na = False)
    final_df.loc[white_mask, "race_white"] = 1

    # group black/african american
    black_mask = race_clean.str.contains("black|african|africian", na = False)
    final_df.loc[black_mask, "race_black"] = 1

    # group asian (including korean)
    asian_mask = race_clean.str.contains("asian|korean", na = False)
    final_df.loc[asian_mask, "race_asian"] = 1

    # filter out categories we handled manually or that count as missing data
    ignore_keywords = [
        "white",
        "black",
        "african american",
        "african",
        "africian",
        "asian",
        "korean",
        "declined",
        "unknown",
        "na",
        "nan",
        "unspecified",
        "missing"
    ]
    is_mapped_or_null = race_clean.apply(lambda x: pd.isna(x) or any(kw in str(x) for kw in ignore_keywords))
    remaining_mask = ~is_mapped_or_null

    new_cols = ["race_white", "race_black", "race_asian"]

    # one-hot encode any remaining values
    if remaining_mask.any():
        race_dummies = pd.get_dummies(
            race_clean.loc[remaining_mask], prefix = "race", dtype = int
        ).reindex(final_df.index, fill_value = 0)

        race_dummies.columns = race_dummies.columns.str.strip()

        # drop columns we mapped manually if they somehow slipped through as substrings
        cols_to_drop = [col for col in race_dummies.columns if any(r in col for r in ["white", "black", "african", "asian", "korean"])]
        race_dummies = race_dummies.drop(columns = cols_to_drop, errors = "ignore")
        
        new_cols.extend(race_dummies.columns.tolist())

        # combine back with the main dataframe slice
        final_df = pd.concat([final_df, race_dummies], axis = 1).fillna(0)

    # encode the hispanic/latino column
    eth_clean = final_df["ethnicity"].astype(str).str.strip()
    eth_clean = eth_clean.mask(final_df["ethnicity"].isnull())
    final_df["hispanic/latino"] = np.nan  # initialize with nan to track uncaught rows

    # map conditions, ensuring that each column you map has not been already mapped
    # set "hispanic/latino" to 0 wherever your missingness indicator column is flagged as 1
    if "eth_is_missing" in final_df.columns:
        final_df.loc[final_df["eth_is_missing"] == 1, "hispanic/latino"] = 0

    is_not_hispanic = eth_clean.str.contains("non|not", na=False) & (final_df["hispanic/latino"].isna())
    final_df.loc[is_not_hispanic, "hispanic/latino"] = 0

    is_hispanic = eth_clean.str.contains("hispanic|latino", na=False) & (final_df["hispanic/latino"].isna())
    final_df.loc[is_hispanic, "hispanic/latino"] = 1

    final_df["hispanic/latino"] = final_df["hispanic/latino"].fillna(0).astype(int)
    new_cols.append("hispanic/latino")

    for col in new_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0).astype(int)

    # determine where to insert the new columns (after the "race_is_missing" column)
    if "race_is_missing" in final_df.columns:
        insert_idx = final_df.columns.get_loc("race_is_missing") + 1
    else:
        insert_idx = final_df.columns.get_loc("race") + 1

    # ensure "hispanic/latino" is at the end of the list
    encoded_cols = [c for c in new_cols if c != "hispanic/latino"] + [
        "hispanic/latino"
    ]

    # insert one-hot encoded columns into the dataframe
    for col in reversed(encoded_cols):
        col_series = final_df.pop(col)
        final_df.insert(loc = insert_idx, column = col, value = col_series)

    final_df = final_df.drop(columns = ["race", "ethnicity"], errors = "ignore")

    return final_df


def main():
    sheet = load_sheet()

    sheet_cleaned = standardize_sheet(sheet)
    sheet_cleaned = fix_typos(sheet_cleaned)
    sheet_cleaned = filter_by_status(sheet_cleaned)
    sheet_cleaned = unmask_missing_data(sheet_cleaned)
    sheet_cleaned = add_missingness_indicators(sheet_cleaned)
    sheet_cleaned = encode_smoking_status(sheet_cleaned)
    sheet_cleaned = impute_bmi_median(sheet_cleaned)
    sheet_encoded = one_hot_encode_demographics(sheet_cleaned)

    if not sheet_encoded.empty:
        clinical_csv_path = "01_data_cleaning/processed_data/processed_clinical_data.csv"
        sheet_encoded.to_csv(clinical_csv_path, index = False)


if __name__ == "__main__":
    main()