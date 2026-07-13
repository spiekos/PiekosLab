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


# exclude patients with "loss to follow-up", "withdraw", or "excluded" (including partial matches) in their "status" column
def filter_by_status(sheet):
    sheet_copy = sheet.copy()

    if "status" in sheet_copy.columns:
        status_clean = sheet_copy["status"].astype(str).str.strip().str.upper()
        
        # build masks for exclusions
        is_excluded = status_clean.str.contains("EXCLUDED", na=False)
        is_loss_to_fu = status_clean.str.contains("LOSS TO FOLLOW-UP|LOSS TO FOLLOW UP", na=False)
        is_withdraw = status_clean.str.contains("WITHDRAW", na=False)
        
        # combine masks to find records we want to keep
        drop_mask = is_excluded | is_loss_to_fu | is_withdraw
        keep_mask = ~drop_mask
        
        sheet_copy = sheet_copy[keep_mask].reset_index(drop=True)
                
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

    race_clean = rest_of_df["race"].astype(str).str.strip().str.upper()

    # map White
    white_mask = race_clean.str.contains("WHITE", na = False)
    rest_of_df.loc[white_mask, "race_WHITE"] = 1

    # group Black/African American
    black_mask = race_clean.str.contains("BLACK|AFRICAN", na = False)
    rest_of_df.loc[black_mask, "race_BLACK"] = 1

    # group Asian (including Korean)
    asian_mask = race_clean.str.contains("ASIAN|KOREAN", na = False)
    rest_of_df.loc[asian_mask, "race_ASIAN"] = 1

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

    # reconstruct the complete dataframe
    # add empty columns to row_2 with 0 values so it matches shape during merge
    new_cols = [col for col in rest_of_df.columns if col not in row_2.columns]
    for col in new_cols:
        row_2[col] = 0
        rest_of_df[col] = rest_of_df[col].fillna(0).astype(int)
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

    final_df = final_df.drop(columns = ["race", "ethnicity"], errors = "ignore")

    return final_df


# creates and returns a table tracking total counts for each race/ethnicity
def get_race_counts(sheet):
    patient_df = sheet.iloc[1:]
    total_patients = len(patient_df)

    race_cols = sorted([col for col in patient_df.columns if col.startswith("race_") and col != "race_is_missing"])
    eth_cols = ["HISPANIC/LATINO"]

    lines = []
    
    def build_section(title, columns):
        section_lines = []
        section_lines.append(f"## {title}")
        section_lines.append("-" * 55)
        section_lines.append(f"{'Feature Name':<40} | {'Count':<6} | {'Percentage':<8}")
        section_lines.append("-" * 55)
        for col in columns:
            if col in patient_df.columns:
                count = (patient_df[col] == 1).sum()
                pct = (count / total_patients) * 100 if total_patients > 0 else 0
                section_lines.append(f"{col:<40} | {count:<6} | {pct:>6.1f}%")
        section_lines.append("-" * 55 + "\n")
        return section_lines

    # Build individual sections
    lines.extend(build_section("RACE CATEGORIES", race_cols))
    
    if "race_is_missing" in patient_df.columns:
        lines.extend(build_section("DATA QUALITY / MISSINGNESS", ["race_is_missing"]))
        
    lines.extend(build_section("ETHNICITY HISTOGRAM", eth_cols))

    return lines, total_patients


# prints outputs to a log file
def print_log(race_table, total_patients):
    log_path = "02_exploratory_analysis/outputs/clinical_sheet_race_counts.txt"

    race_table_str = "\n".join(race_table)

    with open(log_path, "w") as f:
        f.write("Clinical Sheet Demographics Summary Report\n")
        f.write(f"Total Patient Records Analyzed: {total_patients}")
        f.write("\n\n")
        f.write(race_table_str)


def main():
    sheet = load_sheet()

    sheet_cleaned = fix_typos(sheet)
    sheet_cleaned = filter_by_status(sheet_cleaned)
    sheet_cleaned = add_missingness_indicators(sheet_cleaned)
    sheet_encoded = one_hot_encode_demographics(sheet_cleaned)

    race_table, total_patients = get_race_counts(sheet_encoded)

    print_log(race_table, total_patients)

    if not sheet_encoded.empty:
        clinical_csv_path = "01_data_cleaning/processed_data/processed_clinical_data.csv"
        sheet_encoded.to_csv(clinical_csv_path, index = False)


if __name__ == "__main__":
    main()