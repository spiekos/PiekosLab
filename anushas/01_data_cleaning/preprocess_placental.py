import pandas as pd
import numpy as np


# load both sheets and returns both sheets
def load_sheets():
    sheet1 = pd.read_csv("00_raw_data/dp3 3rd set gest age for Tony assessed final.xlsx - Sheet1.csv")
    sheet2 = pd.read_csv("00_raw_data/DP3 slides Tony's analysis batches 1-2.xlsx - Sheet2.csv")
    return sheet1, sheet2


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


# merges both sheets and returns the newly merged sheet
def merge_sheets(sheet1, sheet2):
    # make sure all column names are in all lowercase for uniformity
    sheet1.columns = sheet1.columns.str.lower()
    sheet2.columns = sheet2.columns.str.lower()

    sheet1, sheet2 = rename_columns(sheet1, sheet2)

    # merge both sheets
    merged = pd.concat([sheet1, sheet2], ignore_index = True)

    return merged


# rename columns of both sheets so that column names are consistent across sheets
def rename_columns(sheet1, sheet2):
    sheet1 = sheet1.rename(columns = {
    "gest_age_del": "gestational_age_at_delivery",
    "decual_arteriopathy_membrane_roll/_basal_plate/_both": "decidual_arteriopathy_membrane_role/basal_plate/both",
    "segmental_avascular_villi_small/_intermediate/_large": "segmental_avascular_villi_small/intermediate/large",
    "stem_vessel_obliteration_/_fibromuscular_sclerosis": "stem_vessel_obliteration/fibromuscular_sclerosis",
    "maternal_inflammatory_response_stage_/_grade": "maternal_inflammatory_response_stage/grade",
    "fetal_inflammatory_response_stage_/_grade_/_location": "fetal_inflammatory_response_stage/grade/location",
    "villitis_of_unknown_etiology_high_/_low_grade_focal_/_diffuse": "villitis_of_unknown_etiology,_high/low_grade,_focal/diffuse"
    })

    sheet2 = sheet2.rename(columns = {
    "dp3_participant_#": "id",
    "notes": "mr",
    "gestational_age": "gestational_age_at_delivery",
    "placental_infarctions": "placental_infarction",
    "accelerated_villous_maturity": "accelerated_villous_maturation",
    "syncytial_knots_-_(tony:_considered_with_accelerated_villous_maturation)": "increased_syncytial_knots",
    "thrombosis": "vascular_thrombosis",
    "villous_stromal-vascular_karyorrhexis": "villous_stromal_vascular_karyorrhexis",
    "stem_vessel_obliteration/_fibromuscular_sclerosis": "stem_vessel_obliteration/fibromuscular_sclerosis",
    "delayed_villous_maturation_focal/diffuse": "delayed_villous_maturation",
    "maternal_inflammatory_response_stage,grade": "maternal_inflammatory_response_stage/grade",
    "fetal_inflammatory_response_stage,grade,location": "fetal_inflammatory_response_stage/grade/location",
    "villitis_of_unknown_etiology_high/low_grade,_focal/diffuse": "villitis_of_unknown_etiology,_high/low_grade,_focal/diffuse",
    "massive_perivillous_fibrin_deposition": "increased_perivillous_fibrin_deposition"
    })

    return sheet1, sheet2


# drop patients with no slides and patients with all x's in the columns "slide a", "slide b", and "slide membrane roll"
def delete_patients(sheet):
    # drop patients with no slides
    sheet = sheet[sheet["not_in_file"] != "no_slides"]

    # drop patients with all x's
    sheet = sheet[~((sheet["slide_a"] == "x") & (sheet["slide_b"] == "x") & (sheet["slide_membrane_roll"] == "x"))]

    return sheet


# delete the following unnecessary columns: "slide a", "slide b", "slide membrane roll", "not in file", "mr"
# delete all empty columns, i.e. columns representing histopathology that is not present in any patient
def delete_columns(sheet):
    sheet = sheet.drop(columns = [
        "slide_a", "slide_b", "slide_membrane_roll", "not_in_file", "mr",
        "retroplacental_hemorrhage", "vascular_thrombosis", "villous_stromal_vascular_karyorrhexis", "vascular_intramural_fibrin_deposition", 
        "stem_vessel_obliteration/fibromuscular_sclerosis", "vascular_ectasia", "fetal_inflammatory_response_stage/grade/location", "diffuse_villous_edema", 
        "placental_hypoplasia"
])
    return sheet


# condition_col represents the column that we are checking the value of (true/false)
# change_col represents the column that will be changed by this function. for each row in the sheet, if the cell in condition_col is false (has a value of 0),
# the cell in change_col will be changed to "na". otherwise, it will remain unchanged.
def fillna(sheet, condition_col, change_col):
    sheet.loc[sheet[condition_col] == 0, change_col] = np.nan

    return sheet


# encode all columns from strings to 0, 1, 2, etc.
def encode(sheet):
    # encode yes/no to 1/0
    yes_no_mapping = {"yes": 1}
    yes_no_columns = [
        "placental_infarction", "distal_villous_hypoplasia_focal/diffuse", "accelerated_villous_maturation", "increased_syncytial_knots", 
        "delayed_villous_maturation", "increased_perivillous_fibrin_deposition", "chorangiosis"
    ]
    sheet[yes_no_columns] = sheet[yes_no_columns].apply(lambda x: x.map(yes_no_mapping))
    
    sheet["maternal_inflammatory_response_stage/grade"] = sheet["maternal_inflammatory_response_stage/grade"].map({
        "s1/g1": 1,
        "s2/g1": 2,
        "s3/g1": 3
    })

    sheet["villitis_of_unknown_etiology,_high/low_grade,_focal/diffuse"] = sheet["villitis_of_unknown_etiology,_high/low_grade,_focal/diffuse"].map({
        "yes/l": 1,
        "yes,_lg,_focal": 1,
        "yes/h": 2,
        "yes,_hg,_focal": 2
    })

    sheet["decidual_arteriopathy_membrane_role/basal_plate/both"] = sheet["decidual_arteriopathy_membrane_role/basal_plate/both"].map({
        "yes": 1,
        "yes_(thrombosed)": 1,
        "yes,_roll": 1
    })

    sheet["segmental_avascular_villi_small/intermediate/large"] = sheet["segmental_avascular_villi_small/intermediate/large"].map({
        "yes,_small": 1,
        "yes/i": 2,
        "yes,_large": 3
    })

    sheet["gestational_age_at_delivery"] = pd.to_numeric(sheet["gestational_age_at_delivery"], errors = "coerce")

    # fill all encoded columns with 0 as the corresponding n/a value
    numeric_cols = sheet.select_dtypes(include = "number").columns.tolist()
    numeric_cols.remove("gestational_age_at_delivery")
    sheet[numeric_cols] = sheet[numeric_cols].fillna(0).astype(int)

    return sheet


# print the total of each appropriate column (excluding text columns and gestational age) into a log file
def print_totals(sheet):
    log_path = "02_exploratory_analysis/outputs/sum_placental_histo_features.txt"
    numeric_cols = sheet.select_dtypes(include = "number").columns
    numeric_cols = numeric_cols.tolist()

    with open(log_path, "w") as f:
        f.write("This file contains the sum of the values in each column corresponding to a placental histopathology feature.\n\n")
        f.write("Note that the following columns were dropped from the dataset because these features were not present in any patient:\n")
        f.write("retroplacental hemorrhage, vascular thrombosis, villous stromal vascular karyorrhexis, vascular intramural fibrin deposition,\n")
        f.write("stem vessel obliteration/fibromuscular sclerosis, vascular ectasia, fetal inflammatory response stage/grade/location, diffuse villous edema,\n")
        f.write("and placental hypoplasia.\n\n")

        for col in numeric_cols:
            total = sheet[col].sum()
            if total != 0:
                f.write(f"{col}: {total}\n")


def main():
    sheet1, sheet2 = load_sheets()
    sheet1 = standardize_sheet(sheet1)
    sheet2 = standardize_sheet(sheet2)
    merged = merge_sheets(sheet1, sheet2)
    merged = delete_patients(merged)
    merged = delete_columns(merged)
    merged = encode(merged)

    print_totals(merged)

    # write sheet to an output file
    merged.to_csv("01_data_cleaning/processed_data/processed_placental_data.csv", index = False)


if __name__ == "__main__":
    main()