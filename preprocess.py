import pandas as pd

# load both sheets and returns both sheets
def load_sheets():
    sheet1 = pd.read_csv("dp3 3rd set gest age for Tony assessed final.xlsx - Sheet1.csv")
    sheet2 = pd.read_csv("DP3 slides Tony's analysis batches 1-2.xlsx - Sheet2.csv")
    return sheet1, sheet2

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
        "gest age del": "gestational age",
        "decual arteriopathy membrane roll/ basal plate/ both": "decidual arteriopathy membrane role/basal plate/both",
        "segmental avascular villi small/ intermediate/ large": "segmental avascular villi small/intermediate/large",
        "stem vessel obliteration / fibromuscular sclerosis": "stem vessel obliteration/fibromuscular sclerosis",
        "maternal inflammatory response stage / grade": "maternal inflammatory response stage/grade",
        "fetal inflammatory response stage / grade / location": "fetal inflammatory response stage/grade/location",
        "villitis of unknown etiology high / low grade focal / diffuse": "villitis of unknown etiology, high/low grade, focal/diffuse"
    })

    sheet2 = sheet2.rename(columns = {
        "dp3 participant #": "id",
        "placental infarctions": "placental infarction",
        "accelerated villous maturity": "accelerated villous maturation",
        "syncytial knots - (tony: considered with accelerated villous maturation)": "increased syncytial knots",
        "thrombosis": "vascular thrombosis",
        "villous stromal-vascular karyorrhexis": "villous stromal vascular karyorrhexis",
        "stem vessel obliteration/ fibromuscular sclerosis": "stem vessel obliteration/fibromuscular sclerosis",
        "delayed villous maturation focal/diffuse": "delayed villous maturation",
        "maternal inflammatory response stage,grade": "maternal inflammatory response stage/grade",
        "fetal inflammatory response stage,grade,location": "fetal inflammatory response stage/grade/location",
        "villitis of unknown etiology high/low grade, focal/diffuse": "villitis of unknown etiology, high/low grade, focal/diffuse",
        "massive perivillous fibrin deposition": "increased perivillous fibrin deposition"
    })

    return sheet1, sheet2

# drop patients with no slides and patients with all X's in the columns "slide a", "slide b", and "slide membrane roll"
def delete_patients(sheet):
    # drop patients with no slides
    sheet = sheet[sheet["not in file"] != "no slides"]

    # drop patients with all X's
    sheet = sheet[~((sheet["slide a"] == "X") & (sheet["slide b"] == "X") & (sheet["slide membrane roll"] == "X"))]

    return sheet

# delete the following columns: "slide a", "slide b", "slide membrane roll", "not in file"
def delete_columns(sheet):
    sheet = sheet.drop(columns = ["slide a", "slide b", "slide membrane roll", "not in file"])
    return sheet

# encode all columns from strings to 0, 1, 2, etc.
def encode(sheet):
    # encode yes/no to 1/0
    yes_no_mapping = {"yes": 1}
    yes_no_columns = [
        "placental infarction", "retroplacental hemorrhage", "distal villous hypoplasia focal/diffuse", "accelerated villous maturation", 
        "increased syncytial knots", "vascular thrombosis", "villous stromal vascular karyorrhexis", "vascular intramural fibrin deposition", 
        "stem vessel obliteration/fibromuscular sclerosis", "vascular ectasia", "delayed villous maturation", "fetal inflammatory response stage/grade/location", 
        "diffuse villous edema", "increased perivillous fibrin deposition", "chorangiosis", "placental hypoplasia"
    ]
    sheet[yes_no_columns] = sheet[yes_no_columns].apply(lambda x: x.map(yes_no_mapping))
    
    sheet["maternal inflammatory response stage/grade"] = sheet["maternal inflammatory response stage/grade"].map({
        "S1/G1": 1,
        "S2/G1": 2,
        "S3/G1": 3
    })

    sheet["villitis of unknown etiology, high/low grade, focal/diffuse"] = sheet["villitis of unknown etiology, high/low grade, focal/diffuse"].map({
        "yes/L": 1,
        "yes, LG, focal": 1,
        "yes/H": 2,
        "yes, HG, focal": 2
    })

    sheet["decidual arteriopathy membrane role/basal plate/both"] = sheet["decidual arteriopathy membrane role/basal plate/both"].map({
        "yes": 1,
        "yes (thrombosed)": 1,
        "yes, roll": 1
    })

    sheet["segmental avascular villi small/intermediate/large"] = sheet["segmental avascular villi small/intermediate/large"].map({
        "yes, small": 1,
        "yes/I": 2,
        "yes, large": 3
    })

    # fill all encoded columns with 0 as the corresponding N/A value
    numeric_cols = sheet.select_dtypes(include = "number").columns
    numeric_cols = numeric_cols.tolist()
    numeric_cols.remove("gestational age")
    sheet[numeric_cols] = sheet[numeric_cols].fillna(0).astype(int)

    return sheet

# print the total of each appropriate column (excluding text columns and gestational age) into a log file
def print_totals(sheet):
    log_path = "log.txt"
    numeric_cols = sheet.select_dtypes(include = "number").columns
    numeric_cols = numeric_cols.tolist()
    numeric_cols.remove("gestational age")

    with open(log_path, "w") as f:
        for col in numeric_cols:
            total = sheet[col].sum()
            f.write(f"{col}: {total}\n")

def main():
    sheet1, sheet2 = load_sheets()
    merged = merge_sheets(sheet1, sheet2)
    merged = delete_patients(merged)
    merged = delete_columns(merged)
    merged = encode(merged)

    # write sheet to an output file
    merged.to_csv("output.csv", index = False)

    print_totals(merged)

if __name__ == "__main__":
    main()