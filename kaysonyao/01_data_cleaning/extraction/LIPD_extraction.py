### Lipidomics data extraction from xlsx files
### Piekos Lab, Kayla Xu
### 01/26/2026

# set up environment
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Lipidomics Files:
### 060525_Sadovsky Placenta Lipids Untargeted_ALL.xlsx
### 072925 Sadovsky Plasma Lipids Untargeted_ALL copy.xlsx

# extract batch info
def get_batch(df):
    is_sample = df.columns.notna()
    temp = df.iloc[0:1,is_sample]
    temp = temp.set_index("Sample ID").transpose()
    temp.columns = ["batch"]
    try:
        temp["batch"] = [s.split(": ")[1].split("_")[0] for s in temp["batch"]]
    except:
        temp["batch"] = [s.split("_")[0] for s in temp["batch"]]
    temp.index = temp.index.rename("Sample_ID")
    return temp

# extract compound metadata
def get_compounds(df):
    not_sample = df.columns.isna()
    temp = df.iloc[:, not_sample]
    temp.columns = temp.iloc[0,:]
    temp = temp.drop(temp.index[0])
    temp.index = temp.index.rename("Export Order")
    return temp

# extract expression data
def get_expression(df, comp):
    is_sample = df.columns.notna()
    temp = df.iloc[:, is_sample].drop(columns="Sample ID").drop(df.index[0])
    #temp.index = comp["Name"].fillna(comp.index.to_series())
    temp.index = comp.index
    temp = temp.transpose()
    temp.index = temp.index.rename("compound")
    return temp

# call all csv generating function
def generate_files(df, file_output, polarity):
    get_batch(df).to_csv(file_output / f"{polarity}_batch.csv")

    comp = get_compounds(df)
    comp.to_csv(file_output / f"{polarity}_compounds.csv")

    get_expression(df, comp).to_csv(file_output / f"{polarity}_expression.csv"
    )

# clean header of csv
def clean_df(df):
    df = df.copy()
    df.columns = df.iloc[0,:]
    df = df.iloc[1:,:]
    i = 1
    while i < len(df) and pd.isna(df.iloc[i, 0]):
        df.iat[i, 0] = f"c{i}"  # placeholder names for experimental controls
        i += 1
    df.index = df.iloc[:,0]
    df = df.iloc[:, 1:]
    return df

# helper function
def extract_data(file_input, file_output, tissue):
    sheet_names = {
        "plasma": {
            "pos": "Plasma POS Lipids",
            "neg": "Plasma NEG Lipids",
        },
        "placenta": {
            "pos": "POS Lipids",
            "neg": "NEG Lipids",
        },
    }

    file_pos = pd.read_excel(
        file_input,
        sheet_name=sheet_names[tissue]["pos"],
        header=None,
    ).dropna(how="all")

    file_neg = pd.read_excel(
        file_input,
        sheet_name=sheet_names[tissue]["neg"],
        header=None,
    ).dropna(how="all")

    file_pos = file_pos.iloc[1:, :].copy()
    file_neg = file_neg.iloc[1:, :].copy()
    # remove empty rows and set index/columns
    file_pos = clean_df(file_pos)
    file_neg = clean_df(file_neg)

    #generate files
    generate_files(file_pos, file_output, "pos")
    generate_files(file_neg, file_output, "neg")


def main():
    root = Path(__file__).resolve().parents[2]
    # Run example: python LIPD_extraction.py plasma "your_workbook.xlsx"
    tissue = sys.argv[1].lower()
    input_filename = sys.argv[2]

    # Raw XLSX workbook: <root>/data/raw/original/LIPD/<tissue>/<input_filename>
    file_input = (root/ "data"/ "raw"/ "original"/ "LIPD"/ tissue/ input_filename)

    # Extracted CSV output directory:
    # <root>/data/raw/extracted/LIPD/<tissue>/
    file_output = (root/ "data"/ "raw"/ "extracted"/ "LIPD"/ tissue)
    file_output.mkdir(parents=True, exist_ok=True)

    extract_data(str(file_input), file_output, tissue)

if __name__ == "__main__":
    main()