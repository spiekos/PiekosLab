### Metabolomics data extraction from xlsx files
### Piekos Lab, Kayla Xu
### 01/26/2026

# set up environemtn
import pandas as pd
import numpy as np
import sys

# Metabolomics Files:
### 050725_Sadovsky DP3 Placenta Polar Untargeted_ALL copy.xlsx
### 050725_Sadovsky DP3 Plasma Polar Untargeted_ALL copy.xlsx

# extract batch info
def get_batch(df):
    is_sample = df.columns.notna()
    temp = df.iloc[0:1,is_sample]
    temp = temp.set_index("Sample ID").transpose()
    temp.columns = ["batch"]
    temp["batch"] = [s.split(": ")[1].split("_")[0] for s in temp["batch"]]
    return temp

# extract compound metadata
def get_compounds(df):
    not_sample = df.columns.isna()
    temp = df.iloc[:, not_sample]
    temp.columns = temp.iloc[0,:]
    temp = temp.drop(temp.index[0])
    temp.iloc[0,0] = "01" # two moleclues not in the export order, positive controls
    temp.iloc[1,0] = "02" # slightly different for each file
    return temp

# extract expression data
def get_expression(df, ids):
    is_sample = df.columns.notna()
    temp = df.iloc[:, is_sample].drop(columns="Sample ID").drop(df.index[0])
    temp.index = ids
    temp = temp.transpose()
    return temp


# call all csv generating function
def generate_files(df, file_output, e):
    get_batch(df).to_csv(file_output + "/" + e +"_batch.csv")
    comp = get_compounds(df)
    comp.to_csv(file_output + "/" + e + "_compounds.csv")
    get_expression(df, comp.iloc[:,0]).to_csv(file_output + "/" + e + "_expression.csv")


# helper function
def extract_data(file_input, file_output):
    file_pos = pd.read_excel(file_input, sheet_name="POS Compounds",header=None).dropna(how='all')
    file_neg = pd.read_excel(file_input, sheet_name="NEG Compounds",header=None).dropna(how='all')
    file_pos.columns = file_pos.iloc[0,:]
    file_neg.columns = file_neg.iloc[0,:]
    file_pos = file_pos.drop(file_pos.index[0])
    file_neg = file_neg.drop(file_neg.index[0])

    #generate files
    generate_files(file_pos, file_output, "pos")
    generate_files(file_neg, file_output, "neg")

def main():
    file_input = sys.argv[1]
    file_output = sys.argv[2]
    # read in positive and negative compound sheets separately
    extract_data(file_input, file_output)

if __name__ == "__main__":
    main()
