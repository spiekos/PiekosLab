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

# extract area information
def get_area(df):
    is_sample = df.columns.notna()
    temp = df.iloc[0:1,is_sample]
    temp = temp.set_index("Sample ID").transpose()
    temp.columns = ["Area"]
    return temp

# extract compound metadata
def get_compounds(df):
    not_sample = df.columns.isna()
    temp = df.iloc[:, not_sample]
    temp.columns = temp.iloc[0,:]
    temp = temp.drop(temp.index[0])
    temp.iloc[0,0] = "01" # two moleclues not in the export order
    temp.iloc[1,0] = "02" # slightly different for each file
    return temp



# extract expression data
def get_expression(df, ids):
    is_sample = df.columns.notna()
    temp = df.iloc[:, is_sample].drop(columns="Sample ID").drop(df.index[0])
    temp.index = ids
    temp = temp.transpose()
    return temp


# helper function
def extract_data(file_input, file_output):
    #file = open(file_input, mode="r")
    file = file_input
    file_pos = pd.read_excel(file, sheet_name="POS Compounds",header=None).dropna(how='all')
    file_neg = pd.read_excel(file, sheet_name="NEG Compounds",header=None).dropna(how='all')
    file_pos.columns = file_pos.iloc[0,:]
    file_neg.columns = file_neg.iloc[0,:]
    file_pos = file_pos.drop(file_pos.index[0])
    file_neg = file_neg.drop(file_neg.index[0])

    #extract positive compound files
    get_area(file_pos).to_csv(file_output + "/pos_sampleArea.csv")
    pos_comp = get_compounds(file_pos)
    pos_comp.to_csv(file_output + "/pos_compounds.csv")
    get_expression(file_pos, pos_comp.iloc[:,0]).to_csv(file_output + "/pos_expression.csv")

    #extract negative compound files
    get_area(file_neg).to_csv(file_output + "/neg_sampleArea.csv")
    neg_comp = get_compounds(file_neg)
    neg_comp.to_csv(file_output + "/neg_compounds.csv")
    get_expression(file_neg, neg_comp.iloc[:,0]).to_csv(file_output + "/neg_expression.csv")


def main():
    file_input = sys.argv[1]
    file_output = sys.argv[2]
    # read in positive and negative compound sheets separately
    extract_data(file_input, file_output)

if __name__ == "__main__":
    main()
