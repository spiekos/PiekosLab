### Metabolomics data extraction from xlsx files
### Piekos Lab, Kayla Xu
### 01/26/2026

# set up environment
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
def generate_files(df, file_output, e):
    get_batch(df).to_csv(file_output + "/" + e +"_batch.csv")
    comp = get_compounds(df)
    comp.to_csv(file_output + "/" + e + "_compounds.csv")
    get_expression(df, comp).to_csv(file_output + "/" + e + "_expression.csv")

# clean header of csv
def clean_df(df):
    df.columns = df.iloc[0,:]
    df = df.iloc[1:,:]
    i = 1
    while pd.isna(df.iloc[i,0]):
        df.iloc[i,0] = "c" + str(i) # placeholder names for experimental controls
        i += 1
    df.index = df.iloc[:,0]
    df = df.iloc[:, 1:]
    return df

# helper function
def extract_data(file_input, file_output):
    #file = open(file_input, mode="r")
    file = file_input
    try:
        file_pos = pd.read_excel(file, sheet_name="POS Compounds",header=None).dropna(how='all')
        file_neg = pd.read_excel(file, sheet_name="NEG Compounds",header=None).dropna(how='all')
    except:
        if "Plasma" in file:
            file_pos = pd.read_excel(file, sheet_name="Plasma POS Lipids",header=None).dropna(how='all').iloc[1:,:]
            file_neg = pd.read_excel(file, sheet_name="Plasma NEG Lipids",header=None).dropna(how='all').iloc[1:,:]
        else:
            file_pos = pd.read_excel(file, sheet_name="POS Lipids",header=None).dropna(how='all').iloc[1:,:]
            file_neg = pd.read_excel(file, sheet_name="NEG Lipids",header=None).dropna(how='all').iloc[1:,:]

    # remove empty rows and set index/columns
    file_pos = clean_df(file_pos)
    file_neg = clean_df(file_neg)

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
