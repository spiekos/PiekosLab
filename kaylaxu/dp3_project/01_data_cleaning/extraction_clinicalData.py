import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# set up arguments
parser = argparse.ArgumentParser(prog="extraction_clinicalData",
                                 description="Module for extraction of relevant clinical data measurements in specified csv file")
parser.add_argument('-i', '--input', help="Path to excel sheet of DP3 metadata (i.e. 'dp3 master table v2.xlsx')")
parser.add_argument('-c', '--categorical_variables', help="List of categorical clinical variables of interest", default=["race", "ethnicity", "infant sex"])
parser.add_argument('-n', '--numeric_variables', help="List of numeric clinical variables of interest", default=["prepregnancy BMI self or record", "maternal age", "parity"])
args = parser.parse_args()


# get current directory of script
cwd = Path.cwd()

c_clinical = []
n_clinical = []

# read input
df = pd.read_excel(args.input, sheet_name = "variables of interest")

# extract columns from clinical data sheet that have a name in the first row 
# extract rows from clinical data sheet where the ID is a patient ID
df = df.loc[df["ID"].str.contains('^DP3'), ~df.columns.str.contains('^Unnamed')]

# rename ID to SubjectID for consistency
df = df.rename(columns={"ID": "SubjectID"})

# coerce all non-numeric values in numeric columns to NaN
for n in n_clinical:
    df[n] = pd.to_numeric(df[n], errors="coerce")


# since categories have to be hard coded, parser argument is more informational
# i.e. changing -c won't change anything about the code
df["infant sex"] = np.where(df["infant sex"].str.upper() == "NA", "MISSING", df["infant sex"].str.upper())

# reassign race
df["race"] = np.where(df["race"]  == "WHITE", "WHITE", 
            np.where(df["race"]  == "BLACK", "BLACK", 
            np.where(df["race"]  == "AFRICIAN AMERICAN", "BLACK", 
            np.where(df["race"]  == "na", "MISSING", "OTHER"))))

# reassign ethnicity
df["ethnicity"] = np.where(df["ethnicity"] == "NOT HISPANIC OR LATINO", "NONHISPANIC", 
                    np.where(df["ethnicity"] == "Non Hispanic", "NONHISPANIC", 
                    np.where(df["ethnicity"] == "na", "MISSING", "OTHER")))

# combine race and ethnicity 
df["race_ethnicity"] = np.where((df["ethnicity"] == "NONHISPANIC") & (df["race"] == "WHITE"), "NONHISPANIC_WHITE", 
                            np.where((df["ethnicity"] == "NONHISPANIC") & (df["race"] == "BLACK"), "NONHISPANIC_BLACK", 
                            np.where((df["ethnicity"] == "MISSING") & (df["race"] == "MISSING"), "MISSING", "OTHER"))) # only missing if both race and ethnicity are missing?

df.to_csv(f"{str(cwd)}/../data/processed/dp3_clinical_data.csv")