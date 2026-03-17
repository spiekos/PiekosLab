### Kayla Xu, Piekos Lab
### 03/16/2026

#### Outlier-Based Biomarker Discovery for Pregnancy Complications
# Objective: Identify blood, urine, or gut/vaginal microbiome analytes that show extreme deviation in pregnancy complications relative to term controls, 
# prioritizing these with presistent elevation across multiple timepoints for potential diagnostic biomarker development. 

import pandas as pd
from scipy import stats
import sys

TIMEPOINTS = ["A", "B", "C", "D", "E"]

# Study Design:
#   Cases: Pregnancy complications (n=77)
#   Control: Term deliveries (n=55)
#   Timepoints: Up to 5 collections per tissue per patient (some patients have as few as 2 due to preterm delivery)
#   Data types: Multiple omics platforms (proteomics, metabolomics, etc.)
# Data Status: Already preprocessed, QC'd, batch-corrected, normalized, filtered for missingness

# Step 1: Calculate Reference Distribution (per analyte)
#   for each tissue x timepoint x data type combination
# 1.1 Subset to term controls only
# 1.2 Calculate per-analyte statistics
#   Median
#   median avsolute deviation 
#       if control_MAD = 0, flag as "zero_variance" in log and exclude from outlier analysis
# 1.3 Document control statistics
#   Create reference table with columns as tissue, timepoint, datatype, analyte ID, control median, control_MAD
#   output: control_reference_statistics_<tissue>_<timepoint>_<datatype>.csv

# calculate median and MAD 
# return: dataframe with analyte_ID, tissue, datatype, timepoint, median, and MAD
def getStats(df, t, datatype, tissue):
    temp = pd.DataFrame(index=df.columns)
    temp["analyte_ID"] = df.columns
    temp["tissue"] = tissue
    temp["datatype"] = datatype
    temp["timepoint"] = t
    temp["control_median"] = df.median()
    temp["control_MAD"] = stats.median_abs_deviation(df)
    return temp

# combine batch-split files at timepoint t
def mergeBatches(batches, dir_input, t):
    temp = pd.DataFrame()
    for b in batches:
        try:
            samples = pd.read_csv(dir_input + "/Samples_" + str(b) + "_" + t + ".csv", index_col=0)
            temp = pd.concat([temp, samples])
        except:
            continue
    return temp

# extract and format control reference values
# return: dictionary with keys = timepoints and values = control reference median and MAD values
def controlRefStats(dir_input, dir_output, datatype, tissue, batches):
    controlAll = {} # dict to return with keys = timepoints and values = control reference median and MAD values
    for t in TIMEPOINTS:
        temp = mergeBatches(batches, dir_input, t)
        temp = temp.loc[temp["group"] == "Control"].drop(columns="group")
        controlAll[t] = getStats(temp, t, datatype, tissue)
        controlAll[t].to_csv(dir_output + "/control_reference_statistics_" + tissue + "_" + t + "_" + datatype + ".csv")
    return controlAll

# calculate MAD scores and return score matrix
def MADscores(dir_input, controlRef, metadata, batches):
    for t in TIMEPOINTS:
        temp = mergeBatches(batches, dir_input, t)
        group = temp["group"]
        temp = temp.drop(columns="group")
        # get median at t
        med = controlRef[t].loc[temp.index, "control_median"]
        # get MAD at t
        mad = controlRef[t].loc[temp.index, "control_MAD"]



    
# primary wrapper function for Outlier Analysis
def OutlierAnalysis(dir_input, dir_output, datatype, tissue, metadata, batches):
    controlRef = controlRefStats(dir_input, dir_output, datatype, tissue, batches)
    #scoreMatrix = MADscores(dir_input, controlRef, metadata, batches)


def main():
    dir_input = sys.argv[1] # e.g. /Users/kaylaxu/Desktop/data/clean_data/MTBL_plasma
    dir_output = sys.argv[2] # e.g. /Users/kaylaxu/Desktop/data/MAD_analyses
    metadata = sys.argv[3] # e.g. /Users/kaylaxu/Desktop/data/raw_data/dp3 master table v2.xlsx
    batches = pd.read_csv(dir_input + "/pos_batch.csv")["batch"].unique().tolist()
    if "MTBL" in dir_input:
        datatype = "MTBL"
    else:
        datatype = "LIPD"
    if "plasma" in dir_input:
        tissue = "plasma"
    else:
        tissue = "placenta"
    OutlierAnalysis(dir_input, dir_output, datatype, tissue, metadata, batches)
    

if __name__ == "__main__":
    main()
