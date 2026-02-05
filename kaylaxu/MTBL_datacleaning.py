### Data Cleaning and Quality Control for Pregnancy Deep Phenotyping Metabolomics Data
##### All data cleaning and quality control steps are outlined in the data/README.md
### Kayla Xu, Piekos Lab
### 01/28/2026

# set up environment
import pandas as pd
import numpy as np
import scipy.stats as sp
from scipy import ndimage as nd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
import sys

##############################################################
# DATA CLEANING OVERVIEW 
# 1. Convert improper values to standard missing value.
# 2. Separate QC and biological samples.
# 3. Calculate median absolute deviation (MAD) for the standard compound runs (01 and 02), then filter out biological samples with a standard compound expression more than 5*MAD away from median (i.e. threshold = median +- (5 x MAD)). Track any samples that fail this test in a log file.
# 4. Filter out biological samples with >50% missingness. Track any samples that fail this test in a log file.
# 5. Calculate relative standard deviation (RSD = SD/Mean * 100) of all compounds in the QC pools, split by batch. Remove any metabolites with an RSD > 30%.
# 6. Remove any metabolites with >20% missingness. Double-check that no group has significantly differential patterns of missingness for discard analyses.
# 7. Generate a non-batch corrected PCA plot.
# 8. Batch normalization using the biological replicates (better than quantile normalization):
#       1. Identify and log the number of batch replicates (samples that are present in both batches).
#       2. Calculate the ratio for each sample replicate pair (batch 2 value / batch 1 value) for each metabolite. Store in a list for each metabolite, making sure to exclude any outliers. (what counts as an outlier?)
#       3. Take the median ratio for each metabolite.
#       4. Calculate the correction factor = 1 / median ratio for each metabolite.
#       5. Apply the correction factors to all batch 2 samples
# 9. Generate a batch-corrected PCA plot in which the color of the sample dots match the batch they were in.
# 10. For the samples ran in both batches - average metabolic expression.
# 11. Perform log2 transformation
# 12. Perform median normalization
# 13. Combine the POS and NEG compounds:
#       1. Identify compounds present in both
#       2. Check the correlation between the POS and NEG values
#               1. If it is r >= 0.9, then average the two samples (or if one is missing, take the non-missing vaue)
#               2. If r < 0.9, keep both separately and append _POS or _NEG to the compound name respectively.
# 14. Perform 1/2 minimum imputation of missing data.
# 15. Handle all additional formatting considerations described above. 

# GLOBAL VARIABLES
RSD = 0.3
SAMPLE_MISSING = 0.5
MTBL_MISSING = 0.2

# converts improper or missing expression data to np.nan values
def convert_missing(x):
    try:
        val = float(x)
        if val == 0:
            return np.nan
        else:
            return val
    except:
        return np.nan

# replace inappropriate values with np.nan
def handle_missing(exp1, exp2):
    exp1 = exp1.map(convert_missing)
    exp2 = exp2.map(convert_missing)
    return exp1, exp2

# split expression data by batch and by samples/pooled, saving in exp_data dictionary
def split_exp(exp, batch, e, exp_data):
    is_pooled = ["Pooled" in s for s in exp.index]
    pooled = exp.iloc[is_pooled,:]
    pooled['batch'] = batch["batch"][is_pooled]

    is_sample = ["Pooled" not in s for s in exp.index]
    sample = exp.iloc[is_sample,:]
    sample['batch'] = batch["batch"][is_sample]

    unique_batches = batch["batch"].unique()
    for b in unique_batches:
        exp_data["Pooled_" + str(b) + "_" + e] = pooled[pooled['batch'] == b]
        exp_data["Samples_" + str(b) + "_" + e] = sample[sample['batch'] == b]

# calculate the median absolute deviation and identify samples that fail threshold
def mad_failed(c):
    mad = sp.median_abs_deviation(c)
    med = nd.median(c)
    return (c < med - 5*mad) | (c > med + 5*mad) 

# filter out samples with >5 median absolute deviation in internal controls (c1 and c2)
def controls_MAD(exp_data):
    for temp in exp_data.keys(): 
        if "Samples" in temp: # only filter in sample expression
            mask = mad_failed(exp_data[temp]["c1"]) | mad_failed(exp_data[temp]["c2"]) # mask of samples that failed the internal control for either c1 or c2
            for s in exp_data[temp].index[mask]:
                logging.warning("QC: Sample " + s + " in " + temp + " failed Median Absolute Deviation threshold test.")
                exp_data[temp] = exp_data[temp].drop(index=s)

# remove samples with >50% missingness
def sample_missingness(exp_data): 
    for temp in exp_data.keys(): 
        if "Samples" in temp: # only test in sample expression
            fail = exp_data[temp].isna().sum(axis=1)/len(exp_data[temp].index) > SAMPLE_MISSING
            for s in exp_data[temp].index[fail]:
                logging.warning("QC: Sample " + s + " in " + temp + " failed >50% missingness test.")
                exp_data[temp] = exp_data[temp].drop(index=s)

    

# helper function for data cleaning
def clean(pos_exp, pos_batch, pos_comp, neg_exp, neg_batch, neg_comp):
# 1. handle missing
    pos_exp, neg_exp = handle_missing(pos_exp, neg_exp)
# 2. split batches and pools
    exp_data = {}
    split_exp(pos_exp, pos_batch, "pos", exp_data) # naming format = [Pooled, Samples]_[batch]_[charge]
    split_exp(neg_exp, neg_batch, "neg", exp_data)
# 3. MAD test 
    controls_MAD(exp_data)
# 4. sample >50% missingness test
    sample_missingness(exp_data, "sample")
# 5. RSD test




# 6. metabolite >20% missingness test
# 7. generate PCA
# 8. normalize
    # 9. generate PCA
# 10. merge replicates
# 11, 12.log2 transformation + median normalization
# 13. combine pos and neg
# 14. 1/2 minium imputation of missing



# Given folder with both postive and negative metabolomic expression, batch info, and compound metadata
def main():
    dir_input = sys.argv[1] # e.g. /Users/kaylaxu/Desktop/data/clean_data/MTBL_placenta
    # initiate log file
    logging.basicConfig(
        filename='MTBL_cleaning.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'  # Use 'w' to overwrite the file each run, or 'a' to append
    )
    logging.info("Initializing metabolomics cleaning...")
    # get files
    pos_exp = pd.read_csv(dir_input + "/pos_expression.csv", index_col=0)
    pos_batch = pd.read_csv(dir_input + "/pos_batch.csv", index_col=0)
    pos_comp = pd.read_csv(dir_input + "/pos_compounds.csv", index_col=0)
    neg_exp = pd.read_csv(dir_input + "/neg_expression.csv", index_col=0)
    neg_batch = pd.read_csv(dir_input + "/neg_batch.csv", index_col=0)
    neg_comp = pd.read_csv(dir_input + "/neg_compounds.csv", index_col=0)
    # call cleaning functions
    clean(pos_exp, pos_batch, pos_comp, neg_exp, neg_batch, neg_comp)
    #close log file
    logging.shutdown()

if __name__ == "__main__":
    main()
