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
from itertools import compress
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
# 12. Combine the POS and NEG compounds:
#       1. Identify compounds present in both
#       2. Check the correlation between the POS and NEG values
#               1. If it is r >= 0.9, then average the two samples (or if one is missing, take the non-missing vaue)
#               2. If r < 0.9, keep both separately and append _POS or _NEG to the compound name respectively.
# 13. Perform 1/2 minimum imputation of missing data.
# 14. Handle all additional formatting considerations described above. 

# GLOBAL VARIABLES
RSD = 30
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


#### Merge these two functions
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
################################

#### Clean up this merged function + by group check
# mode rsd = calculate RSD (sd/mean * 100) in QC pools and remove metabolites with RSD > 30%
# mode >20% missing = remove metabolites with >20% missing
# check missing is not disproportionate between groups
def rsd_missing(exp_data, mode):
    if mode == "RSD":
        x = "Pooled"
    elif mode == ">20% missing":
        x = "Sample"
    else: 
        logging.warning("Invalid mode in rsd_missing()...")
        return
    d = list(compress(list(exp_data.keys()), [mode in k for k in exp_data.keys()])) # get keys of all pooled data frames
    df = pd.DataFrame() # initialize empty dataframe for RSD calculations
    for k in d: # calculate RSD
        if mode == "RSD":
            r = (exp_data[k].std()/exp_data[k].mean())*100 
            df[k] = r
        else:
            r = exp_data[k].isna().sum()
            df[k] = r
    if mode == "RSD":
        remove = df.index[(df > RSD).sum(axis=1) > 0] # get mtbl that failed the RSD test in at least one of the pools
    else:
        remove = df.index[df.sum(axis=1) / len(df) > MTBL_MISSING]
    for m in remove: # for every failed mtbl, log failure and drop from all data frames
        logging.warning("QC: Compound " + str(m) + " failed the " + mode + " check " + 10*'\t' + "RSD values: " + list(df.loc[m]))
        for k in exp_data.keys:
            exp_data[k] = exp_data[k].drop(columns=m)
##################################

# generate pca from expression data
def generate_pca(exp_data, title):
    df = pd.DataFrame()
    for k in exp_data.keys():
        if "Sample" in k:
            df = pd.concat(df, exp_data[k])
    batch = df["batch"]
    df = df.drop(["batch"], axis=1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['batch'] = list(batch)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PC1', 
        y='PC2', 
        hue='batch', 
        data=pca_df, 
        palette='viridis', 
        s=100,      # Marker size
        alpha=0.8   # Transparency
    )
    # Add titles and labels
    plt.title(title, fontsize=15)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance)', fontsize=12)
    plt.grid(True)
    plt.show()


# use biolgoical replicates to conduct batch normalization
def normalization(exp_data):
    



    batch1 = exp_data["Samples_32425"].drop(['batch'], axis=1)
batch2 = exp_data["Samples_62323"].drop(['batch'], axis=1)
replicates = list(set(batch1.index) & set(batch2.index))
logging.info("Initializing batch normalization with " + str(len(replicates)) + " biological replicates.")
rep1 = batch1.loc[replicates,:]
rep2 = batch2.loc[replicates,:]
ratios = rep2 / rep1
mad = sp.median_abs_deviation(ratios)
med = ratios.median()
upper = med + 5*mad
lower = med - 5*mad
med_ratios = {}
# calculate median ratio, excluding outilers from this calculation (should outliers be removed completely?)
for m in ratios.columns:
    temp = ratios.loc[list(ratios[m] < upper[m]) and list(ratios[m] > lower[m]), m]
    med_ratios[m] = temp.median()
correction_factors = {k: 1/v for k, v in med_ratios.items()}
len(correction_factors)
for m in batch2.columns:
    exp_data["Samples_62323"][m] = batch2[m]*correction_factors[m]

# average replicate expression values
def merge_rep():
    pass

# do log2 transformation on expression data
def log2_transform():
    pass

# combine pos and neg expression in same file
    # if r >=0.9, average
    # else, keep both (_POS and _NEG)
def combine_pos_neg():
    pass

# perform 1/2 minimum imputation
def min_imputation(a=0.5):
    pass

# final formatting of expression files
def formatting():
    pass
    

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
# 5. RSD test (consider metabolite as missing), check there's not one group thats dispproportionally missing 
    rsd_missing(exp_data, "RSD")
# 6. metabolite >20% missingness test, check there's not one group thats dispproportionally missing 
    # split by group, then check this
    rsd_missing(exp_data, ">20% missing")
# 7. generate PCA
    generate_pca(exp_data, "Unnormalized MTBL Expression")
# 8. normalize
    normalization()
    # 9. generate PCA
    generate_pca(exp_data, "Batch Normalized MTBL Expression")
# 10. merge replicates
    merge_rep()
# 11, 12.log2 transformation 
    log2_transform()
# 13. combine pos and neg
    combine_pos_neg()
# 14. 1/2 minium imputation of missing
    min_imputation()
# do all formatting steps
    # add group columns
    # replace n1, n2, n3... with metabolite names
    # split files by timepoint
    formatting()



# Given folder with both postive and negative metabolomic expression, batch info, and compound metadata
def main():
    dir_input = sys.argv[1] # e.g. /Users/kaylaxu/Desktop/data/clean_data/MTBL_placenta

    logging.basicConfig( # initiate log file
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
