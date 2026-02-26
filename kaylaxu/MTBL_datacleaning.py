### Data Cleaning and Quality Control for Pregnancy Deep Phenotyping Metabolomics Data
##### All data cleaning and quality control steps are outlined in the data/README.md
### Kayla Xu, Piekos Lab
### 01/28/2026

# set up environment
import pandas as pd
import numpy as np
import scipy.stats as sp
from scipy import ndimage as nd
from scipy.stats import iqr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import re
import logging
import sys
import warnings
warnings.filterwarnings("ignore")

##############################################################
# DATA CLEANING OVERVIEW 
# 1. Convert improper values to standard missing value.
# 1-1. Calculate quality score for handling multiple parent metabolites (duplicates)
    # QS = peak rating qc (Max) + RSD QC Areas [%] + MS2 + Column Area (Max.) + mzCloud Best Match confidence
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
TIMEPOINTS = ["A", "B", "C", "D", "E"]

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

# calculate quality scores (peak quality + rsd + ms2 + signal intensity + annotation confidence)
def qs(comp):
    # get peak rating qc
    peak = pd.Series([10 if x >= 7.0 else 7 if x >= 5 else 4 if x >= 3 else 1 for x in comp['Peak Rating (Max.)']])
    # get rsd qc areas
    try:
        rsd = pd.Series([10 if x < 10 else 8 if x < 15 else 6 if x < 20 else 4 if x < 25 else 2 if x < 30 else 0 for x in comp["RSD QC Areas [%]"]])
    except:
        rsd = 0
    # get ms2
    ms2 = pd.Series([10 if x == "DDA for preferred ion" else 6 if x == "DDA for other ion" else 4 if x == "DDA available" else 0 for x in comp["MS2"]])
    # get signal intensity (area max)
    scaler = MinMaxScaler(feature_range=(0, 10))
    signal = pd.Series((scaler.fit_transform(comp[["Area (Max.)"]])).flatten())
    # annotation confidence (mzCloud Best Match Confidence)
    temp = (comp.loc[:, ['Annot. Source: Predicted Compositions', 'Annot. Source: mzCloud Search', 'Annot. Source: mzVault Search', 'Annot. Source: Metabolika Search', 'Annot. Source: ChemSpider Search','Annot. Source: MassList Search']])
    full = (temp == "Full match").sum(axis=1)
    notTop = (temp == "Not the top hit").sum(axis=1)
    partial = (temp == "Partial match").sum(axis =1)
    mzCloud = comp["mzCloud Best Match Confidence"]
    ac = pd.Series([10 if mzCloud.iloc[i] >= 90 else 9 if mzCloud.iloc[i] >=80 else 8 if mzCloud.iloc[i] >= 70 else 0 if mzCloud.iloc[i] < 70 else 10 if full.iloc[i] == 6 else 9 if full.iloc[i] == 5 else 8 if full.iloc[i] == 4 else 7 if full.iloc[i] == 3 else 6 if full.iloc[i] == 2 else 5 if full.iloc[i] == 1 else 4 if notTop.iloc[i] >= 1 else 3 if partial.iloc[i] >= 3 else 2 if partial.iloc[i] == 2 else 1 if partial.iloc[i] == 1 else 0 for i in range(len(mzCloud))])
    return list(peak + rsd + ms2 + signal + ac)

# remove duplicate metabolites based on the quality scores
def remove_duplicates(exp, comp):
    comp["Name"] = comp["Name"].fillna(comp.index.to_series())
    comp["Name"] = np.where(comp["Name"] == "Not named", comp.index.to_series(), comp["Name"])    
    comp = comp.sort_values("quality_score", ascending=False)
    dup = comp["Name"].loc[comp.duplicated(subset = ["Name"])]
    for m in dup:
        logging.info("Multiplet parent metabolite handling: " + m + " is a duplicate of another metabolite with a higher quality score.")
    comp = comp.drop_duplicates(subset=["Name"]).sort_index()
    mask = [x in comp.index for x in exp.columns]
    exp = exp.loc[:, mask]
    comp = comp.loc[exp.columns,:]
    exp.columns = comp["Name"]
    return exp

# split expression data by batch and by samples/pooled, saving in exp_data dictionary
def split_exp(exp, batch, e, exp_data, unique_batches):
    is_pooled = ["Pooled" in s for s in exp.index]
    pooled = exp.iloc[is_pooled,:]
    pooled['batch'] = batch["batch"][is_pooled]
    
    is_sample = ["Pooled" not in s for s in exp.index]
    sample = exp.iloc[is_sample,:]
    sample['batch'] = batch["batch"][is_sample]

    for b in unique_batches:
        exp_data["Pooled_" + str(b) + "_" + e] = pooled[pooled['batch'] == int(b)]
        newIndex = []
        for i in range(len(exp_data["Pooled_" + str(b) + "_" + e].index)):
            newIndex.append(exp_data["Pooled_" + str(b) + "_" + e].index[i] + "_" + str(i))
        exp_data["Pooled_" + str(b) + "_" + e].index = newIndex
        exp_data["Pooled_" + str(b) + "_" + e]["batch"] = int(b)
        exp_data["Samples_" + str(b) + "_" + e] = sample[sample['batch'] == int(b)]
        exp_data["Samples_" + str(b) + "_" + e]["batch"] = int(b)

# calculate the median absolute deviation and identify samples that fail threshold
def mad_failed(c):
    mad = sp.median_abs_deviation(c)
    med = nd.median(c)
    return (c < med - 5*mad) | (c > med + 5*mad) 

# filter out samples with >5 median absolute deviation in internal controls (c1 and c2)
# filter out samples with >50% missing
def MAD_or_missing(exp_data, mode, e):
    for temp in exp_data.keys(): 
        if "Samples" in temp and e in temp: # only filter in sample expression with the specified charge
            if mode == "MAD":
                fail = mad_failed(exp_data[temp]["c1"]) | mad_failed(exp_data[temp]["c2"]) # mask of samples that failed the internal control for either c1 or c2
                message = "Median Absolute Deviation threshold"
            elif mode == "sample_missing":
                fail = exp_data[temp].isna().sum(axis=1)/len(exp_data[temp].index) > SAMPLE_MISSING
                message = ">50% missingness"
            else:
                logging.error("Invalid mode specified for MAD or sample missing test.")
                return
            for s in exp_data[temp].index[fail]:
                logging.info("QC: Sample " + s + " in " + temp + " failed the " + message + " test.")
                for x in exp_data.keys():
                    try:
                        exp_data[x] = exp_data[x].drop(index=s)
                    except:
                        continue

#### add by group check
# rsd = calculate RSD (sd/mean * 100) in QC pools and remove metabolites with RSD > 30%
# mode >20% missing = remove metabolites with >20% missing
def rsd_or_missing(exp_data, mode, e, unique_batches):
    if mode == "RSD":
        rsd = pd.DataFrame()
        for b in unique_batches:
            rsd[b] = (exp_data["Pooled_" + b + "_" + e].std()/exp_data["Pooled_" + b + "_" + e].mean())*100 
        failed = rsd.index[(rsd > RSD).sum(axis=1) > 0]
    elif mode == "mtbl_missing":
        missing = pd.DataFrame()
        for b in unique_batches:
            missing[b] = exp_data["Samples_" + b + "_" + e].isna().sum()
        failed = missing.index[(missing > MTBL_MISSING*len(missing.index)).sum(axis=0)]
    else:
        logging.error("Invalid mode for rsd_or_missing(): choose RSD or mtbl_missing")
    for m in failed:
        if mode == "RSD":
            logging.info("QC: Compound " + str(m) + " failed the RSD < 30 check\n" + str(rsd.loc[m,:]))
        else:
            logging.info("QC: Compound " + str(m) + " failed the <20% missing check.")
        for b in unique_batches:
            try:
                exp_data["Pooled_" + b + "_" + e] = exp_data["Pooled_" + b + "_" + e].drop(columns=m)
            except:
                logging.warning(m + " is missing from Pooled_" + b + "_" + e)
            try:    
                exp_data["Samples_" + b + "_" + e] = exp_data["Samples_" + b + "_" + e].drop(columns=m)
            except:
                logging.warning(m + " is missing from Samples" + b + "_" + e)

# generate pca from expression data
def generate_pca(exp_data, title, e, dir_input):
    df = pd.DataFrame()
    for k in exp_data.keys():
        if "Sample" in k and e in k:
            df = pd.concat([df, exp_data[k]])
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
    plt.savefig(dir_input + "/" + title + "_PCA_" + e + ".png")

def correction_factor(rep1, rep2):
    med_ratios = {}
    ratios = rep2 / rep1
    ratios = ratios.dropna(how = "all")
    mad = sp.median_abs_deviation(ratios)
    upper = ratios.median() + 5*mad
    lower = ratios.median() - 5*mad
    for m in ratios.columns:
        temp = ratios.loc[list(ratios[m] < upper[m]) and list(ratios[m] > lower[m]), m]
        med_ratios[m] = temp.median()
    correction = {k: 1/v for k, v in med_ratios.items()}
    return correction

# use biolgoical replicates to conduct batch normalization 
def normalization(exp_data, e, unique_batches, mode):
    bdf = {}
    samples = []
    for b in unique_batches:
        bdf[b] = exp_data["Samples_" + b + "_" + e].drop(['batch'], axis=1)
        samples = samples + list(bdf[b].index)

    replicates = [i for i in set(samples) if samples.count(i) > 1]

    logging.info("Initializing batch normalization with " + str(len(replicates)) + " biological replicates.")
    reps = {}
    for b in unique_batches:
        reps[b] = bdf[b].loc[list(set(bdf[b].index) & set(replicates)),:]

    #110123 is reference batch for plasma
    if mode == "placenta":
        cf = correction_factor(reps["32425"], reps["62323"])
        for m in reps["62323"].columns:
            exp_data["Samples_62323_" + e][m] = reps["62323"][m]*cf[m]
    else:
        cf1 = correction_factor(reps["110123"], reps["51223"])
        cf2 = correction_factor(reps["110123"], reps["112524"])
        for m in reps["51223"].columns:
            exp_data["Samples_51223_" + e][m] = exp_data["Samples_51223_" + e][m]*cf1[m]
        for m in reps["112524"].columns:
            exp_data["Samples_112524_" + e][m] = exp_data["Samples_112524_" + e][m]*cf2[m]


# average replicate expression values
def merge_rep(exp_data, e, unique_batches, mode):
    samples = []
    for b in unique_batches:
        samples = samples + list(exp_data["Samples_" + b + "_" + e].drop(['batch'], axis=1).index)
    logging.info("Getting replicates for merging...")
    replicates = [i for i in set(samples) if samples.count(i) > 1]
    if mode == "placenta":
        logging.info("Placenta mode...")
        rep1 = exp_data["Samples_" + unique_batches[0] + "_" + e].loc[replicates,:].drop(['batch'], axis=1)
        rep2 = exp_data["Samples_" + unique_batches[1] + "_" + e].loc[replicates,:].drop(['batch'], axis=1)
        rep_avg = (rep1 + rep2)/2
        exp_data["Samples_" + str(unique_batches[0]) + "_" + e].loc[rep_avg.index,:] = rep_avg
        exp_data["Samples_" + str(unique_batches[1]) + "_" + e].loc[rep_avg.index,:] = rep_avg
    else:
        logging.info("Plasma mode...")
        reps = {}
        for b in unique_batches:
            reps[b] = exp_data["Samples_" + b + "_" + e].loc[list(set(exp_data["Samples_" + b + "_" + e].index) & set(replicates)),:].drop(['batch'], axis=1)
        avg1 = ((reps["51223"] + reps["110123"])/2).dropna(how="all")
        avg2 = ((reps["112524"] + reps["110123"])/2).dropna(how="all")

        # replace for 511223 and 110123
        exp_data["Samples_51223_" + e].loc[avg1.index,:] = avg1
        exp_data["Samples_110123_" + e].loc[avg1.index,:] = avg1

        # replace for 112524 and 110123
        exp_data["Samples_112524_" + e].loc[avg2.index,:] = avg2
        exp_data["Samples_110123_" + e].loc[avg2.index,:] = avg2
        

# do log2 transformation on expression data
def log2_transform(exp_data,e, unique_batches):
    exp_data["Pooled_" + unique_batches[0] + "_" + e]= np.log2(exp_data["Pooled_" + unique_batches[0] + "_" + e])
    exp_data["Pooled_" + unique_batches[1] + "_" + e] = np.log2(exp_data["Pooled_" + unique_batches[1] + "_" + e])
    exp_data["Samples_" + unique_batches[0] + "_" + e]= np.log2(exp_data["Samples_" + unique_batches[0] + "_" + e])
    exp_data["Samples_" + unique_batches[1] + "_" + e] = np.log2(exp_data["Samples_" + unique_batches[1] + "_" + e])

# combine pos and neg expression in same file
    # if mtbl present in both, choose the one with the better signal intensity (area max)
def combine_pos_neg(mode, batch, exp_data, pos_comp, neg_comp):
    all_s = list(set(exp_data[mode + "_" + batch + "_POS"].index) | set(exp_data[mode + "_" + batch + "_NEG"].index))
    all_m = list(set(exp_data[mode + "_" + batch + "_POS"].columns) | set(exp_data[mode + "_" + batch + "_NEG"].columns))
    combine_best = pd.DataFrame(index=all_s)
    for m in all_m:
        try:
            neg_m = exp_data[mode + "_" + batch + "_NEG"][m]
            try:
                pos_m = exp_data[mode + "_" + batch + "_POS"][m]
                if list(pos_comp.loc[pos_comp["Name"] == m,:]["Area (Max.)"])[0] > list(neg_comp.loc[neg_comp["Name"] == m,:]["Area (Max.)"])[0]:
                    combine_best[m + "_POS"] = pos_m
                    logging.info("Polarity Prioritzation: " + m + " signal intensity is higher in POS for batch " + batch + " - NEG is excluded from final mtbl expression file.")
                else:
                    combine_best[m + "_NEG"] = neg_m
                    logging.info("Polarity Prioritzation: " + m + " signal intensity is higher in NEG for batch " + batch + " - POS is excluded from final mtbl expression file.")
            except:
                neg_m = exp_data[mode + "_" + batch + "_NEG"][m]
                combine_best[m + "_NEG"] = neg_m
        except:
            pos_m = exp_data[mode + "_" + batch + "_POS"][m]
            combine_best[m + "_POS"] = pos_m
    return combine_best

# perform 1/2 minimum imputation
#def min_imputation(a=0.5):
#    pass

# final formatting of expression files
def formatting(final, meta, mode, dir_input):
    for k in final.keys():
        if "Samples" in k:
            #final[k]["group"] = meta.loc[final[k].index,:]["group"]
            group = []
            for id in final[k].index:
                try:
                    group.append(meta.loc[id[:-1].strip(),:]["group"])
                except KeyError:
                    group.append(meta.loc[id[:-1].strip() + "E",:]["group"])
                except KeyError:
                    group.append(meta.loc[id + "E",:]["group"])
                except:
                    logging.error("Issue with indexing " + id + " in meta data.")
            final[k]["group"] = group
            if mode == 'plasma':
                for t in TIMEPOINTS:
                    temp = pd.DataFrame()
                    for id in list(final[k].index):
                        if re.search(t + "$", id):
                            temp = pd.concat([temp, final[k].loc[id,:]], axis=1)
                    if not temp.empty:
                        temp = temp.transpose()
                        temp.to_csv(dir_input + "/" + k + "_" + t + ".csv")
            else:
                final[k].to_csv(dir_input + "/" + k + ".csv")
        else:
            final[k].to_csv(dir_input + "/" + k + ".csv")
    
# do all cleaning steps split by charge
def cleanHelper(exp_data, e, dir_input, unique_batches, mode):
    MAD_or_missing(exp_data, "MAD", e) # MAD test 
    MAD_or_missing(exp_data, "sample_missing", e) # sample >50% missingness test
    rsd_or_missing(exp_data, "RSD", e, unique_batches) # RSD test (consider metabolite as missing)
    rsd_or_missing(exp_data, "mtbl_missing", e, unique_batches) # metabolite >20% missingness test
        #Have not added by group check yet, mtbl doesn't have any missing mtbl expression anyway
    logging.info("Generating unnormalized PCA plot...")
    generate_pca(exp_data, "Unnormalized_MTBL_Expression", e, dir_input) # generate unnormalized PCA
    normalization(exp_data, e, unique_batches, mode) # batch normalization using replicates
    logging.info("Generating batch unnormalized PCA plot")
    generate_pca(exp_data, "Batch_Normalized_MTBL_Expression", e, dir_input) # generate normalized PCA
    merge_rep(exp_data, e, unique_batches, mode) # average replicates expression 
    logging.info("Applying log2 transformation...")
    log2_transform(exp_data, e, unique_batches) # log2 transform all data

# function for data cleaning
def clean(pos_exp, pos_batch, pos_comp, neg_exp, neg_batch, neg_comp, dir_input, meta):
    if "plasma" in dir_input:
        mode = "plasma"
    elif "placenta" in dir_input:
        mode = "placenta"
    else:
        logging.error("Formatting Error: output directory doesn't specify 'plasma' or 'placenta'")
    unique_batches = list(set(pos_batch["batch"]))
    unique_batches = [str(x) for x in unique_batches]
    logging.info("Batches: " + str(unique_batches))
    #  handle missing
    logging.info("Handling NA entires...")
    pos_exp, neg_exp = handle_missing(pos_exp, neg_exp) 

    logging.info("Splitting by batch and run type (Samples or Pooled)...")
    exp_data = {} # dictionary to stroe split data by batches and pools
    split_exp(pos_exp, pos_batch, "POS", exp_data, unique_batches) # naming format = [Pooled, Samples]_[batch]_[charge]
    split_exp(neg_exp, neg_batch, "NEG", exp_data, unique_batches)

    # do cleaning steps split by charge
    cleanHelper(exp_data, "POS", dir_input, unique_batches, mode)
    cleanHelper(exp_data, "NEG", dir_input, unique_batches, mode)

    # remove duplicate metabolite entries based on computed quality scores
    logging.info("Removing multiplet parent metabolites by quality score...")
    neg_comp["quality_score"] = qs(neg_comp)
    pos_comp["quality_score"] = qs(pos_comp)
    for x in exp_data.keys():
        if "POS" in x:
            exp_data[x] = remove_duplicates(exp_data[x], pos_comp)
        else:
            exp_data[x] = remove_duplicates(exp_data[x], neg_comp)

    # combine the pos and neg compounds
    logging.info("Combining POS and NEG polarity into shared file...")
    final = {}
    b = list(set(pos_batch["batch"]))
    s = ["Pooled", "Samples"]
    for i in b:
        for j in s:
            temp = combine_pos_neg(str(j), str(i), exp_data, pos_comp, neg_comp) # combine compounds in both pos and neg
            final[str(j) + "_" + str(i)] = temp

     # 1/2 minium imputation of missing, no mtbl are missing
    #min_imputation()
    logging.info("Exporting csv files...")
    formatting(final, meta, mode, dir_input) # add group columns, split files by timepoint

        

# Give folder with both postive and negative metabolomic expression, batch info, and compound metadata
def main():
    dir_input = sys.argv[1] # e.g. /Users/kaylaxu/Desktop/data/clean_data/MTBL_placenta
    meta_input = sys.argv[2] # e.g. /Users/kaylaxu/Desktop/data/raw_data/dp3 master table v2.xlsx

    logging.basicConfig( # initiate log file
        filename='MTBL_cleaning.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'  # Use 'w' to overwrite the file each run, or 'a' to append
    )
    logging.info("Initializing metabolomics cleaning...")
    logging.info("Reading expression, batch, compound, and metadata files...")
    # get files
    pos_exp = pd.read_csv(dir_input + "/pos_expression.csv", index_col=0)
    pos_batch = pd.read_csv(dir_input + "/pos_batch.csv", index_col=0)
    pos_comp = pd.read_csv(dir_input + "/pos_compounds.csv", index_col=0)
    neg_exp = pd.read_csv(dir_input + "/neg_expression.csv", index_col=0)
    neg_batch = pd.read_csv(dir_input + "/neg_batch.csv", index_col=0)
    neg_comp = pd.read_csv(dir_input + "/neg_compounds.csv", index_col=0)
    meta = pd.read_excel(meta_input, index_col=0)

    # call cleaning functions
    clean(pos_exp, pos_batch, pos_comp, neg_exp, neg_batch, neg_comp, dir_input, meta)

    logging.info("DONE - Metabolomics cleaning pipeline complete")
    #close log file
    logging.shutdown()

if __name__ == "__main__":
    main()
