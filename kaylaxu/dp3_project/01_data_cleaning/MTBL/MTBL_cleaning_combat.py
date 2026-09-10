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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import re
import logging
import sys
import warnings
from inmoose.pycombat import pycombat_norm

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
# 8. Batch normalization using Python pycombat.
# 9. Generate a batch-corrected PCA plot in which the color of the sample dots match the batch they were in.
# 10. For the samples ran in both batches - average metabolic expression.
# 11. Perform log2 transformation
# 12. Combine the POS and NEG compounds.
# 13. Handle all additional formatting: adding group, spliting by timepoint, output to csv.

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

# for lipids only
def _lipid_lsi_score(value)->float:
        """
        Update these regular expressions if Appendix F uses a different
        LipidID convention.

        Examples:
        PC(16:0/18:1(9Z))  -> structure-defined: 10
        PC(16:0/18:1)      -> sn-position: 9
        PC(16:0_18:1)      -> molecular species: 7
        PC(34:1) or PC 34:1 -> species: 5
        PC                 -> class only: 2
        """
        lipid_id=str(value or "").strip()

        if not lipid_id:
            return 0.0

        normalized=re.sub(r"\s+","",lipid_id)

        # Structure-defined: acyl chains plus double-bond geometry/location.
        if re.search(r"\(\d+[ZE]\)",normalized,re.IGNORECASE):
            return 10.0

        # sn-position-resolved species.
        if "/" in normalized:
            return 9.0

        # Molecular species, chains known but sn positions unresolved.
        if "_" in normalized:
            return 7.0

        # Sum-composition / species-level notation, e.g. PC(34:1).
        if re.search(r"\d+:\d+",normalized):
            return 5.0

        # Lipid class/category but no acyl composition.
        if re.match(r"^[A-Za-z]+$",normalized):
            return 2.0

        return 0.0

# calculate quality scores (peak quality + rsd + ms2 + signal intensity + annotation confidence)
def qs(comp):
    # get peak rating qc
    try:
        peak = pd.Series([10 if x >= 7.0 else 7 if x >= 5 else 4 if x >= 3 else 1 for x in comp['Peak Rating (Max.)']])
    except:
        peak = 0
    # get rsd qc areas
    try:
        rsd = pd.Series([10 if x < 10 else 8 if x < 15 else 6 if x < 20 else 4 if x < 25 else 2 if x < 30 else 0 for x in comp["RSD QC Areas [%]"]])
    except:
        rsd = 0
    # get ms2
    try:
        ms2 = pd.Series([10 if x == "DDA for preferred ion" else 6 if x == "DDA for other ion" else 4 if x == "DDA available" else 0 for x in comp["MS2"]])
    except:
        ms2 = 0
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
        if "Samples" in temp and e in temp: 
            if mode == "MAD":
                fail = mad_failed(exp_data[temp]["c1"]) | mad_failed(exp_data[temp]["c2"]) 
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

def rsd(exp_data, e, unique_batches):
    rsd = pd.DataFrame()
    for b in unique_batches:
        rsd[b] = (exp_data["Pooled_" + b + "_" + e].std()/exp_data["Pooled_" + b + "_" + e].mean())*100 
    failed = rsd.index[(rsd > RSD).sum(axis=1) > 0]
    for m in failed:
        logging.info("QC: Compound " + str(m) + " failed the RSD < 30 check\n" + str(rsd.loc[m,:]))
        for b in unique_batches:
            try:
                exp_data["Pooled_" + b + "_" + e] = exp_data["Pooled_" + b + "_" + e].drop(columns=m)
            except:
                logging.warning(m + " is missing from Pooled_" + b + "_" + e)
            try:    
                exp_data["Samples_" + b + "_" + e] = exp_data["Samples_" + b + "_" + e].drop(columns=m)
            except:
                logging.warning(m + " is missing from Samples" + b + "_" + e)

def sample_missing(exp_data, e, unique_batches):
    missing_dict = {}
    for b in unique_batches:
        df_b = exp_data["Samples_" + b + "_" + e]
        missing_dict[b] = df_b.isna().sum() / len(df_b.index)
    missing = pd.DataFrame(missing_dict)
    mask = (missing > MTBL_MISSING).sum(axis=1) > 0
    failed = missing.loc[mask].index
    for m in failed:
        logging.info(f"QC: Compound {m} failed the <20% missing check.")
        for b in unique_batches:
            sample_key = "Samples_" + b + "_" + e
            pooled_key = "Pooled_" + b + "_" + e
            if m in exp_data[sample_key].columns:
                exp_data[sample_key] = exp_data[sample_key].drop(columns=m)
            if m in exp_data[pooled_key].columns:
                exp_data[pooled_key] = exp_data[pooled_key].drop(columns=m)


def generate_pca(exp_data, title, e, dir_input, tissue):
    df = pd.DataFrame()
    for k in exp_data.keys():
        if "Sample" in k and e in k:
            df = pd.concat([df, exp_data[k]])
    batch = df["batch"]
    df = df.drop(["batch"], axis=1)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    imputer = SimpleImputer(missing_values = np.nan, 
                        strategy ='mean')
    imputer = imputer.fit(scaled_data)
    scaled_data = imputer.transform(scaled_data)

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
        s=100,
        alpha=0.8
    )
    plt.title(title, fontsize=15)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance)', fontsize=12)
    plt.grid(True)
    plt.savefig(f"{dir_input}/{title}_PCA_{e}_{tissue}.png")

def combat_normalize_wide(X: pd.DataFrame, batch_labels: pd.Series) -> pd.DataFrame:
    """
    Apply ComBat batch normalization using Python pycombat.
    Preserves original missingness pattern after correction.
    """
    b = batch_labels.reindex(X.index)
    if b.isna().any():
        raise ValueError("ComBat requires non-missing batch labels for all samples.")

    if b.nunique() < 2:
        return X.copy()

    missing_mask = X.isna()

    X_filled = X.copy()
    med = X_filled.median(axis=0, skipna=True)
    X_filled = X_filled.fillna(med).fillna(0.0)

    # checking for 0 variance features
    tol = 1e-8
    df_filtered = X_filled.loc[:, X_filled.var(axis=0) > tol]

    # Step 2: (Optional but recommended) Remove features with zero variance in ANY batch
    valid_features = pd.Series(True, index=df_filtered.columns)

    for batch_id in b.unique():
        # Get the rows (samples) for this specific batch
        batch_rows = df_filtered.loc[b == batch_id]
        
        # Calculate feature variance within this specific batch
        batch_var = batch_rows.var(axis=0)
        
        # Update valid features: must have variance > 0 in this batch
        # (If a batch has only 1 sample, var is NaN, which > 0 safely handles)
        valid_features = valid_features & (batch_var > tol)
        logging.info(f"Number of features: {len(X_filled.columns)}")
        logging.info(f"Number of features with 0 variance across samples in batch {batch_id}: {(batch_var <= 0).sum()}")

    # Apply the strict batch-variance filter (keeping all rows, filtering columns)
    X_filled = df_filtered.loc[:, valid_features]

    # model = Combat()
    # pycombat can error with pandas slicing internals; pass ndarray explicitly.
    # corrected = model.fit_transform(X_filled.to_numpy(dtype=float), b.values)
    # corrected = pycombat(X_filled, b.values)
    corrected_transposed = pycombat_norm(X_filled.T.copy(), b.values.copy(), ref_batch=1)
    
    # Transpose back to standard format (samples as rows)
    corrected = corrected_transposed.T
    Xc = pd.DataFrame(corrected, index=X_filled.index, columns=X_filled.columns)

    Xc = Xc.mask(missing_mask)
    return Xc

def normalization(exp_data, e, unique_batches, mode):
    df_list = []
    batch_list = []
    
    for b in unique_batches:
        # Extract dataframe, dropping batch column. Keep original index as column 
        # to circumvent Duplicate Index errors upon concatenation of biological replicates.
        df_b = exp_data["Samples_" + str(b) + "_" + e].drop(['batch'], axis=1)
        df_list.append(df_b.reset_index())
        batch_list.extend([b] * len(df_b))
        
    X = pd.concat(df_list, axis=0, ignore_index=True)
    print(X.columns)
    original_indices = X['compound']
    X = X.drop(columns=['compound'])
    
    # Ensure mapping of batches maps to 1, 2, 3... to satisfy `ref_batch=1` in provided ComBat function.
    unique_b_list = list(pd.Series(batch_list).unique())
    batch_mapping = {b: i+1 for i, b in enumerate(unique_b_list)}
    batch_labels = pd.Series([batch_mapping[b] for b in batch_list], index=X.index)
    
    logging.info(f"Applying ComBat normalization for {e} charges...")
    Xc = combat_normalize_wide(X, batch_labels)
    Xc.index = original_indices 
    
    # Reconstruct back into separated batch objects in dictionary
    start_idx = 0
    for b in unique_batches:
        key = "Samples_" + str(b) + "_" + e
        n_samples = len(exp_data[key])
        
        df_b_corrected = Xc.iloc[start_idx : start_idx + n_samples].copy()
        df_b_corrected['batch'] = int(b) if str(b).isdigit() else b 
        
        exp_data[key] = df_b_corrected
        start_idx += n_samples

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
        
def log2_transform(exp_data,e, unique_batches):
    for b in unique_batches:
        exp_data["Pooled_" + str(b) + "_" + e]= np.log2(exp_data["Pooled_" + str(b) + "_" + e])
        exp_data["Samples_" + str(b) + "_" + e]= np.log2(exp_data["Samples_" + str(b) + "_" + e])

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

def formatting(final, meta, mode, dir_input, dir_output, tissue, preNorm = False):
    for k in final.keys():
        if "Samples" in k:
            patient = []
            group = []
            subgroup = []
            gest_age = []
            gest_age_collection = []
            column = ""
            if tissue == "plasma":
                column = "Sample ID"
            elif tissue == "placenta":
                column = "ID"
            print(meta.columns)
            for id in final[k].index:
                keys_to_try = [
                    id,
                    id[:-1].strip() + "E" + id[-1],
                    id[:-1].strip() + " " + id[-1]
                ]
                found = False
                for key in keys_to_try:
                    try:
                        print(id)
                        patient.append(list(meta.loc[meta[column] == key, :].index)[0])
                        group.append(list(meta.loc[meta[column] == key, "group"])[0])
                        subgroup.append(list(meta.loc[meta[column] == key, "subgroup"])[0])
                        gest_age.append(list(meta.loc[meta[column] == key, "gest age del"])[0])
                        gest_age_collection.append(list(meta.loc[meta[column] == key,"sample gest Age"])[0])
                        found = True
                        break 
                    except:
                        continue 
                if not found:
                    logging.error(f"Issue with indexing {id} in meta data.")
            final[k]["patient_ID"] = patient
            final[k]["group"] = group
            final[k]["subgroup"] = subgroup
            final[k]["gestational_age"] = gest_age
            final[k]["gestational_age_at_collection"] = gest_age_collection
            if mode == 'plasma':
                for t in TIMEPOINTS:
                    temp = pd.DataFrame()
                    for id in list(final[k].index):
                        if re.search(t + "$", id):
                            temp = pd.concat([temp, final[k].loc[id,:]], axis=1)
                    if not temp.empty:
                        temp = temp.transpose()
                        if preNorm:
                            temp.to_csv(dir_output + "/" + k + "_" + t + "_preNorm.csv")
                        else:
                            temp.to_csv(dir_output + "/" + k + "_" + t + ".csv")
            else:
                if preNorm:
                    final[k].to_csv(dir_output + "/" + k + "_preNorm.csv")
                else:
                    final[k].to_csv(dir_output + "/" + k + ".csv")
        else:
            if preNorm:
                final[k].to_csv(dir_output + "/" + k + "_preNorm.csv")
            else:
                final[k].to_csv(dir_output + "/" + k + ".csv")
    
def cleanHelper(exp_data, e, dir_input, unique_batches, mode, tissue):
    MAD_or_missing(exp_data, "MAD", e) 
    MAD_or_missing(exp_data, "sample_missing", e) 
    rsd(exp_data,  e, unique_batches) 
    sample_missing(exp_data, e, unique_batches) 
    logging.info("Generating unnormalized PCA plot...")
    generate_pca(exp_data, "Unnormalized_MTBL_Expression", e, dir_input, tissue) 
    preNorm = {k: v for k, v in exp_data.items() if e in k}
    normalization(exp_data, e, unique_batches, mode) 
    logging.info("Generating batch unnormalized PCA plot")
    generate_pca(exp_data, "Batch_Normalized_MTBL_Expression", e, dir_input, tissue) 
    merge_rep(exp_data, e, unique_batches, mode)  
    logging.info("Applying log2 transformation...")
    log2_transform(exp_data, e, unique_batches) 
    return preNorm

def clean(pos_exp, pos_batch, pos_comp, neg_exp, neg_batch, neg_comp, dir_input, meta, dir_output, tissue):
    if "plasma" in dir_input:
        mode = "plasma"
    elif "placenta" in dir_input:
        mode = "placenta"
    else:
        logging.error("Formatting Error: output directory doesn't specify 'plasma' or 'placenta'")
    unique_batches = list(set(pos_batch["batch"]))
    unique_batches = [str(x) for x in unique_batches]
    logging.info("Batches: " + str(unique_batches))
    logging.info("Handling NA entires...")
    pos_exp, neg_exp = handle_missing(pos_exp, neg_exp) 

    logging.info("Splitting by batch and run type (Samples or Pooled)...")
    exp_data = {} 
    split_exp(pos_exp, pos_batch, "POS", exp_data, unique_batches) 
    split_exp(neg_exp, neg_batch, "NEG", exp_data, unique_batches)

    preNorm_POS = cleanHelper(exp_data, "POS", dir_input, unique_batches, mode, tissue)
    preNorm_NEG = cleanHelper(exp_data, "NEG", dir_input, unique_batches, mode, tissue)

    preNorm = preNorm_NEG | preNorm_POS

    logging.info("Removing multiplet parent metabolites by quality score...")
    neg_comp["quality_score"] = qs(neg_comp)
    pos_comp["quality_score"] = qs(pos_comp)
    for x in exp_data.keys():
        if "POS" in x:
            exp_data[x] = remove_duplicates(exp_data[x], pos_comp)
            preNorm_POS[x] = remove_duplicates(preNorm_POS[x], pos_comp)
        else:
            exp_data[x] = remove_duplicates(exp_data[x], neg_comp)
            preNorm_NEG[x] = remove_duplicates(preNorm_NEG[x], neg_comp)

    logging.info("Combining POS and NEG polarity into shared file...")
    final = {}
    preNorm_final = {}
    b = list(set(pos_batch["batch"]))
    s = ["Pooled", "Samples"]
    for i in b:
        for j in s:
            temp = combine_pos_neg(str(j), str(i), exp_data, pos_comp, neg_comp) 
            final[str(j) + "_" + str(i)] = temp
            temp = combine_pos_neg(str(j), str(i), preNorm, pos_comp, neg_comp)
            preNorm_final[str(j) + "_" + str(i)] = temp

    logging.info("Exporting csv files...")
    formatting(final, meta, mode, dir_input, dir_output, tissue) 
    formatting(preNorm_final, meta, mode, dir_input, dir_output, tissue, preNorm=True)

def main():
    dir_input = sys.argv[1] 
    meta_input = sys.argv[2] 
    dir_output = sys.argv[3]

    if "MTBL" in dir_input:
        filename= 'MTBL_cleaning.log'
        modality = "MTBL"
    elif "LIPD" in dir_input:
        filename= 'LIPD_cleaning.log'
        modality = "LIPD"

    if "plasma" in dir_input:
        tissue = "plasma"
    elif "placenta" in dir_input:
        tissue = "placenta"

    configFile = f"{modality}_{tissue}_cleaning.log"

    logging.basicConfig(
        filename=configFile,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'  
    )
    logging.info("Initializing metabolomics cleaning...")
    logging.info("Reading expression, batch, compound, and metadata files...")
    pos_exp = pd.read_csv(dir_input + "/pos_expression.csv", index_col=0)
    pos_batch = pd.read_csv(dir_input + "/pos_batch.csv", index_col=0)
    pos_comp = pd.read_csv(dir_input + "/pos_compounds.csv", index_col=0)
    neg_exp = pd.read_csv(dir_input + "/neg_expression.csv", index_col=0)
    neg_batch = pd.read_csv(dir_input + "/neg_batch.csv", index_col=0)
    neg_comp = pd.read_csv(dir_input + "/neg_compounds.csv", index_col=0)
    meta = pd.read_excel(meta_input, sheet_name="n=133 metabolomics")
    meta.index = meta["ID"]
    meta = meta[meta.index.notna()]
    clean(pos_exp, pos_batch, pos_comp, neg_exp, neg_batch, neg_comp, dir_input, meta, dir_output, tissue)

    logging.info("DONE - Metabolomics cleaning pipeline complete")
    logging.shutdown()

if __name__ == "__main__":
    main()