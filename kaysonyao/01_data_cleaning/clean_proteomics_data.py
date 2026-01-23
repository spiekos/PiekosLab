'''
Title: clean_proteomics_data.py
Author: Samantha Piekso
Date: 9/28/21
Description: This takes in a .csv of Olink full proteomics data. It removes columns (proteins) with
25% or more of the values missing. It then performs minimum imputation on the remaining missing
values in the dataframe and saves to a new .csv. Prior to input convert all red cell values (olink
determined below the level of detection) and red text values (sample failed quality control for
that olink panel) using VBA in excel.

@file_input 	path to olink .csv input file
@file_output 	path to save minimum imputed data .csv file
'''


import sys
import pandas as pd
import math
import numpy as np
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()
    

CUTOFF_PERCENT_MISSING = 0.20

#FIXME: Combine batches before missingness filtering

def count_missingness_group_distribution(df, protein, dict_group_count):
    stats = importr('stats')
    d = {}
    m = np.zeros(shape=(2,len(dict_group_count.keys())))
    p = 1
    for key in dict_group_count.keys():
        d[key] = 0
    for index, data in df.iterrows():
        if math.isnan(data[protein]):
            d[data['Group']] += 1
    c = 0
    for k, v in d.items():
        percent = round(v*100/dict_group_count[k], 2)
        print(k + ': ' + str(v) + ' (' + str(percent) +'%)')
        m[:, c] = [v, dict_group_count[k]-v]
        c += 1
    if np.all((m[1, :] == 0)):
        print('p-value:  1.0')
    else:
        res = stats.fisher_test(m,  simulate_p_value=True)
        print('p-value: {}'.format(res[0][0]))
        p = res[0][0]
    print('\n')
    return(p)


def count_samples_in_each_group(df):
    d = {}
    for item in df['Group']:
        if item not in d:
            d[item] = 1
        else:
            d[item] += 1
    return d


def perform_benjamini_hochberg(dict_missing_p_values, alpha=0.05):
    '''
    This function performs the Benjamini-Hochberg correction for multiple hypothesis testing
    on the p-values provided in dict_missing_p_values. It prints out the proteins that are found
    to have a significant difference in missingness distribution between groups after correction.
    
    :param dict_missing_p_values: dict
        Mapping {analyte_name: p_value}
    :param alpha: float
        Desired FDR control level
    '''
    # Sort p-values while keeping track of their keys
    sorted_items = sorted(dict_missing_p_values.items(), key=lambda x: x[1])
    n_failed_proteins = len(sorted_items)
    
    print('Number of proteins tested: ' + str(n_failed_proteins))
    print('Benjamini-Hochberg FDR control level (alpha): ' + str(alpha))
    print('Proteins found significant after BH correction:')
    
    # Find the largest rank (i) such that P(i) <= (i/n) * alpha
    significant_keys = []
    for i, (k, v) in enumerate(sorted_items):
        rank = i + 1
        bh_threshold = (rank / n_failed_proteins) * alpha
        if v <= bh_threshold:
            significant_keys.append(k)
    
    # Print results (In BH, if p(i) is significant, all p(j) where j < i are also significant)
    # The logic below identifies the "last" significant index to properly control FDR.
    for k in significant_keys:
        print(k)
        
    print('Number of proteins with significant difference: ' + str(len(significant_keys)))


def remove_cols_failing_missingness_cutoff(df):
    dict_missing_p_values = {}
    dict_group_count = count_samples_in_each_group(df)
    df_new = df['Group'].to_frame()
    print("Protein Failing Missingness Cutoff")
    print("Group: Number of Samples Missing Protein Measurement (%)")
    for name, data in df.iteritems():
        percent_missing = data.isna().sum()/len(data)
        if percent_missing < CUTOFF_PERCENT_MISSING:
            df_new[name] = data
            df_new = df_new.copy()
        else:
            print(name)
            print('Total: ' + str(data.isna().sum()) + ' (' + str(round(percent_missing*100, 1)) + '%)')
            dict_missing_p_values[name] = count_missingness_group_distribution(df, name, dict_group_count)
    print('\n\n')
    print('Number of samples: ' + str(df_new.shape[0]))
    print('Number of proteins: ' + str(df_new.shape[1]))
    print('\n')
    perform_benjamini_hochberg(dict_missing_p_values)
    return df_new

def apply_quantile_normalization(df):
    '''
    Quantile normalize a samples x features matrix (rows=samples, cols=proteins)
    Returns a DataFrame of the same shape with quantile-normalized values
    :param df: DataFrame
        DataFrame of shape (n_samples, n_features)
    '''
    X = df.copy()
    # Pivot to samples x proteins matrix
    X = df.pivot(index='SampleID', columns='Assay', values='NPX')
    # Convert to numpy for speed
    arr = X.to_numpy(dtype=float)

    n_samples, n_proteins = arr.shape
    out = np.full_like(arr, np.nan)

    # Sort each sample's values
    sorted_vals = []
    sort_orders = []
    finite_counts = []

    for i in range(n_samples):
        row = arr[i, :]
        mask = np.isfinite(row)
        vals = row[mask]

        order = np.argsort(vals)
        sorted_vals.append(vals[order])
        sort_orders.append(order)
        finite_counts.append(len(vals))

    max_k = max(finite_counts)
    if max_k == 0:
        # All values missing; nothing to normalize
        return X

    # Compute mean value at each rank across samples
    rank_means = np.zeros(max_k)
    for k in range(max_k):
        vals_at_rank = [
            sorted_vals[i][k]
            for i in range(n_samples)
            if finite_counts[i] > k
        ]
        rank_means[k] = np.mean(vals_at_rank)

    # Assign normalized values back to original positions
    for i in range(n_samples):
        mask = np.isfinite(arr[i, :])
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        order = sort_orders[i]
        idx_sorted = idx[order]
        out[i, idx_sorted] = rank_means[:len(idx_sorted)]

    # Return as DataFrame with labels preserved
    X_qn = pd.DataFrame(out, index=X.index, columns=X.columns)

    return X_qn


def half_min_imputation_of_missing_data(file_input, file_output):
    df = pd.read_csv(file_input, header=0, index_col=0, sep=';')
    if 'QC_Warning' in df.columns:
        df.loc[df['QC_Warning'].astype(str).str.upper() != 'PASS', 'NPX'] = np.nan
    if 'Assay_Warning' in df.columns:
        df.loc[df['Assay_Warning'].astype(str).str.upper() != 'PASS', 'NPX'] = np.nan
    df_normalized = apply_quantile_normalization(df)
    #FIXME: Combine batches before missingness filtering
    df_imputed = remove_cols_failing_missingness_cutoff(df_normalized)
    for name, data in df_imputed.iteritems():
        df_imputed[name].fillna(value=0.5*df[name].min(), inplace=True)
    df_imputed.to_csv(file_output)


def main():
    file_input = sys.argv[1]
    file_output = sys.argv[2]
    half_min_imputation_of_missing_data(file_input, file_output)


if __name__ == "__main__":
    main()
