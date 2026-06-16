import sys
import subprocess

# force-install the required packages directly into the active environment
try:
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests
except ModuleNotFoundError:
    print("--> Packages missing. Installing scipy and statsmodels now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "statsmodels"])
    
    # try importing them again after installation
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests
    print("--> Installation successful! Continuing script...")

import pandas as pd
import numpy as np
# from scipy.stats import spearmanr
# from statsmodels.stats.multitest import multipletests

# loads both sheets (one contains the placental histopathology data, and the other contains the variables of interest data) and returns both sheets
def load_sheets():
    sheet1 = pd.read_csv("01_data_cleaning/preprocess_slides_data/output.csv")
    sheet2 = pd.read_csv("02_exploratory_analysis/explore_correlations/dp3 master table v2.xlsx - variables of interest.csv")
    return sheet1, sheet2

def check_spearman_correlation(df_placental,df_delivery, placental_vars, delivery_vars, fdr_threshold = 0.05):
    # make sure column names are consistent across dataframes
    df_delivery = df_delivery.rename(columns = {"ID": "id"})

    # align and merge datasets based on patient ID
    # only grab the ID column and variables of interest
    df_placental_sub = df_placental[["id"] + placental_vars]
    df_delivery_sub = df_delivery[["id"] + delivery_vars]

    merged_df = pd.merge(df_placental_sub, df_delivery_sub, on="id", how="inner")

    # drop rows with missing values in the columns of interest
    all_vars = placental_vars + delivery_vars
    merged_df = merged_df.dropna(subset=all_vars)

    # vectorized spearman correlation loop
    # loop over placental variables and vectorize across delivery variables
    records = []
    for p_var in placental_vars:
        # for each delivery variable, calculate its correlation with each placental variable
        for d_var in delivery_vars:
            # check if either column is constant (has only 1 unique value) or is empty
            if merged_df[p_var].nunique() <= 1 or merged_df[d_var].nunique() <= 1:
                rho, p = None, None
                print(f"Skipping: {p_var} or {d_var} has constant values.")
            else:
                rho, p = spearmanr(merged_df[p_var], merged_df[d_var])

            # save this to records
            records.append({
                "placental_var": p_var,
                "delivery_var": d_var,
                "rho": rho,
                "p_value": p
            })

    results_table = pd.DataFrame(records)

    # multiple hypothesis testing (FDR/Benjamini-Hochberg)
    # adjusts p-values to account for the fact that we're running a large number of tests
    reject, q_values, _, _ = multipletests(results_table["p_value"], alpha = fdr_threshold, method = "fdr_bh")
    results_table["FDR"] = q_values

    # filter results for significance and direction
    significant_pairs = results_table[results_table["FDR"] <= fdr_threshold]
    pos_associations = significant_pairs[significant_pairs["rho"] > 0]
    neg_associations = significant_pairs[significant_pairs["rho"] < 0]

    unique_pos = list(pos_associations["delivery_var"].unique())
    unique_neg = list(neg_associations["delivery_var"].unique())

    analysis_assets = {
        "master_results": results_table,
        "pos_delivery_vars": unique_pos,
        "neg_delivery_vars": unique_neg
    }

    return analysis_assets


# print all calculated correlation data into a log file
def print_log(df):
    pos_log_path = "02_exploratory_analysis/explore_correlations/positively_associated_delivery_vars.txt"
    neg_log_path = "02_exploratory_analysis/explore_correlations/negatively_associated_delivery_vars.txt"

    # write into the positively associated delivery var file
    with open(pos_log_path, "w") as pos_file:
        # write to file only if at least one delivery var passes the FDR threshold
        if df["pos_delivery_vars"]:
            pos_file.write("\n".join(df["pos_delivery_vars"]))
        else:
            pos_file.write("")

    # write into the negatively associated delivery var file
    with open(neg_log_path, "w") as neg_file:
        # write to file only if at least one delivery var passes the FDR threshold
        if df["neg_delivery_vars"]:
            neg_file.write("\n".join(df["neg_delivery_vars"]))
        else:
            neg_file.write("")

def main():
    placental_metrics = [
        "placental infarction", "distal villous hypoplasia focal/diffuse", "accelerated villous maturation", "increased syncytial knots", 
        "decidual arteriopathy membrane role/basal plate/both", "segmental avascular villi small/intermediate/large", "delayed villous maturation", 
        "maternal inflammatory response stage/grade", "villitis of unknown etiology, high/low grade, focal/diffuse", "increased perivillous fibrin deposition", 
        "chorangiosis", "fetal inflammatory response stage/grade/location"
    ]
    delivery_metrics = ["apgar 1", "apgar 5", "nicu days", "birthweight", "gest age del"]
    placental_df, delivery_df = load_sheets()
    
    master_results = check_spearman_correlation(
        df_placental=placental_df,
        df_delivery=delivery_df,
        placental_vars=placental_metrics,
        delivery_vars=delivery_metrics,
        fdr_threshold=0.05
    )

    print_log(master_results)

if __name__ == "__main__":
    main()