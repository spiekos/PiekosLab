import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


# loads both sheets (sheet1 contains the placental histopathology data, and sheet2 contains the variables of interest data)
def load_sheets():
    sheet1 = pd.read_csv("01_data_cleaning/processed_data/processed_placental_data.csv")
    sheet2 = pd.read_csv("00_raw_data/dp3 master table v2.xlsx - variables of interest.csv")
    return sheet1, sheet2


# runs vectorized Spearman tests across given sets of independent and dependent variables
def run_spearman_core(merged_df, independent_vars, dependent_vars, id_col="id", fdr_threshold=0.05):
    records = []

    for ind_var in independent_vars:
        for dep_var in dependent_vars:
            if ind_var not in merged_df.columns or dep_var not in merged_df.columns:
                continue

            # drop NaNs for just these two columns
            pair_df = merged_df[[ind_var, dep_var]].dropna()

            if pair_df[ind_var].dtype == "object" or pair_df[dep_var].dtype == "object":
                continue

            # don't perform correlation test if inputs are constant
            if pair_df[ind_var].nunique() <= 1 or pair_df[dep_var].nunique() <= 1:
                continue

            # ensure sample size is big enough to test
            if len(pair_df) < 10:
                continue
            
            rho, p = spearmanr(pair_df[ind_var], pair_df[dep_var])

            records.append({
                "independent_var": ind_var,
                "dependent_var": dep_var,
                "rho": rho,
                "p_value": p,
                "sample_size_N": len(pair_df)
            })

    if not records:
        print("Warning: No valid numerical variable pairs found to correlate.")
        return {"master_results": pd.DataFrame(), "pos_dependent_vars": [], "neg_dependent_vars": []}

    results_table = pd.DataFrame(records)

    # multiple hypothesis testing (FDR/Benjamini-Hochberg)
    reject, q_values, _, _ = multipletests(results_table["p_value"], alpha=fdr_threshold, method="fdr_bh")
    results_table["FDR"] = q_values

    # filter results for significance and direction
    significant_pairs = results_table[results_table["FDR"] <= fdr_threshold]
    pos_associations = significant_pairs[significant_pairs["rho"] > 0]
    neg_associations = significant_pairs[significant_pairs["rho"] < 0]

    unique_pos = list(pos_associations["dependent_var"].unique())
    unique_neg = list(neg_associations["dependent_var"].unique())

    analysis_assets = {
        "master_results": results_table,
        "pos_dependent_vars": unique_pos,
        "neg_dependent_vars": unique_neg
    }

    return analysis_assets


# test 1: runs correlation test between placental histopathology variables and delivery metrics
def run_test_1_placental_vs_delivery(df_placental, df_delivery, placental_vars, delivery_vars, fdr_threshold=0.05):
    df_delivery = df_delivery.rename(columns={"ID": "id"})

    # clean trailing spaces
    df_placental.columns = df_placental.columns.str.strip()
    df_delivery.columns = df_delivery.columns.str.strip()

    # isolate target columns
    df_placental_sub = df_placental[["id"] + [v for v in placental_vars if v in df_placental.columns]]
    df_delivery_sub = df_delivery[["id"] + [v for v in delivery_vars if v in df_delivery.columns]]

    # align matrices
    merged_df = pd.merge(df_placental_sub, df_delivery_sub, on="id", how="inner")
    
    return run_spearman_core(merged_df, placental_vars, delivery_vars, id_col="id", fdr_threshold=fdr_threshold)


# test 2: runs correlation test comparing fitbit variables with placental histopathology and delivery metrics
def run_test_2_fitbit_vs_all_outcomes(master_fitbit_path, clinical_vars, placental_vars, fdr_threshold=0.05):
    # ensure the file analysis_fitbit.py has been run and data has been prepared for this analysis
    try:
        master_df = pd.read_csv(master_fitbit_path)
    except FileNotFoundError:
        print(f"Warning: Could not find master Fitbit path: {master_fitbit_path}. Run analyze_fitbit.py first!")
        return {"master_results": pd.DataFrame(), "pos_dependent_vars": [], "neg_dependent_vars": []}

    master_df = master_df.rename(columns={"Record ID": "id"})
    master_df.columns = master_df.columns.str.strip()

    # compile the two lists we are comparing
    fitbit_features = [col for col in master_df.columns if "Trimester" in col]
    all_outcome_targets = [v for v in (clinical_vars + placental_vars) if v in master_df.columns]

    if not fitbit_features:
        print("Warning: No columns prefixed with 'Trimester' found in the master Fitbit file.")
        return {"master_results": pd.DataFrame(), "pos_dependent_vars": [], "neg_dependent_vars": []}

    return run_spearman_core(master_df, fitbit_features, all_outcome_targets, id_col="id", fdr_threshold=fdr_threshold)


# print calculated correlation data into respective file destinations
def print_log(df_assets, fdr_threshold, prefix=""):
    # if master results came back empty, break early
    if df_assets["master_results"].empty:
        print(f"No results to write to log files for pass: {prefix}")
        return

    # Add a unique file prefix (e.g., 'test1_' or 'test2_') so they do not overwrite each other
    pos_log_path = f"02_exploratory_analysis/outputs/{prefix}positively_associated_vars.txt"
    neg_log_path = f"02_exploratory_analysis/outputs/{prefix}negatively_associated_vars.txt"
    full_table_log_path = f"02_exploratory_analysis/outputs/{prefix}full_correlation_table.txt"
    filtered_table_log_path = f"02_exploratory_analysis/outputs/{prefix}filtered_correlation_table.txt"

    # write into the positively associated file
    with open(pos_log_path, "w") as pos_file:
        if df_assets["pos_dependent_vars"]:
            pos_file.write("\n".join(df_assets["pos_dependent_vars"]))
        else:
            pos_file.write("--- No Positively Associated Variables ---")

    # write into the negatively associated file
    with open(neg_log_path, "w") as neg_file:
        if df_assets["neg_dependent_vars"]:
            neg_file.write("\n".join(df_assets["neg_dependent_vars"]))
        else:
            neg_file.write("--- No Negatively Associated Variables ---")

    # formats table by sorting and applying a mask to hide duplicate label rows
    def get_clean_markdown(table):
        if table.empty:
            return "--- No Rows Passed Selection ---"
        sorted_t = table.sort_values(by = ["independent_var", "FDR"]).copy()
        sorted_t["independent_var"] = sorted_t["independent_var"].mask(sorted_t["independent_var"].duplicated(), "")
        return sorted_t.to_markdown(index = False)

    # write full table
    with open(full_table_log_path, "w") as full_table_file:
        full_table_file.write(get_clean_markdown(df_assets["master_results"]))
        
    # write filtered table, which contains only the variables that passed the FDR significance threshold 
    with open(filtered_table_log_path, "w") as filtered_table_file:
        passed_fdr = df_assets["master_results"][df_assets["master_results"]["FDR"] <= fdr_threshold]
        filtered_table_file.write(get_clean_markdown(passed_fdr))


def main():
    placental_metrics = [
        "placental infarction", "distal villous hypoplasia focal/diffuse", "accelerated villous maturation", "increased syncytial knots", 
        "decidual arteriopathy membrane role/basal plate/both", "segmental avascular villi small/intermediate/large", "delayed villous maturation", 
        "maternal inflammatory response stage/grade", "villitis of unknown etiology, high/low grade, focal/diffuse", "increased perivillous fibrin deposition", 
        "chorangiosis"
    ]
    delivery_metrics = [
        "maternal age", "weight (kg)", "prepregnancy weight self or record", "prepregnancy BMI self or record", "gravida", "parity", "diabetes", "chtn"
    ]
    strict_delivery_metrics = [
        "gest age del", "birthweight", "apgar 1", "apgar 5", "nicu days"
    ]

    placental_df, delivery_df = load_sheets()
    
    # run and log test 1: placental histopathology variables vs delivery variables
    test1_assets = run_test_1_placental_vs_delivery(
        df_placental = placental_df,
        df_delivery = delivery_df,
        placental_vars = placental_metrics,
        delivery_vars = delivery_metrics,
        fdr_threshold = 0.05
    )
    print_log(test1_assets, fdr_threshold=0.05, prefix="placenta_")

    # run and log test 2: fitbit data vs placental histopathology + delivery variables
    master_fitbit_csv = "01_data_cleaning/processed_data/master_fitbit_clinical_correlation_data.csv"
    
    test2_assets = run_test_2_fitbit_vs_all_outcomes(
        master_fitbit_path = master_fitbit_csv,
        clinical_vars = strict_delivery_metrics,
        placental_vars = placental_metrics,
        fdr_threshold = 0.05
    )
    
    print_log(test2_assets, fdr_threshold = 0.05, prefix = "fitbit_")


if __name__ == "__main__":
    main()