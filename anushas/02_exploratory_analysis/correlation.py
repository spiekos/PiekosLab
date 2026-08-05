import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import ks_2samp, anderson_ksamp


# loads sheets (sheet1 contains the placental histopathology data, sheet2 contains the variables of interest data, sheet3 contains the cleaned clinical data)
def load_sheets():
    sheet1 = pd.read_csv("01_data_cleaning/processed_data/processed_placental_data.csv")
    sheet2 = pd.read_csv("00_raw_data/dp3 master table v2.xlsx - variables of interest.csv")
    sheet3 = pd.read_csv("01_data_cleaning/processed_data/processed_clinical_data.csv")
    sheet4 = pd.read_csv("01_data_cleaning/processed_data/processed_fitbit_data.csv")

    return sheet1, sheet2, sheet3, sheet4


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
    fitbit_features = [col for col in master_df.columns if any(prefix in col for prefix in ["activities", "heart_rate", "sleep"])]
    all_outcome_targets = [v for v in (clinical_vars + placental_vars) if v in master_df.columns]

    if not fitbit_features:
        print("Warning: No columns containing 'activities', 'heart_rate', or 'sleep' found in the master Fitbit file.")
        return {"master_results": pd.DataFrame(), "pos_dependent_vars": [], "neg_dependent_vars": []}

    return run_spearman_core(master_df, fitbit_features, all_outcome_targets, id_col="id", fdr_threshold=fdr_threshold)


# test 3: runs correlation test comparing the following variables with each other:
# age, race, ethnicity, fetal sex, prepregnancy BMI, smoking
# builds a correlation matrix, creates an annotated heatmap figure, and calculates VIF scores
def run_test_3_demographics_collinearity(df):
    # skip first structural header/metadata row if present
    if df.index.max() > 0:
        df = df.iloc[1:].copy()

    df.columns = df.columns.str.strip()

    # encode the "fetal sex" column: male = 1, female = 0
    if "infant_sex" in df.columns:
        df["infant_sex_encoded"] = df["infant_sex"].astype(str).str.strip().str.upper().map({"M": 1, "MALE": 1, "F": 0, "FEMALE": 0}).fillna(0).astype(int)
    else:
        df["infant_sex_encoded"] = np.nan

    # encode the "smoking" column
    if "smoking" in df.columns:
        df["smoking_encoded"] = df["smoking"].astype(str).str.strip().str.upper().map({
            "NEVER": 0,
            "QUIT": 0,
            "YES": 1
        })
    else:
        df["smoking_encoded"] = np.nan

    race_cols = [c for c in df.columns if c.startswith("race_") and c != "race_is_missing"]

    # create column list to run correlation test on
    target_base = ["maternal_age", "infant_sex_encoded", "prepregnancy_BMI_self_or_record", "smoking_encoded", "hispanic/latino"]    
    analysis_vars = [v for v in target_base if v in df.columns and not df[v].isna().all()] + race_cols

    # extract modeling slice and drop complete NaN arrays
    modeling_df = df[analysis_vars].dropna().apply(pd.to_numeric, errors='coerce').dropna()
    
    modeling_df = modeling_df.dropna(how = "all")
    
    if modeling_df.empty or len(modeling_df) < 5:
        print("Warning: Insufficient numeric overlap entries to evaluate Test 3.")
        return

    # build the Spearman correlation matrix
    corr_matrix = modeling_df.corr(method = "spearman")

    # render and save the annotated heatmap figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1,
        square=True, 
        linewidths=0.5
    )
    plt.title("Baseline Demographics Correlation Matrix (Spearman $\\rho$)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig("04_results_and_figures/correlations/test3/demographics_correlation_heatmap.png", dpi=300)
    plt.close()

    # process Variance Inflation Factors (VIF)
    vif_df_clean = modeling_df.dropna(how = "any").copy()
    if len(vif_df_clean) > 10:
        vif_records = []
        X = vif_df_clean.copy()
        X['intercept'] = 1.0

        for col in vif_df_clean.columns:
            col_idx = X.columns.get_loc(col)
            vif_val = variance_inflation_factor(X.values, col_idx)
            vif_records.append({"Variable": col, "VIF": vif_val})

        vif_df = pd.DataFrame(vif_records).sort_values(by="VIF", ascending=False)
        with open("04_results_and_figures/correlations/test3/demographics_vif.txt", "w") as vif_file:
            vif_file.write(vif_df.to_markdown(index=False))
    else:
        print(f"VIF Warning: Too many missing values across columns (only {len(vif_df_clean)} complete rows). Skipping VIF calculation to avoid biased scores.")
    
    return corr_matrix


# compares distributions of metrics between control and complications across timepoints
# @param: test_type: 'ks' for Kolmogorov-Smirnov or 'ad' for Anderson-Darling
def run_test_4_compare_populations(df, feature_cols, test_type='ks'):
    records = []
    
    df['pregnancy_week'] = (df['timepoint'] // 7) + 1
    weeks = df["pregnancy_week"].unique()
    # filter to only include timepoints during the first trimester
    weeks = weeks[weeks < 14]
    
    for wk in weeks:
        df_tp = df[df["pregnancy_week"] == wk]
        
        control_data = df_tp[df_tp["group_bin"] == 1]
        comp_data = df_tp[df_tp["group_bin"] == 0]
        
        for feature in feature_cols:
            x = control_data[feature].dropna()
            y = comp_data[feature].dropna()
            
            if len(x) < 2 or len(y) < 2 or x.nunique() <= 1 or y.nunique() <= 1:
                continue
                
            if test_type == 'ks':
                # two-sample Kolmogorov-Smirnov test
                stat, p_val = ks_2samp(x, y)
            elif test_type == 'ad':
                # two-sample Anderson-Darling test
                res = anderson_ksamp([x, y], variant='midrank')
                stat = res.statistic
                p_val = res.pvalue
            else:
                raise ValueError("test_type must be 'ks' or 'ad'")
                
            records.append({
                'week': wk,
                'feature': feature,
                'test_statistic': stat,
                'p_value': p_val
            })
            
    results_df = pd.DataFrame(records)
    
    if results_df.empty:
        print("No valid comparisons made. Check data formatting and column names.")
        return results_df

    # apply Benjamini-Hochberg multiple hypothesis correction
    reject, pvals_corrected, _, _ = multipletests(
        results_df['p_value'], 
        alpha=0.05, 
        method='fdr_bh'
    )
    
    results_df['p_value_adjusted'] = pvals_corrected
    results_df['is_differentially_distributed'] = reject
    
    return results_df


# print calculated correlation data into respective file destinations
def print_log(df_assets, fdr_threshold, test, suffix):
    # if master results came back empty, break early
    if df_assets["master_results"].empty:
        print(f"No results to write to log files for pass: {suffix}")
        return

    # Add a unique file suffix so they do not overwrite each other
    pos_log_path = f"04_results_and_figures/correlations/{test}/positively_associated_vars_{suffix}.txt"
    neg_log_path = f"04_results_and_figures/correlations/{test}/negatively_associated_vars_{suffix}.txt"
    full_table_log_path = f"04_results_and_figures/correlations/{test}/full_correlation_table_{suffix}.txt"
    filtered_table_log_path = f"04_results_and_figures/correlations/{test}/filtered_correlation_table_{suffix}.txt"

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
    placental_df, delivery_df, clinical_df, fitbit_df = load_sheets()

    placental_metrics = [
        "placental_infarction", "distal_villous_hypoplasia_focal/diffuse", "accelerated_villous_maturation", "increased_syncytial_knots", 
        "decidual_arteriopathy_membrane_role/basal_plate/both", "segmental_avascular_villi_small/intermediate/large", "delayed_villous_maturation", 
        "maternal_inflammatory_response_stage/grade", "villitis_of_unknown_etiology,_high/low_grade,_focal/diffuse", "increased_perivillous_fibrin_deposition", 
        "chorangiosis"
    ]
    delivery_metrics = [
        "maternal_age", "weight_(kg)", "prepregnancy_weight_self_or_record", "prepregnancy_bmi_self_or_record", "gravida", "parity", "diabetes", "chtn"
    ]
    strict_delivery_metrics = [
        "gest_age_del", "birthweight", "apgar_1", "apgar_5", "nicu_days"
    ]
    fitbit_metrics = [col for col in fitbit_df.columns if col.startswith(("activities", "sleep", "heart_rate"))]

    # run and log test 1: placental histopathology variables vs delivery variables
    test1_assets = run_test_1_placental_vs_delivery(
        df_placental = placental_df,
        df_delivery = delivery_df,
        placental_vars = placental_metrics,
        delivery_vars = delivery_metrics,
        fdr_threshold = 0.05
    )
    print_log(test1_assets, fdr_threshold=0.05, test="test1", suffix="placental")

    # run and log test 2: fitbit data vs placental histopathology + delivery variables
    master_fitbit_csv = "01_data_cleaning/processed_data/master_fitbit_clinical_correlation_data.csv"
    
    test2_assets = run_test_2_fitbit_vs_all_outcomes(
        master_fitbit_path = master_fitbit_csv,
        clinical_vars = strict_delivery_metrics,
        placental_vars = placental_metrics,
        fdr_threshold = 0.05
    )
    print_log(test2_assets, fdr_threshold = 0.05, test="test2", suffix = "fitbit")

    test3_results = run_test_3_demographics_collinearity(clinical_df)
    test3_results.to_csv("04_results_and_figures/correlations/test3/demographics_correlation_matrix.csv")

    test4_results = run_test_4_compare_populations(fitbit_df, fitbit_metrics)
    sig_test4_results = test4_results[test4_results['is_differentially_distributed'] == True].sort_values('p_value_adjusted')
    test4_results.to_csv("04_results_and_figures/correlations/test4/fitbit_differential_distribution.csv")
    sig_test4_results.to_csv("04_results_and_figures/correlations/test4/fitbit_differential_distribution_significant.csv")


if __name__ == "__main__":
    main()