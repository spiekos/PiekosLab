import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy


# set up basic logger configuration
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()


# loads the fitbit and placental sheets
def load_sheets():
    fitbit_sheet = pd.read_csv("01_data_cleaning/processed_data/processed_fitbit_data.csv")
    placental_sheet = pd.read_csv("01_data_cleaning/processed_data/processed_placental_data.csv")
    clinical_sheet = pd.read_csv("01_data_cleaning/processed_data/processed_clinical_data.csv")
    return fitbit_sheet, placental_sheet, clinical_sheet


# detects extreme outliers in a numerical pandas series using the IQR method
def detect_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers.any()


# runs a generalized linear model (GLM) for each combination of Fitbit metrics (x) and clinical variables (y),
# adjusting for specified covariates and interaction terms.
def run_glm_fitbit(dat, fitbit_metrics, clinical_vars):
    # set up a file handler to log model outputs and progress
    logger = logging.getLogger(__name__)
    fhandler = logging.FileHandler(
        filename = "04_results_and_figures/models/glm_fitbit_analysis.log", mode="w"
    )
    logger.addHandler(fhandler)

    count = 0
    completed = 0
    results = []

    # loop through each of the Fitbit metrics (x)
    for x in fitbit_metrics:
        # loop through each clinical variable (y)
        for y in clinical_vars:
            # dynamically filter covariates to exclude Y if Y is a clinical variable
            base_covariates = [
                "maternal_age",
                "prepregnancy_bmi_self_or_record",
                "race",
                "infant_sex",
                "smoking",
                "parity"
            ]
            active_covariates = [cov for cov in base_covariates if cov != y]

            # extract complete subset with dynamic covariates
            cols_to_extract = ["id", y, x] + [c for c in active_covariates if c in dat.columns]
            sub = dat[cols_to_extract].copy()

            # rename columns temporarily for clean formula handling
            sub.rename(columns={y: "Y_Clinical", x: "X_Metric"}, inplace=True)

            # ensure numerical variables are correctly typed as float
            sub['Y_Clinical'] = pd.to_numeric(sub['Y_Clinical'], errors='coerce')
            sub['X_Metric'] = pd.to_numeric(sub['X_Metric'], errors='coerce')
            if 'maternal_age' in sub.columns:
                sub['maternal_age'] = pd.to_numeric(sub['maternal_age'], errors='coerce')
            if 'prepregnancy_bmi_self_or_record' in sub.columns:
                sub['prepregnancy_bmi_self_or_record'] = pd.to_numeric(sub['prepregnancy_bmi_self_or_record'], errors='coerce')
            if 'parity' in sub.columns:
                sub['parity'] = pd.to_numeric(sub['parity'], errors='coerce')

            data_subset = sub.dropna()

            data_subset['Y_Clinical'] = data_subset['Y_Clinical'].astype(float)
            data_subset['X_Metric'] = data_subset['X_Metric'].astype(float)

            # safety checks before running GLM
            if len(data_subset) < 15:
                print(f"Skipping {x} vs {y}: Insufficient overlapping data ({len(data_subset)} rows)")
                continue
            if data_subset['X_Metric'].nunique() <= 1:
                print(f"Skipping {x} vs {y}: Zero variance in predictor {x}")
                continue
            if data_subset['X_Metric'].std() < 1e-8 or np.isnan(data_subset['X_Metric'].std()):
                print(f"Skipping {x} vs {y}: Near-zero variance in aligned cohort for {x}")
                continue

            is_binary_outcome = set(data_subset['Y_Clinical'].unique()).issubset({0, 1})

            if is_binary_outcome:
                family_type = sm.families.Binomial()
                family_type.link = sm.families.links.Logit()
                family_name = "Binomial (Logistic)"
                family_link = "Logit"
            else:
                family_type = sm.families.Gaussian()
                family_type.link = sm.families.links.Identity()
                family_name = "Gaussian"
                family_link = "Identity"

                # check normality and skewness on outcome Y_Clinical
                _, ks_p_value = scipy.stats.kstest(data_subset['Y_Clinical'], "norm")

                # if continuous outcome is heavily right-skewed and non-negative, use Gamma family
                if (abs(data_subset['Y_Clinical'].skew()) > 1.5 or ks_p_value < 0.05) and (data_subset['Y_Clinical'] >= 0).all():
                    # impute small offsets for zeros in outcome Y_Clinical
                    if (data_subset['Y_Clinical'] == 0).any():
                        min_val = data_subset.loc[data_subset['Y_Clinical'] > 0, 'Y_Clinical'].min()
                        min_val = (min_val / 2.0 if pd.notna(min_val) else 1e-6)
                        data_subset.loc[data_subset['Y_Clinical'] == 0, 'Y_Clinical'] = min_val

                    family_type = sm.families.Gamma()
                    family_type.link = sm.families.links.Log()
                    family_name = "Gamma"
                    family_link = "Log"

            # check for extreme outliers in outcome variable
            extreme_outliers = detect_outliers(data_subset["Y_Clinical"])

            try:
                # build dynamic formula based on active covariates
                formula_terms = ["X_Metric"]
                for cov in active_covariates:
                    if cov in ["race", "infant_sex", "smoking"]:
                        formula_terms.append(f"C({cov})")
                    else:
                        formula_terms.append(cov)

                ols_model = "Y_Clinical ~ " + " + ".join(formula_terms)

                # initialize the GLM model using statsmodels formula API (smf)
                model = smf.glm(
                    ols_model, data=data_subset, family=family_type, missing="drop"
                )

                # fit the model (use robust standard errors 'hc0' if extreme outliers are detected)
                if extreme_outliers:
                    fitted_model = model.fit(cov_type="hc0", maxiter=100)
                else:
                    fitted_model = model.fit(maxiter=100)

                # calculate pseudo R-squared to evaluate model fit quality
                null_deviance = fitted_model.null_deviance
                residual_deviance = fitted_model.deviance
                pseudo_r_squared = (
                    1 - (residual_deviance / null_deviance)
                    if null_deviance != 0
                    else np.nan
                )

                row_dict = {
                    "Fitbit_Metric": x,
                    "Outcome": y,
                    "N": len(fitted_model.fittedvalues),
                    "Model_Family": family_name,
                    "Link_Function": family_link,
                    "Extreme_Outliers": extreme_outliers,
                    "Converged": fitted_model.converged,
                    "Pseudo_R_Squared": pseudo_r_squared
                }

                # append parameters and p-values
                for param, val in fitted_model.params.items():
                    row_dict[f"coef_{param}"] = val
                for param, pval in fitted_model.pvalues.items():
                    row_dict[f"p_{param}"] = pval

                results.append(row_dict)
                completed += 1

            except Exception as e:
                logger.info(
                    f"Failed Fitbit metric {x} and outcome {y} with error {str(e)}"
                )

            count += 1

    return pd.DataFrame(results)


def main():
    fitbit_sheet, placental_sheet, clinical_sheet = load_sheets()

    # clean placeholder strings/invalid flags across Fitbit metrics
    fitbit_sheet["heart_rate_resting_heart_rate"] = pd.to_numeric(
        fitbit_sheet["heart_rate_resting_heart_rate"].replace("no value", np.nan),
        errors="coerce"
    )

    # if -1.0 is an invalid placeholder for activity score, convert it to NaN
    if "activities_summary_activescore" in fitbit_sheet.columns:
        fitbit_sheet["activities_summary_activescore"] = fitbit_sheet[
            "activities_summary_activescore"
        ].replace(-1.0, np.nan)

    merged = fitbit_sheet.merge(placental_sheet, on='id', how='inner')
    merged = merged.merge(clinical_sheet, on='id', how='inner')

    # merge all columns ending in _x and _y
    # identify all base column names that ended up with _x and _y
    x_cols = [c for c in merged.columns if c.endswith("_x")]

    for x_col in x_cols:
        base_name = x_col[:-2]  # remove '_x'
        y_col = f"{base_name}_y"

        if y_col in merged.columns:
            # fill missing values in _x using _y
            merged[base_name] = merged[x_col].fillna(merged[y_col])
            # drop the original suffix columns
            merged.drop(columns=[x_col, y_col], inplace=True)

    fitbit_metrics = [
        'activities_summary_activitycalories', 'activities_summary_caloriesout', 'activities_summary_fairlyactiveminutes', 
        'activities_summary_lightlyactiveminutes', 'activities_summary_sedentaryminutes', 'heart_rate_resting_heart_rate', 'heart_rate_zone:_fat_burn_minutes',
        'heart_rate_zone:_cardio_minutes', 'heart_rate_zone:_peak_minutes', 'sleep_summary_stages_deep', 'sleep_summary_stages_light', 'sleep_summary_stages_rem', 
        'sleep_summary_stages_wake', 'sleep_summary_total_sleep_records', 'sleep_summary_total_time_in_bed', 'activities_summary_steps', 
        'activities_summary_totaldistances', 'activities_summary_veryactiveminutes', 'sleep_summary_total_minutes_asleep'
    ]

    # filter your fitbit_metrics list dynamically to only include columns with actual valid data
    valid_fitbit_metrics = []
    for metric in fitbit_metrics:
        if metric in fitbit_sheet.columns:
            # check how many non-null numeric values remain after conversion
            clean_series = pd.to_numeric(fitbit_sheet[metric].replace(["no value", -1.0], np.nan), errors="coerce")
            if clean_series.dropna().shape[0] > 10:  # ensure there's a minimum threshold of data
                valid_fitbit_metrics.append(metric)

    '''outcomes = [
        'distal_villous_hypoplasia_focal/diffuse', 'accelerated_villous_maturation', 'increased_syncytial_knots',
        'decidual_arteriopathy_membrane_role/basal_plate/both', 'segmental_avascular_villi_small/intermediate/large', 'delayed_villous_maturation',
        'maternal_inflammatory_response_stage/grade', 'villitis_of_unknown_etiology,_high/low_grade,_focal/diffuse',
        'increased_perivillous_fibrin_deposition', 'chorangiosis', 'spontaneous_preterm_birth'
    ]'''

    clinical_vars = [
        "maternal_age", "weight_(kg)", "prepregnancy_weight_self_or_record", "prepregnancy_bmi_self_or_record", "gravida", "parity",
        "gest_age_del", "birthweight", "apgar_1", "apgar_5", "nicu_days"
    ]

    results_df = run_glm_fitbit(merged, fitbit_metrics, clinical_vars)
    
    results_df.to_csv('04_results_and_figures/models/final_glm_results.csv', index=False)


if __name__ == "__main__":
    main()