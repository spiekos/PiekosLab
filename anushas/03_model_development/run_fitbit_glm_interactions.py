import logging
from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links
from statsmodels.genmod.families import family
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


# runs a generalized linear model (GLM) for each combination of Fitbit metrics (x) and placental outcome variables (y),
# adjusting for specified covariates and interaction terms.
def run_glm_fitbit(dat, fitbit_metrics, outcomes):
    # set up a file handler to log model outputs and progress
    fhandler = logging.FileHandler(
        filename = "04_results_and_figures/models/glm_fitbit_analysis.log", mode="w"
    )
    logger.addHandler(fhandler)

    count = 0
    completed = 0
    results = []

    # loop through each of the Fitbit metrics (x)
    for x in fitbit_metrics:
        # loop through each placental outcome variable (y)
        for y in outcomes:
            # isolate and clean the series
            data_subset = dat[[x, y]].dropna()
            data_subset[x] = pd.to_numeric(data_subset[x], errors="coerce")
            data_subset = data_subset.dropna()

            # safety checks before running GLM
            if len(data_subset) < 15:  # too few samples to model reliably
                print(f"Skipping {x}: Insufficient overlapping data ({len(data_subset)} rows)")
                continue
            if data_subset[x].nunique() <= 1:
                print(f"Skipping {x}: Zero variance (constant values)")
                continue
            if data_subset[x].std() < 1e-8 or np.isnan(data_subset[x].std()):
                print(f"Skipping {x}: Near-zero variance in aligned cohort")
                continue

            is_binary_outcome = set(data_subset[y].dropna().unique()).issubset({0, 1})

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

                # evaluate data normality using a Kolmogorov-Smirnov test
                _, ks_p_value = scipy.stats.kstest(data_subset[x], "norm")

                # if data is heavily skewed or fails normality, use a Gamma test with a log link
                if (abs(data_subset[x].skew()) > 1.5 or ks_p_value < 0.05):
                    # handle potential zero values if Gamma is chosen (Gamma requires strictly positive values)
                    if (data_subset[x] == 0).any():
                        min_val = data_subset.loc[data_subset[x] > 0, x].min()
                        min_val = (min_val / 2.0 if pd.notna(min_val) else 1e-6)
                        data_subset.loc[data_subset[x] == 0, x] = min_val

                    family_type = sm.families.Gamma()
                    family_type.link = sm.families.links.Log()
                    family_name = "Gamma"
                    family_link = "Log"

            # subset the data to include the outcome (y), the predictor (x), and all covariates
            sub = dat[
                [
                    "id",
                    y,
                    x,
                    "maternal_age",
                    "prepregnancy_bmi_self_or_record",
                    "race",
                    "infant_sex",
                    "smoking",
                    "parity"
                ]
            ].copy()

            # rename columns temporarily for clean formula handling
            sub.rename(columns={y: "Y_Outcome", x: "X_Metric"}, inplace=True)

            # ensure numerical variables are correctly typed as float
            sub['Y_Outcome'] = pd.to_numeric(sub['Y_Outcome'], errors='coerce')
            sub['X_Metric'] = pd.to_numeric(sub['X_Metric'], errors='coerce')
            sub['maternal_age'] = pd.to_numeric(sub['maternal_age'], errors='coerce')
            sub['prepregnancy_bmi_self_or_record'] = pd.to_numeric(sub['prepregnancy_bmi_self_or_record'], errors='coerce')
            sub['parity'] = pd.to_numeric(sub['parity'], errors='coerce')

            # check for extreme outliers in the outcome variable to decide on robust covariance
            extreme_outliers = detect_outliers(sub["Y_Outcome"])

            try:
                # define the GLM formula:
                # y ~ x + age + x*age + bmi + x*bmi + race + x*race + infant_sex + smoking
                ols_model = (
                    "Y_Outcome ~ X_Metric + maternal_age + "
                    "prepregnancy_bmi_self_or_record + "
                    "C(race) + "
                    "C(infant_sex) + "
                    "C(smoking) + "
                    "parity"
                )

                # initialize the GLM model using statsmodels formula API (smf)
                model = smf.glm(
                    ols_model, data=sub, family=family_type, missing="drop"
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

                results.append(
                    (
                        x,
                        y,
                        len(fitted_model.fittedvalues),
                        family_name,
                        family_link,
                        extreme_outliers,
                        fitted_model.converged,
                        *fitted_model.params,
                        *fitted_model.pvalues,
                        pseudo_r_squared
                    )
                )
                completed += 1

            except Exception as e:
                logger.info(
                    f"Failed Fitbit metric {x} and outcome {y} with error {str(e)}"
                )

            count += 1

    # convert results into a structured Pandas DataFrame for downstream analysis
    if results:
        res_columns = [
            "Fitbit_Metric",
            "Outcome",
            "N",
            "Model Family",
            "Link Function",
            "Extreme Outliers",
            "Converged",
            *fitted_model.params.index,
            *[str(col) + "_p" for col in fitted_model.pvalues.index],
            "Pseudo_R_Squared"
        ]
        df_results = pd.DataFrame(results, columns=res_columns)
        return df_results

    return pd.DataFrame()


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

    outcomes = [
        'distal_villous_hypoplasia_focal/diffuse', 'accelerated_villous_maturation', 'increased_syncytial_knots',
        'decidual_arteriopathy_membrane_role/basal_plate/both', 'segmental_avascular_villi_small/intermediate/large', 'delayed_villous_maturation',
        'maternal_inflammatory_response_stage/grade', 'villitis_of_unknown_etiology,_high/low_grade,_focal/diffuse',
        'increased_perivillous_fibrin_deposition', 'chorangiosis', 'spontaneous_preterm_birth'
    ]

    results_df = run_glm_fitbit(merged, fitbit_metrics, outcomes)
    
    results_df.to_csv('04_results_and_figures/models/final_glm_results.csv', index=False)


if __name__ == "__main__":
    main()