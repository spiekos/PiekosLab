import logging
from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links
from statsmodels.genmod.families import family


# set up basic logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        filename="04_results_and_figures/models/glm_fitbit_analysis.log", mode="a"
    )
    logger.addHandler(fhandler)

    count = 0
    completed = 0
    results = []

    # loop through each of the Fitbit metrics (x)
    for x in fitbit_metrics:
        # loop through each placental outcome variable (y)
        for y in outcomes:
            # 1. Force Gaussian distribution as requested
            family_type = family.Gaussian()
            family_type.link = links.Identity()  # identity link function for Gaussian
            family_name = "Gaussian"
            family_link = "Identity"

            # subset the data to include the outcome (y), the predictor (x), and all covariates
            sub = dat[
                [
                    "record_id",
                    y,
                    x,
                    "maternal_age",
                    "prepregnancy_bmi_self_or_record",
                    "race",
                    "infant_sex",
                    "smoking",
                ]
            ].copy()

            # rename columns temporarily for clean formula handling
            sub.rename(columns={y: "Y_Outcome", x: "X_Metric"}, inplace=True)

            # ensure numerical variables are correctly typed as float
            sub['Y_Outcome'] = pd.to_numeric(sub['Y_Outcome'], errors='coerce')
            sub['X_Metric'] = pd.to_numeric(sub['X_Metric'], errors='coerce')
            sub['maternal_age'] = pd.to_numeric(sub['maternal_age'], errors='coerce')
            sub['prepregnancy_bmi_self_or_record'] = pd.to_numeric(sub['prepregnancy_bmi_self_or_record'], errors='coerce')

            # check for extreme outliers in the outcome variable to decide on robust covariance
            extreme_outliers = detect_outliers(sub["Y_Outcome"])

            try:
                # define the GLM formula:
                # y ~ x + age + x*age + bmi + x*bmi + race + x*race + infant_sex + smoking
                ols_model = (
                    "Y_Outcome ~ X_Metric * maternal_age + "
                    "X_Metric * prepregnancy_bmi_self_or_record + "
                    "X_Metric * C(race) + "
                    "C(infant_sex) + "
                    "C(smoking)"
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
                        pseudo_r_squared,
                    )
                )
                completed += 1

            except Exception as e:
                logger.info(
                    f"Failed Fitbit metric {x} and outcome {y} with error {str(e)}"
                )

            count += 1

            break # DEBUGGING. DELETE

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
            "Pseudo_R_Squared",
        ]
        df_results = pd.DataFrame(results, columns=res_columns)
        return df_results

    return pd.DataFrame()


def main():
    fitbit_sheet, placental_sheet, clinical_sheet = load_sheets()

    print(fitbit_sheet.columns)

    merged = fitbit_sheet.merge(placental_sheet, left_on='record_id', right_on='id', how='inner')
    merged = merged.merge(clinical_sheet, left_on='record_id', right_on='id', how='inner')

    fitbit_metrics = [
        'activities_summary_activescore', 'activities_summary_activitycalories',
       'activities_summary_caloriesbmr', 'activities_summary_caloriesout',
       'activities_summary_fairlyactiveminutes',
       'activities_summary_lightlyactiveminutes',
       'activities_summary_marginalcalories',
       'activities_summary_sedentaryminutes', 'heart_rate_resting_heart_rate',
       'heart_rate_zone:_out_of_range_caloriesout',
       'heart_rate_zone:_out_of_range_min',
       'heart_rate_zone:_out_of_range_max',
       'heart_rate_zone:_out_of_range_minutes',
       'heart_rate_zone:_fat_burn_caloriesout',
       'heart_rate_zone:_fat_burn_min', 'heart_rate_zone:_fat_burn_max',
       'heart_rate_zone:_fat_burn_minutes',
       'heart_rate_zone:_cardio_caloriesout', 'heart_rate_zone:_cardio_min',
       'heart_rate_zone:_cardio_max', 'heart_rate_zone:_cardio_minutes',
       'heart_rate_zone:_peak_caloriesout', 'heart_rate_zone:_peak_min',
       'heart_rate_zone:_peak_max', 'heart_rate_zone:_peak_minutes',
       'sleep_summary_stages_deep', 'sleep_summary_stages_light',
       'sleep_summary_stages_rem', 'sleep_summary_stages_wake',
       'sleep_summary_total_sleep_records', 'sleep_summary_total_time_in_bed',
       'activities_summary_steps', 'activities_summary_totaldistances',
       'activities_summary_veryactiveminutes',
       'sleep_summary_total_minutes_asleep'
    ]

    print("Length of fitbit_metrics list: ", len(fitbit_metrics))

    for metric in fitbit_metrics:
        print(merged[metric].value_counts())
        print("\n")

    # Returns non-null, non-zero counts for all metrics at once
    non_zero_counts = (merged[fitbit_metrics].notna() & (merged[fitbit_metrics] != 0)).sum()
    print(non_zero_counts)
    
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