import logging
from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links

# Set up basic logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# loads the fitbit and placental sheets
def load_sheets():
    fitbit_sheet = pd.read_csv("01_data_cleaning/processed_data/processed_fitbit_data.csv")
    placental_sheet = pd.read_csv("01_data_cleaning/processed_data/processed_placental_data.csv")
    return fitbit_sheet, placental_sheet


def detect_outliers(series):
    """Detects extreme outliers in a numerical pandas Series using the IQR method.

    Returns True if any extreme outliers are found, otherwise False.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers.any()


def run_glm_fitbit(dat, fitbit_metrics, outcomes):
    """Runs a Generalized Linear Model (GLM) for each combination of

    Fitbit metrics (x) and outcome variables (y) adjusting for specified covariates
    and interaction terms.
    """

    # Set up a file handler to log model outputs and progress
    fhandler = logging.FileHandler(
        filename="output/glm_fitbit_analysis.log", mode="a"
    )
    logger.addHandler(fhandler)

    count = 0
    completed = 0
    results = []

    # Loop through each of the 19 Fitbit metrics (x)
    for x in fitbit_metrics:
        # Loop through each outcome variable (y)
        for y in outcomes:

            # 1. Force Gaussian distribution as requested
            family_type = family.Gaussian()
            family_type.link = links.identity()  # Identity link function for Gaussian
            family_name = "Gaussian"
            family_link = "Identity"

            # 2. Subset the data to include the outcome (y), the predictor (x), and all covariates.
            sub = dat[
                [
                    "Patient-ID",
                    y,
                    x,
                    "maternal_age",
                    "bmi",
                    "race",
                    "fetal_sex",
                    "smoking_status",
                ]
            ].copy()

            # Rename columns temporarily for clean formula handling
            sub.rename(columns={y: "Y_Outcome", x: "X_Metric"}, inplace=True)

            # Ensure numerical variables are correctly typed as float
            sub["Y_Outcome"] = sub["Y_Outcome"].astype(np.float64)
            sub["X_Metric"] = sub["X_Metric"].astype(np.float64)
            sub["maternal_age"] = sub["maternal_age"].astype(np.float64)
            sub["bmi"] = sub["bmi"].astype(np.float64)

            # Check for extreme outliers in the outcome variable to decide on robust covariance
            extreme_outliers = detect_outliers(sub["Y_Outcome"])

            try:
                # 3. Define the GLM Formula:
                # y ~ x + age + x*age + bmi + x*bmi + race + x*race + fetal_sex + smoking_status
                ols_model = (
                    "Y_Outcome ~ X_Metric * maternal_age + "
                    "X_Metric * bmi + "
                    "X_Metric * C(race) + "
                    "C(fetal_sex) + "
                    "C(smoking_status)"
                )

                # Initialize the GLM model using statsmodels formula API (smf)
                model = smf.glm(
                    ols_model, data=sub, family=family_type, missing="drop"
                )

                # Fit the model (use robust standard errors 'hc0' if extreme outliers are detected)
                if extreme_outliers:
                    fitted_model = model.fit(cov_type="hc0", maxiter=100)
                else:
                    fitted_model = model.fit(maxiter=100)

                # Calculate pseudo R-squared to evaluate model fit quality
                null_deviance = fitted_model.null_deviance
                residual_deviance = fitted_model.deviance
                pseudo_r_squared = (
                    1 - (residual_deviance / null_deviance)
                    if null_deviance != 0
                    else np.nan
                )

                # Save model results parameters and p-values into our results list
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
                # Catch and log any errors that occur during individual model fitting
                logger.info(
                    f"Failed Fitbit metric {x} and outcome {y} with error {str(e)}"
                )

            count += 1

    # Convert results into a structured Pandas DataFrame for downstream analysis
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
    fitbit_sheet, placental_sheet = load_sheets()

    print(fitbit_sheet.columns)

    merged = fitbit_sheet.merge(placental_sheet, left_on='record_id', right_on='id', how='inner')

    # 3. Define your lists of variables
    fitbit_metrics = [
        'activities_summary_steps',
        'activities_summary_totaldistances',
       'activities_summary_veryactiveminutes',
       'sleep_summary_total_minutes_asleep',
       'activities_-_summary_-_activescore',
       'activities_-_summary_-_activitycalories',
       'activities_-_summary_-_caloriesbmr',
       'activities_-_summary_-_caloriesout',
       'activities_-_summary_-_totaldistances',
       'activities_-_summary_-_fairlyactiveminutes',
       'activities_-_summary_-_lightlyactiveminutes',
       'activities_-_summary_-_marginalcalories',
       'activities_-_summary_-_sedentaryminutes',
       'activities_-_summary_-_steps',
       'activities_-_summary_-_veryactiveminutes',
       'heart_rate_-_resting_heart_rate',
       'heart_rate_-_zone:_out_of_range_-_caloriesout',
       'heart_rate_-_zone:_out_of_range_-_min',
       'heart_rate_-_zone:_out_of_range_-_max',
       'heart_rate_-_zone:_out_of_range_-_minutes',
       'heart_rate_-_zone:_fat_burn_-_caloriesout',
       'heart_rate_-_zone:_fat_burn_-_min',
       'heart_rate_-_zone:_fat_burn_-_max',
       'heart_rate_-_zone:_fat_burn_-_minutes',
       'heart_rate_-_zone:_cardio_-_caloriesout',
       'heart_rate_-_zone:_cardio_-_min', 'heart_rate_-_zone:_cardio_-_max',
       'heart_rate_-_zone:_cardio_-_minutes',
       'heart_rate_-_zone:_peak_-_caloriesout',
       'heart_rate_-_zone:_peak_-_min', 'heart_rate_-_zone:_peak_-_max',
       'heart_rate_-_zone:_peak_-_minutes', 'sleep_-_summary_-_stages_-_deep',
       'sleep_-_summary_-_stages_-_light', 'sleep_-_summary_-_stages_-_rem',
       'sleep_-_summary_-_stages_-_wake',
       'sleep_-_summary_-_total_minutes_asleep',
       'sleep_-_summary_-_total_sleep_records',
       'sleep_-_summary_-_total_time_in_bed'
    ]

    print(fitbit_metrics.__len__())
    
    outcomes = [
        'outcome_1', 'outcome_2'
    ] # Replace with your actual outcome column names

    # 4. Run the analysis function on the unified master dataframe
    results_df = run_glm_fitbit(merged, fitbit_metrics, outcomes)
    
    # 5. Save the final results to a CSV file
    results_df.to_csv('output/final_glm_results.csv', index=False)
    print("Analysis complete! Results saved.")


if __name__ == "__main__":
    main()