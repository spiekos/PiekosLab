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
    logger = logging.getLogger("GLM_Logger")
    logger.addHandler(fhandler)

    count = 0
    completed = 0
    results = []

    primary_predictor = "maternal_age"

    # loop through each of the Fitbit metrics (x)
    for x in fitbit_metrics:
        # loop through each placental outcome variable (y)
        for y in outcomes:
            required_cols = [
                "id",
                y,
                primary_predictor,
                x,
                "prepregnancy_bmi_self_or_record",
                "race",
                "infant_sex",
                "smoking",
                "parity"
            ]

            if not all(col in dat.columns for col in required_cols):
                continue

            sub = dat[required_cols].copy()

            numeric_cols = [
                y,
                primary_predictor,
                x,
                "prepregnancy_bmi_self_or_record",
                "parity"
            ]
            for col in numeric_cols:
                sub[col] = pd.to_numeric(sub[col], errors="coerce")

            sub = sub.dropna()

            if len(sub) < 15:
                print(
                    f"Skipping {y} ~ {primary_predictor} ({x}): Insufficient overlapping rows ({len(sub)})"
                )
                continue
            if (
                sub[primary_predictor].nunique() <= 1
                or sub[primary_predictor].std() < 1e-8
            ):
                print(
                    f"Skipping {primary_predictor}: Zero/near-zero variance in cohort"
                )
                continue

            # determine distribution family based on outcome (y) variable
            is_binary_outcome = set(sub[y].unique()).issubset({0, 1})

            if is_binary_outcome:
                family_type = sm.families.Binomial(link=sm.families.links.Logit())
                family_name = "Binomial (Logistic)"
                family_link = "Logit"
            else:
                # evaluate normality of continuous outcome Y
                _, ks_p_value = scipy.stats.kstest(sub[y], "norm")

                # if continuous outcome y is heavily skewed or non-negative non-normal, use Gamma distribution
                if (abs(sub[y].skew()) > 1.5 or ks_p_value < 0.05) and (sub[y] >= 0).all():
                    # shift zeros slightly positive if Gamma family is selected
                    if (sub[y] == 0).any():
                        min_val = sub.loc[sub[y] > 0, y].min()
                        min_val = (min_val / 2.0 if pd.notna(min_val) else 1e-6)
                        sub.loc[sub[y] == 0, y] = min_val

                    family_type = sm.families.Gamma(link=sm.families.links.Log())
                    family_name = "Gamma"
                    family_link = "Log"
                else:
                    family_type = sm.families.Gaussian(link=sm.families.links.Identity())
                    family_name = "Gaussian"
                    family_link = "Identity"

            extreme_outliers = detect_outliers(sub[y])

            try:
                # GLM formula modeling continuous outcome/predictor with Fitbit metric covariate and interaction
                formula = (
                    f"{y} ~ {primary_predictor} * {x} + "
                    "prepregnancy_bmi_self_or_record + "
                    "C(race) + C(infant_sex) + C(smoking) + parity"
                )

                model = smf.glm(
                    formula, data=sub, family=family_type, missing="drop"
                )

                if extreme_outliers:
                    fitted_model = model.fit(cov_type="hc0", maxiter=100)
                else:
                    fitted_model = model.fit(maxiter=100)

                # pseudo R-squared calculation
                null_dev = fitted_model.null_deviance
                res_dev = fitted_model.deviance
                pseudo_r2 = (
                    1 - (res_dev / null_dev)
                    if (null_dev is not None and null_dev != 0)
                    else np.nan
                )

                res_dict = {
                    "Outcome": y,
                    "Primary_Predictor": primary_predictor,
                    "Fitbit_Metric": x,
                    "N": len(fitted_model.fittedvalues),
                    "Model_Family": family_name,
                    "Link_Function": family_link,
                    "Extreme_Outliers": extreme_outliers,
                    "Converged": fitted_model.converged,
                    "Pseudo_R_Squared": pseudo_r2
                }

                # attach coefficients and p-values explicitly
                for param_name, val in fitted_model.params.items():
                    res_dict[f"coef_{param_name}"] = val
                    res_dict[f"pval_{param_name}"] = fitted_model.pvalues[param_name]

                results.append(res_dict)
                completed += 1

            except Exception as e:
                logger.info(
                    f"Failed model for {y} ~ {primary_predictor} + {x} with error: {str(e)}"
                )

            count += 1

    if results:
        return pd.DataFrame(results)

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