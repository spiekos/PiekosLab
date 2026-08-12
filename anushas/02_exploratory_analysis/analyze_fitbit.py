import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from scipy import stats
from statsmodels.stats.multitest import multipletests
from matplotlib.backends.backend_pdf import PdfPages

# load and return the fitbit dataset
def load_sheet():
    sheet1 = pd.read_csv("01_data_cleaning/processed_data/processed_fitbit_data.csv", low_memory = False)
    sheet2 = pd.read_csv("01_data_cleaning/processed_data/processed_clinical_data.csv")
    return sheet1, sheet2


# filters the sheet to only include "fitbit data" events and only include events during pregnancy
def filter_sheet(sheet):
    # filter out all events from the sheet except the "fitbit data" ones
    sheet_filtered = sheet[sheet["event_name"] == "fitbit data"].copy()

    # only include events during pregnancy
    # if gestational age at delivery is not in the dataset, assume that delivery occurred at 40 weeks
    sheet_filtered["current_weeks"] = sheet_filtered["timepoint"] / 7
    sheet_filtered = sheet_filtered[(sheet_filtered["current_weeks"] < sheet_filtered["gest_age_del"]) | 
                                    (sheet_filtered["gest_age_del"].isna() & sheet_filtered["current_weeks"] < 40)]
    return sheet_filtered


# splits the filtered dataset into four smaller datasets, based on which trimester of the pregnancy each datapoint is in
# the four datasets are: 1st trimester, early 2nd trimester, late 2nd/early 3rd trimester, late 3rd trimester
# returns a list containing these four datasets
# note that the input dataset has already been filtered and only includes events during pregnancy
def bucket_data(sheet):
    local_sheet = sheet.copy()

    local_sheet["current_weeks"] = pd.to_numeric(local_sheet["current_weeks"], errors="coerce")

    bins = [float("-inf"), 14, 22, 32, 37, float("inf")]
    labels = ["first", "early_second", "late_second_early_third", "mid_third", "late_third"]

    local_sheet["bin"] = pd.cut(local_sheet["current_weeks"], bins = bins, labels = labels, right = False)

    outputs = []
    
    for label in labels:
        group_sheet = local_sheet[local_sheet["bin"] == label].drop(columns = ["bin"])
        outputs.append(group_sheet)

    return outputs


# returns the total number of unique patients, after data has been filtered
def get_total_patients(sheet):
    # filter for rows for which record id starts with "DP3-"
    # this ensures we only count actual patients
    dp3_patients = sheet[sheet["id"].astype(str).str.startswith("DP3-")]
    return dp3_patients["id"].nunique()


# returns the total number of missing (aka "na") values in the dataset across all columns. excludes the "na" values corresponding to the general information 
# rows for each patient, as these do not represent missing values in the data.
def count_total_missing(sheet):
    # sum all nan values across the whole sheet
    return sheet.isna().sum().sum()


# returns a table containing the number of days missing per patient
# a day is only considered missing if every value for that day is nan
def get_missing_per_patient(sheet, feature_cols):
    local_sheet = sheet.copy()
    # check if all values in each row are nan
    local_sheet["is_missing"] = local_sheet[feature_cols].isna().all(axis = 1)

    # sum the true values by patient id
    result = (
        local_sheet.groupby("id")["is_missing"]
        .sum()
        .reset_index()
        .rename(columns = {"is_missing": "missing_days"})
    )

    return result


def get_missing_patients_per_feature_per_timeframe(sheets, feature_cols, timeframe_names):
    summary_data = []

    for df, timeframe in zip(sheets, timeframe_names):
        df_cleaned = df.copy()

        total_patients_in_timeframe = df_cleaned["id"].nunique()

        for feature in feature_cols:
            # calculate number of patients who have valid (non-missing) data for this specific feature
            sub_df = df_cleaned.dropna(subset=["id", feature])
            valid_patients = sub_df["id"].nunique()

            missing_patients = total_patients_in_timeframe - valid_patients

            summary_data.append(
                {
                    "Timeframe": timeframe,
                    "Feature": feature,
                    "Total_Missing_Patients": missing_patients,
                }
            )

    return pd.DataFrame(summary_data)


# finds all patients that have consistently missing data across all features and bins
# also finds the patients with omics data that have consistently missing data across all features and bins
def get_consistently_missing_patients(clinical_sheet, sheets, feature_cols, timeframe_names):
    summary_data = []

    for df, timeframe in zip(sheets, timeframe_names):
        df_cleaned = df.copy()

        for _, row in df_cleaned.iterrows():
            patient_id = row["id"]

            for feature in feature_cols:
                if feature in df_cleaned.columns:
                    val = row[feature]
                    is_missing = pd.isna(val) or (str(val).strip() == "no value") or (val == -1.0)
                else:
                    is_missing = True

                summary_data.append({
                    "id": patient_id,
                    "Feature_Bin": f"{feature} [{timeframe}]",
                    "Missing": int(is_missing)
                })

    df_tracker = pd.DataFrame(summary_data)

    if df_tracker.empty:
        return [], []

    # Use pivot_table to safely handle any duplicate id/Feature_Bin pairs
    df_wide = df_tracker.pivot_table(
        index="id", 
        columns="Feature_Bin", 
        values="Missing", 
        aggfunc="max"
    ).reset_index()
    
    # fill any missing ID-bin entries with 1
    df_wide = df_wide.fillna(1)

    # calculate how many present (0) values each participant has across all feature-bin columns
    df_wide["total_present"] = (df_wide.iloc[:, 1:] == 0).sum(axis=1)

    # filter for participants who have 0 valid data points across the entire dataset
    consistently_missing = df_wide[df_wide["total_present"] == 0]
    missing_ids = consistently_missing["id"].tolist()
    
    # filter for those where omics data exists
    clinical_dropped = clinical_sheet[clinical_sheet["id"].isin(missing_ids)]
    with_omics = clinical_dropped[clinical_dropped["omics_set#"] > 0]["id"].tolist()

    return missing_ids, with_omics


# returns a table containing the maximum consecutive number of days missing per feature per patient
def get_max_consecutive_missing(sheet, feature_cols):
    feature_results = {}

    # ensure the dataframe is sorted chronologically per patient
    sorted_sheet = sheet.sort_values(by = ["id", "date"]).reset_index(drop = True)

    for col in feature_cols:
        is_missing = sorted_sheet[col].isna()

        # create a block_id that changes only when a non-nan value is seen
        # i.e. if a non-nan value is seen, the is_missing value is false. therefore, the ~is_missing value is true. adding this to a sum would add 1 to the sum.
        # this means consecutive missing values will all share the same block_id number
        block_id = (~is_missing).cumsum()

        # group is_missing by (patient id, block id). since block_id only changes when a non-nan value is seen, these groups will each contain streaks of nan 
        # values, organized in chronological order, for each patient.
        # then sum the true values (by groups) in is_missing. this gives the lengths of every missing streak for that patient
        streak_lengths = is_missing.groupby([sorted_sheet["id"], block_id]).sum()

        # find the maximum streak length for each patient
        max_streak = streak_lengths.groupby("id").max()

        feature_results[col] = max_streak

    # combine results for all features into a table
    final_table = pd.DataFrame(feature_results).reset_index()

    return final_table


# returns total number of unique dates recorded across all patients
def count_unique_dates(sheet):
    return sheet["date"].nunique()


# returns median + interquartile range for each relevant metric:
# gestational age at start of study, gestational age at delivery, steps, total distance, very active minutes, total minutes asleep
def calc_summary_stats_median(sheet, feature_cols):
    new_cols = feature_cols + ["gestational_age_by_reported_lmp", "gest_age_del"]

    def iqr(x):
        return x.quantile(0.75) - x.quantile(0.25)
    
    # force-convert to numeric
    summary_df = sheet[new_cols].copy()
    for col in new_cols:
        summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")
    
    # aggregate across the entire dataset for median and iqr
    summary = summary_df.agg(["median", iqr]).T

    summary = summary.reset_index()
    summary.columns = ["Feature", "Median", "IQR"]

    return summary


# returns mean, standard deviation, and FDR for each significant Fitbit metric, split by control vs complication groups
# only includes values in the first trimester
def calc_summary_stats_mean(sheet, feature_cols):
    # filter metric list to only include differentially distributed metrics
    diff_distribution_sheet = pd.read_csv("04_results_and_figures/correlations/test4/fitbit_differential_distribution_significant.csv")
    diff_distribution_sheet["feature"] = (diff_distribution_sheet["feature"].astype(str).str.strip())

    # map minimum FDR q-value (p_value_adjusted) per feature across weeks
    ks_fdr_map = diff_distribution_sheet.groupby("feature")["p_value_adjusted"].min().to_dict()
    sig_metrics = list(ks_fdr_map.keys())
    metric_cols_adjusted = [c for c in feature_cols if c in sig_metrics]

    # force-convert to numeric
    summary_df = sheet[metric_cols_adjusted + ["group_bin"]].copy()
    for col in metric_cols_adjusted:
        summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

    df_control = summary_df[summary_df["group_bin"] == 0]
    df_complication = summary_df[summary_df["group_bin"] == 1]

    ctrl_stats = df_control[metric_cols_adjusted].agg(["mean", "std"]).T
    ctrl_stats.columns = ["Control Mean", "Control SD"]

    comp_stats = df_complication[metric_cols_adjusted].agg(["mean", "std"]).T
    comp_stats.columns = ["Complications Mean", "Complications SD"]

    summary = pd.concat([ctrl_stats, comp_stats], axis=1).reset_index()
    summary.rename(columns={"index": "Feature"}, inplace=True)

    # attach FDR q-values to the summary table
    summary["FDR"] = summary["Feature"].str.strip().map(ks_fdr_map)
    
    return summary


# returns a table containing the number of patients with fitbit data for at least one metric during each timeframe
# the timeframes are: 1st trimester, early 2nd trimester, late 2nd/early 3rd trimester, late 3rd trimester
# @param sheets: a list containing four datasets, each containing the data for one of the above timeframes
def get_patients_per_timeframe(sheets, feature_cols, timeframe_names):
    summary_data = []

    for df, timeframe in zip(sheets, timeframe_names):
        # drop rows for which all metric columns are missing
        df_cleaned = df.dropna(subset = feature_cols, how = "all")

        patient_count = df_cleaned["id"].nunique()

        summary_data.append({
            "Timeframe": timeframe,
            "Patient Count": patient_count
        })

    return pd.DataFrame(summary_data)


# returns a table of unique patient counts per feature per bin, split by control and adverse outcomes
def get_patients_per_feature_per_bin(sheets, feature_cols, timeframe_names):
    summary_data = []

    for df, timeframe in zip(sheets, timeframe_names):
        df_cleaned = df.copy()

        df_cleaned["is_control"] = (df_cleaned["group"].astype(str).str.strip().str.lower() == "control")

        for feature in feature_cols:
            sub_df = df_cleaned.dropna(subset=["id", feature, "group"])

            if sub_df.empty:
                continue

            # count unique patients per control status for this feature in this bin
            grouped = (sub_df.groupby("is_control")["id"].nunique().to_dict())

            control_count = grouped.get(True, 0)
            adverse_count = grouped.get(False, 0)

            summary_data.append(
                {
                    "Timeframe": timeframe,
                    "Feature": feature,
                    "Control": control_count,
                    "Adverse_Outcome": adverse_count,
                }
            )

    return pd.DataFrame(summary_data)


def get_omics_patients_per_feature_per_bin(sheets, feature_cols, timeframe_names, clinical_sheet):
    summary_data = []

    valid_omics_patients = set(clinical_sheet.loc[clinical_sheet["omics_set#"] > 0, "id"].dropna())

    for df, timeframe in zip(sheets, timeframe_names):
        df_cleaned = df.copy()

        df_cleaned = df_cleaned[df_cleaned["id"].isin(valid_omics_patients)]

        if df_cleaned.empty:
            continue

        df_cleaned["is_control"] = (df_cleaned["group"].astype(str).str.strip().str.lower() == "control")

        for feature in feature_cols:
            sub_df = df_cleaned.dropna(subset=["id", feature, "group"])

            if sub_df.empty:
                continue

            # count unique patients per control status for this feature in this bin
            grouped = (sub_df.groupby("is_control")["id"].nunique().to_dict())

            control_count = grouped.get(True, 0)
            adverse_count = grouped.get(False, 0)

            summary_data.append(
                {
                    "Timeframe": timeframe,
                    "Feature": feature,
                    "Control": control_count,
                    "Adverse_Outcome": adverse_count,
                }
            )

    return pd.DataFrame(summary_data)


# returns a true/false matrix (patients x metrics) showing whether each patient has non-missing data for at least 80% of their valid pregnancy tracking days
# also returns a table containing the number of metrics with 80+% of valid data, per patient
# also returns a table containing the number of patients with 80+% of valid data, per metric
def get_metric_representation_matrices(sheet, feature_cols):
    df = sheet.sort_values(by = ["id", "timepoint"]).copy()

    # extract start and end dates per patient
    # start date marks the day the patient enrolled/started data collection
    # end data marks delivery
    enrollment_days = df.groupby("id")["timepoint"].min()
    max_recorded_days = df.groupby("id")["timepoint"].max()

    delivery_weeks = df.groupby("id")["gest_age_del"].first()
    delivery_days = delivery_weeks * 7

    # apply fallback logic for the end of the tracking window:
    # use gestational age at delivery if present; otherwise, use latest recorded day
    end_days = delivery_days.fillna(max_recorded_days)

    recorded_data_lengths = (end_days - enrollment_days) + 1
    recorded_data_lengths = recorded_data_lengths.clip(lower = 1) # ensure values are positive

    # count how many days of valid data exist per metric per patient
    active_days_per_metric = df.groupby("id")[feature_cols].agg(lambda x: x.notna().sum())

    # contains percent of data that is valid per metric per patient
    representation_matrix = active_days_per_metric.div(recorded_data_lengths, axis = 0)
    
    final_table = representation_matrix >= 0.80
    final_table = final_table.reset_index()

    # contains the number of metrics with 80+% of valid data, per patient
    pt_summary = pd.DataFrame({
        "Patient ID": final_table["id"],
        "Compliant Metrics Count": final_table[feature_cols].sum(axis = 1)
    })
    pt_summary = pt_summary.sort_values(by = "Compliant Metrics Count", ascending = False).reset_index(drop = True)

    # contains the number of patients with 80+% of valid data, per metric
    patient_counts_per_metric = final_table[feature_cols].sum(axis = 0)
    metric_summary = pd.DataFrame({
        "Fitbit Metric": patient_counts_per_metric.index,
        "Patients With >= 80% Density": patient_counts_per_metric.values
    })
    metric_summary = metric_summary.sort_values(by = "Patients With >= 80% Density", ascending = False).reset_index(drop = True)

    return final_table, pt_summary, metric_summary


# data preparation for histogram
# processes raw data and returns two datasets:
# one contains all daily patient counts (including post-delivery)
# the other has been filtered to only include datapoints during pregnancy (excluding post-delivery)
# patients are counted if any of the metrics are non-null
# note that we only include the datapoints for which the column Event.Name != "General"
def prepare_pregnancy_counts_histogram(df):
    df_filtered = df[df["event_name"] != "general"].copy()

    # only keep rows for which any of the metrics are non-null
    feature_cols = [col for col in df_filtered.columns if col.startswith(("activities", "sleep", "heart_rate"))]
    df_clean = df_filtered.dropna(subset = feature_cols, how = "all").copy()

    df_clean["current_weeks"] = df_clean["timepoint"] / 7

    # construct dataset 1: all valid Fitbit updates
    all_data_counts = (
        df_clean.groupby("current_weeks")["id"]
        .nunique()
        .reset_index(name = "patient_count")
    )

    # construct dataset 2: pregnancy only (stop counting datapoints past delivery)
    pregnancy_only_df = df_clean[df_clean["current_weeks"] <= df_clean["gest_age_del"]]

    pregnancy_counts = (
        pregnancy_only_df.groupby("current_weeks")["id"]
        .nunique()
        .reset_index(name = "patient_count")
    )

    return all_data_counts, pregnancy_counts


# plotting function
# takes both dataframes over time (all datapoints, pregnancy only) and plots the two histograms
# plot 1: patients with valid Fitbit updates per day
# plot 2: patients with valid Fitbit updates per day (during pregnancy only)
def make_histograms_pdf(all_data, pregnancy_data):
    output_filename = "04_results_and_figures/data_analysis/fitbit/pregnancy_plots_report.pdf"

    with PdfPages(output_filename) as pdf:
    
        def draw_plot(data, title):
            fig = plt.figure(figsize = (12, 6))

            # plot the data
            plt.bar(data["current_weeks"], data["patient_count"], width = 0.1, color = "skyblue")

            # add trimester lines
            plt.axvline(x = 14, color = "red", linestyle = "--", linewidth = 1.5, label = "End of Trimester 1 (Week 14)")
            plt.axvline(x = 22, color = "orange", linestyle = "--", linewidth = 1.5, label = "End of Early Trimester 2 (Week 22)")
            plt.axvline(x = 32, color = "blue", linestyle = "--", linewidth = 1.5, label = "End of Early Trimester 3 (Week 32)")
            plt.axvline(x = 37, color = "green", linestyle = ":", linewidth = 1.5, label = "Typical Delivery (Week 37)")

            # clean up x-axis ticks by only making them visible every 5 weeks
            ax = plt.gca()
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.xaxis.set_minor_locator(ticker.NullLocator())

            plt.xlim(0, data["current_weeks"].max() + 2)

            plt.title(title, fontsize = 14, fontweight = "bold")
            plt.xlabel("Week of Pregnancy", fontsize = 12)
            plt.ylabel("Number of Patients with Fitbit Data", fontsize = 12)
            plt.legend(loc = "upper right")
            plt.grid(axis = "y", alpha = 0.3)
            plt.tight_layout()

            pdf.savefig(fig, dpi = 300)
            plt.close(fig)

        # generate plot 1: all valid datapoints
        draw_plot(
            data = all_data,
            title = "Number of Patients per Day (All Valid Fitbit Updates)",
        )

        # generate plot 2: only datapoints during pregnancy
        draw_plot(
            data = pregnancy_data,
            title = "Number of Patients per Day (Strictly During Pregnancy)",
        )


# generates combined violin/box plots (split by control vs complications) for each metric for the first trimester
# @param normalize: if True, normalizes the data using Z-scores. useful if data ranges vary wildly between metrics
# @param significant: if True, only plots the Fitbit metrics that are differentially distributed between control/complication groups
def make_metric_violin_box_plots(df, metric_cols, normalize=True, significant=True):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'font.size': 13})

    colors = {0: 'cornflowerblue', 1: 'crimson'}
    labels = {0: 'Control', 1: 'Complications'}

    valid_metrics = []
    ctrl_data_list = []
    comp_data_list = []
    q_values = []
    q_value_map = {}

    if significant:
        # filter metric list to only include differentially distributed metrics
        diff_distribution_sheet = pd.read_csv("04_results_and_figures/correlations/test4/fitbit_differential_distribution_significant.csv")
        diff_distribution_sheet["feature"] = (diff_distribution_sheet["feature"].astype(str).str.strip())
        # Group by feature and take the minimum (most significant) q-value across weeks
        min_q = diff_distribution_sheet.groupby("feature")["p_value_adjusted"].min()
        q_value_map = min_q[min_q < 0.05].to_dict()
        metric_cols_adjusted = [c for c in metric_cols if c.strip() in q_value_map]
    else:
        metric_cols_adjusted = metric_cols

    for metric in metric_cols_adjusted:
        if metric not in df.columns or "group_bin" not in df.columns:
            print(f"Skipping {metric}: Column missing from DataFrame.")
            continue

        valid_df = df[["group_bin", metric]].dropna()
        ctrl_vals = valid_df[valid_df["group_bin"] == 0][metric].values
        comp_vals = valid_df[valid_df["group_bin"] == 1][metric].values

        if len(ctrl_vals) == 0 or len(comp_vals) == 0:
            print(f"Skipping {metric}: Missing control or complication data.")
            continue

        # Use pre-computed q-value from Kolmogorov-Smirnov test results if available
        if metric in q_value_map:
            q_val = q_value_map[metric]
        else:
            _, p_val = stats.mannwhitneyu(ctrl_vals, comp_vals, alternative='two-sided')
            q_val = p_val

        # Z-score standardization per metric if raw scales vary drastically
        if normalize:
            all_vals = np.concatenate([ctrl_vals, comp_vals])
            mean, std = np.mean(all_vals), np.std(all_vals)
            if std > 0:
                ctrl_vals = (ctrl_vals - mean) / std
                comp_vals = (comp_vals - mean) / std

        valid_metrics.append(metric)
        ctrl_data_list.append(ctrl_vals)
        comp_data_list.append(comp_vals)
        q_values.append(q_val)

    if not valid_metrics:
        print("No valid metrics to plot.")
        return None

    n_metrics = len(valid_metrics)
    metric_spacing = 1.8
    group_offset = 0.28

    centers = np.arange(n_metrics) * metric_spacing
    ctrl_positions = centers - group_offset
    comp_positions = centers + group_offset

    fig_width = max(10, n_metrics * 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # helper function to render violins and boxplots per group
    def render_layer(data_list, positions, color):
        vparts = ax.violinplot(
            data_list,
            positions=positions,
            widths=0.45,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )
        for pc in vparts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(0.35)

        bp = ax.boxplot(
            data_list,
            positions=positions,
            widths=0.12,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False
        )
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)

        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(bp[element], color='black', linewidth=1.5)

    render_layer(ctrl_data_list, ctrl_positions, colors[0])
    render_layer(comp_data_list, comp_positions, colors[1])

    # add significance annotations (stars) above the violins based on FDR-adjusted q-values
    all_data_max = max(max(np.max(c), np.max(m)) for c, m in zip(ctrl_data_list, comp_data_list))
    all_data_min = min(min(np.min(c), np.min(m)) for c, m in zip(ctrl_data_list, comp_data_list))
    y_range = all_data_max - all_data_min
    bracket_height = y_range * 0.03
    text_offset = y_range * 0.01

    max_annotation_y = all_data_max

    for i in range(n_metrics):
        q = q_values[i]
        print(f"Metric: {valid_metrics[i]}, Q-value: {q:.6f}")
        if q < 0.0001:
            star = '****'
        elif q < 0.001:
            star = '***'
        elif q < 0.01:
            star = '**'
        elif q < 0.05:
            star = '*'
        else:
            star = ''

        if star:
            # Measure local peak height for this metric
            local_max = max(np.max(ctrl_data_list[i]), np.max(comp_data_list[i]))
            bar_y = local_max + y_range * 0.04
            
            # Draw significance bracket connecting control and complication violins
            ax.plot(
                [ctrl_positions[i], ctrl_positions[i], comp_positions[i], comp_positions[i]],
                [bar_y, bar_y + bracket_height, bar_y + bracket_height, bar_y],
                lw=1.2, color='black'
            )
            
            # Draw significance star centered above the bracket
            ax.text(
                centers[i], bar_y + bracket_height + text_offset, star,
                ha='center', va='bottom', fontsize=11, fontweight='bold'
            )
            
            max_annotation_y = max(max_annotation_y, bar_y + bracket_height + text_offset + y_range * 0.05)

    # Adjust top y-limit so brackets/stars don't get clipped
    ax.set_ylim(top=max_annotation_y + y_range * 0.05)

    label_map = {
        "activities_summary_caloriesout": "Calories Out",
        "activities_summary_lightlyactiveminutes": "Lightly Active Minutes",
        "heart_rate_resting_heart_rate": "Resting Heart Rate",
        "sleep_summary_stages_deep": "Deep Sleep Minutes",
        "sleep_summary_stages_rem": "REM Sleep Minutes",
        "activities_summary_steps": "Steps",
        "activities_summary_veryactiveminutes": "Very Active Minutes"
    }
    formatted_labels = [label_map.get(metric, metric) for metric in valid_metrics]

    ax.set_xticks(centers)
    ax.set_xticklabels(formatted_labels, rotation=25, ha='right', fontweight='bold')
    
    ylabel_str = "Standardized Value (Z-Score)" if normalize else "Value"
    ax.set_ylabel(ylabel_str)
    ax.set_title("First Trimester Metrics: Control vs. Complications", pad=15, fontweight='bold')

    legend_patches = [
        mpatches.Patch(color=colors[0], alpha=0.85, label=labels[0]),
        mpatches.Patch(color=colors[1], alpha=0.85, label=labels[1])
    ]
    ax.legend(handles=legend_patches, loc='upper left', frameon=True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


# computes min and max feature sample sizes across gestational bins, split by control vs complication status
def get_sample_size_range_by_bin(sheets, feature_cols, trimester_names):
    records = []

    group_map = {0: "Control", 1: "Complication", 0.0: "Control", 1.0: "Complication", "0": "Control", "1": "Complication"}

    for idx, df in enumerate(sheets):
        stage_label = trimester_names[idx] if idx < len(trimester_names) else f"Trimester_{idx+1}"

        # map 0/1 values to "control"/"complication"
        df_stage = df.copy()
        df_stage["group_bin"] = df_stage["group_bin"].replace(group_map)

        groups = df_stage["group_bin"].dropna().unique()
        
        for grp in groups:
            sub = df_stage[df_stage["group_bin"] == grp]
            
            if len(sub) == 0:
                min_n, max_n = 0, 0
            else:
                counts = sub[feature_cols].notna().sum()
                min_n = int(counts.min()) if not counts.empty else 0
                max_n = int(counts.max()) if not counts.empty else 0

            records.append({
                "Gestational Bin": stage_label,
                "Group": grp,
                "Min N": min_n,
                "Max N": max_n
            })

    summary_df = pd.DataFrame(records)

    poster_table = summary_df.pivot(
        index="Gestational Bin",
        columns="Group",
        values=["Min N", "Max N"]
    )

    # order bins chronologically
    ordered_bins = [s for s in trimester_names if s in poster_table.index]
    poster_table = poster_table.reindex(ordered_bins)

    # swap header levels to put group status (control vs complication) at top
    poster_table.columns = poster_table.columns.swaplevel(0, 1)
    poster_table.sort_index(axis=1, level=0, inplace=True)

    final_df = poster_table.reset_index()

    return final_df


# print all calculated data into a log file
def print_log(total_patients, total_missing, per_patient, consistently_missing_patients, consistently_missing_with_omics, max_con_missing, 
              unique_dates, summary_stats_median, summary_stats_mean, patients_per_timeframe, metric_matrix, pt_summary, metric_summary, num_metrics, 
              sample_size_table):
    log_path = "04_results_and_figures/data_analysis/fitbit/fitbit_data_analysis.txt"

    with open(log_path, "w") as f:
        f.write("Following are various statistics about the Fitbit dataset. \n")
        f.write("Note that this data has been filtered to only include datapoints during pregnancy.\n\n\n")

        f.write(f"Total number of patients: {total_patients}")
        f.write("\n\n\n")

        f.write(f"Total number of missing values: {total_missing}")
        f.write("\n\n\n")

        f.write("Number of missing days per patient:\n")
        f.write(per_patient.to_string(index = False))
        f.write("\n\n\n")

        f.write(f"Patients with consistently missing data across features and bins: ({len(consistently_missing_patients)} patients total)\n")
        f.write(str(consistently_missing_patients))
        f.write("\n\n\n")

        f.write(f"Patients with consistently missing data across features and bins AND omics data: ({len(consistently_missing_with_omics)} patients total)\n")
        f.write(str(consistently_missing_with_omics))
        f.write("\n\n\n")

        f.write("Maximum consecutive number of days missing per feature per patient:\n")
        f.write(max_con_missing.to_string(index = False))
        f.write("\n\n\n")

        f.write(f"Total number of unique dates recorded across all patients: {unique_dates}")
        f.write("\n\n\n")

        f.write("Summary statistics by metric (median/IQR):\n")
        f.write(summary_stats_median.to_string(index = False))
        f.write("\n\n\n")

        f.write("First trimester summary statistics by metric, split by control vs complications (mean/SD):\n")
        f.write(summary_stats_mean.to_string(index = False))
        f.write("\n\n\n")

        f.write("Number of unique patients per timeframe:\n")
        f.write(patients_per_timeframe.to_string(index = False))
        f.write("\n\n\n")

        f.write("Which patients contributed valid data for at least 80% of their pregnancy, per feature:\n")
        f.write(metric_matrix.to_string(index = False))
        f.write("\n\n\n")

        f.write(f"Number of the {num_metrics} metrics for which each patient achieved >= 80% data density:\n")
        f.write(pt_summary.to_string(index = False))
        f.write("\n\n\n")

        f.write(f"Number of the {total_patients} patients that achieved >= 80% data density, per metric:\n")
        f.write(metric_summary.to_string(index = False))
        f.write("\n\n\n")

        f.write("Sample size ranges by gestational bin:\n")
        f.write(sample_size_table.to_string(index = False))
        f.write("\n\n\n")


def main():
    fitbit_sheet, clinical_sheet = load_sheet()

    sheet_filtered = filter_sheet(fitbit_sheet)

    feature_cols = [col for col in sheet_filtered.columns if col.startswith(("activities", "sleep", "heart_rate"))]
    numeric_cols = feature_cols + ["gestational_age_by_reported_lmp", "gest_age_del"]

    # forcefully convert all feature columns + gest age columns into numeric types
    sheet_filtered[numeric_cols] = sheet_filtered[numeric_cols].apply(pd.to_numeric, errors = "coerce")
    
    sheets_bucketed = bucket_data(sheet_filtered)

    timeframe_names = ["First Trimester", "Early Second Trimester", "Late Second and Early Third Trimester", "Mid Third Trimester", "Late Third Trimester"]

    total_patients = get_total_patients(sheet_filtered)
    total_missing = count_total_missing(sheet_filtered)
    per_patient = get_missing_per_patient(sheet_filtered, feature_cols)
    missing_per_feature_per_bin = get_missing_patients_per_feature_per_timeframe(sheets_bucketed, feature_cols, timeframe_names)
    consistently_missing_patients, consistently_missing_with_omics = get_consistently_missing_patients(
        clinical_sheet, sheets_bucketed, feature_cols, timeframe_names
    )
    max_con_missing = get_max_consecutive_missing(sheet_filtered, feature_cols)
    unique_dates = count_unique_dates(sheet_filtered)
    summary_stats_median = calc_summary_stats_median(sheet_filtered, feature_cols)
    summary_stats_mean = calc_summary_stats_mean(sheets_bucketed[0], feature_cols)
    patients_per_timeframe = get_patients_per_timeframe(sheets_bucketed, feature_cols, timeframe_names)
    patients_per_feature_per_bin = get_patients_per_feature_per_bin(sheets_bucketed, feature_cols, timeframe_names)
    omics_patients_per_feature_per_bin = get_omics_patients_per_feature_per_bin(sheets_bucketed, feature_cols, timeframe_names, clinical_sheet)
    metric_matrix, pt_summary, metric_summary = get_metric_representation_matrices(sheet_filtered, feature_cols)
    sample_size_table = get_sample_size_range_by_bin(sheets_bucketed, feature_cols, timeframe_names)

    all_counts, preg_counts = prepare_pregnancy_counts_histogram(fitbit_sheet)
    make_histograms_pdf(all_counts, preg_counts)

    violin_box_plots = make_metric_violin_box_plots(sheets_bucketed[0], feature_cols)
    with PdfPages("04_results_and_figures/data_analysis/fitbit/violin_box_plots.pdf") as pdf:
        pdf.savefig(violin_box_plots, bbox_inches='tight')

    print_log(total_patients, total_missing, per_patient, consistently_missing_patients, consistently_missing_with_omics, max_con_missing, 
              unique_dates, summary_stats_median, summary_stats_mean, patients_per_timeframe, metric_matrix, pt_summary, metric_summary, 
              len(feature_cols), sample_size_table)

    patients_per_feature_per_bin.to_csv("04_results_and_figures/data_analysis/fitbit/patients_per_feature_per_bin.csv", index=False)
    omics_patients_per_feature_per_bin.to_csv("04_results_and_figures/data_analysis/fitbit/omics_patients_per_feature_per_bin.csv", index=False)
    missing_per_feature_per_bin.to_csv("04_results_and_figures/data_analysis/fitbit/missing_per_feature_per_bin.csv", index=False)


if __name__ == "__main__":
    main()