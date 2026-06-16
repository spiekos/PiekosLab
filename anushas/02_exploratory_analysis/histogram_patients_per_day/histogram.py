import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loads and returns the dataset
def load_sheet():
    sheet = pd.read_csv("02_exploratory_analysis/histogram_patients_per_day/DP3_playset.csv")
    return sheet

# data preparation
# processes raw data and returns two datasets:
# one contains all daily patient counts (including post-delivery)
# the other has been filtered to only include datapoints during pregnancy (excluding post-delivery)
# patients are counted if all four activity/sleep metrics are non-null
# note that we only include the datapoints for which the column Event.Name != "General"
def prepare_pregnancy_counts(df):
    df_filtered = df[df["Event.Name"] != "General"].copy()

    # only keep rows for which all four activity/sleep metrics are non-null
    data_cols = [
        "Activities...Summary...steps", "Activities...Summary...totalDistances", 
        "Activities...Summary...veryActiveMinutes", "Sleep...Summary...total.minutes.asleep"
    ]
    df_clean = df_filtered.dropna(subset = data_cols, how = "any").copy()

    df_clean["delivery_day_limit"] = df_clean["gest age del"] * 7

    # construct dataset 1: all valid Fitbit updates
    all_data_counts = (
        df_clean.groupby("timepoint")["Record.ID"]
        .nunique()
        .reset_index(name = "patient_count")
    )

    # construct dataset 2: pregnancy only (stop counting datapoints past delivery)
    pregnancy_only_df = df_clean[df_clean["timepoint"] <= df_clean["delivery_day_limit"]]

    pregnancy_counts = (
        pregnancy_only_df.groupby("timepoint")["Record.ID"]
        .nunique()
        .reset_index(name = "patient_count")
    )

    return all_data_counts, pregnancy_counts






def main():
    sheet = load_sheet()
    prepare_pregnancy_counts(sheet)



if __name__ == "__main__":
    main()