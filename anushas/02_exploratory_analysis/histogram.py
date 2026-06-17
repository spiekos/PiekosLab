import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# loads and returns the dataset
def load_sheet():
    sheet = pd.read_csv("00_raw_data/DP3_playset.csv")
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

    df_clean["current_weeks"] = df_clean["timepoint"] / 7

    # construct dataset 1: all valid Fitbit updates
    all_data_counts = (
        df_clean.groupby("current_weeks")["Record.ID"]
        .nunique()
        .reset_index(name = "patient_count")
    )

    # construct dataset 2: pregnancy only (stop counting datapoints past delivery)
    pregnancy_only_df = df_clean[df_clean["current_weeks"] <= df_clean["gest age del"]]

    pregnancy_counts = (
        pregnancy_only_df.groupby("current_weeks")["Record.ID"]
        .nunique()
        .reset_index(name = "patient_count")
    )

    return all_data_counts, pregnancy_counts

# plotting function
# takes both dataframes (all datapoints, pregnancy only) and plots the two histograms
def make_histograms(all_data, pregnancy_data):
    def draw_plot(data, title, filename):
        plt.figure(figsize = (12, 6))

        # plot the data
        plt.bar(data["current_weeks"], data["patient_count"], width = 0.1, color = "skyblue")

        # add trimester lines
        plt.axvline(x = 13, color = "red", linestyle = "--", linewidth = 1.5, label = "End of Trimester 1 (Week 13)")
        plt.axvline(x = 26, color = "orange", linestyle = "--", linewidth = 1.5, label = "End of Trimester 2 (Week 26)")
        plt.axvline(x = 40, color = "green", linestyle = ":", linewidth = 1.5, label = "Typical Delivery (Week 40)")

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
        plt.savefig(filename, dpi = 300)

    # generate plot 1: all valid datapoints
    draw_plot(
        data = all_data,
        title = "Number of Patients per Day (All Valid Fitbit Updates)",
        filename = "patients_per_day_all.png"
    )

    # generate plot 2: only datapoints during pregnancy
    draw_plot(
        data = pregnancy_data,
        title = "Number of Patients per Day (Strictly During Pregnancy)",
        filename = "patients_per_day_during_pregnancy.png"
    )

    plt.show()

def main():
    sheet = load_sheet()
    all_counts, preg_counts = prepare_pregnancy_counts(sheet)
    make_histograms(all_counts, preg_counts)

if __name__ == "__main__":
    main()