### Kayla Xu, Piekos Lab
### 03/16/2026

#### Outlier-Based Biomarker Discovery for Pregnancy Complications
# Objective: Identify blood, urine, or gut/vaginal microbiome analytes that show extreme deviation in pregnancy complications relative to term controls, 
# prioritizing these with presistent elevation across multiple timepoints for potential diagnostic biomarker development. 

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import logging
import ast
import sys

TIMEPOINTS = ["A", "B", "C", "D", "E"]
MAD_THRESHOLD = 3
COMPLICATIONS = ["FGR", "HDP", "sPTB"]

# Study Design:
#   Cases: Pregnancy complications (n=77)
#   Control: Term deliveries (n=55)
#   Timepoints: Up to 5 collections per tissue per patient (some patients have as few as 2 due to preterm delivery)
#   Data types: Multiple omics platforms (proteomics, metabolomics, etc.)
# Data Status: Already preprocessed, QC'd, batch-corrected, normalized, filtered for missingness

# Step 1: Calculate Reference Distribution (per analyte)
#   for each tissue x timepoint x data type combination
# 1.1 Subset to term controls only
# 1.2 Calculate per-analyte statistics
#   Median
#   median avsolute deviation 
#       if control_MAD = 0, flag as "zero_variance" in log and exclude from outlier analysis
# 1.3 Document control statistics
#   Create reference table with columns as tissue, timepoint, datatype, analyte ID, control median, control_MAD
#   output: control_reference_statistics_<tissue>_<timepoint>_<datatype>.csv

# Step 2: Calculate MAD Scores (All Samples)
# For each sample in each tissue at each timepoint for each data type:
# 2.1 Calculate MAD Score
#   For each analyte measurement:
#       MAD_score = (observed_value - Control_Median) / Control_MAD
#   Use the control_median and control_MAD from the appropriate tissue/timepoint/data_type combination
# 2.2 Create MAD score matrix
#   Generate a data structure with:
#       Rows = Sample_ID
#       Columns = Analytes
#       Values = MAD scores
#       Additional metadata = patient_ID, group (control/FGR/GHDP/sPTB), group_subtype, gestational_age, gestational_age_at_sample_collection
# Output: mad_scores_matrix_<tissue>_<timepoint>_<data_type>.csv

# Step 4: Restrict to tissue analytes with >=2 outlier timepoints
# 4.1 For each individual patient x analyte combination
#   Count how many timeoints showed that analyte as an outlier (elevated or decreased)
#   Record the direction at each outlier timepoint
# 4.2 Filter criteria
#   Keep only analyte x patient combinations where:
#       Number of outlier timepoints >= 2
#       Direction is consistent (all elevated OR all decreased, not mixed)
# 4.3 Create filtered dataset
#   Table with columns:
#       Patient_ID
#       Analyte_ID
#       Group
#       Subgroup
#       Total_Timepoints_Avaliable (2-5)
#       Outlier_Timepoints_Count(2-5)
#       Outlier_Direction(elevated/decreased)
#       Outlier_Timepoints_List (e.g., ["T1", "T3", "T4"])
#       MAD_Scores_List (list of MAD scores at outlier timepoints)
#       Mean_MAD_Score (average across outlier timepoints)

# Step 5: Generate prioritized biomarker lists by tissue type
# Create 5 separate lits focusing on elevated analyses only for each tissue type examining outliers across all data types (more clinicall interpretable for diagnostics)





# calculate median and MAD 
# return: dataframe with analyte_ID, tissue, datatype, timepoint, median, and MAD
def getStats(df, t, datatype, tissue):
    temp = pd.DataFrame(index=df.columns)
    temp["analyte_ID"] = df.columns
    temp["tissue"] = tissue
    temp["datatype"] = datatype
    temp["timepoint"] = t
    temp["control_median"] = df.median()
    temp["control_MAD"] = stats.median_abs_deviation(df)
    return temp

# combine batch-split files at timepoint t
def mergeBatches(batches, dir_input, t):
    temp = pd.DataFrame()
    for b in batches:
        try:
            samples = pd.read_csv(dir_input + "/Samples_" + str(b) + "_" + t + ".csv", index_col=0)
            temp = pd.concat([temp, samples])
        except:
            continue
    return temp

# extract and format control reference values
# return: dictionary with keys = timepoints and values = control reference median and MAD values
#         list of all control IDs (including timepoint suffix)
# output: control_reference_statistics_<tissue>_<timepoint>_<data_type>.csv
def controlRefStats(samples, dir_output, datatype, tissue, batches):
    controlAll = {} # dict to return with keys = timepoints and values = control reference median and MAD values
    control_IDs = []
    for t in TIMEPOINTS:
        control_IDs.extend(list(samples[t].index))
        controlAll[t] = getStats(samples[t], t, datatype, tissue)
        controlAll[t].to_csv(dir_output + "/control_reference_statistics_" + tissue + "_" + t + "_" + datatype + ".csv")
    with open(dir_output + '/control_IDs.txt', 'w') as f:
        for line in control_IDs:
            f.write(f"{line}\n")
    return controlAll, control_IDs

# calculate MAD scores based on timepoint
def getMADscores(df, controlRef, t):
    scores_dict = {}
    for m in df.columns:
        if controlRef[t].loc[m, "control_MAD"] == 0:
            logging.info(m + " at timepoint " + t + " has zero_variance/a MAD value of 0 and has been removed from downstream analyses")
        else:
            try:
                temp = (df[m] - controlRef[t].loc[m, "control_median"]) / controlRef[t].loc[m, "control_MAD"]
                scores_dict[m] = temp
            except:
                logging.warning("A issue has occured when calculating MAD score of " + m + " at timepoint " + t + ": control_median = " + str(controlRef[t].loc[m, "control_median"]) + ", control_MAD = " + str(controlRef[t].loc[m, "control_MAD"]))
    scores = pd.DataFrame(scores_dict, index=df.index)
    return scores

# calculate MAD scores for all samples and analytes
# return: dictionary where keys = timepoints and values = MAD score matrices with group, subgroup, gestational age, and gestational age at sample collection per sample
# output: mad_scores_matrix_<tissue>_<timepoint>_<data_type>.csv
def MADscores(samples, dir_output, controlRef, batches, tissue, datatype):
    scoreMatrix = {}
    for t in TIMEPOINTS:
        scoreMatrix[t] = getMADscores(samples[t], controlRef, t)
        scoreMatrix[t].to_csv(dir_output + "/mad_scores_matrix_" + tissue + "_" + t + "_" + datatype + ".csv")
    return scoreMatrix

# flag MAD score > 3 or < -3
# return: dictionary of matrices by timepoint with 1 = elevated, -1 = decreased, 0 = outlier 
# output: outlier_flags_matrix_<tissue>_<timepoint>_<data_type>.csv
def flagOutliers(dir_output, scoreMatrix, tissue, datatype):
    outliers = {}
    for t in TIMEPOINTS:
        outliers[t] = scoreMatrix[t].map(lambda x: 1 if x > MAD_THRESHOLD else (-1 if x < -MAD_THRESHOLD else 0))
        outliers[t].to_csv(dir_output + "/outlier_flags_matrix_" + tissue + "_" + t + "_" + datatype + ".csv")
    return outliers

# remove metadata from dataframe and save in a separate dictionary
# return: metadata dictionary of keys = timepoint, values = dataframe of sample ID, group, subgroup, gest age, and gest age at collection
#         sample dictionary of keys = timepoint, values = dataframe of batch normalized and log2 transformed metabolite expression        
def splitData(dir_input, batches):
    allMeta = {}
    allSamples = {}
    for t in TIMEPOINTS:
        temp = mergeBatches(batches, dir_input, t)
        meta = temp[["patient_ID","group", "subgroup", "gestational_age", "gestational_age_at_collection"]]
        meta["group"] = meta["group"].replace("sptb", "sPTB")

        samples = temp.drop(columns=["patient_ID", "group", "subgroup", "gestational_age", "gestational_age_at_collection"])
        allMeta[t] = meta
        allSamples[t] = samples
    return allMeta, allSamples

# helper function for filterOutliers
# return: dictionary of indices of each patient in each timepoint dataframe
def t_to_p(outlierMatrix, patient_metadata):
    temp = {t: {} for t in TIMEPOINTS}
    for t in TIMEPOINTS:
        for idx in outlierMatrix[t].index:
            for p in patient_metadata.keys():
                if p in idx:
                    temp[t][p] = idx
                    break # Assuming one match per patient per timepoint
    return temp

# Filter for patient x analyte combinations that have >= 2 outlier timepoints and all outliers are directionally consistent (all elevated OR decreased)
# return: dataframe with rows = patient x analytes and columns = persistent outlier info
# output: persistent_outliers_<tissue>_<data_type>.csv
def filterOutliers(dir_output, outlierMatrix, scoreMatrix, meta, tissue, datatype):
    results = []
    unique_patients = pd.concat([meta[t] for t in TIMEPOINTS], ignore_index=True).drop_duplicates(subset=["patient_ID"])
    patient_metadata = unique_patients.set_index("patient_ID")[["group", "subgroup"]].to_dict('index')
    # Pre-compute index mappings to avoid O(N) string matching in the inner loop
    # This creates a mapping of: timepoint -> {patient_id: exact_index_name}
    t_to_p_index = t_to_p(outlierMatrix, patient_metadata)
    for p, metadata in patient_metadata.items():
        group = metadata["group"]
        subgroup = metadata["subgroup"]
        for m in outlierMatrix["A"].columns:
            total_timepoints = 0
            outlier_values = []
            outlier_timepoints = []
            outlier_mads = []
            for t in TIMEPOINTS:
                if p not in t_to_p_index[t]: # check if patient is in this timepoint
                    continue
                idx = t_to_p_index[t][p]
                try:
                    if abs(outlierMatrix[t].at[idx, m]) > 0:
                        outlier_values.append(list(outlierMatrix[t].loc[[p in x for x in outlierMatrix[t].index], m])[0])
                        outlier_timepoints.append(t)
                        outlier_mads.append(list(scoreMatrix[t].loc[[p in x for x in scoreMatrix[t].index], m])[0])
                    total_timepoints += 1
                except:
                    continue
            total_outlier_timepoints = len(outlier_timepoints)
            if total_outlier_timepoints >= 2:
                if abs(sum(outlier_values)) == total_outlier_timepoints:
                    direction = "elevated" if sum(outlier_values) > 0 else "decreased"
                    results.append({
                        "patient_ID": p,
                        "analyte_ID": m,
                        "group": group,
                        "subgroup": subgroup,
                        "total_timepoints": total_timepoints,
                        "outlier_timepoint_count": total_outlier_timepoints,
                        "outlier_direction": direction,
                        "outlier_timepoints": outlier_timepoints,
                        "outlier_mad_scores": outlier_mads,
                        "mean_outlier_mad": sum(outlier_mads)/len(outlier_mads)
                    })
    persistent = pd.DataFrame(results)
    persistent.to_csv(dir_output + "/persistent_outliers_" + tissue + "_" + datatype + ".csv")
    return persistent

# List 1: Most Prevelent
#   Goal: analytes elevated in the most complication patients
#   Steps:
#       1. Filter to complication samples (exclude controls)
#       2. For each analyte, count number of unique patients showing elevation
#       3. Calculate % complications affected = (n_patients / total complications in data for this tissue) * 100
#       4. Rank analytes by % complication affected (descending)
#       5. Select top 50 analytes
#   Include in output:
#       analyte_ID
#       n_patients_affected
#       percent_complications_affected
#       mean_outlier_timepoints_per_patient
#       complication_types_represented
#   Output: biomarker_most_prevalent_<tissue>.csv
def mostPrevalent(dir_output, persistentMatrix, meta, analytes, tissue):
    results = []
    complicationOnly = persistentMatrix.loc[persistentMatrix["group"] != "Control",:]
    totalComplications = len(meta.loc[meta["group"] != "Control",:].index)
    #for m in analytes:
    #    elevatedCounts = Counter(complicationOnly.loc[complicationOnly["analyte_ID"] == a,:]["group"])
    for m in analytes:
        mOnly = complicationOnly.loc[complicationOnly["analyte_ID"] == m,:]
        if len(mOnly.index) == 0:
            continue
        n_patients = len(mOnly.index)
        percentAffected = (n_patients / totalComplications) * 100
        results.append({"analyte_ID": m,
                        "n_patients_affected": n_patients,
                        "percent_complications_affected": percentAffected,
                        "mean_outlier_timepoints_per_patient": mOnly["outlier_timepoint_count"].sum() / len(mOnly.index),
                        "complication_types_represented": mOnly["group"].str.upper().unique().tolist()})
    prevalent50 = pd.DataFrame(results).sort_values(by=["percent_complications_affected"], ascending=False).iloc[0:50,:]
    prevalent50.to_csv(dir_output + "/biomarker_most_prevalent_" + tissue + ".csv")
    return prevalent50
        

# List 2: Most Persistent
#   Goal: Analytes showing sustained elevation across pregnancy
#   Steps:
#       1. For each analyte (complication samples only):
#           Calculate average number of outlier timepoints per affected individual
#           Calculate average proportion: (outlier_timepoints / total_available_timepoints)
#       2. Filter to analytes affecting >= 5 patients
#       3. Rank by average proportion of timepoints (descending)
#       4. Select top 50 analytes
#   Inlcude in output:
#       Analyte_ID
#       n_patients_affected
#       mean_outlier_timepoints_per_patient_affected
#       mean_proportion_timepoints (outlier timeopints / available timepoints)
#       max_consecutive timepoints (longest stretch of consecutive outlier timepoints)
#   Output: biomarker_most_persistent_<tissue>.csv
def mostPersistent(dir_output, persistentMatrix, analytes, tissue):
    results = []
    complicationOnly = persistentMatrix.loc[persistentMatrix["group"] != "Control",:]
    for m in analytes:
        mOnly = complicationOnly.loc[complicationOnly["analyte_ID"] == m,:]
        if len(mOnly.index) < 5:
            continue
        meanOutlierTP = mOnly["outlier_timepoint_count"].sum() / len(mOnly.index)
        meanTP = mOnly["total_timepoints"].sum() / len(mOnly.index)
        proportion = meanOutlierTP / meanTP
        maxConsecutive = ""
        for p in mOnly["patient_ID"]:
            raw = mOnly.loc[mOnly["patient_ID"] == p,:]["outlier_timepoints"].iloc[0]
            outlierTP = ast.literal_eval(raw)
            maxConsecutive = []
            outlierTPstring = "".join(outlierTP)
            mergedTP = "".join(TIMEPOINTS)
            if outlierTPstring in mergedTP:
                if len(outlierTPstring) > len(maxConsecutive):
                    maxConsecutive = outlierTP
        results.append({"analyte_ID": m,
                        "n_patients_affected": len(mOnly.index),
                        "mean_outlier_timepoints_per_patient_affected": meanOutlierTP,
                        "mean_proportion_timepoints": proportion,
                        "max_consecutive_timepoints": maxConsecutive})
    persistent = pd.DataFrame(results).sort_values(by=["mean_proportion_timepoints"], ascending=False).iloc[0:50,:]
    persistent.to_csv(dir_output + "/biomarker_most_persistent_" + tissue + ".csv")
    return persistent


# List 3: Early Warning
#   Goal: Analytes elevated at earlist available timepoints
#   Steps:
#       1. Define early timepoints as first 2 available collections
#       2. For each analyte in complication samples:
#           Count patients showing elevation at their earliest available timepoint
#           Count patients showing elevation at both of their first 2 timepoints (if available)
#       3. Filter to analytes elevated early in >= 10 patients
#       4. Rank by:
#           Primary: % of patients elevated at earliest timepoint
#           Secondary: % elevated at first 2 timepoints
#   Include in output:
#       analyte_ID
#       n_patients_elevated_at_earliest
#       percent_elevated_at_earliest
#       n_patient_elevated_at_first_two
#       percent_elevated_at_first_two
#       mean_MAD_score_at_earliest
#       mean_MAD_score_at_first_two
#   Output: biomarker_early_warning_<tissue>.csv
def earlyWarning(dir_output, persistentMatrix, analytes, tissue):
    # early_timepoints = A and B
    results = []
    complicationOnly = persistentMatrix.loc[persistentMatrix["group"] != "Control",:]
    for m in analytes:
        mOnly = complicationOnly.loc[complicationOnly["analyte_ID"] == m,:]
        if len(mOnly.index) == 0:
            continue
        hasA = mOnly.loc[["A" in x for x in mOnly["outlier_timepoints"]],:]
        hasB = mOnly.loc[["B" in x for x in mOnly["outlier_timepoints"]],:]       
        early = hasA.merge(hasB, how="outer")
        if len(early.index) < 10:
            continue
        both = hasA.merge(hasB, how="inner")
        results.append({"analyte_ID": m,
                        "n_patients_elevated_at_earliest": len(early.index),
                        "percent_elevated_at_earliest": len(early["patient_ID"]) / len(mOnly["patient_ID"]),
                        "n_patient_elevated_at_first_two": len(both["patient_ID"]),
                        "percent_elevated_at_first_two": len(both["patient_ID"]) / len(mOnly["patient_ID"]),
                        "mean_MAD_score_at_earliest": early["mean_outlier_mad"].sum() / len(early["patient_ID"]),
                        "mean_MAD_score_at_first_two": 0 if len(both["patient_ID"]) == 0 else both["mean_outlier_mad"].sum() / len(both["patient_ID"])})
    warning = pd.DataFrame(results).sort_values(by=['percent_elevated_at_earliest', 'percent_elevated_at_first_two'], ascending=False)
    warning.to_csv(dir_output + "/biomarker_early_warning_" + tissue + ".csv")
    return warning


# List 4: Complication-Specific
#   Goal: Analytes enriched in specific complication subtypes
#   Steps:
#       1. For each complication type (FGR, HDP, sPTB) separately
#           Calculate % of that complication type showing each analyte elevated
#       2. For each analyte:
#           Identify which complication tye shows highest %
#           Calculate enrichement (% in top complication / % in other complications)
#           If % in other complications = 0, set enrichment = InF (or a very large number like 999). If both numerator and denominator = 0, exclude analyte from this list
#       3. Filter to analytes with:
#           30% prevelence in at least one complication type
#           enrichment ratio >= 2 (at least x2 higher in one complication vs another)
#       4. Rank by enrichment ratio (descending)
#   Include in output:
#       analyte_ID
#       primary_complication_type
#       percent_in_primary_complication
#       percent_in_other_complications
#       enrichment_ratio
#       n_patients_primary_complication
#   Output: biomarker_complication_specific_<tissue>.csv
def complicationSpecific(dir_output, persistentMatrix, meta, analytes, tissue):
    results = []
    complicationOnly = persistentMatrix.loc[persistentMatrix["group"] != "Control",:]
    for m in analytes:
        mOnly = complicationOnly.loc[complicationOnly["analyte_ID"] == m,:]
        maxValue = 0
        maxKey = ""
        n_patients = 0
        compPercent = {}
        for condition in COMPLICATIONS:
            total = len(mOnly.loc[mOnly["group"].str.upper() == condition.upper(),:]["patient_ID"]) 
            percent = total / len(meta.loc[meta["group"].str.upper() == condition.upper(),:]["patient_ID"].unique())
            if percent > maxValue:
                maxValue = percent
                maxKey = condition
                n_patients = total
            compPercent[condition] = percent
        if maxValue < 0.3:
            continue
        enrichment = [float('inf') if x == 0 else maxValue/x for x in compPercent.values() if x != maxValue]
        enrichment.sort(reverse=True)
        if all(x < 2 for x in enrichment):
            continue
        otherKeys = list(compPercent.keys())
        otherKeys.remove(maxKey)
        results.append({"analyte_ID": m,
                        "primary_complication_type": maxKey,
                        "percent_in_primary_complication": maxValue,
                        "percent_in_other_complications": [compPercent[k] for k in otherKeys],
                        "enrichment_ratio": enrichment,
                        "n_patients_primary_complications": n_patients})
    specific = pd.DataFrame(results).sort_values(by="enrichment_ratio", ascending=False)
    specific.to_csv(dir_output + "/biomarker_complication_specific_" + tissue + ".csv")
    return specific
        
# List 5: Most Extreme
#   Goal: Analytes with highest magnitude deviations
#   Steps:
#       1. For each analyte (complications only):
#           Calculate median MAD score across all outlier instances
#           Calculate 99th percentile MAD score
#           Calculate max MAD score observed
#       2. Filter to analytes affecting >=5 patients
#       3. Rank by median MAD score (descending)
#       4. Select top 50 analytes
#   Include in output:
#       analyte_ID
#       n_patients_affected
#       median_MAD_score
#       percentile_99_MAD_score
#       max_MAD_score
#       patient_with_max (patient_ID showing maximum deviation)
#   Output: biomarker_most_extreme_<tissue>.csv
def mostExtreme(dir_output, persistentMatrix, analytes, tissue):
    results = []
    complicationOnly = persistentMatrix.loc[persistentMatrix["group"] != "Control",:]
    for m in analytes:
        mOnly = complicationOnly.loc[complicationOnly["analyte_ID"] == m,:]
        if len(mOnly.index) < 5:
            continue
        all_mad_scores = list(mOnly["mean_outlier_mad"])
        #for i in complicationOnly.index:
        #    all_mad_scores.extend(ast.literal_eval(complicationOnly.loc[i,"outlier_mad_scores"]))
        results.append({"analyte_ID": m,
                        "n_patients_affected": len(mOnly.index),
                        "median_MAD_score": np.median(all_mad_scores),
                        "percentile_99_MAD_score": np.percentile(all_mad_scores, 99),
                        "max_MAD_score": max(all_mad_scores),
                        "patient_with_max":  mOnly.loc[mOnly["mean_outlier_mad"] == max(all_mad_scores),"patient_ID"].iloc[0]
                        })
    extreme = pd.DataFrame(results).sort_values(by="median_MAD_score").iloc[0:50,:]
    extreme.to_csv(dir_output + "/biomarker_most_extreme_" + tissue + ".csv")
    return extreme

# helper function for running all biomarker identification functions
def identifyBiomarkers(dir_output, persistentMatrix, meta, outlierMatrix, tissue):
    # elevated only
    elevatedOnly = persistentMatrix.loc[persistentMatrix["outlier_direction"] == "elevated",:]
    mergedMeta = pd.concat([meta[t] for t in TIMEPOINTS], ignore_index=True).drop_duplicates(subset=["patient_ID"])
    prevalentMarkers = mostPrevalent(dir_output, elevatedOnly, mergedMeta, outlierMatrix["A"].columns, tissue)
    persistentMarkers = mostPersistent(dir_output, elevatedOnly, outlierMatrix["A"].columns, tissue)
    earlyMarkers = earlyWarning(dir_output, elevatedOnly, outlierMatrix["A"].columns, tissue)
    specificMarkers = complicationSpecific(dir_output, elevatedOnly, mergedMeta, outlierMatrix["A"].columns, tissue)
    extremeMarkers = mostExtreme(dir_output, elevatedOnly, outlierMatrix["A"].columns, tissue)
    return prevalentMarkers, persistentMarkers, earlyMarkers, specificMarkers, extremeMarkers

# generate crosswalk matrix
# return: dataframe where rows = metabolites, columns = category of outlier, values = 1 if in category, 0 if not
def crosswalkMatrix(dir_output, analytes, prevalent, persistent, early, specific, extreme, tissue):
    df = pd.DataFrame(0, index=analytes, columns=["most_persistent", "most_prevalent", "early_warning", "complication_specific", "most_extreme", "super_candidate"])
    for m in df.index:
        if m in set(prevalent["analyte_ID"]):
            df.loc[m, "most_prevalent"] = 1
        if m in set(persistent["analyte_ID"]):
            df.loc[m, "most_persistent"] = 1
        if m in set(early["analyte_ID"]):
            df.loc[m, "early_warning"] = 1
        if m in set(specific["analyte_ID"]):
            df.loc[m, "complication_specific"] = 1
        if m in set(extreme["analyte_ID"]):
            df.loc[m, "most_extreme"] = 1
    keep = [x > 0 for x in (list(df.sum(axis=1)))]
    df = df.loc[keep,:]
    df["super_candidate"] = [x >= 3 for x in (list(df.sum(axis=1)))]
    logging.info(str(df["super_candidate"].sum()) + " super candidate metabolites (in >=3 lists) identified.")
    df.to_csv(dir_output + "/biomarker_summary_crosswalk_" + tissue + ".csv")
    return df


    
# primary wrapper function for Outlier Analysis
def OutlierAnalysis(dir_input, dir_output, datatype, tissue, batches):
    meta, samples = splitData(dir_input, batches)
    logging.info("Calculating control reference statistics...")
    controlRef, control_IDs = controlRefStats(samples, dir_output, datatype, tissue, batches)
    logging.info("Calculating sample MAD scores...")
    scoreMatrix = MADscores(samples, dir_output, controlRef, batches, tissue, datatype)
    logging.info("Flagging outliers by patient x analyte across timepoints...")
    outlierMatrix = flagOutliers(dir_output, scoreMatrix, tissue, datatype)
    logging.info("Identifying persistent and consistent outliers...")
    #persistentMatrix = filterOutliers(dir_output, outlierMatrix, scoreMatrix, meta, tissue, datatype)
    persistentMatrix = pd.read_csv("/Users/kaylaxu/Desktop/data/MAD_analyses/persistent_outliers_plasma_MTBL.csv", index_col=0)
    # identify biomarkers
    prevalentMarkers, persistentMarkers, earlyMarkers, specificMarkers, extremeMarkers = identifyBiomarkers(dir_output, persistentMatrix, meta, outlierMatrix, tissue)
    # get crosswalk matrix
    crosswalk = crosswalkMatrix(dir_output, outlierMatrix["A"].columns, prevalentMarkers, persistentMarkers, earlyMarkers, specificMarkers, extremeMarkers, tissue)


    return


def main():
    dir_input = sys.argv[1] # e.g. /Users/kaylaxu/Desktop/data/clean_data/MTBL_plasma
    dir_output = sys.argv[2] # e.g. /Users/kaylaxu/Desktop/data/MAD_analyses

    batches = pd.read_csv(dir_input + "/pos_batch.csv")["batch"].unique().tolist()

    if "MTBL" in dir_input:
        datatype = "MTBL"
    else:
        datatype = "LIPD"
    if "plasma" in dir_input:
        tissue = "plasma"
    else:
        tissue = "placenta"

    logging.basicConfig( # initiate log file
        filename= datatype + '_outlierAnalysis.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'  # Use 'w' to overwrite the file each run, or 'a' to append
    )
    logging.info("Initializing " + datatype + " MAD outlier analysis...")

    OutlierAnalysis(dir_input, dir_output, datatype, tissue, batches)

    logging.info("DONE - " + datatype + " MAD outlier analysis complete!")
    #close log file
    logging.shutdown()
    return
    

if __name__ == "__main__":
    main()
