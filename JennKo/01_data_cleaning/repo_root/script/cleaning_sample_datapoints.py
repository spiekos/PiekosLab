import pandas as pd

###Load dp3 file with gest/sample ages
dp3_age = pd.read_excel("/Users/jennko/Desktop/Penn/Piekos Lab/PiekosLab/JennKo/01_data_cleaning/repo_root/data/dp3 master table v2.xlsx", sheet_name="n=133 proteomics")
dp3_clinical = pd.read_excel("/Users/jennko/Desktop/Penn/Piekos Lab/PiekosLab/JennKo/01_data_cleaning/repo_root/data/dp3 master table v2.xlsx", sheet_name="clinical data", header=1)
print(dp3_clinical)

###Data Processing
##Editing ID (include first 8 char)
dp3_age=dp3_age[['ID','sample Id','gest age del','sample gest Age', 'group']].copy()
dp3_age['ID']=dp3_age['ID'].astype(str).str[:8]

##Calculating differences in gestational age between delivery and sample collection
dp3_age['gest_age_del'] = dp3_age['gest age del'].round(1)
dp3_age['gest_age_samp'] = dp3_age['sample gest Age'].round(1)
dp3_age['gest_age_diff'] = dp3_age['gest_age_del'] - dp3_age['gest_age_samp']

##Categorization of samples collection by trimester
def categorize_sample_time(gest_age_samp):
    if gest_age_samp >= 6.0 and gest_age_samp <= 13.9:
        return '1'
    elif gest_age_samp >= 14.0 and gest_age_samp <= 21.9:
        return '2'
    elif gest_age_samp >= 22.0 and gest_age_samp <= 31.9:
        return '3'
    elif gest_age_samp >= 32.0 and gest_age_samp <= 36.9:
        return '4'
    elif gest_age_samp >= 37.0:
        return '5'
    else:
        return None

dp3_age['sample_time_cat']=dp3_age['gest_age_samp'].apply(categorize_sample_time)

dp3_age['sample_time_cat_name']=dp3_age['sample_time_cat'].map({
    '1':'6.0-13.9',
    '2':'14.0-21.9',
    '3':'22.0-31.9',
    '4':'32.0-36.9',
    '5':'37.0+'
})

cat_order=['6.0-13.9','14.0-21.9','22.0-31.9','32.0-36.9','37.0+']

# Total number of samples per time category
sample_counts=dp3_age['sample_time_cat_name'].value_counts().reindex(cat_order)
print("Total number of samples per time category:")
print(sample_counts)
print("Total:",sample_counts.sum())

# Unique IDs per category
uid_counts=dp3_age.groupby('sample_time_cat_name')['ID'].nunique().reindex(cat_order)
print("\nTotal number of unique IDs per sample category:")
print(uid_counts)
print("Total unique IDs:",dp3_age['ID'].nunique())

for g,df_g in dp3_age.groupby('group'):
    tbl=(
        df_g.groupby('sample_time_cat_name')
        .agg(
            total_samples=('ID','size'),
            unique_ids=('ID','nunique')
        )
        .reindex(cat_order)
        .reset_index()
        .rename(columns={'sample_time_cat_name':'time_cat'})
    )
    print(f"\nGroup: {g}")
    print(tbl.to_string(index=False))

##When add delivery samples, they trigger if statement first 
##Categorization of labor onset
dp3_clinical=dp3_clinical[['ID','LABOR_ONSET','INDICATED_ONSET (1=INDICATED; 0=NOT INDICATED)']].copy()
dp3_clinical=dp3_clinical.rename(columns={'INDICATED_ONSET (1=INDICATED; 0=NOT INDICATED)':'indicated_onset_flag'})

def categorize_labor_onset(row):
    if row['LABOR_ONSET'] == 'SPONTANEOUS':
        return 1
    else:
        return 0
dp3_clinical['spont_labor_flag'] = dp3_clinical.apply(categorize_labor_onset, axis=1)

def cat_labor_onset(row):
    if row['LABOR_ONSET'] == 'SPONTANEOUS' or row['indicated_onset_flag'] == 0:
        return 1
    else:
        return 0
dp3_clinical['cat_labor_onset_flag'] = dp3_clinical.apply(cat_labor_onset, axis=1)

##Indicator for samples collected within 0.1 weeks of delivery
dp3_age['gest_sample_flag'] = dp3_age['gest_age_diff'].apply(lambda x: 1 if x <= 0.1 else 0)

###Flag samples collected within 0.1 weeks of spontaneous/non-indicated delivery, samples taken after delivery; start: 503
##Within 0.1 weeks of spontaneous/indicated delivery
dp3_age_filt1=dp3_age[~((dp3_age['gest_sample_flag']==1) & (dp3_clinical['cat_labor_onset_flag']==1))].copy()
n_all=len(dp3_age)
n_filt1=len(dp3_age_filt1)
print(f'rows filtered: {n_filt1} (dropped {n_all-n_filt1})') #27 dropped - 503 
##Samples taken after delivery 
dp3_age_filt2=dp3_age_filt1[dp3_age_filt1["gest_age_diff"]>=0]
n_filt2=len(dp3_age_filt2)
print(f'rows filtered: {n_filt2} (dropped {n_filt1-n_filt2})') #4 dropped - 499
dp3_age_filt2
#break down by sample type (FGR, )

#Check for samples after delivery for amount of time after