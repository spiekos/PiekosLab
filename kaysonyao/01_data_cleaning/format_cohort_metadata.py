import numpy as np
import pandas as pd
import sys


DICT_RENAME_COLS = {\
    'FDELTYPE': 'DeliveryMethod',\
    'MaternalAge/Age': 'MaternalAge',\
    'Smoke': 'Smoker',\
    'Study_ID/ID': 'Patient-ID',\
    }


LIST_COLS_SUBSET = [\
    'Study_ID/ID',\
    'Condition',\
    'LaborInitiation',\
    'MaternalAge/Age',\
    'Height_Meters/HeightMeters',\
    'Race',\
    'Ethnicity',\
    'WksGest',\
    'PrePregWt_Kg',\
    'Grav',\
    'Para',\
    'LaborOnset',\
    'Smoke',\
    'MTOXCOC',\
    'MTOXNARC',\
    'MTOXOTHR',\
    'MTOXTHC',\
    'IWITHDRAW',\
    'FDELTYPE',\
    'MCVDHTN',\
    'MOBHTN',\
    'InfSex',\
    'Birthweight',\
    ]


LIST_COLS_FINAL = [\
    'Patient-ID',\
    'Condition',\
    'LaborInitiation',\
    'MaternalAge',\
    'isAsian',\
    'isBlack',\
    'isNativeAmerican',\
    'isPacificIslander',\
    'isWhite',\
    'RaceMissing',\
    'Ethnicity',\
    'WksGest',\
    'PregravidBMI',\
    'Grav',\
    'Para',\
    'LaborOnset',\
    'Smoker',\
    'IllicitDrugUser',\
    'DeliveryMethod',\
    'PregnancyRelatedHypertension',\
    'FetalGrowthRestriction',\
    'Preeclampsia',\
    'InfSex',\
    'Birthweight',\
    ]


def rename_cols(df):
    df = df.rename(columns=DICT_RENAME_COLS)
    return df


def check_hyp_status(chronic_hypertension, pregnancy_hypertension):
    total = chronic_hypertension + pregnancy_hypertension
    if total >= 1:
        return 1
    return 0


def check_preeclampsia_status(pregnancy_hypertension, condition):
    list_preeclampsia_codes = [2, 3, 5]
    if pregnancy_hypertension in list_preeclampsia_codes:
        return 1
    if 'PE' in condition:
        return 1
    return 0


def check_FGR_status(condition):
    if 'FGR' in condition:
        return 1
    return 0


def format_int(i):
    if i != i:
        return 0
    if i:
        return int(i)
    return 0


def format_conditions(df):
    df['FetalGrowthRestriction'] = 0
    df['Preeclampsia'] = 0
    df['PregnancyRelatedHypertension'] = 0
    for index, row in df.iterrows():
        condition = row['Condition']
        chronic_hypertension = format_int(row['MCVDHTN'])
        pregnancy_hypertension = format_int(row['MOBHTN'])
        df.at[index, 'FetalGrowthRestriction'] = check_FGR_status(condition)
        df.at[index, 'Preeclampsia'] = check_preeclampsia_status(pregnancy_hypertension, condition)
        df.at[index, 'PregnancyRelatedHypertension'] = check_hyp_status(chronic_hypertension, pregnancy_hypertension)
    return df


def add_drug_use(df):
    df['IllicitDrugUser'] = np.nan
    df_temp = df.fillna(0)
    for index, row in df_temp.iterrows():
        if int(row['MTOXCOC']) or int(row['MTOXNARC']) or int(row['MTOXOTHR']) or int(row['MTOXTHC']) or int(row['IWITHDRAW']):
            df.at[index, 'IllicitDrugUser'] = 1
        else:
            df.at[index, 'IllicitDrugUser'] = 0
    return df


def calculate_bmi(height, weight):
    return weight/height**2


def add_bmi(df):
    df['PregravidBMI'] = np.nan
    for index, row in df.iterrows():
        height = float(row['Height_Meters/HeightMeters'])
        weight = float(row['PrePregWt_Kg'])
        if height and weight:
            df.at[index, 'PregravidBMI'] = calculate_bmi(height, weight)
    return df


def format_metadata(df):
    df = add_bmi(df)
    df = add_drug_use(df)
    df = format_conditions(df)
    df['isAsian'] = df['Race']
    df['isBlack'] = df['Race']
    df['isNativeAmerican'] = df['Race']
    df['isPacificIslander'] = df['Race']
    df['isWhite'] = df['Race']
    df['RaceMissing'] = df['Race']
    df = df.replace({'Condition' : {'FGR+PE Severe': 'FGR+hyp', 'FGR+PE mild': 'FGR+hyp', 'FGR+PE Mild': 'FGR+hyp', 'FGR+Transient hypertension': 'FGR+hyp', 'FGR+Chronic Hypertension':  'FGR+hyp'}})
    df = df.replace({'FDELTYPE' : {'2': 1, '1': 0, '0': 0, np.nan : -1}})
    df = df.replace({'InfSex' : { 'F': 0, 'M': 1, np.nan: -1 }})
    df = df.replace({'LaborOnset' : {'Induced' : 1, 'Induced (does not include PROM)' : 1, 'No labor': 1, 'PROM' : 0,  'Spontaneous': 0, np.nan : 0 }})
    df = df.replace({'Smoke': {'Yes, former smoker quit more than 12 months ago':0, 'Never smoker':0, 'Yes, smoker, current status unknown':0, 'Yes, former smoker quit within the past 12 months':0,'Yes, smoke everyday':1,'Yes, smoke some days':1, np.nan : -1 }})
    df = df.replace({'Para' : {'0':0, '1': 1, '2': 1, '3':1, '4':1, '5':1, '6':2, '7':2, np.nan : -1}})
    df = df.replace({'Grav' : {'0':0, '1': 0, '2': 1, '3':1, '4':1, '5':1, '6':2, '7':2, '8':2, '9':2, '10':2, '11':2, '12':2, np.nan : -1}})
    df = df.replace({'Ethnicity' : {'NOT HISPANIC OR LATINO': 0, 'HISPANIC OR LATINO': 1, 'UNSPECIFIED':-1, 'DECLINED':-1, np.nan: -1 }})
    df = df.replace({'isAsian' : {'B': 0, 'C': 1, 'D': 0, 'E': 1, 'F': 1, 'G': 0, 'I': 1, 'J': 1, 'K': 1, 'L': 0, 'N': 0, 'P': 0, 'Q': 0, 'S': 0, 'V': 1, 'W': 0, '9': 0, np.nan: 0 }})
    df = df.replace({'isBlack' : {'B': 1, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0, 'N': 0, 'P': 0, 'Q': 0, 'S': 0, 'V': 0, 'W': 0, '9': 0, np.nan: 0 }})
    df = df.replace({'isNativeAmerican' : {'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 1, 'N': 1, 'P': 0, 'Q': 0, 'S': 0, 'V': 0, 'W': 0, '9': 0, np.nan: 0 }})
    df = df.replace({'isPacificIslander' : {'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 1, 'I': 0, 'J': 0, 'K': 0, 'L': 0, 'N': 0, 'P': 1, 'Q': 1, 'S': 1, 'V': 0, 'W': 0, '9': 0, np.nan: 0 }})
    df = df.replace({'isWhite' : {'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0, 'N': 0, 'P': 0, 'Q': 0, 'S': 0, 'V': 0, 'W': 1, '9': 0, np.nan: 0 }})
    df = df.replace({'RaceMissing' : {'B': 0, 'C': 0, 'D': 1, 'E': 0, 'F': 0, 'G': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0, 'N': 0, 'P': 0, 'Q': 0, 'S': 0, 'V': 0, 'W': 0, '9': 1, np.nan: 1 }})
    return df

def clean_file(file_input, file_output):
    df = pd.read_csv(file_input)[LIST_COLS_SUBSET]
    df = df.iloc[1:]
    df = format_metadata(df)
    df = rename_cols(df)
    df = df[LIST_COLS_FINAL]
    df.to_csv(file_output, index=False)


def main():
    file_input =  sys.argv[1]
    file_output= sys.argv[2]
    clean_file(file_input, file_output)


if __name__ == "__main__":
    main()
