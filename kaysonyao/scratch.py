import pandas as pd
import math
import numpy as np
import sys
import logging
import os

wd = os.getcwd()
input_path = os.path.join(wd, 'kaysonyao/00_raw_data/placenta', 
                        'Q-07626_Barak_TissueLysate_NPX_2023-06-12', 
                        'Q-07626_Barak_TissueLysate_NPX_2023-06-12.csv')

# Load data
if os.path.exists(input_path):
    data = pd.read_csv(input_path, sep=';')
    print("Data loaded successfully.")

    data.info()
    print(data['QC_Warning'].unique())
    print(data['Assay_Warning'].unique())
    print(data['Sample_Type'].unique())
