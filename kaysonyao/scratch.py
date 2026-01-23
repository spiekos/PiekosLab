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
    data = pd.read_csv(input_path)
    print("Data loaded successfully.")
    print(data.head(5))

# Mask QC failures with NA values
if 'QC Warning' in data.columns:
    data.loc[data['QC Warning'] == 1, data.columns[3:]] = np.nan
    print("QC failures masked.")
else:
    print("No 'QC Warning' column found.")
    print('Proteins failing Bonferroni correction:')
