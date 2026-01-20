'''
Title: clean_proteomics_data.py
Author: Samantha Piekso
Date: 9/28/21
Description: This takes in a .csv of Olink full proteomics data. It removes columns (proteins) with
25% or more of the values missing. It then performs minimum imputation on the remaining missing
values in the dataframe and saves to a new .csv. Prior to input convert all red cell values (olink
determined below the level of detection) and red text values (sample failed quality control for
that olink panel) using VBA in excel.

@file_input 	path to olink .csv input file
@file_output 	path to save minimum imputed data .csv file
'''


import sys
import pandas as pd
import math
import numpy as np
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()


CUTOFF_PERCENT_MISSING = 0.20


def count_missingness_group_distribution(df, protein, dict_group_count):
	stats = importr('stats')
	d = {}
	m = np.zeros(shape=(2,len(dict_group_count.keys())))
	p = 1
	for key in dict_group_count.keys():
		d[key] = 0
	for index, data in df.iterrows():
		if math.isnan(data[protein]):
			d[data['Group']] += 1
	c = 0
	for k, v in d.items():
		percent = round(v*100/dict_group_count[k], 2)
		print(k + ': ' + str(v) + ' (' + str(percent) +'%)')
		m[:, c] = [v, dict_group_count[k]-v]
		c += 1
	if np.all((m[1, :] == 0)):
		print('p-value:  1.0')
	else:
		res = stats.fisher_test(m,  simulate_p_value=True)
		print('p-value: {}'.format(res[0][0]))
		p = res[0][0]
	print('\n')
	return(p)


def count_samples_in_each_group(df):
	d = {}
	for item in df['Group']:
		if item not in d:
			d[item] = 1
		else:
			d[item] += 1
	return d


def perform_bonferroni_correction(dict_missing_p_values):
	c = 0
	n_failed_proteins = len(dict_missing_p_values.keys())
	a_bonferroni = 0.05/n_failed_proteins
	print('Number of proteins failing cutoff: ' + str(n_failed_proteins))
	print('Bonferroni Correction alpha: ' + str(a_bonferroni))
	print('Proteins with significantly different percentage missing between groups following Bonferroni correction:')
	for k, v in dict_missing_p_values.items():
		if v < a_bonferroni:
			print(k)
			c += 1
	print('Number of proteins failing cutoff with significant difference in distribution of missingess between groups: ' + str(c))


def remove_cols_failing_missingness_cutoff(df):
	dict_missing_p_values = {}
	dict_group_count = count_samples_in_each_group(df)
	df_new = df['Group'].to_frame()
	print("Protein Failing Missingness Cutoff")
	print("Group: Number of Samples Missing Protein Measurement (%)")
	for name, data in df.iteritems():
		percent_missing = data.isna().sum()/len(data)
		if percent_missing < CUTOFF_PERCENT_MISSING:
			df_new[name] = data
			df_new = df_new.copy()
		else:
			print(name)
			print('Total: ' + str(data.isna().sum()) + ' (' + str(round(percent_missing*100, 1)) + '%)')
			dict_missing_p_values[name] = count_missingness_group_distribution(df, name, dict_group_count)
	print('\n\n')
	print('Number of samples: ' + str(df_new.shape[0]))
	print('Number of proteins: ' + str(df_new.shape[1]))
	print('\n')
	perform_bonferroni_correction(dict_missing_p_values)
	return df_new


def min_imputation_of_missing_data(file_input, file_output):
	df = pd.read_csv(file_input, header=0, index_col=0)
	df = df.replace(["FGR+PE Severe", "FGR+PE mild", "FGR+PE Mild", "FGR+Chronic Hypertension", "FGR+Transient hypertension"], "FGR+Hypertension")
	df_min_imputed = remove_cols_failing_missingness_cutoff(df)
	for name, data in df_min_imputed.iteritems():
		df_min_imputed[name].fillna(value=df[name].min(), inplace=True)
	df_min_imputed.to_csv(file_output)


def main():
	file_input = sys.argv[1]
	file_output = sys.argv[2]
	min_imputation_of_missing_data(file_input, file_output)


if __name__ == "__main__":
    main()
