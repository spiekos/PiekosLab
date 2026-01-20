# load envrionment
import math
import numpy as np
import rpy2.robjects.numpy2ri
import pandas as pd
import sys

from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()


# declare universal variables
CUTOFF_PERCENT_MISSING = 0.80
CUTOFF_MIN_COUNT = 500


def read_file_to_dict(f, k_col, v_col):
	d = {}
	df = pd.read_csv(f)
	for index, row in df.iterrows():
		key = str(row[k_col])
		key = key.split(' ')[0]
		value = row[v_col]
		if key == key:
			d[key] = value
	return d


def format_patient_ids(df, d, dict_subgroups):
	df.insert(1, 'Group', '')
	for index, row in df.iterrows():
		pat_id = str(row['Patient-ID'])
		pat_id = pat_id.replace('.', '-')
		if pat_id in d.keys():
			pat_id = d[pat_id]
		df.loc[index, 'Patient-ID'] = pat_id
		subgroup = dict_subgroups[pat_id]
		df.loc[index, 'Group'] = subgroup
	df = df.replace(["FGR+PE Severe", "FGR+PE mild", "FGR+PE Mild", "FGR+chronic", "FGR+Transient hypertension"], "FGR+Hypertension")
	df.iloc[:, 2:] = df.iloc[:, 2:].astype(int)
	df = df.replace(0, np.nan)
	return df


def count_missingness_group_distribution(df, analyte, dict_group_count):
	stats = importr('stats')
	d = {}
	m = np.zeros(shape=(2,len(dict_group_count.keys())))
	p = 1
	for key in dict_group_count.keys():
		d[key] = 0
	for index, data in df.iterrows():
		if math.isnan(data[analyte]):
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
	n_failed_analytes = len(dict_missing_p_values.keys())
	a_bonferroni = 0.05/n_failed_analytes
	print('Number of analytes failing cutoff: ' + str(n_failed_analytes))
	print('Bonferroni Correction alpha: ' + str(a_bonferroni))
	print('Analytes with significantly different percentage missing between groups following Bonferroni correction:')
	for k, v in dict_missing_p_values.items():
		if v < a_bonferroni:
			print(k)
			c += 1
	print('Number of analytes failing cutoff with significant difference in distribution of missingess between groups: ' + str(c))


def sum_list(l):
	s = 0
	l = list(l)
	for i in l:
		if i == i:
			i = int(i)
			s += i
	return s 


def remove_cols_failing_missingness_cutoff(df):
	dict_missing_p_values = {}
	count_fail_cutoff = 0
	dict_group_count = count_samples_in_each_group(df)
	df_new = df[['Patient-ID', 'Group']]
	print("Analyte Failing Missingness Cutoff")
	print("Group: Number of Samples Missing Analyte Measurement (%)")
	for name, data in df.iloc[:, 2:].items():
		percent_missing = data.isna().sum()/len(data)
		if percent_missing < CUTOFF_PERCENT_MISSING:
			if sum(data) > CUTOFF_MIN_COUNT:
				df_new[name] = data
				df_new = df_new.copy()
			else:
				count_fail_cutoff += 1
		else:
			print(name)
			print('Total: ' + str(data.isna().sum()) + ' (' + str(round(percent_missing*100, 1)) + '%)')
			dict_missing_p_values[name] = count_missingness_group_distribution(df, name, dict_group_count)
	print('\nNumber of analytes that pass the missingness cutoff, but fail the minimum count cutoff:', str(count_fail_cutoff), '\n\n')
	print('\n\n')
	print('Number of samples:', str(df_new.shape[0]))
	print('Number of analytes:', str(df_new.shape[1]))
	perform_bonferroni_correction(dict_missing_p_values)
	return df_new


def format_file(file_descriptor, file_input, file_output):
	dict_descriptor = read_file_to_dict(file_descriptor, 'Sample ID', 'Study ID')
	dict_subgroups = read_file_to_dict(file_descriptor, 'Study ID', 'Condition')
	df_input = pd.read_csv(file_input, header=None, low_memory=False).T
	df_input.iloc[0, 0] = 'Patient-ID'
	df_input.columns = df_input.iloc[0]
	df_input = df_input.tail(-1)
	df_formatted = format_patient_ids(df_input, dict_descriptor, dict_subgroups)

	# min impute formatted df
	df_min_imputed = remove_cols_failing_missingness_cutoff(df_formatted)
	for name, data in df_min_imputed.iteritems():
		df_min_imputed[name].fillna(value=df_formatted[name].min(), inplace=True)
	df_min_imputed.to_csv(file_output)


def main():
	file_descriptor = sys.argv[1]
	file_input = sys.argv[2]
	file_output = sys.argv[3]
	format_file(file_descriptor, file_input, file_output)


if __name__ == "__main__":
    main()
