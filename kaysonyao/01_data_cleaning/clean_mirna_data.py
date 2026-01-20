import pandas as pd
import sys


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


def format_patient_ids(df, d):
	for index, row in df.iterrows():
		pat_id = str(row['Patient-ID'])
		pat_id = pat_id.replace('.', '-')
		if pat_id in d.keys():
			value = d[pat_id]
			df.loc[index,'Patient-ID'] = value
		else:
			df.loc[index,'Patient-ID'] = pat_id
	return df


def format_file(file_descriptor, file_input, file_output):
	dict_descriptor = read_file_to_dict(file_descriptor, 'Sample ID', 'Study ID')
	df_input = pd.read_csv(file_input, header=None, low_memory=False).T
	df_input.iloc[0, 0] = 'Patient-ID'
	df_input.columns = df_input.iloc[0]
	df_input = df_input.tail(-1)
	df_final = format_patient_ids(df_input, dict_descriptor)
	df_final.to_csv(file_output)


def main():
	file_descriptor = sys.argv[1]
	file_input = sys.argv[2]
	file_output = sys.argv[3]
	format_file(file_descriptor, file_input, file_output)


if __name__ == "__main__":
    main()
