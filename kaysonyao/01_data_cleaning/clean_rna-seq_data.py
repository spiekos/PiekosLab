import pandas as pd
import sys


def save_meta_file(df, file_output_meta):
	df.rename(columns={'name': 'Transcript'}, inplace=True)
	df['Transcript'].to_csv(file_output_meta, index=False)


def create_name_dict(df):
	d = {}
	for index, row in df.iterrows():
		k = row['initial_alias']
		v = row['name']
		d[k] = v
	return d


def format_meta_data_file(file_name_converter, file_output_meta):
	df = pd.read_csv(file_name_converter, header=0)
	df.loc[df['name'] == 'None', 'name'] = df.loc[df['name'] == 'None', 'initial_alias']
	dict_names = create_name_dict(df)
	save_meta_file(df, file_output_meta)
	return dict_names


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


def rename_transcripts(df, d):
	for index, row in df.iterrows():
		name = row[0]
		if name in d.keys():
			value = d[name]
			df.loc[index, 0] = value
		else:
			df.loc[index, 0] = name
	return df


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


def format_data_file(file_descriptor, file_input, file_output, dict_names):
	dict_descriptor = read_file_to_dict(file_descriptor, 'Sample ID', 'Study ID')
	df_input = pd.read_csv(file_input, header=None, low_memory=False)
	df_input = rename_transcripts(df_input, dict_names)
	df_input = df_input.T
	df_input.iloc[0, 0] = 'Patient-ID'
	df_input.columns = df_input.iloc[0]
	df_input = df_input.tail(-1)
	df_final = format_patient_ids(df_input, dict_descriptor)
	df_final.to_csv(file_output)


def format_files(file_descriptor, file_input, file_name_converter, file_output, file_output_meta):
	dict_names = format_meta_data_file(file_name_converter, file_output_meta)
	format_data_file(file_descriptor, file_input, file_output, dict_names)


def main():
	file_descriptor = sys.argv[1]
	file_input = sys.argv[2]
	file_name_converter = sys.argv[3]
	file_output = sys.argv[4]
	file_output_meta = sys.argv[5]
	format_files(file_descriptor, file_input, file_name_converter, file_output, file_output_meta)


if __name__ == "__main__":
    main()
