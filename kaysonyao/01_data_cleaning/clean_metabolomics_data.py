# set up environment
import numpy as np
import pandas as pd
import sys


def gen_dict_from_file(filepath, key_name, value_name):
	d = {}
	file = open(filepath, mode='r')
	df = pd.read_csv(file, sep=',', header=0)
	for index, row in df.iterrows():
		key = str(row[key_name])
		value = str(row[value_name])
		d[key] = value
	file.close()
	return d


def rename_indices(df, file):
	d = gen_dict_from_file(file, 'PARENT_SAMPLE_NAME', 'CLIENT_IDENTIFIER')
	df.rename(index=d, inplace=True)
	return df


def rename_columns(df, file):
	d = gen_dict_from_file(file, 'CHEM_ID', 'CHEMICAL_NAME')
	df.rename(columns=d, inplace=True)
	return df


def clean_metabolite_file(file_input, file_patient_key, file_chemical_key, file_output):
	file = open(file_input, mode='r')
	df_metabolites = pd.read_csv(file, sep=',', header=0, index_col=0)
	df_metabolites = rename_indices(df_metabolites, file_patient_key)
	df_metabolites = rename_columns(df_metabolites, file_chemical_key)
	df_metabolites[df_metabolites < 0] = 'NA'  # replace negative values with NA
	#df_metabolites = np.log2(df_metabolites)
	df_metabolites = df_metabolites.fillna('NA') # indicate missing values with NA
	df_metabolites.rename_axis('Patient-ID', inplace=True)
	df_metabolites.to_csv(file_output)
	file.close()


def main():
	file_input = sys.argv[1]
	file_patient_key = sys.argv[2]
	file_chemical_key = sys.argv[3]
	file_output = sys.argv[4]
	clean_metabolite_file(file_input, file_patient_key, file_chemical_key, file_output)


if __name__ == "__main__":
    main()
