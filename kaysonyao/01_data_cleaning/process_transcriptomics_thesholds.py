# import environment
import numpy as np
import pandas as pd
import sys


# Declare Universal Variable
THRESH_MISSINGNESS = 0.8
THRESH_UNIQUENESS = 0.5
THRESH_RNA_READ_COUNTS = 500


def filter_data(df):
	df_filtered = df.shape[0] * THRESH_MISSINGNESS
	df.dropna(thresh=df_filtered, axis=1, inplace=True)
	minUnique = df.nunique()/len(df)
	col_drop = minUnique[minUnique < THRESH_UNIQUENESS].index
	df.drop(columns=col_drop, inplace=True)
	col_sums = df.sum()
	col_sums.drop(col_sums.index[1], inplace=True)
	col_drop = col_sums[col_sums < THRESH_RNA_READ_COUNTS].index
	df.drop(columns=col_drop, inplace=True)
	return df


def impute_missing_data(df):
	float_columns = [col for col in df.columns if col != 'Patient-ID']
	df[float_columns] = df[float_columns].astype(float)
	df.fillna(0, inplace=True)

	# Calculate the minimum values for each numeric column
	df_filtered= df[df[float_columns] != 0]
	min_values = df_filtered[float_columns].min()/2

	# Replace 0 values in the numeric columns with their respective minimum values
	df[float_columns] = df[float_columns].apply(lambda col: np.where(col == 0, min_values[col.name], col))

	return df


def read_analyte_df(file_input):
	df = pd.read_table(file_input, sep=',', dtype={'Patient-ID':str})
	#df_temp = pd.DataFrame(df['Patient-ID'])
	df = filter_data(df)
	df = impute_missing_data(df)
	#df = pd.concat([df_temp, df], axis=1)
	print(df.head())
	print(df.shape)
	return df


def format_data(file_transcripts, file_clinical, file_output):
	df = read_analyte_df(file_transcripts)
	df_clinical = pd.read_table(file_clinical, sep=',', dtype={'Patient-ID':str})
	df_clinical = df_clinical[df_clinical['Condition']!='Control PTD']  # remove Control PTD placentas
	df_clinical = df_clinical[df_clinical['InfSex']>=0]
	df_clinical.head()
	df = df[df['Patient-ID'].isin(df_clinical['Patient-ID'])]
	df = df.drop(df.columns[[0]], axis=1) # drop unnamed column
	print(df.head())
	print(df.shape)
	df.to_csv(file_output, index=False, sep=',')
	return


def main():
	file_transcripts = sys.argv[1]
	file_clinical = sys.argv[2]
	file_output = sys.argv[3]
	format_data(file_transcripts, file_clinical, file_output)


if __name__ == '__main__':
	main()
