'''
Author: Samantha Piekos
Date: 12/8/22
Updated_Date: 5/19/23
Description: This takes in the placental histopathology report received from the
clinician and convert it into a cleaned file that denotes each of the four
features evaluated as an ordianl value. This relies on the input file to have
the following columns: case, MVM, FVM, AI, CI, and Comments. It requires the
presence of the features to be indicated by a 'Y' and expects the phrase of
'low', 'moderate', or 'high' followed by 'grade' followed by 'FVM', 'AI', or
'CI' to indicate the severity of the FVM, AI, or CI features in the Comments.
In addition, the following binary features noted from the placental slides:
''

python3 format_placental_histopathology_report.py file_input file_export

@file_input1	filepath to the csv file containing placental histopathology reports
@file_input2	filepath to the csv file containing placental slide features
@file_export	filepath to cleaned csv file formatting features as ordinal values
'''


# load environment
import pandas as pd
import sys


# declare universal variables
DF2_COLUMNS=[
	'Patient-ID',
	'MOMI-ID',
	'Condition',
	'hvarClusters',
	'gestationalAge',
	'DVH',
	'AVM',
	'Syncytial Knots',
	'Segmental Avascular Villi',
	'DVM',
	'Villitis of Unknown Etiology',
	'Diffuse Villous Edema',
	'Chorangiosis'
	]

DF2_LIMITED_COLUMNS=[
	'Patient-ID',
	'DVH',
	'AVM',
	'Syncytial Knots',
	'Segmental Avascular Villi',
	'DVM',
	'Villitis of Unknown Etiology',
	'Diffuse Villous Edema',
	'Chorangiosis'
	]



def format_slide_features_df(file_input):
	'''
	Read in the additional placental histopathology features from slide analysis.
	Format them into a pandas dataframe containing the patient-id, condition, and
	features only.
	'''
	df = pd.read_csv(file_input, header=None)
	df.columns = DF2_COLUMNS
	df = df[DF2_LIMITED_COLUMNS]
	df = df.iloc[2:]
	return df


def evaluate_3_grade_scale(comments, feature, index):
	'''
	Used to evaluate FVM and CI level which is recorded as low or high. This returns
	a 1 for a low grade and 2 for a high grade of the feature.
	'''
	comments = comments.lower()
	str_low = ('low grade ' + feature).lower()
	str_high = ('high grade ' + feature).lower()
	if str_low in comments:
		return 1
	if str_high  in comments:
		return 2
	print('Error! Expected phrase concerning ' + feature + ' not found in comments!')
	print('Does not match: ', str_low, ' or ', str_high)
	print('Row : ' + str(index))
	print(comments)
	print('\n')
	return -999


def evaluate_4_grade_scale(comments, feature, index):
	'''
	Used to evaluate AI level which is recorded as low, moderate, or high. This 
	returns a 1 for a low grade, 2 for moderate grade, and 3 for a high grade of the 
	feature.
	'''
	comments = comments.lower()
	str_low = ('low grade ' + feature).lower()
	str_moderate_1 = ('moderate grade ' + feature).lower()
	str_moderate_2 = ('moderate ' + feature).lower()
	str_high = ('high grade ' + feature).lower()
	if str_low in comments:
		return 1
	if str_moderate_1 in comments or str_moderate_2 in comments:
		return 2
	if str_high in comments:
		return 3
	print('Error! Expected phrase concerning ' + feature + ' not found in comments!')
	print('Does not match: ', str_low, ', ', str_moderate_1, ', ', str_moderate_2, ', or ', str_high)
	print('Row : ' + str(index))
	print(comments)
	print('\n')
	return -999


def evaluate_histopathological_features(row, dict_new_row, index):
	'''
	Convert histopathological features to binary or ordinal scales.

	MVM	0	Absent
		1 	Present

	FVM	0	None
		1 	Low Grade
		2 	High Grade

	AI	0	None
		1 	Low Grade
		2 	Moderate Grade
		3 	High Grade

	CI 	0 	None
		1 	Low Grade
		2 	High Grade

	PlacentalAbnormality	0 Present
							1 Absent
	'''
	if row['MVM'] == 'Y':
			dict_new_row['MVM'] = 1
	if row['FVM'] == 'Y':
		dict_new_row['FVM'] = evaluate_3_grade_scale(row['Comments'], 'FVM', index)
	if row['AI'] == 'Y':
		dict_new_row['AI'] = evaluate_4_grade_scale(row['Comments'], 'AI', index)
	if row['CI'] == 'Y':
		dict_new_row['CI'] = evaluate_3_grade_scale(row['Comments'], 'CI', index)
	if dict_new_row['MVM'] + dict_new_row['FVM'] + dict_new_row['AI'] + dict_new_row['CI'] > 0:
		dict_new_row['Placental Abnormality'] = 1
	return dict_new_row


def format_histopathology_report_df(file_input):
	'''
	Read input histopathology report and create and return a new dataframe that has
	converted the values for the features into binary or oridianl values.
	'''
	df = pd.read_excel(file_input, engine='openpyxl')
	df_new = pd.DataFrame(columns=['Patient-ID', 'MVM', 'FVM', 'AI', 'CI', 'Placental Abnormality'])
	for index, row in df.iterrows():
		dict_new_row = {'Patient-ID': '', 'MVM': 0, 'FVM': 0, 'AI': 0, 'CI': 0, 'Placental Abnormality': 0}
		dict_new_row['Patient-ID'] = row['case'].replace(' ', '-')
		dict_new_row = evaluate_histopathological_features(row, dict_new_row, index)
		df_row = pd.DataFrame(dict_new_row, index=[0])
		df_new = pd.concat([df_new[:], df_row]).reset_index(drop=True)
	return df_new


def create_clean_df(file_input, file_input2, file_output, file_output2):
	df1 = format_histopathology_report_df(file_input)
	df1.to_csv(file_output, index=False)
	df2 = format_slide_features_df(file_input2)
	df2.to_csv(file_output2, index=False)


def main():
	file_input = sys.argv[1]
	file_input2 = sys.argv[2]
	file_output = sys.argv[3]
	file_output2 = sys.argv[4]
	create_clean_df(file_input, file_input2, file_output, file_output2)


if __name__ == "__main__":
    main()
