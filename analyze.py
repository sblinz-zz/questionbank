"""
###################################################################################################
#
# The Question Bank
# 
# File: analyze.py
# Desc: methods for computing score and grade data
#
###################################################################################################
"""

import pandas as pd

def normalize_scores(df, max_scores):
	"""
	Replace each score with its value normalized by the max score possible on the given question
	"""
	return df.apply(lambda row : row/max_scores, axis=1) 

def get_aux_score_data(df):
	"""
	Derive auxiliary score data
	"""
	grades_df = pd.DataFrame()
	grades_df['attempted'] = df.count(axis=1)
	grades_df['total score'] = df.sum(axis=1)
	return grades_df

def get_grades_as_average_score(df, attempted):
	"""
	Compuate grades Series as average of scores on attempted questions
	"""
	return df.sum(axis=1)/attempted

def get_grades_using_max_scores(df, grades_df, max_scores):
	"""
	Compute grades Series as ratio of total score to max possible score on attempted questions
	"""
	student_max_score = pd.Series(index=df.index)
	grades = pd.Series(index=df.index)

	if isinstance(max_scores, float):
			student_max_score = grades_df['attempted']*max_scores
			grades = grades_df['total score']/student_max_score

	elif isinstance(max_scores, pd.Series):
		for index, row in df.iterrows():
			attempted = df.ix[index].apply(lambda score : score >= 0)
			attempted_cols = df.ix[index, attempted].index
			max_score_possible = max_scores[attempted_cols].sum()
			student_max_score.ix[index] = max_score_possible
			grades.ix[index] = grades_df.ix[index,'total score']/max_score_possible

	return student_max_score, grades