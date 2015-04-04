"""
###################################################################################################
#
# The Question Bank
#	- A machine learning library for analyzing student test question data
#	- Analyze, Recommend Effective Questions, Predict Student Responses
# 
# By Sam Blinstein, March-April 2015
# 
# File: question_bank.py
# Desc: main class representing a session with a given dataset
#
###################################################################################################
"""

import imp
import pandas as pd
import numpy as np

an = imp.load_source('', 'analyze.py')
plot = imp.load_source('', 'plot.py')

class QuestionBank:
	"""
	Main wrapper for session
	"""

	def __init__(self, df=None, normalized=False):
		self.df = df
		self.normalized = normalized

	def __init__(self, filepath, type='csv', normalized=False):
		if type == 'csv':
			self.df = pd.DataFrame.from_csv(filepath)

		self.normalized = normalized

	def Reshape(self, students, questions, scores, max_scores=None):
		"""
		Pivot the DF to assign student id's as index, question id's as features, and scores as values
		If max scores are given generate a separate Series for them indexed by question id's
		Place NaN if no entry for a given (student, question) pair

		Params:
			@students, @questions, @scores: Series names in self.df containing the respective data
			@max_scores: Series name in self.df containing the max possible score on the given question
		"""
		self.df.reset_index(inplace=True)
		if max_scores != None:
			self.max_scores = self.df.pivot_table(index=questions, values=max_scores, aggfun=max)

		self.df = self.df.pivot_table(index=students, columns=questions, values=scores)

	def AddMaxScores(self, max_scores):
		"""
		Add a Series or value representing max possible score on each question
		Assumes self.df.columns is question id's (e.g., after Reshape())

		Params:
			@max_scores: Series or single value representing max possible score on each question				
				If non-positive value then 1 is assumed
				If Series, index must match question id's in data (= columns of df after reshaping)
		"""
		#If max_scores are given as a Series its index must match the question id's
		if isinstance(max_scores, pd.Series) and self.df.columns.equals(max_scores.index):
			#Error check the max_scores Series
			max_scores.replace([np.inf, -np.inf, np.nan], 1.0, inplace=True)
			max_scores[max_scores <= 0] = 1.0
			self.max_scores = max_scores

		else: 
			try:
				if float(max_scores) <= 0:
					max_scores = 1.0
			except:
				max_scores = 1.0
			self.max_scores = float(max_scores)

	"""
	Analysis Methods
	"""

	def Normalize(self):
		try:
			self.df = an.NormalizeScores(self.df, self.max_scores)
			self.normalized = True

		except AttributeError:
			print "[Err] Can't normalize scores: no maximum scores defined. Use AddMaxScores()."

	def Grade(self):
		"""
		Compute auxiliary score data and grades
		If scores were normalized or max scores are not defined, report accuracy as average score on attempted questions
		Otherwise report accuracy as the ratio of total score to max possible score on attempted questions
		"""
		self.grades_df = an.GetAuxScoreData(self.df)

		if not self.normalized:
			try:
				self.grades_df['max score'], self.grades_df['grade'] = an.GetGradesUsingMaxScores(self.df, self.grades_df, self.max_scores)
				return
			except AttributeError:
				pass

		self.grades_df['grade'] = an.GetGradesAsAverageScore(self.df, self.grades_df['attempted'])

	"""
	Plot Methods
	"""

	def PlotHists(self, plot_total_scores=False):
		"""
		Plot histograms of number of attempted questions, accuracy, and total scores (optional)
		"""
		plot.PlotSeriesHist(self.grades_df['attempted'], "No. of Questions Attempted", "No. of Questions Attempted", "Freq.")
		plot.PlotSeriesHist(self.grades_df['grade'], "Grades", "Grade", "Freq")
		if plot_total_scores:
			plot.PlotSeriesHist(self.grades_df['total score'], "Total Scores", "Total Score", "Freq")

	def PlotScatter(self, jitter=None, alpha=None):
		"""
		Plot scatter plot of grade vs. number of questions attempted
		"""
		plot.PlotSeriesScatter(self.grades_df['attempted'], self.grades_df['grade'], "Grades vs. No. of Questions Attempted", "No. of Questions Attempted", "Grade (%)", jitter=jitter, alpha=alpha)