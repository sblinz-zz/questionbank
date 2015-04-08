"""
###################################################################################################
#
# The Question Bank
#	- A machine learning library for analyzing student test question data
#		- Analyze
#		- Recommend Effective Questions
#		- Predict Student Responses
# 
# By Sam Blinstein, March-April 2015
# 
# File: question_bank.py
# Desc: main class representing a session with a given dataset
#
###################################################################################################
"""
import numpy as np
import pandas as pd
import analyze as an
import plot
import predictor as pred

class QuestionBank:
	"""
	Main wrapper for session
	"""

	"""
	Setup
	"""

	def __init__(self, type='csv', filepath=None, df=None, normalized=False):
		if type == 'csv':
			self.df = pd.DataFrame.from_csv(filepath)
		elif type =='df':
			self.df = df

		self.normalized = normalized

	def reshape(self, students, questions, scores, q_ratings=None):
		"""
		Pivot the DF to assign student id's as index, question id's as features, and scores as values
		If max scores are given generate a separate Series for them indexed by question id's
		Places NaN if no entry for a given (student, question) pair

		Params:
			@students, @questions, @scores: Series names in self.df containing the respective data
			@q_ratings: Series name in self.df containing the question difficulty rating
			@max_scores: Series name in self.df containing the max possible score on the given question
		"""
		self.df.reset_index(inplace=True)
		if q_ratings != None:
			self.q_ratings = pd.DataFrame(self.df.pivot_table(index=questions, values=q_ratings))
			self.q_ratings.columns= ['difficulty rating']

		self.df = self.df.pivot_table(index=students, columns=questions, values=scores)

	def add_max_scores(self, max_scores):
		"""
		Add a Series or value representing max possible score on each question
		Assumes self.df.columns is question id's (e.g., after reshape())

		Params:
			@max_scores: Series or single value representing max possible score on each question				
				If non-positive value then 1 is assumed
				If Series, index must match question id's in data (= columns of df after reshaping)
		"""
		#If max_scores are given as a Series its index must match the question id's
		if isinstance(max_scores, pd.Series) and self.df.columns.equals(max_scores.index):
			#Clean the max_scores Series
			max_scores.replace([np.inf, -np.inf, np.nan], 1.0, inplace=True)
			max_scores[max_scores <= 0] = 1.0
			self.max_scores = max_scores

		else: 
			try:
				if float(max_scores) <= 0:
					max_scores = 1.0

			except TypeError:
				max_scores = 1.0
			self.max_scores = float(max_scores)

	def make_question_diff_ser(self, questions, diff):
		"""
		Add a Series with question id's as index and difficult rating as values

		Params:
			@diff: Series name in self.df containing question difficulty
		"""

	"""
	Analysis Methods
	"""

	def normalize(self):
		try:
			self.df = an.normalize_scores(self.df, self.max_scores)
			self.normalized = True

		except AttributeError:
			print "[Err] Can't normalize scores: no maximum scores defined. Use AddMaxScores()."

	def grade(self, include_total_score=False):
		"""
		Compute auxiliary score data and grades
		If scores were normalized or max scores are not defined, report accuracy as average score on attempted questions
		Otherwise report accuracy as the ratio of total score to max possible score on attempted questions
		"""
		self.grades_df = an.get_aux_score_data(self.df, include_total_score=include_total_score)

		if not self.normalized:
			try:
				self.grades_df['max score'], self.grades_df['grade'] = an.get_grades_using_max_scores(self.df, self.grades_df, self.max_scores)
				return

			except AttributeError:
				#if max scores not defined, compuate grade using average score as in normalized case
				pass

		self.grades_df['grade'] = an.get_grades_as_average_score(self.df, self.grades_df['attempted'])

	"""
	Plot Methods
	"""

	def plot_hists(self, plot_num_attempted=True, plot_total_scores=False):
		"""
		Histograms of 
			grades
			number of attempted questions (optional)
			total scores (optional)
		"""
		plot.series_hist(self.grades_df['grade'], "Grade")
		if plot_num_attempted:
			plot.series_hist(self.grades_df['attempted'], "No. of Questions Attempted")
		if plot_total_scores:
			plot.series_hist(self.grades_df['total score'], "Total Scores", "Total Score")

	def plot_scatters(self, jitter=None, alpha=None):
		"""
		Density-colored scatter plot of 
			grade vs. number of questions attempted
		"""
		plot.series_scatter(self.grades_df['attempted'], self.grades_df['grade'], "Grades vs. No. of Questions Attempted", "No. of Questions Attempted", "Grade (%)", jitter=jitter, alpha=alpha)

	"""
	Prediction Methods
	"""
	def load_DCF(self):
		"""
		Instantiate an instance of the DistanceCollaborativeFilter and load our data
		"""
		self.dcf= pred.DistanceCollaborativeFilter(self.df)

	def get_DCF_prediction_df(self, metric, k, r=None):
		"""
		Get a copy of the internal DataFrame with missing values filled in using distance collaborative filtering

		Params:
			@metric: metric to use in computing k nearest neighbors
			@k: number of nearest neighbors to use for predictions
			@r: parameter for Minkowski metric: 1 = Manhattan, 2 = Euclidean
		"""
		predictions =  self.dcf.predict(metric, k, r)
		pred_df = self.df.copy(deep=True)
		#Add predicted values to pred_df
		for student in predictions:
			for question in predictions[student]:
				pred_df.ix[student,question] = predictions[student][question]
		return pred_df

	def get_DCF_prediction_student(self, student, metric, k, r=None):
		"""
		Get predictions for an individual student using distance collaborative filtering

		Params:
			@metric: metric to use in computing k nearest neighbors
			@k: number of nearest neighbors to use for predictions
			@r: parameter for Minkowski metric: 1 = Manhattan, 2 = Euclidean
		"""
		return self.dcf.predict_student(student, metric, k)

	"""
	Selection Methods
	"""
	def get_questions_by_variance_ceiling(self, var):
		"""
		Return a list of question id's whose variance is below the argument
		"""
		var_ser = self.df.var(axis=0) #make a series of the variance of each question (column)
		return var_ser.loc[var_ser < var].index

