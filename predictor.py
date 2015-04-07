"""
###################################################################################################
#
# The Question Bank - Prediction
# 
# File: predictor.py
# Desc: interface classes for student response prediction
#
###################################################################################################
"""

import numpy as np
from sklearn.preprocessing import Imputer

class Predictor:
	"""
	Parent class for student response prediction
	Works internally with numpy array
	"""

	def __init__(self, df, give_stats=True):
		"""
		Assumes DataFrame is student id's x question id's

		Params:
			@give_stats: whether to return statistics on predictions made; depends on method used
		"""
		self.matrix = df.as_matrix()	#numpy array
		self.questions_idx = df.columns
		self.students_idx = df.index

		self.give_stats = give_stats	

	def round(self, precision=0):
		"""
		round all values in matrix to specified precision.
		0 means to nearest whole number
		negative rounds to that may digits LEFT of decimal point
		"""
		self.matrix.around(decimals=precision)

	def get_predicted_df(self):
		"""
		Returns a DataFrame built out of the internal matrix
		"""
		return pd.DataFrame(self.matrix, index=self.students_idx, columns=self.questions_idx)

	def predict(self, method='distance', metric='cosine'):
		"""
		Predicts 
		"""

