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

	def __init__(self, df):
		"""
		Assumes DataFrame is student id's x question id's
		"""
		self.matrix = df.as_matrix()	#numpy array
		self.questions_idx = df.columns
		self.students_idx = df.index
		self.pred_dict = {}				#dictionary of dictionaries to store predictions

	def round(self, precision=0):
		"""
		round all values in matrix to specified precision.
		0 means to nearest whole number
		negative rounds to that may digits LEFT of decimal point
		"""
		self.matrix.around(decimals=precision)

	def get_augmented_df(self):
		"""
		Returns initial DataFrame with predictions filled in
		"""
		return pd.DataFrame(self.matrix, index=self.students_idx, columns=self.questions_idx)

	def get_predicted_dict(self):
		"""
		Returns dictionary of predictions made
		{ user1 : { pred_question1 : pred_score, pred_question2 : pred_score, ...}, user2 : { ... }, ... }
		"""
		return self.pred_dict

