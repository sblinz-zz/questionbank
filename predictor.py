"""
###################################################################################################
#
# The Question Bank - Prediction
# 
# File: predictor.py
# Desc: interface classes for student response prediction
#
# See derived class notes for references
#
###################################################################################################
"""

import numpy as np
from sklearn.preprocessing import Imputer

class Predictor:
	"""
	Base class for student response prediction
	Works internally with numpy array representing matrix of student id's x question id's
	All indices based on internal array rows and columns
	Should not be instantiated

	To Do: implement as abstract class using ABCMeta in abc module
	"""

	def __init__(self, df):
		"""
		Assumes DataFrame is student id's x question id's
		"""
		self.matrix = df.as_matrix()	#numpy array
		self.questions_idx = df.columns
		self.students_idx = df.index

		#dictionary of dictionaries to store new predictions
		#{ user1 : { pred_question1 : pred_score, pred_question2 : pred_score, ...}, user2 : { ... }, ... }
		self.pred_dict = {}				

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
		Returns dictionary of new predictions made
		{ user1 : { pred_question1 : pred_score, pred_question2 : pred_score, ...}, user2 : { ... }, ... }
		"""
		return self.pred_dict

class DistanceCollaborativeFilter(Predictor):
	"""
	Implements ordinary distance-based collaborative filtering with multiple metrics
	Adapted from: http://guidetodatamining.com/chapter-2/
	"""

	def __init__(self, df, metric, minkowski_param=None):
		"""

		"""
		Predictor.__init__(self, df)
		self.metric = metric
		if metric == 'cosine':
			self.metric_fn = self.cosine_similarity
		elif metric == 'pearson':
			self.metric_fn = self.pearson_similarity
		elif metric == 'minkowski':
			try:
				self.minkowski_param = int(minkowski_param)
				self.metric_fn = self.minkowski_similarity
			except TypeError:
				raise ValueError("Invalid Minkowski metric parameter. Must be a positive integer.")		
		else:
			bad_metric_txt = "Invalid metric supplied for instance of DistanceCollaborativeFilter.\n"
			bad_metric_txt += "Support metric arguments:\n"
			bad_metric_txt += "\tcosine\n"
			bad_metric_txt += "\tpearson\n"
			bad_metric_txt += "\tminkowski"
			raise ValueError(bad_metric_txt)

	def get_k_nearest_neighbors(self, student_idx):
		pass

	def cosine_similarity(student1_idx, student2_idx):
		pass

	def pearson_similarity(student1_idx, student2_idx):
		pass

	def minkowski_similarity(student1_idx, student2_idx):
		pass

	def predict(self):
		pass

class LatentCollaborativeFilter(Predictor): 

