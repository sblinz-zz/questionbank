"""
###################################################################################################
#
# The Question Bank - Prediction
# 
# File: predictor.py
# Desc: interface classes for student response prediction
#
# See class notes for references
#
###################################################################################################
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from math import sqrt

class DistanceCollaborativeFilter:
	"""
	Implements ordinary distance-based collaborative filtering with multiple metrics
	Adapted from: http://guidetodatamining.com/chapter-2/

	To Do:
		Implement get_k_nearest_neighbors using a max heap
		Implement automatic determination of k based on number of students/number of missing values
	"""
	def __init__(self, df, metric, k, minkowski_param=None):
		"""
		Assumes df is given as student id's x question id's
			e.g., after question_bank.reshape() or pd.pivot_table()

		Params:
			@metric: 
				distance metric to use (as a string).
				Throws ValueError if unmatched
			@k:
				number of neighbors to get predictions from
			@minkowski_param: 
				parameter for Minkowski metric: 1 = Manhattan, 2 = Euclidean, etc. 
				Throws ValueError if not (convertable to) a positive int
		"""

		#Dictionaries of dictionaries for observed scores and predicted scores
		#{ student1_id : { question1_id : score, question2_id : score, ... }, ... }
		self.scores = self.get_dictionary_of_valid_scores(df)
		self.predictions = {}

		try:
			bad_k_msg = "Invalid number of neighbors parameter k: must be  a positive integer and less than number of students ."
			if int(k) > 0:
				self.k = int(k)
			else:
				raise ValueError(bad_k_msg)

		except (TypeError, ValueError):
			raise ValueError(bad_k_msg)

		self.metric = metric
		if metric == 'cosine':
			self.metric_fn = self.cosine_similarity
		elif metric == 'fast-pearson':
			self.metric_fn = self.fast_pearson_similarity
		elif metric == 'pearson':
			self.metric_fn = self.pearson_similarity
		elif metric == 'minkowski':
			try:
				self.minkowski_param = int(minkowski_param)
				self.metric_fn = self.minkowski_similarity
			except TypeError:
				raise ValueError("Invalid Minkowski metric parameter: must be (convertable to) a positive integer.")	

		else:
			bad_metric_msg = "Invalid metric supplied for instance of DistanceCollaborativeFilter. Supported metric arguments:"
			bad_metric_msg += "\tcosine\n"
			bad_metric_msg += "\tpearson\n"
			bad_metric_msg += "\tfast-pearson\n"
			bad_metric_msg += "\tminkowski"
			raise ValueError(bad_metric_msg)

	def get_dictionary_of_valid_scores(self, df):
		"""
		Returns dictionary of dictionaries with non-null scores as:
			{ student1_id : { question1_id : score, question2_id : score, ... }, ... }

		Take transpose because to_dict() uses columns as main keys but we assume questions are columns
		"""
		scores = df.transpose().to_dict()
		for stu in scores:
			scores[stu] = { key : value for key, value in scores[stu].iteritems() if pd.notnull(value) }
		return scores

	def fast_pearson_similarity(self, scores1, scores2):
		"""
		Approximate Pearson value for overlapping question scores
		Requires single pass through data
		"""
		intersection = list(scores1.viewkeys() & scores2.viewkeys())
		if len(intersection) != 0:
			n = len(intersection)
			xs = [scores1[question] for question in intersection]
			ys = [scores2[question] for question in intersection]

			sum_xy = sum([x*y for x, y in zip(xs, ys)])
			sum_x = sum(xs)
			sum_y = sum(ys)
			sum_x2 = sum([x**2 for x in xs])
			sum_y2 = sum([y**2 for y in ys])
		else:
			return 0
	
		denom = sqrt(sum_x2 - (sum_x**2 / n))*sqrt(sum_y2 - (sum_y**2 / n))
		if denom == 0:
			return 0
		else:
			return (sum_xy - (sum_x * sum_y) / n) / denom

	def pearson_similarity(self, scores1, scores2):
		"""
		Pearson value for overlapping question scores
		"""
		#Get overlapping questions
		intersection = list(scores1.viewkeys() & scores2.viewkeys())
		if len(intersection) != 0:
			x = [scores1[question] for question in intersection]
			y = [scores2[question] for question in intersection]
			return pearsonr(x, y)[0] #pearsonr returns pearson value and p-value
		else:
			return 0

	def minkowski_similarity(self, scores1, scores2):
		"""

		"""

	def get_k_nearest_neighbors(self, student):
		"""
		Get k of the nearest neighbors by the given metric sorted from closest
		
		To Do:
			Implement with a max heap of size k for better efficieny
		"""
		nbrs = []
		for nbr in self.scores:
			if nbr != student:
				distance = self.metric_fn(self.scores[nbr], self.scores[student])
				if pd.notnull(distance):
					nbrs.append((nbr, distance))
		nbrs.sort(key = lambda nbr_tuple : nbr_tuple[1], reverse=True)
		return nbrs[:self.k]

	def predict_student(self, student):
		"""
		Predict question scores for a given student using weighted average of scores of k nearest neighbors

		To Do:
			If totalDistance == 0 use simple average of all contributing neighbor scores, not max
		"""
		predictions = {}
		k_nbrs = self.get_k_nearest_neighbors(student)
		stu_scores = self.scores[student]
		totalDistance = 0.0

		for i in range(self.k):
			totalDistance += k_nbrs[i][1]
			
		for i in range(self.k):
			if totalDistance != 0:
				weight = k_nbrs[i][1] / totalDistance
			else:
				#set dummy value and keep best prediction found
				weight = -1 
			
			nbr = k_nbrs[i][0]
			nbr_scores = self.scores[nbr]

			#Iterate over neighbors scores which student has not answered
			#Predict student answer using weighted or average of neighbors scores on that question
			for question in nbr_scores:
				if question not in stu_scores:
					if question not in predictions:
						if weight != -1:
							predictions[question] = nbr_scores[question]*weight
						else:
							predictions[question] = nbr_scores[question]
					else:
						if weight != -1:
							predictions[question] = predictions[question] + nbr_scores[question]*weight
						else:
							#If totalDistance was 0 keep the best score predicted by these "exact match" neighbors
							#Should take average over all contributing "exact match" neighbors
							predictions[question] = max(predictions[question], nbr_scores[question])

		return predictions

	def predict(self):
		"""
		Return predicted question scores for all students using k nearest neighbors
		"""
		for student in scores:
			self.predictions[student] = predict_student(student)
		return self.predictions
