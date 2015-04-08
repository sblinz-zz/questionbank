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
from heapq import *

class DistanceCollaborativeFilter:
	"""
	Implements ordinary distance-based collaborative filtering with multiple metrics
	Adapted from: http://guidetodatamining.com/chapter-2/

	To Do:
		Implement get_k_nearest_neighbors using a max heap
		Implement automatic determination of k based on number of students/number of missing values
		Implement Spearman rank correlation metric
	"""

	"""
	Setup
	"""
	def __init__(self, df):
		"""
		Assumes df is given as student id's x question id's e.g., after question_bank.reshape() or pd.pivot_table()
		"""
		#Dictionaries of dictionaries for observed scores and predicted scores
		#{ student1_id : { question1_id : score, question2_id : score, ... }, ... }
		self.scores = self.get_dictionary_of_valid_scores(df)
		self.predictions = {}

	def check_and_get_k_value(self, k):
		try:
			bad_k_msg = "Invalid number of neighbors parameter k: "
			bad_k_msg += "must be a positive integer and (should be a lot) less than number of students."
			if int(k) > 0:
				return int(k)
			else:
				raise ValueError(bad_k_msg)

		except (TypeError, ValueError):
			raise ValueError(bad_k_msg)

	def check_and_get_metric(self, metric, r=None):
		if metric == 'cosine':
			metric_fn = self.cosine_distance
		elif metric == 'fast-pearson':
			metric_fn = self.fast_pearson_distance
		elif metric == 'pearson':
			metric_fn = self.pearson_distance
		elif metric == 'minkowski':
			try:
				if int(r) > 0:
					r = int(r)
				else:
					bad_r_msg = "Invalid Minkowski metric parameter r: must be (convertable to) a positive integer."
					raise ValueError
				metric_fn = self.minkowski_distance
			except (TypeError, ValueError):
				raise ValueError(bad_r_msg)	

		else:
			bad_metric_msg = "Invalid metric supplied for instance of DistanceCollaborativeFilter. Supported metric arguments:"
			bad_metric_msg += "\tcosine\n"
			bad_metric_msg += "\tpearson\n"
			bad_metric_msg += "\tfast-pearson\n"
			bad_metric_msg += "\tminkowski"
			raise ValueError(bad_metric_msg)

		return metric_fn, r

	def get_dictionary_of_valid_scores(self, df):
		"""
		Returns dictionary of dictionaries with non-null scores as:
			{ student1_id : { question1_id : score, question2_id : score, ... }, ... }

		Take transpose because to_dict() uses columns as main keys but we assume questions are columns
		"""	
		scores = df.transpose().to_dict()
		for student in scores:
			scores[student] = { key : value for key, value in scores[student].iteritems() if pd.notnull(value) }
		return scores

	"""
	Distance methods
	"""

	def fast_pearson_distance(self, scores1, scores2):
		"""
		Approximate Pearson r value for overlapping question scores
		Requires single pass through data

		Params: both are dictionaries of scores of the form: { question1_id : score1, question2_id : score2, ... }
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

	def pearson_distance(self, scores1, scores2):
		"""
		Pearson r value for overlapping question scores

		Params: both are dictionaries of scores of the form: { question1_id : score1, question2_id : score2, ... }
		"""
		#Get overlapping questions
		intersection = list(scores1.viewkeys() & scores2.viewkeys())
		if len(intersection) != 0:
			x = [scores1[question] for question in intersection]
			y = [scores2[question] for question in intersection]
			return pearsonr(x, y)[0] #pearsonr returns pearson value and p-value
		else:
			return 0

	def minkowski_distance(self, scores1, scores2):
		"""
		Minkowski distance for overlapping question scores
		Params: both are dictionaries of scores of the form: { question1_id : score1, question2_id : score2, ... }

		Note: r = 1 => Manhattan distance, r = 2 => Euclidean distance
		"""
		intersection = list(scores1.viewkeys() & scores2.viewkeys())
		if len(intersection) != 0:
			return sum([abs(scores1[question] - scores2[question])**self.r for question in intersection])
		else:
			return 0

	def cosine_distance(self, scores1, scores2):
		"""
		Cosine of angle between overlapping question score vectors

		Params: both are dictionaries of scores of the form: { question1_id : score1, question2_id : score2, ... }
		"""
		dot_prod = lambda v, w: sum(x*y for x,y in zip(v, w))
		norm = lambda v : sqrt(dot_prod(v, v))

		intersection = list(scores1.viewkeys() & scores2.viewkeys())
		if len(intersection) != 0:
			xs = [scores1[question] for question in intersection]
			ys = [scores2[question] for question in intersection]

			norm_x = norm(xs)
			norm_y = norm(ys)
			if norm_x != 0 and norm_y != 0:
				return dot_prod(xs,ys)/(norm_x*norm_y)
		return 0

	"""
	Prediction methods
	"""

	def get_k_nearest_neighbors(self, student, dbg=False):
		"""
		Get k of the nearest neighbors by the given metric sorted from closest
		
		To Do:
			Implement with max heap for Minkowski metric
		"""
		#For metrics where larger measures mean higher similarity use a min heap of self.k elements
		#For now, only Minkowski metric gives smaller measures for higher similarity 
		if dbg:
			use_min_heap = False
		else:
			use_min_heap = (self.metric_fn != self.minkowski_distance)

		nbrs = []
		for nbr in self.scores:
			if nbr != student:
				distance = self.metric_fn(self.scores[nbr], self.scores[student])
				if pd.notnull(distance):
					if use_min_heap:
						if len(nbrs) < self.k:
							heappush(nbrs, (distance, nbr))
						elif distance > nbrs[0][0]:
							heappushpop(nbrs, (distance, nbr))
					else:
						nbrs.append((distance, nbr))

		if not use_min_heap:
			nbrs.sort()

		return nbrs[:self.k]

	def predict_student(self, student, metric, k, r=None):
		"""
		Predict question scores for a given student using weighted average of scores of k nearest neighbors

		Params:
			@student: student identifier from df

			@metric: metric to use in computing k nearest neighbors
			
			@k: number of nearest neighbors to use for predictions

			@r: 
				parameter for Minkowski metric: 1 = Manhattan, 2 = Euclidean 
				Throws ValueError if not (convertable to) a positive int

		To Do:
			If totalDistance == 0 use simple average of all contributing neighbor scores, not max
		"""
		self.k = self.check_and_get_k_value(k)
		self.metric_fn, self.r = self.check_and_get_metric(metric, r)

		predictions = {}
		k_nbrs = self.get_k_nearest_neighbors(student)
		stu_scores = self.scores[student]
		totalDistance = 0.0

		for i in range(self.k):
			totalDistance += k_nbrs[i][0]

		for i in range(self.k):
			if totalDistance != 0:
				weight = k_nbrs[i][0] / totalDistance
			else:
				#set dummy value and keep best prediction found
				weight = -1 
			
			nbr = k_nbrs[i][1]
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

	def predict(self, metric, k, r=None):
		"""
		Return predicted question scores for all students using k nearest neighbors

		Params:
			@metric: metric to use in computing k nearest neighbors

			@k: number of nearest neighbors to use for predictions

			@r: 
				parameter for Minkowski metric: 1 = Manhattan, 2 = Euclidean 
				Throws ValueError if not (convertable to) a positive int
		"""
		self.k = self.check_and_get_k_value(k)
		self.metric_fn, self.r = self.check_and_get_metric(metric, r)

		for student in self.scores:
			self.predictions[student] = self.predict_student(student, metric, k, r)
		return self.predictions

	"""
	Cleanup methods
	"""

	def round(self, decimals=0, inplace=True):
		"""
		Round all the values in the predictions dictionary to the specified decimals
		"""
		if inplace:
			preds = self.predictions
		else:
			preds = {}

		for student in self.predictions:
			preds[student] = { key : np.around(value, decimals=decimals) for key, value in self.predictions[student].iteritems() }

		return preds
