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
from math import sqrt

class DistanceCollaborativeFilter():
	"""
	Implements ordinary distance-based collaborative filtering with multiple metrics
	Adapted from: http://guidetodatamining.com/chapter-2/

	To Do:
		Implement get_k_nearest_neighbors using a max heap
		Implement automatic determination of k based in number of students
	"""
	def __init__(self, df, metric, k, minkowski_param=None):
		"""
		Params:
			@metric: 
				distance metric to use (as a string).
				Throws ValueError if unmatched
			@k:
				number of neighbors to get predictions from
			@minkowski_param: 
				parameter for Minkowski metric: 1 = Manhattan, 2 = Euclidean, etc. 
				Thows ValueError if not positive int
		"""
		self.scores = self.get_dictionary_of_valid_scores(df)
		self.k = k
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
				raise ValueError("Invalid Minkowski metric parameter. Must be (convertable to) a positive integer.")	

		else:
			bad_metric_msg = "Invalid metric supplied for instance of DistanceCollaborativeFilter.\n"
			bad_metric_msg += "Supported metric arguments:\n"
			bad_metric_msg += "\tcosine\n"
			bad_metric_msg += "\tpearson\n"
			bad_metric_msg += "\tfast-pearson\n"
			bad_metric_msg += "\tminkowski"
			raise ValueError(bad_metric_msg)

	def get_dictionary_of_valid_scores(self, df):
		scores = df.transpose().to_dict()
		for stu in scores:
			scores[stu] = { key : value for key, value in scores[stu].iteritems() if pd.notnull(value) }
		return scores

	def fast_pearson_similarity(self, rating1, rating2):
		sum_xy = 0
		sum_x = 0
		sum_y = 0
		sum_x2 = 0
		sum_y2 = 0
		n = 0
		for key in rating1:
			if key in rating2:
				n += 1
				x = rating1[key]
				y = rating2[key]
				sum_xy += x * y
				sum_x += x
				sum_y += y
				sum_x2 += pow(x, 2)
				sum_y2 += pow(y, 2)
		if n == 0:
			return 0
		# now compute denominator
		denominator = (sqrt(sum_x2 - pow(sum_x, 2) / n)*sqrt(sum_y2 - pow(sum_y, 2) / n))
		if denominator == 0:
			return 0
		else:
			return (sum_xy - (sum_x * sum_y) / n) / denominator

	def get_k_nearest_neighbors(self, student):
		"""
		Get k of the nearest neighbors by the given metric sorted from closest
		Implement with a max heap of size k for better efficieny
		"""
		nbrs = []
		for nbr in self.scores:
			if nbr != student:
				distance = self.metric_fn(self.scores[nbr],self.scores[student])
				nbrs.append((nbr, distance))
		nbrs.sort(key = lambda nbr_tuple : nbr_tuple[1], reverse=True)

		#return up to k in case k exceeds number of students-1
		return nbrs[:max(self.k, len(nbrs))]

	def predict(self, user):
		recommendations = {}
		# first get list of users  ordered by nearness
		nearest = self.get_k_nearest_neighbors(user)
		#
		# now get the ratings for the user
		#
		userRatings = self.scores[user]
		#
		# determine the total distance
		totalDistance = 0.0
		for i in range(self.k):
			totalDistance += nearest[i][1]
		# now iterate through the k nearest neighbors
		# accumulating their ratings
		for i in range(self.k):
			# compute slice of pie 
			weight = nearest[i][1] / totalDistance
			# get the name of the person
			name = nearest[i][0]
			# get the ratings for this person
			neighborRatings = self.scores[name]
			# get the name of the person
			# now find bands neighbor rated that user didn't
			for artist in neighborRatings:
				if not artist in userRatings:
					if artist not in recommendations:
						recommendations[artist] = (neighborRatings[artist]* weight)
					else:
						recommendations[artist] = (recommendations[artist]+ neighborRatings[artist]* weight)
		# now make list from dictionary
		recommendations = list(recommendations.items())
		#recommendations = [(self.convertProductID2name(k), v) for (k, v) in recommendations]
		# finally sort and return
		recommendations.sort(key=lambda artistTuple: artistTuple[1],reverse = True)
		# Return the first n items
		#return recommendations[:self.n]
		return recommendations