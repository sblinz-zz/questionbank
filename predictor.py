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
from math import sqrt
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
		self.num_students = len(self.matrix)
		self.num_questions = len(self.matrix[0])

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

	To Do:
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
				parameter for Minkowski metric. 1 = Manhattan, 2 = Euclidean. 
				Thows ValueError if not positive int
		"""
		Predictor.__init__(self, df)
		self.k = k
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
		distances = []
        for neighbor_idx in range(self.num_students):
            if neighbor_idx != student_idx:
                distance = self.metric_fn(self.matrix[student_idx], self.matrix[neighbor_idx])
                distances.append((neighbor_idx, distance))
        # sort based on distance -- closest first
        distances.sort(key = lambda student_tuple: student_tuple[1], reverse=True)
        return distances

	def cosine_similarity(self, scores1, scores2):
		pass

	def pearson_similarity(self, scores1, scores2):
		sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        n = 0
        for question_idx in range(self.num_questions):
        	x = scores1[question_idx]
        	y = scores2[question_idx]
        	if x != np.nan and y != np.nan:
                n += 1
                sum_xy += x * y
                sum_x += x
                sum_y += y
                sum_x2 += pow(x, 2)
                sum_y2 += pow(y, 2)
        if n == 0:
            return 0
        # now compute denominator
        denominator = (sqrt(sum_x2 - pow(sum_x, 2) / n)
                       * sqrt(sum_y2 - pow(sum_y, 2) / n))
        if denominator == 0:
            return 0
        else:
            return (sum_xy - (sum_x * sum_y) / n) / denominator

	def minkowski_similarity(self, scores1, scores2):
		pass

	def predict(self, student):
		"""Give list of recommendations"""
       recommendations = {}
       # first get list of users ordered by nearness
       nearest = self.computeNearestNeighbor(user)
       #
       # now get the ratings for the user
       #
       userRatings = self.matrix[student]
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
          neighborRatings = self.matrix[name]
          # get the name of the person
          # now find bands neighbor rated that user didn't
          for artist in neighborRatings:
             if not artist in userRatings:
                if artist not in recommendations:
                   recommendations[artist] = (neighborRatings[artist]
                                              * weight)
                else:
                   recommendations[artist] = (recommendations[artist]
                                              + neighborRatings[artist]
                                              * weight)
       # now make list from dictionary
       recommendations = list(recommendations.items())
       recommendations = [(self.convertProductID2name(k), v)
                          for (k, v) in recommendations]
       # finally sort and return
       recommendations.sort(key=lambda artistTuple: artistTuple[1],
                            reverse = True)
       # Return the first n items
       num_recs = max(self.num_questions, len(recommendations))
       return recommendations[:num_recs]

class LatentCollaborativeFilter(Predictor): 
	pass

