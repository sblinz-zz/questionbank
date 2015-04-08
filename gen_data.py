import numpy as np
import pandas as pd
import random
from scipy.stats import gumbel_l
from scipy.stats import gumbel_r
from scipy.stats import norm

def generate_data(num_students, num_questions, num_questions_attempted):
	f = open('gen_data.csv', 'w')
	f.write('question_id,question_rating,student_id,score\n')
	
	q_ids = [x for x in range(num_questions)]
	s_ids = [x for x in range(num_students)]

	student_attempted_q = {}

	for s in s_ids:
		jitter = np.random.randint(-5, 6)
		student_attempted_q[s] = random.sample(range(num_questions), num_questions_attempted+jitter)

	for q in q_ids:
		#randomly set the question difficult as the mean of the normally distributed scores
		q_rating = np.random.randint(1, 4)
		#hard question
		if q_rating == 3:
			mean = 0.60
			std = 0.1
			score_fn = gumbel_r
		if q_rating == 2:
			mean = 0.75
			std = 0.1
			score_fn = norm
		if q_rating == 1:
			mean = 0.85
			std = 0.1
			score_fn = gumbel_l

		for s in s_ids:
			if q in student_attempted_q[s]:
				score = score_fn.rvs(loc=mean, scale=std, size=1)
				if score < 0:
					score = 0
				elif score > 1:
					score = 1
				np.around(score, decimals=2)
				f.write(str(q) + ',' + str(q_rating) + ',' + str(s) + ',' + '%.2f\n' % score)
	f.close()