"""
###################################################################################################
#
# The Question Bank
# 
# File: plot.py
# Desc: methods for graphing data
#
###################################################################################################
"""

from matplotlib import pyplot as plt
from matplotlib import cm as cm
from scipy.stats import gaussian_kde
import numpy as np

def series_hist(ser, xlabel):
	"""
	Plot histogram of Series object and approximating Guassian KDE PDF
	"""
	fig, ax1 = plt.subplots()

	n, bins, patches = ax1.hist(ser.values, color='burlywood', histtype='bar', label=xlabel)
	xs = np.linspace(bins[0], bins[-1], 200)

	ax1.set_xlabel(xlabel)
	ax1.set_ylabel("Frequency", color='burlywood')

	density = gaussian_kde(ser.values)

	ax2 = ax1.twinx()
	ax2.plot(xs,density(xs), color='crimson', linewidth=2, label="Fit")
	ax2.set_ylabel('Gaussian KDE', color='crimson')

	plt.title(xlabel)
	plt.show()


def series_scatter(ser_x, ser_y, title, xlabel, ylabel, jitter=None, alpha=None):
	"""
	Plot scatter plot of Series objects with density coloring
	"""
	if jitter != None:
		x = ser_x + np.random.uniform(-jitter, jitter, len(ser_x))
		title += " (jitter = " + str(jitter) + ")"
	else:
		x = ser_x

	stack = np.vstack([x.values, ser_y.values])
	colors = gaussian_kde(stack)(stack)

	plt.scatter(x, ser_y, alpha=alpha, c=colors)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()