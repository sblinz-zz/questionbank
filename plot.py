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
from scipy.interpolate import UnivariateSpline
from scipy.stats import gaussian_kde
import numpy as np

def PlotSeriesHist(ser, title, xlabel, ylabel, plot_fit=True):
	plt.hist(ser.values, color='blue', histtype='bar', label=xlabel)

	if plot_fit:
		hist, bins = np.histogram(ser.values)
		#convert bin edges to centers for fitting
		bins = bins[:-1] + (bins[1] - bins[0])/2			
		f = UnivariateSpline(bins, hist)
		plt.plot(bins, f(bins), "--", color='green', linewidth=2, label="Fit")
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()
	plt.show()

def PlotSeriesScatter(ser_x, ser_y, title, xlabel, ylabel, jitter=None, alpha=None):
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