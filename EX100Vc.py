from __future__ import division
import time
import numpy as np
import numpy.random as npr
import blspricer as bp
import customML as cm
import scipy.stats as scs
from doe_lhs import lhs
from scipy.stats.distributions import norm

class EX100V(object):
	def __init__(self):
		# --- Parameters ---
		self.D = 100    # num. of underlyings
		self.S0 = 100.; self.mu = 0.08
		self.rfr = 0.05; self.K = 100.
		self.tau = 0.04; self.T = 0.1
		self.c = 876.8636; self.perc = 0.99    # 99th percentile of the portfolio loss distribution

		# Volatility vector
		self.sig_vec = np.zeros(self.D)
		self.sig_vec[0:30] = 0.5
		self.sig_vec[30:70] = 0.3
		self.sig_vec[70:100] = 0.1

		# Correlation matrix
		corr_mat = np.zeros((self.D, self.D))
		for i in range(10):
		    corr_mat[i*10:i*10+10,i*10:i*10+10] = 0.2
		for j in range(self.D):
		    for i in range(self.D):
		        if i == j:
		            corr_mat[i, j] = 1.0
		self.cho_mat = np.linalg.cholesky(corr_mat)

		self.I_pred = 2**15

		# --- portfolio price @ t = 0 ---
		opt_data = map(bp.blsprice2,[self.S0]*self.D,[self.K]*self.D,\
						[self.rfr]*self.D,[self.T]*self.D,self.sig_vec)
		self.Value0 = -10. * np.sum(opt_data)

	def ns(self,kk,beta=1.):
		''' Standard Nested Simulations
		'''
		pass

	def regr_data_prep(self,kk,N_i=1):
		''' Regression data preparation via nested simulations
		'''
		pass

	def poly_regr(self,deg=2):
		''' Polynomial Regression
		'''
		pass

	def svr(self,C=1000.,gamma=1e-2):
		''' Support Vector Regression
		'''
		pass

def re_poly(kk,N_i,deg=2):
	pass

def re_svr(kk,N_i):
	pass

def conv(K,N_i=1,L=1000,regr_method=re_svr,filename='EX100Vc'):
	pass

if __name__ == "__main__":
	import EX100Vc as EX100
	K = [ii**5 for ii in range (4,23)]
	EX100.conv(K, N_i=1, L=1000, regr_method=re_svr, filename='EX100Vc')