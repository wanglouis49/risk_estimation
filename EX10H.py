from __future__ import division
import time
import numpy as np
import numpy.random as npr
import blspricer as bp
import customML as cm
import scipy.stats as scs
from scipy.stats.distributions import norm
from doe_lhs import lhs

class EX10(object):
	def __init__(self):
		# --- Parameters ---
		self.D = 10
		self.S0 = 100.; self.mu = 0.08; self.sigma = 0.3
		self.rfr = 0.05; self.T = 0.1; self.tau = 0.04
		self.K = 100.; self.H = 95.
		self.c = 306.8763; self.perc = 0.99

		self.I_pred = 2.**15
		self.dS = 1e-6

		# --- portfolio price @ t = 0 ---
		opt_val = lambda S: 5.*bp.blsprice(S,self.K,self.rfr,self.T,self.sigma,'put') \
							+ 10.*bp.rear_end_dnOutCall(S,self.K,self.rfr,self.T,self.sigma,self.H,self.tau,0.)
		self.Value0 = 0.
		self.delta = np.zeros(self.D)
		for d in range(self.D):
			self.delta[d] = (opt_val(self.S0+self.dS)-opt_val(self.S0-self.dS))/self.dS
			self.Value0 += -opt_val(self.S0) + self.S0 * self.delta[d]
		#print opt_val(self.S0)
		print self.delta

	def analy(self,N_o):
		''' Analytical solution
		'''
		# --- portfolio loss distribution @ t = \tau via analytical formulas ---
		opt_val = lambda S: 5.*bp.blsprice(S,self.K,self.rfr,self.T-self.tau,self.sigma,'put') \
							+ 10.*bp.dnOutCall(S,self.K,self.rfr,self.T-self.tau,self.sigma,self.H)

		#ran1 = npr.multivariate_normal(np.zeros(self.D),np.eye(self.D),N_o)
		#ran1 = npr.standard_normal((N_o,self.D))
		ran1 = norm(loc=0,scale=1).ppf(lhs(self.D,samples=N_o))
		S1 = np.zeros((N_o,self.D))
		S1[:,:] = self.S0
		S1[:,:] = S1[:,:] * np.exp((self.mu - 0.5 * self.sigma**2) * self.tau\
				 + self.sigma * np.sqrt(self.tau) * ran1[:,:])

		ValueTau = np.zeros(N_o)
		for n in range(N_o):
			for d in range(self.D):
				ValueTau[n] += -opt_val(S1[n,d]) + S1[n,d] * self.delta[d]

		L_analy = np.sort(self.Value0 - ValueTau)
		#L_analy = np.sort(-ValueTau + self.Value0 * np.exp(self.rfr*self.tau))
		print L_analy
		var = scs.scoreatpercentile(L_analy, self.perc*100.)
		eel = np.mean(np.maximum(L_analy-var,0))
		return (var, eel)