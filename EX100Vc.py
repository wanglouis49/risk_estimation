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
		self.corr_mat = np.zeros((self.D, self.D))
		for i in range(10):
		    self.corr_mat[i*10:i*10+10,i*10:i*10+10] = 0.2
		for j in range(self.D):
		    for i in range(self.D):
		        if i == j:
		            self.corr_mat[i, j] = 1.0
		self.cho_mat = np.linalg.cholesky(self.corr_mat)

		self.groups = 10
		self.I_pred = 2**15

		# --- portfolio price @ t = 0 ---
		opt_data = map(bp.blsprice2,[self.S0]*self.D,[self.K]*self.D,\
						[self.rfr]*self.D,[self.T]*self.D,self.sig_vec)
		self.Value0 = -10. * np.sum(opt_data)

	def ns(self,kk,beta=1.):
		''' Standard Nested Simulations
		'''
		# --- Computation budget allocatoin ---
		N_o = np.int(np.ceil(kk**(2./3.)*beta))
		print "Outer loop: %d" % N_o
		N_i = np.int(np.ceil(kk**(1./3.)/beta))
		print "Inner loop: %d" % N_i
		print "Total: %d" % (N_o*N_i)


		# --- Analytical solution from 10**7 samples ---
		VaR_true = 8.768636132637976e2
		EEL_true = 1.818317262496906

		# --- portfolio loss distribution @ t = \tau via analytical formulas---
		t0 = time.time()
		ran1 = npr.multivariate_normal(np.zeros(self.D),self.corr_mat,N_o)
		S1 = np.zeros((N_o,self.D))
		S1[:,:] = self.S0
		S1[:,:] = S1[:,:] * np.exp((self.mu - 0.5 * np.dot(np.ones((N_o,1)),self.sig_vec[np.newaxis,:])**2) * self.tau \
			+ np.dot(np.ones((N_o,1)),self.sig_vec[np.newaxis,:]) * np.sqrt(self.tau) * ran1[:,:])

		ValueTau = np.zeros((N_o,1))
		for dim in range(self.D):
			ran2 = npr.standard_normal((N_o,N_i))
			S2 = np.zeros((N_o,N_i))
			S2[:,:] = np.dot(S1[:,dim,np.newaxis],np.ones((1,N_i))) * np.exp((self.rfr - 0.5*self.sig_vec[dim]**2)\
				*(self.T-self.tau) + self.sig_vec[dim] * np.sqrt(self.T-self.tau) * ran2[:,:])

			V1 = np.dot((np.maximum(self.K-S2[:,:],0)),np.ones((N_i,1))) / float(N_i) * np.exp(-self.rfr*(self.T-self.tau))
			V2 = np.dot((np.maximum(S2[:,:]-self.K,0)),np.ones((N_i,1))) / float(N_i) * np.exp(-self.rfr*(self.T-self.tau))

			ValueTau[:] += -10. * (V1+V2)
		y = self.Value0 - ValueTau
		t_ns = time.time()-t0
		print "%.1fs spent in re-valuation" % t_ns

		L_ns = np.sort(y)
		var = scs.scoreatpercentile(L_ns, self.perc*100.)
		el = L_ns - self.c
		eel = np.mean(el.clip(0))
		print "VaR estimated: %e (true: %e)" % (var, VaR_true)
		print "EEL estimated: %e (true: %e)" % (eel, EEL_true)

		return eel

	def regr_data_prep(self,kk,N_i=1):
		''' Regression data preparation via nested simulations
		'''
		import scipy.io

		# --- Computation budget allocatoin ---
		N_o = int(kk/N_i)

		# --- portfolio price @ t = \tau via regression ---
		t0 = time.time()
		ValueTau = np.zeros((N_o,self.groups))
		ran1 = npr.multivariate_normal(np.zeros(self.D),self.corr_mat,N_o)
		S1 = np.zeros((N_o,self.D))
		S1[:,:] = self.S0
		S1[:,:] = S1[:,:] * np.exp((self.mu - 0.5 * np.dot(np.ones((N_o,1)),self.sig_vec[np.newaxis,:])**2) * self.tau \
			+ np.dot(np.ones((N_o,1)),self.sig_vec[np.newaxis,:]) * np.sqrt(self.tau) * ran1[:,:])
		for group in range(self.groups):
			for dim in range(group*10,(group+1)*10):
				ran2 = npr.standard_normal((N_o,N_i))
				S2 = np.zeros((N_o,N_i))
				S2[:,:] = np.dot(S1[:,dim,np.newaxis],np.ones((1,N_i))) * np.exp((self.rfr - 0.5*self.sig_vec[dim]**2)*(self.T-self.tau) \
				            + self.sig_vec[dim] * np.sqrt(self.T-self.tau) * ran2[:,:])
				V1 = np.dot((np.maximum(self.K-S2[:,:],0)),np.ones((N_i,1))) / float(N_i) * np.exp(-self.rfr*(self.T-self.tau))
				V2 = np.dot((np.maximum(S2[:,:]-self.K,0)),np.ones((N_i,1))) / float(N_i) * np.exp(-self.rfr*(self.T-self.tau))
				ValueTau[:,group,np.newaxis] += -10. * (V1+V2)
		t_ns = time.time()-t0

		#print "%es spent in re-valuation" % t_ns

		# prediction samples
		mat = scipy.io.loadmat('sobol_100_32768.mat')
		ran3_uniform = mat['sobol_100_32768']
		ran3 = scs.norm.ppf(ran3_uniform)
		ran3 = np.dot(self.cho_mat, ran3.T).T
		S_pred = np.zeros((self.I_pred,self.D))
		S_pred[:,:] = self.S0
		S_pred[:,:] = S_pred[:,:] * np.exp((self.mu - 0.5 * np.dot(np.ones((self.I_pred,1)),self.sig_vec[np.newaxis,:])**2) * self.tau \
			+ np.dot(np.ones((self.I_pred,1)),self.sig_vec[np.newaxis,:]) * np.sqrt(self.tau) * ran3[:,:])

		self.X_all = S1
		self.X_pred_all = S_pred
		self.y_all = ValueTau

	def poly_regr(self,deg=2):
		''' Polynomial Regression
		'''
		y_lr = np.zeros((self.I_pred,self.groups))
		for group in range(self.groups):
			X = self.X_all[:,group*10:(group+1)*10]
			y = self.y_all[:,group]
			X_pred = self.X_pred_all[:,group*10:(group+1)*10]

			# Training
			phi = cm.naivePolyFeature(X,deg=deg,norm=True)
			#rank = np.linalg.matrix_rank(phi)
			U, s, V = np.linalg.svd(phi,full_matrices=False)
			r = np.dot(V.T,np.dot(U.T,y[:,np.newaxis])/s[:,np.newaxis])

			# Predicting
			phi_pred = cm.naivePolyFeature(X_pred,deg=deg,norm=True)
			y_lr[:,group,np.newaxis] = np.dot(phi_pred,r)

		print self.Value0-self.c-np.sum(y_lr,axis=1)
		print max(self.Value0-self.c-np.sum(y_lr,axis=1))

		return np.mean(np.maximum(self.Value0-self.c-np.sum(y_lr,axis=1),0))

	def svr(self,C=1000.,gamma=1e-2):
		''' Support Vector Regression
		'''
		from sklearn.svm import SVR

		y_svr = np.zeros((self.I_pred,self.groups))
		for group in range(self.groups):
			X = self.X_all[:,group*10:(group+1)*10]
			y = self.y_all[:,group]
			X_pred = self.X_pred_all[:,group*10:(group+1)*10]

			# Training
			svr = SVR(kernel='rbf', C=C, gamma=gamma)
			svr.fit(X/self.S0, y)

			# Predicting
			y_svr[:,group] = svr.predict(X_pred/self.S0)

		print self.Value0-self.c-np.sum(y_svr,axis=1)
		print max(self.Value0-self.c-np.sum(y_svr,axis=1))

		return np.mean(np.maximum(self.Value0-self.c-np.sum(y_svr,axis=1),0))

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