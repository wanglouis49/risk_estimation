from __future__ import division
import time
import numpy as np
import numpy.random as npr
import blspricer as bp
import customML as cm
import scipy.stats as scs
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
		ValueTau = np.zeros((N_o,1))
		ran1 = npr.multivariate_normal(np.zeros(self.D),self.corr_mat,N_o)
		S1 = np.zeros((N_o,self.D))
		S1[:,:] = self.S0
		S1[:,:] = S1[:,:] * np.exp((self.mu - 0.5 * np.dot(np.ones((N_o,1)),self.sig_vec[np.newaxis,:])**2) * self.tau \
			+ np.dot(np.ones((N_o,1)),self.sig_vec[np.newaxis,:]) * np.sqrt(self.tau) * ran1[:,:])
		for dim in range(self.D):
			ran2 = npr.standard_normal((N_o,N_i))
			S2 = np.zeros((N_o,N_i))
			S2[:,:] = np.dot(S1[:,dim,np.newaxis],np.ones((1,N_i))) * np.exp((self.rfr - 0.5*self.sig_vec[dim]**2)*(self.T-self.tau) \
				            + self.sig_vec[dim] * np.sqrt(self.T-self.tau) * ran2[:,:])

			V1 = np.dot((np.maximum(self.K-S2[:,:],0)),np.ones((N_i,1))) / float(N_i) * np.exp(-self.rfr*(self.T-self.tau))
			V2 = np.dot((np.maximum(S2[:,:]-self.K,0)),np.ones((N_i,1))) / float(N_i) * np.exp(-self.rfr*(self.T-self.tau))

			ValueTau[:] += -10. * (V1+V2)
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

		self.X = S1
		self.X_pred = S_pred
		self.y = ValueTau

	def poly_regr(self,deg=2):
		''' Polynomial Regression
		'''
		# Training
		t0 = time.time()
		phi = cm.naivePolyFeature(self.X,deg=deg,norm=True)
		U, s, V = np.linalg.svd(phi,full_matrices=False)
		r = np.dot(V.T,np.dot(U.T,self.y)/s[:,np.newaxis])
		t_tr = time.time() - t0

		# Predicting
		t0 = time.time()
		phi_pred = cm.naivePolyFeature(self.X_pred,deg=deg,norm=True)
		y_lr = np.dot(phi_pred,r)
		t_pr = time.time() - t0

		eel = np.mean(np.maximum(self.Value0-self.c-np.sum(y_lr,axis=1),0))
		return (eel, t_tr, t_pr)

	def poly_ridge(self,deg=2):
		''' Polynomial Ridge Regression
		'''
		from sklearn import linear_model
		# Training
		t0 = time.time()
		phi = cm.naivePolyFeature(self.X,deg=deg,norm=True)
		lm = linear_model.RidgeCV(alphas=np.logspace(-4,0,5))
		lm.fit(phi,self.y)
		#print lm.alpha_
		t_tr = time.time() - t0

		# Predicting
		t0 = time.time()
		phi_pred = cm.naivePolyFeature(self.X_pred,deg=deg,norm=True)
		y_lr = lm.predict(phi_pred)
		t_pr = time.time() - t0

		eel = np.mean(np.maximum(self.Value0-self.c-np.sum(y_lr,axis=1),0))
		return (eel, t_tr, t_pr)

def re_poly2(kk,N_i):
	from EX100V2c import EX100V
	import time
	import numpy as np
	if kk/N_i < 1.:
		eel = (np.nan, 0., 0., 0.)
	else:
		port = EX100V()
		t0 = time.time()
		port.regr_data_prep(kk,N_i)
		t_ns = time.time() - t0
		eel = port.poly_regr(deg=2)
		eel += (t_ns,)
	return eel

def re_poly5(kk,N_i):
	from EX100V2c import EX100V
	import time
	import numpy as np
	if kk/N_i < 1.:
		eel = (np.nan, 0., 0., 0.)
	else:
		port = EX100V()
		t0 = time.time()
		port.regr_data_prep(kk,N_i)
		t_ns = time.time() - t0
		eel = port.poly_regr(deg=5)
		eel += (t_ns,)
	return eel

def re_poly8(kk,N_i):
	from EX100V2c import EX100V
	import time
	import numpy as np
	if kk/N_i < 1.:
		eel = (np.nan, 0., 0., 0.)
	else:
		port = EX100V()
		t0 = time.time()
		port.regr_data_prep(kk,N_i)
		t_ns = time.time() - t0
		eel = port.poly_regr(deg=8)
		eel += (t_ns,)
	return eel

def re_ridge2(kk,N_i):
	from EX100V2c import EX100V
	import time
	import numpy as np
	if kk/N_i < 1.:
		eel = (np.nan, 0., 0., 0.)
	else:
		port = EX100V()
		t0 = time.time()
		port.regr_data_prep(kk,N_i)
		t_ns = time.time() - t0
		eel = port.poly_ridge(deg=2)
		eel += (t_ns,)
	return eel

def re_ridge5(kk,N_i):
	from EX100V2c import EX100V
	import time
	import numpy as np
	if kk/N_i < 1.:
		eel = (np.nan, 0., 0., 0.)
	else:
		port = EX100V()
		t0 = time.time()
		port.regr_data_prep(kk,N_i)
		t_ns = time.time() - t0
		eel = port.poly_ridge(deg=5)
		eel += (t_ns,)
	return eel

def re_ridge8(kk,N_i):
	from EX100V2c import EX100V
	import time
	import numpy as np
	if kk/N_i < 1.:
		eel = (np.nan, 0., 0., 0.)
	else:
		port = EX100V()
		t0 = time.time()
		port.regr_data_prep(kk,N_i)
		t_ns = time.time() - t0
		eel = port.poly_ridge(deg=8)
		eel += (t_ns,)
	return eel

def conv(K,N_i=1,L=100,regr_method=re_poly2,filename='EX100Vc'):
	import scipy.io
	import ipyparallel
	from multiprocessing import cpu_count
	import os

	print
	print "##################################"
	print "# Risk Estimation via Regression #"
	print "##################################"
	print
	print "regression method: %s" % str(regr_method)
	print "Output file: %s.mat" % filename
	print "N_i = %d; L = %d" % (N_i, L)
	print
	print "Setting up ipyparallel"
	print "CPU count: %d" % cpu_count()
	print
	rc = ipyparallel.Client()
	print("Check point 1")
	rc.block = True
	print("Check point 2")
	view = rc.load_balanced_view()
	print("Check point 3")
	dview = rc[:]
	print("Check point 4")
	dview.map(os.chdir, ['/home/l366wang/Code/risk_regr/']*cpu_count())
	print("Check point 5")
	print
	print "Checks done. Commensing computations..."
	print


	EEL_true = 1.818317262496906

	mse = np.zeros(len(K))
	bias2 = np.zeros(len(K))
	var = np.zeros(len(K))
	t_tr = np.zeros(len(K))
	t_pr = np.zeros(len(K))
	t_ns = np.zeros(len(K))

	for k_idx, kk in enumerate(K):
		print "K = %d" % kk
		t0 = time.time()
		eel_data = view.map(regr_method,[kk]*L,[N_i]*L)
		eel = [eel_data[ii][0] for ii in range(L)]

		mse[k_idx] = np.mean((np.array(eel)-EEL_true)**2)
		bias2[k_idx] = (np.mean(eel)-EEL_true)**2
		var[k_idx] = np.mean((np.array(eel)-np.mean(eel))**2)

		t_tr[k_idx] = np.mean([eel_data[ii][1] for ii in range(L)])
		t_pr[k_idx] = np.mean([eel_data[ii][2] for ii in range(L)])
		t_ns[k_idx] = np.mean([eel_data[ii][3] for ii in range(L)])

		print "%.2fs elapsed" % (time.time()-t0)
		scipy.io.savemat('./Data/EX100V2/'+filename+'.mat',mdict={'K':K,'N_i':N_i,'L':L,\
			'mse':mse,'bias2':bias2,'var':var,'t_tr':t_tr, 't_pr':t_pr,\
			't_ns':t_ns})
		print

if __name__ == "__main__":
	import EX100V2c as EX100
	K = [ii**5 for ii in range(4,20)]
	EX100.conv(K, N_i=1, L=100, regr_method=re_poly2, filename='EX100Vc')