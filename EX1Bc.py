from __future__ import division
import time
import numpy as np
import numpy.random as npr
import blspricer as bp
import customML as cm
import scipy.stats as scs
from scipy.stats.distributions import norm
from doe_lhs import lhs

class EX1B(object):
	def __init__(self):
		# --- Parameters ---
		self.D = 1
		self.S0 = 100.; self.mu = 0.08; self.sigma = 0.2
		self.rfr = 0.03; self.T = 1./12.; self.tau = 1./52.
		self.K = np.array([101., 110., 114.5])
		self.H = np.array([91., 100., 104.5])
		self.pos = np.array([1., 1., -1.])
		self.c = 0.3608; self.perc = 0.95

		self.I_pred = 10**4

		# --- portfolio price @ t = 0 ---
		self.n = 3
		V0 = np.zeros(self.n)
		for i in range(self.n):
			V0[i] = bp.rear_end_dnOutPut(self.S0, self.K[i], self.rfr, self.T,\
				self.sigma, self.H[i], self.tau, q=0.)
		self.Value0 = np.sum(self.pos * V0)


	def analy(self,N_o):
		''' Analytical solution
		'''
		# --- Analytical solution from 10**7 samples ---
		VaR_true = 0.361879
		CVaR_true = 0.770773
		EEL_true = 0.020454    # not conditional

		# --- portfolio loss distribution @ t = \tau via analytical formulas---
		ran = bp.gen_sn(0,N_o,anti_paths=True,mo_match=True)
		S = np.zeros(N_o)
		S[:] = self.S0
		S[:] = S[:] * np.exp((self.mu - 0.5 * self.sigma**2.) * self.tau +\
							self.sigma * np.sqrt(self.tau) * ran)

		t0 = time.time()
		Vtau = np.zeros((N_o, self.n))
		ValueTau = np.zeros(N_o)   # be careful of the vector size
		for i in range(N_o):
			for j in range(self.n):
				Vtau[i][j] = bp.dnOutPut(S[i], self.K[j], self.rfr, self.T-self.tau,\
										 self.sigma, self.H[j])
			ValueTau[i] = np.sum(self.pos * Vtau[i])
		y = self.Value0 - ValueTau
		t_analy = time.time()-t0
		print "%fs spent in re-valuation" % t_analy

		L_analy = np.sort(y)
		var = scs.scoreatpercentile(L_analy, self.perc*100.)
		idx = np.int(np.ceil(N_o*self.perc))
		cvar = ((np.float(idx)/np.float(N_o) - self.perc)*L_analy[idx] + 
		    np.sum(L_analy[idx+1:])/np.float(N_o))/(1.-self.perc)

		EL = y - VaR_true
		EEL = np.mean(EL.clip(0))

		print "Analytical results:"
		print "VaR: %e vs. %e (true)" % (var, VaR_true)
		print "CVaR: %e vs. %e (true)" % (cvar, CVaR_true)
		print "EEL: %e vs. %e (true)" % (EEL, EEL_true)

		return {'eel':EEL,'t':t_analy}

	def ns(self,kk,beta=1.):
		''' Standard nested simulations
		'''
		# --- Computation budget allocatoin ---
		N_o = np.int(np.ceil(kk**(2./3.)*beta))
		#print "Outer loop: %d" % N_o
		N_i = np.int(np.ceil(kk**(1./3.)/beta))
		#print "Inner loop: %d" % N_i
		#print "Total: %d" % (N_o*N_i)

		# --- portfolio loss distribution @ t = \tau via analytical formulas---
		t0 = time.time()
		ran1 = npr.standard_normal((N_o,1))
		S1 = np.zeros((N_o,1))
		S1[:] = self.S0
		S1[:] = S1[:] * np.exp((self.mu - 0.5*self.sigma*self.sigma)*self.tau + self.sigma *\
								np.sqrt(self.tau) * ran1[:])

		ran2 = npr.standard_normal((N_o,N_i))
		S2 = np.zeros((N_o,N_i))
		S2[:,:] = np.dot(S1[:],np.ones((1,N_i))) * np.exp((self.rfr - 0.5*self.sigma*self.sigma)*(self.T-self.tau) \
						+ self.sigma * np.sqrt(self.T-self.tau) * ran2[:,:])

		prob0 = (1.-np.exp(-2.*(np.log(np.dot(S1[:],np.ones((1,N_i)))/(self.H[0]*np.ones((N_o,N_i))))\
			                   *np.log(S2[:,:]/(self.H[0]*np.ones((N_o,N_i))))/(self.sigma**2)/(self.T-self.tau))))\
		                       *(np.dot(S1[:],np.ones((1,N_i))) >= self.H[0]*np.ones((N_o,N_i))).astype(float)\
		                       *(S2[:,:] >= self.H[0]*np.ones((N_o,N_i))).astype(float)
		prob1 = (1.-np.exp(-2.*(np.log(np.dot(S1[:],np.ones((1,N_i)))/(self.H[1]*np.ones((N_o,N_i))))\
			                   *np.log(S2[:,:]/(self.H[1]*np.ones((N_o,N_i))))/(self.sigma**2)/(self.T-self.tau))))\
		                       *(np.dot(S1[:],np.ones((1,N_i))) >= self.H[1]*np.ones((N_o,N_i))).astype(float)\
		                       *(S2[:,:] >= self.H[1]*np.ones((N_o,N_i))).astype(float)
		prob2 = (1.-np.exp(-2.*(np.log(np.dot(S1[:],np.ones((1,N_i)))/(self.H[2]*np.ones((N_o,N_i))))\
			                   *np.log(S2[:,:]/(self.H[2]*np.ones((N_o,N_i))))/(self.sigma**2)/(self.T-self.tau))))\
		                       *(np.dot(S1[:],np.ones((1,N_i))) >= self.H[2]*np.ones((N_o,N_i))).astype(float)\
		                       *(S2[:,:] >= self.H[2]*np.ones((N_o,N_i))).astype(float)


		Vtau0 = np.dot((np.maximum(self.K[0]-S2[:,:],0)*prob0), np.ones((N_i,1))) / \
				float(N_i) * np.exp(-self.rfr*(self.T-self.tau))
		Vtau1 = np.dot((np.maximum(self.K[1]-S2[:,:],0)*prob1), np.ones((N_i,1))) / \
				float(N_i) * np.exp(-self.rfr*(self.T-self.tau))
		Vtau2 = np.dot((np.maximum(self.K[2]-S2[:,:],0)*prob2), np.ones((N_i,1))) / \
				float(N_i) * np.exp(-self.rfr*(self.T-self.tau))


		ValueTau = Vtau0*self.pos[0] + Vtau1*self.pos[1] + Vtau2*self.pos[2]

		y = self.Value0 - ValueTau

		t_ns = time.time()-t0
		#print "%es spent in re-valuation" % t_ns

		L_ns = np.sort(y)
		var = scs.scoreatpercentile(L_ns, self.perc*100.)
		el = L_ns - self.c
		eel = np.mean(el.clip(0))
		#print "EEL estimated: %e" % eel

		return eel


	def regr_data_prep(self,kk,N_i=1):
		''' Regression data preparation vias nested simulations
		'''
		import customML as cm

		# --- Computation budget allocatoin ---
		N_o = int(kk/N_i)

		# --- portfolio price @ t = \tau via Nested simulations---
		t0 = time.time()
		ran1 = npr.standard_normal((N_o,1))
		S1 = np.zeros((N_o,1))
		S1[:] = self.S0
		S1[:] = S1[:] * np.exp((self.mu - 0.5*self.sigma*self.sigma)*self.tau + \
								self.sigma * np.sqrt(self.tau) * ran1[:])

		ran2 = npr.standard_normal((N_o,N_i))
		S2 = np.zeros((N_o,N_i))
		S2[:,:] = np.dot(S1[:],np.ones((1,N_i))) * np.exp((self.rfr - 0.5*self.sigma*self.sigma)*(self.T-self.tau) \
						+ self.sigma * np.sqrt(self.T-self.tau) * ran2[:,:])

		prob0 = (1.-np.exp(-2.*(np.log(np.dot(S1[:],np.ones((1,N_i)))/(self.H[0]*np.ones((N_o,N_i))))\
			                   *np.log(S2[:,:]/(self.H[0]*np.ones((N_o,N_i))))/(self.sigma**2)/(self.T-self.tau))))\
		                       *(np.dot(S1[:],np.ones((1,N_i))) >= self.H[0]*np.ones((N_o,N_i))).astype(float)\
		                       *(S2[:,:] >= self.H[0]*np.ones((N_o,N_i))).astype(float)
		prob1 = (1.-np.exp(-2.*(np.log(np.dot(S1[:],np.ones((1,N_i)))/(self.H[1]*np.ones((N_o,N_i))))\
			                   *np.log(S2[:,:]/(self.H[1]*np.ones((N_o,N_i))))/(self.sigma**2)/(self.T-self.tau))))\
		                       *(np.dot(S1[:],np.ones((1,N_i))) >= self.H[1]*np.ones((N_o,N_i))).astype(float)\
		                       *(S2[:,:] >= self.H[1]*np.ones((N_o,N_i))).astype(float)
		prob2 = (1.-np.exp(-2.*(np.log(np.dot(S1[:],np.ones((1,N_i)))/(self.H[2]*np.ones((N_o,N_i))))\
			                   *np.log(S2[:,:]/(self.H[2]*np.ones((N_o,N_i))))/(self.sigma**2)/(self.T-self.tau))))\
		                       *(np.dot(S1[:],np.ones((1,N_i))) >= self.H[2]*np.ones((N_o,N_i))).astype(float)\
		                       *(S2[:,:] >= self.H[2]*np.ones((N_o,N_i))).astype(float)


		Vtau0 = np.dot((np.maximum(self.K[0]-S2[:,:],0)*prob0), np.ones((N_i,1))) / \
						float(N_i) * np.exp(-self.rfr*(self.T-self.tau))
		Vtau1 = np.dot((np.maximum(self.K[1]-S2[:,:],0)*prob1), np.ones((N_i,1))) / \
						float(N_i) * np.exp(-self.rfr*(self.T-self.tau))
		Vtau2 = np.dot((np.maximum(self.K[2]-S2[:,:],0)*prob2), np.ones((N_i,1))) / \
						float(N_i) * np.exp(-self.rfr*(self.T-self.tau))


		ValueTau = Vtau0*self.pos[0] + Vtau1*self.pos[1] + Vtau2*self.pos[2]

		t_ns = time.time() - t0

		# prediction samples
		#ran3 = norm(loc=0, scale=1).ppf(lhs(D, samples=I_pred))
		stratified_gaussian  = np.array([(i-0.5)/self.I_pred for i in range(1,self.I_pred+1)])
		ran3 = norm.ppf(stratified_gaussian[:,np.newaxis])
		S_pred = np.zeros((self.I_pred,1))
		S_pred[:] = self.S0
		S_pred[:] = S_pred[:] * np.exp((self.mu - 0.5*self.sigma*self.sigma)*self.tau +\
										self.sigma*np.sqrt(self.tau) * ran3[:])

		self.X = S1
		self.X_pred = S_pred
		self.y = ValueTau


	def poly_regr(self,deg=2):
		''' Polynomial linear Regression
		'''
		from sklearn import linear_model

		t0 = time.time()  # training
		phi = cm.naivePolyFeature(self.X,deg=deg,norm=True)
		#rank = np.linalg.matrix_rank(phi)
		lr = linear_model.LinearRegression()
		lr.fit(phi,self.y)
		t_tr = time.time() - t0

		t0 = time.time()  # prediction
		phi_pred = cm.naivePolyFeature(self.X_pred,deg=deg,norm=True)
		y_lr = lr.predict(phi_pred)
		t_pr = time.time() - t0
		t_tt = t_tr + t_pr

		#self.eel_poly_regr = np.mean(np.maximum(self.Value0-self.c-y_lr,0))
		return np.mean(np.maximum(self.Value0-self.c-y_lr,0))

	def spec_regr(self):
		''' Problem-specific basis polynomial Regression
		'''
		from sklearn import linear_model

		t0 = time.time()  # training
		phi = cm.specifiedFeature(self.X,deg=2,norm=True)
		#rank = np.linalg.matrix_rank(phi)
		lr = linear_model.LinearRegression()
		lr.fit(phi,self.y)
		t_tr = time.time() - t0

		t0 = time.time()  # prediction
		phi_pred = cm.specifiedFeature(self.X_pred,deg=2,norm=True)
		y_lr = lr.predict(phi_pred)
		t_pr = time.time() - t0

		t_tt = t_tr + t_pr

		#self.eel_spec_regr = np.mean(np.maximum(self.Value0-self.c-y_lr,0))
		return np.mean(np.maximum(self.Value0-self.c-y_lr,0))

	def knn(self):
		''' K nearest neighbors Regression
		'''
		from sklearn import neighbors
		from sklearn.grid_search import GridSearchCV

		if len(self.y) == 2**5:
			knn = GridSearchCV(neighbors.KNeighborsRegressor(n_neighbors=5,weights='uniform'),\
								param_grid={"n_neighbors": [5,10,20]})
		elif len(self.y) == 3**5:
			knn = GridSearchCV(neighbors.KNeighborsRegressor(n_neighbors=10,weights='uniform'),\
								param_grid={"n_neighbors": [10,20,50,70,100,150]})
		else:
			knn = GridSearchCV(neighbors.KNeighborsRegressor(n_neighbors=10,weights='uniform'),\
								param_grid={"n_neighbors": [10,20,50,70,100,200,300,400]})

		y_knn = knn.fit(self.X/self.S0, self.y).predict(self.X_pred/self.S0)

		return np.mean(np.maximum(self.Value0-self.c-y_knn,0))

	def svr(self,C=1000.,gamma=1e-2):
		''' Support Vector Regression
		'''
		from sklearn.svm import SVR

		# training
		svr = SVR(kernel='rbf', C=C, gamma=gamma)
		svr.fit(self.X/self.S0, self.y[:,0])

		y_svr = svr.predict(self.X_pred/self.S0)

		return np.mean(np.maximum(self.Value0-self.c-y_svr,0))

	def svrCV(self):
		''' Support Vector Regression with Cross-Validation
		'''
		from sklearn.svm import SVR
		from sklearn.grid_search import GridSearchCV

		# training
		svr = GridSearchCV(SVR(C=1000.,kernel='rbf',gamma=1e-4), cv=5,
                       param_grid={"C": [100., 1000.],
                                   "gamma": [1e-3,1e-2,1e-1]}, n_jobs=-1)
		svr.fit(self.X/self.S0, self.y[:,0])

		y_svr = svr.predict(self.X_pred/self.S0)

		return np.mean(np.maximum(self.Value0-self.c-y_svr,0))

def re_poly(kk,N_i,deg=2):
	if kk/N_i < 1.:
		eel = np.nan
	else:
		port = EX1B()
		port.regr_data_prep(kk,N_i)
		eel = port.poly_regr(deg=deg)
	return eel

def re_svr(kk,N_i):
	if kk/N_i < 1.:
		eel = np.nan
	else:
		port = EX1B()
		port.regr_data_prep(kk,N_i)
		eel = port.svr()
	return eel

def conv(K,N_i,L=1000):
	import scipy.io
	import ipyparallel
	from multiprocessing import cpu_count
	import os

	print "#CPUs: %d" % cpu_count()
	print
	rc = ipyparallel.Client()
	print("Check 1")
	rc.block = True
	print("Check 2")
	view = rc.load_balanced_view()
	print("Check 3")
	dview = rc[:]
	print("Check 4")
	dview.map(os.chdir, ['/home/l366wang/Code/f2c/New/']*cpu_count())
	print("Check 5")


	EEL_true = 0.0203852730

	mse = np.zeros(len(K))
	bias2 = np.zeros(len(K))
	var = np.zeros(len(K))
	t = np.zeros(len(K))

	for k_idx, kk in enumerate(K):
		print "K = %d" % kk
		t0 = time.time()
		eel_data = view.map(re_svr,[kk]*L,[N_i]*L)

		mse[k_idx] = np.mean((np.array(eel_data)-EEL_true)**2)
		bias2[k_idx] = (np.mean(eel_data)-EEL_true)**2
		var[k_idx] = np.mean((np.array(eel_data)-np.mean(eel_data))**2)
		t[k_idx] = time.time()-t0
		print "%.2fs elapsed" % (time.time()-t0)
		scipy.io.savemat('EX1Bc.mat',mdict={'mse':mse,'bias2':bias2,'var':var,'t':t})
		print

if __name__ == "__main__":
	import EX1Bc as EX1
	K = [ii**5 for ii in range(2,23)]
	EX1.conv(K, N_i=1, L=1000)



