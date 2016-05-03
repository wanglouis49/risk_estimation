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
		''' Regression data preparation via nested simulations
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
		lr = linear_model.LinearRegression()
		lr.fit(phi,self.y)
		t_tr = time.time() - t0

		t0 = time.time()  # prediction
		phi_pred = cm.naivePolyFeature(self.X_pred,deg=deg,norm=True)
		y_lr = lr.predict(phi_pred)
		t_pr = time.time() - t0

		eel = np.mean(np.maximum(self.Value0-self.c-y_lr,0))
		return (eel, t_tr, t_pr)

	def spec_regr(self):
		''' Problem-specific basis polynomial Regression
		'''
		from sklearn import linear_model

		t0 = time.time()  # training
		phi = cm.specifiedFeature(self.X,deg=2,norm=True)
		lr = linear_model.LinearRegression()
		lr.fit(phi,self.y)
		t_tr = time.time() - t0

		t0 = time.time()  # prediction
		phi_pred = cm.specifiedFeature(self.X_pred,deg=2,norm=True)
		y_lr = lr.predict(phi_pred)
		t_pr = time.time() - t0

		eel = np.mean(np.maximum(self.Value0-self.c-y_lr,0))
		return (eel, t_tr, t_pr)

	def spec_regr_full(self):
		''' Problem-specific basis polynomial Regression with full basis set
		'''
		from sklearn import linear_model

		t0 = time.time()  # training
		phi = cm.specifiedFeatureFull(self.X,deg=2,norm=True)
		lr = linear_model.LinearRegression()
		lr.fit(phi,self.y)
		t_tr = time.time() - t0

		t0 = time.time()  # prediction
		phi_pred = cm.specifiedFeatureFull(self.X_pred,deg=2,norm=True)
		y_lr = lr.predict(phi_pred)
		t_pr = time.time() - t0

		eel = np.mean(np.maximum(self.Value0-self.c-y_lr,0))
		return (eel, t_tr, t_pr)

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

		eel = np.mean(np.maximum(self.Value0-self.c-y_knn,0))
		return eel

	def svr(self,C=1000.,gamma=1e-2):
		''' Support Vector Regression
		'''
		from sklearn.svm import SVR

		# training
		t0 = time.time()
		svr = SVR(kernel='rbf', C=C, gamma=gamma)
		svr.fit(self.X/self.S0, self.y[:,0])
		t_tr = time.time() - t0

		# predicting
		t0 = time.time()
		y_svr = svr.predict(self.X_pred/self.S0)
		t_pr = time.time() - t0

		eel = np.mean(np.maximum(self.Value0-self.c-y_svr,0))
		return (eel, t_tr, t_pr)

	def svrCV(self):
		''' Support Vector Regression with Cross-Validation
		'''
		from sklearn.svm import SVR
		from sklearn.grid_search import GridSearchCV

		# training
		t0 = time.time()
		svr = GridSearchCV(SVR(C=1000.,kernel='rbf',gamma=1e-4), cv=5,
                       param_grid={"C": [100., 1000.],
                                   "gamma": [1e-3,1e-2,1e-1]}, n_jobs=-1)
		svr.fit(self.X, self.y[:,0])
		t_tr = time.time() - t0

		# predicting
		t0 = time.time()
		y_svr = svr.predict(self.X_pred)
		t_pr = time.time() - t0

		eel = np.mean(np.maximum(self.Value0-self.c-y_svr,0))
		return (eel, t_tr, t_pr)

def re_poly2(kk,N_i):
	from EX1Bc import EX1B
	import time
	if kk/N_i < 1.:
		eel = (np.nan, 0., 0., 0.)
	else:
		port = EX1B()
		t0 = time.time()
		port.regr_data_prep(kk,N_i)
		t_ns = time.time() - t0
		eel = port.poly_regr(deg=2)
		eel += (t_ns,)
	return eel

def re_poly5(kk,N_i):
	from EX1Bc import EX1B
	import time
	if kk/N_i < 1.:
		eel = (np.nan, 0., 0., 0.)
	else:
		port = EX1B()
		t0 = time.time()
		port.regr_data_prep(kk,N_i)
		t_ns = time.time() - t0
		eel = port.poly_regr(deg=5)
		eel += (t_ns,)
	return eel

def re_poly8(kk,N_i):
	from EX1Bc import EX1B
	import time
	if kk/N_i < 1.:
		eel = (np.nan, 0., 0., 0.)
	else:
		port = EX1B()
		t0 = time.time()
		port.regr_data_prep(kk,N_i)
		t_ns = time.time() - t0
		eel = port.poly_regr(deg=8)
		eel += (t_ns,)
	return eel

def re_spec(kk,N_i):
	from EX1Bc import EX1B
	import time
	if kk/N_i < 1.:
		eel = (np.nan, 0., 0., 0.)
	else:
		port = EX1B()
		t0 = time.time()
		port.regr_data_prep(kk,N_i)
		t_ns = time.time() - t0
		eel = port.spec_regr()
		eel += (t_ns,)
	return eel

def re_spec_full(kk,N_i):
	from EX1Bc import EX1B
	import time
	if kk/N_i < 1.:
		eel = (np.nan, 0., 0., 0.)
	else:
		port = EX1B()
		t0 = time.time()
		port.regr_data_prep(kk,N_i)
		t_ns = time.time() - t0
		eel = port.spec_regr_full()
		eel += (t_ns,)
	return eel

def re_svr(kk,N_i):
	from EX1Bc import EX1B
	import time
	if kk/N_i < 1.:
		eel = (np.nan, 0., 0., 0.)
	else:
		port = EX1B()
		t0 = time.time()
		port.regr_data_prep(kk,N_i)
		t_ns = time.time() - t0
		eel = port.svr()
		eel += (t_ns,)
	return eel

def conv(K,N_i=1,L=1000,regr_method=re_svr,filename='EX1Bc'):
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


	EEL_true = 0.0203852730

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
		scipy.io.savemat('./Data/EX1B/'+filename+'.mat',mdict={'K':K,'N_i':N_i,'L':L,\
			'mse':mse,'bias2':bias2,'var':var,'t_tr':t_tr, 't_pr':t_pr,\
			't_ns':t_ns})
		print

if __name__ == "__main__":
	import EX1Bc as EX1
	K = [ii**5 for ii in range(2,23)]
	EX1.conv(K, N_i=1, L=1000, regr_method=re_poly2, filename='re_poly2_1')
	EX1.conv(K, N_i=1, L=1000, regr_method=re_poly5, filename='re_poly5_1')
	EX1.conv(K, N_i=1, L=1000, regr_method=re_poly8, filename='re_poly8_1')
	EX1.conv(K, N_i=1, L=1000, regr_method=re_spec, filename='re_spec_1')
	EX1.conv(K, N_i=1, L=1000, regr_method=re_spec_full, filename='re_spec_full_1')

	EX1.conv(K, N_i=10, L=1000, regr_method=re_poly2, filename='re_poly2_10')
	EX1.conv(K, N_i=10, L=1000, regr_method=re_poly5, filename='re_poly5_10')
	EX1.conv(K, N_i=10, L=1000, regr_method=re_poly8, filename='re_poly8_10')
	EX1.conv(K, N_i=10, L=1000, regr_method=re_spec, filename='re_spec_10')
	EX1.conv(K, N_i=10, L=1000, regr_method=re_spec_full, filename='re_spec_full_10')

	EX1.conv(K, N_i=100, L=1000, regr_method=re_poly2, filename='re_poly2_100')
	EX1.conv(K, N_i=100, L=1000, regr_method=re_poly5, filename='re_poly5_100')
	EX1.conv(K, N_i=100, L=1000, regr_method=re_poly8, filename='re_poly8_100')
	EX1.conv(K, N_i=100, L=1000, regr_method=re_spec, filename='re_spec_100')
	EX1.conv(K, N_i=100, L=1000, regr_method=re_spec_full, filename='re_spec_full_100')
