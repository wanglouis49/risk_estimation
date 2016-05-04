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
		self.c = 246.42939112403042; self.perc = 0.99

		self.I_pred = 2**15

		# --- portfolio price @ t = 0 ---
		opt_val = lambda S: - 5.*bp.blsprice(S,self.K,self.rfr,self.T,self.sigma,'put') \
							- 10.*bp.rear_end_dnOutCall(S,self.K,self.rfr,self.T,self.sigma,self.H,self.tau,0.)
		self.Value0 = np.sum(map(opt_val,[self.S0]*self.D))

	def analy(self,N_o):
		''' Analytical solution
		'''
		# --- portfolio loss distribution @ t = \tau via analytical formulas ---
		opt_val = lambda S: - 5.*bp.blsprice(S,self.K,self.rfr,self.T-self.tau,self.sigma,'put') \
							- 10.*bp.dnOutCall(S,self.K,self.rfr,self.T-self.tau,self.sigma,self.H)

		#ran1 = npr.multivariate_normal(np.zeros(self.D),np.eye(self.D),N_o)
		#ran1 = npr.standard_normal((N_o,self.D))
		ran1 = norm(loc=0,scale=1).ppf(lhs(self.D,samples=N_o))
		S1 = np.zeros((N_o,self.D))
		S1[:,:] = self.S0
		S1[:,:] = S1[:,:] * np.exp((self.mu - 0.5 * self.sigma**2) * self.tau\
				 + self.sigma * np.sqrt(self.tau) * ran1[:,:])
		
		t0 = time.time()
		ValueTau = np.zeros(N_o)
		for n in range(N_o):
			ValueTau[n] = np.sum(map(opt_val,S1[n,:]))
		print "%.2fs eclipsed" % (time.time() - t0)

		L_analy = np.sort(self.Value0 - ValueTau)
		#L_analy = np.sort(-ValueTau + self.Value0 * np.exp(self.rfr*self.tau))
		var = scs.scoreatpercentile(L_analy, self.perc*100.)
		eel = np.mean(np.maximum(L_analy-var,0))
		return (var, eel)

	def ns(self,kk,beta=1.):
		''' Standard nested simulations
		'''
		# --- Computation budget allocation ---
		N_o = np.int(np.ceil(kk**(2./3.)*beta))
		N_i = np.int(np.ceil(kk**(1./3.)/beta))
		print "Outer loop: %d" % N_o
		print "Inner loop: %d" % N_i
		print "Total: %d" % (N_o*N_i)

		# --- True values from 2**15 LHS ---
		var_true = 246.42939112403042
		eel_true = 0.45553892616275565

		# --- portfolio loss distribution @ t = \tau via analytical formulas---
		t0 = time.time()
		ran1 = npr.standard_normal((N_o,self.D))
		S1 = np.zeros((N_o,self.D))
		S1[:,:] = self.S0
		S1[:,:] = S1[:,:] * np.exp((self.mu - 0.5*self.sigma*self.sigma)*self.tau + self.sigma *\
								np.sqrt(self.tau) * ran1[:,:])

		ValueTau = np.zeros((N_o,1))
		for dim in range(self.D):
			ran2 = npr.standard_normal((N_o,N_i))
			S2 = np.zeros((N_o,N_i))
			S2[:,:] = np.dot(S1[:,dim,np.newaxis],np.ones((1,N_i))) * np.exp((self.rfr - 0.5*self.sigma**2)*(self.T-self.tau) + self.sigma \
				* np.sqrt(self.T-self.tau) * ran2[:,:])

			prob = (1.-np.exp(-2.*(np.log(np.dot(S1[:,dim,np.newaxis],np.ones((1,N_i)))/self.H)\
			                   *np.log(S2[:,:]/self.H)/(self.sigma**2)/(self.T-self.tau))))\
		                       *(np.dot(S1[:,dim,np.newaxis],np.ones((1,N_i))) >= self.H*np.ones((N_o,N_i))).astype(float)\
		                       *(S2[:,:] >= self.H*np.ones((N_o,N_i))).astype(float)

			C2do = np.dot((np.maximum(S2[:,:]-self.K,0)*prob), np.ones((N_i,1))) / \
				float(N_i) * np.exp(-self.rfr*(self.T-self.tau))

			P2 = np.dot((np.maximum(self.K-S2[:,:],0)),np.ones((N_i,1))) / \
				float(N_i) * np.exp(-self.rfr*(self.T-self.tau))

			ValueTau[:] += -10. * C2do - 5. * P2
		t_ns = time.time() - t0
		print "%.2fs eclipsed" % t_ns

		L_ns = np.sort(self.Value0 - ValueTau)
		var = scs.scoreatpercentile(L_ns, self.perc*100.)
		eel = np.mean(np.maximum(L_ns-var,0))

		print "VaR estimated: %e (true: %e)" % (var, var_true)
		print "EEL estimated: %e (true: %e)" % (eel, eel_true)

		return (eel, t_ns)

	def regr_data_prep(self,kk,N_i=1):
		''' Regression data preparation via nested simulations
		'''
		# --- Computation budget allocation ---
		N_o = int(kk/N_i)

		# --- portfolio loss distribution @ t = \tau via analytical formulas---
		t0 = time.time()
		ran1 = npr.standard_normal((N_o,self.D))
		S1 = np.zeros((N_o,self.D))
		S1[:,:] = self.S0
		S1[:,:] = S1[:,:] * np.exp((self.mu - 0.5*self.sigma*self.sigma)*self.tau + self.sigma *\
								np.sqrt(self.tau) * ran1[:,:])

		ValueTau = np.zeros((N_o,1))
		for dim in range(self.D):
			ran2 = npr.standard_normal((N_o,N_i))
			S2 = np.zeros((N_o,N_i))
			S2[:,:] = np.dot(S1[:,dim,np.newaxis],np.ones((1,N_i))) * np.exp((self.rfr - 0.5*self.sigma**2)*(self.T-self.tau) + self.sigma \
				* np.sqrt(self.T-self.tau) * ran2[:,:])

			prob = (1.-np.exp(-2.*(np.log(np.dot(S1[:,dim,np.newaxis],np.ones((1,N_i)))/self.H)\
			                   *np.log(S2[:,:]/self.H)/(self.sigma**2)/(self.T-self.tau))))\
		                       *(np.dot(S1[:,dim,np.newaxis],np.ones((1,N_i))) >= self.H*np.ones((N_o,N_i))).astype(float)\
		                       *(S2[:,:] >= self.H*np.ones((N_o,N_i))).astype(float)

			C2do = np.dot((np.maximum(S2[:,:]-self.K,0)*prob), np.ones((N_i,1))) / \
				float(N_i) * np.exp(-self.rfr*(self.T-self.tau))

			P2 = np.dot((np.maximum(self.K-S2[:,:],0)),np.ones((N_i,1))) / \
				float(N_i) * np.exp(-self.rfr*(self.T-self.tau))

			ValueTau[:] += -10. * C2do - 5. * P2
		t_ns = time.time() - t0

		ran3 = norm(loc=0,scale=1).ppf(lhs(self.D,samples=self.I_pred))
		S_pred = np.zeros((self.I_pred,self.D))
		S_pred[:,:] = self.S0
		S_pred[:,:] = S_pred[:,:] * np.exp((self.mu - 0.5*self.sigma*self.sigma)*self.tau + self.sigma *\
								np.sqrt(self.tau) * ran3[:,:])

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

	def spec_regr(self,deg=5):
		pass

def re_ns(kk):
	pass

def re_poly2(kk,N_i):
	from EX10 import EX10
	import time
	if kk/N_i < 1.:
		eel = (np.nan, 0., 0., 0.)
	else:
		port = EX10()
		t0 = time.time()
		port.regr_data_prep(kk,N_i)
		t_ns = time.time() - t0
		eel = port.poly_regr(deg=2)
		eel += (t_ns,)
	return eel

def re_poly5(kk,N_i):
	from EX10 import EX10
	import time
	if kk/N_i < 1.:
		eel = (np.nan, 0., 0., 0.)
	else:
		port = EX10()
		t0 = time.time()
		port.regr_data_prep(kk,N_i)
		t_ns = time.time() - t0
		eel = port.poly_regr(deg=5)
		eel += (t_ns,)
	return eel

def re_poly8(kk,N_i):
	from EX10 import EX10
	import time
	if kk/N_i < 1.:
		eel = (np.nan, 0., 0., 0.)
	else:
		port = EX10()
		t0 = time.time()
		port.regr_data_prep(kk,N_i)
		t_ns = time.time() - t0
		eel = port.poly_regr(deg=8)
		eel += (t_ns,)
	return eel

def conv(K,N_i=1,L=100,regr_method=re_poly2,filename='EX10'):
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


	EEL_true = 0.45553892616275565

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
		scipy.io.savemat('./Data/EX10/'+filename+'.mat',mdict={'K':K,'N_i':N_i,'L':L,\
			'mse':mse,'bias2':bias2,'var':var,'t_tr':t_tr, 't_pr':t_pr,\
			't_ns':t_ns})
		print

if __name__ == "__main__":
	import EX10
	K = [ii**5 for ii in range(2,23)]
	EX10.conv(K,N_i=1,L=100,regr_method=re_poly2,filename='re_poly2_1')
	EX10.conv(K,N_i=1,L=100,regr_method=re_poly5,filename='re_poly5_1')
	EX10.conv(K,N_i=1,L=100,regr_method=re_poly8,filename='re_poly8_1')