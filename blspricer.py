"""
Option pricing module for Risk Management research at the Scientific Computing group 
at University of Waterloo

Developed by Luyu Wang
2015-2016
"""
from __future__ import division
import numpy as np
import numpy.random as npr
from math import log, sqrt, exp
from scipy import stats

##########################################################################################
# Random Number Generators
##########################################################################################

def gen_sn(M, I, anti_paths=False, mo_match=False):
	''' Function to generate random numbers for simulation with variance reduction

	from Y. Hilpisch, Python for Finance

	Parameters
	==========
	M : int
	    number of time intervals for discretization
	I : int
	    number of paths to be simulated
	anti_paths : Boolean
	    use of antithetic variates
	mo_match : Boolean
	    use of moment matching
	'''

	if anti_paths is True:
		sn = npr.standard_normal((M + 1, I / 2))
		sn = np.concatenate((sn, -sn), axis=1)
	else:
		sn = npr.standard_normal((M + 1, I))
	if mo_match is True:
		sn = (sn - sn.mean()) / sn.std()
	return sn


def gen_sn2(D, I, anti_paths=True, mo_match=True):
	''' Function to generate random numbers for simulation.

    from Y. Hilpisch, Python for Finance
    Modified by L. Wang for handling high dimensional data generation.

	Parameters
	==========
	D : int
	    number of dimensions
	I : int
	    number of paths to be simulated
	anti_paths : Boolean
	    use of antithetic variates
	mo_match : Boolean
	    use of moment matching
	'''
	import numpy as np
	import numpy.random as npr
	if anti_paths is True:
		sn = npr.standard_normal((D, I / 2))
		sn = np.concatenate((sn, -sn), axis=1)
	else:
		sn = npr.standard_normal((D, I))
	if mo_match is True:
		sn = (sn - sn.mean()) / sn.std()
	return sn


def normcdfM(x):
	''' Gives the normal cumulative density function probabilities
	    Author: Sivakumar Batthala
	'''
	from math import log, sqrt, exp, pi

	a1 = 0.319381530; a2 = -0.356563782; a3 = 1.781477937; a4 = -1.821255978; a5 = 1.330274429
	gamma = 0.2316419

	k = 1./(1.+(gamma*x))
	nprime = (1./sqrt(2.*pi)) * (exp(-(x**2.)/2.))

	if (x >= 0.):
		n = 1. - (nprime * (a1*k + a2*k**2. + a3*k**3. + a4*k**4. + a5*k**5.))
	else:
		n = 1. - normcdfM(-x)

	return n

def bivnormcdf(a,b,rho):
	''' Gives the bivariate normal distribution function probabilities
	    As given in Hull's textbook
	    Author: Sivakumar Batthala
	'''
	from math import log, sqrt, exp, pi
	import numpy as np

	if a <= 0. and b <= 0. and rho <= 0. :
		aprime = a/sqrt(2.*(1.-rho**2.))
		bprime = b/sqrt(2.*(1.-rho**2.))
		A = np.array([0.3253030, 0.4211071, 0.1334425, 0.006374323])
		B = np.array([0.1337764, 0.6243247, 1.3425378, 2.2626645])

		t = 0.

		for i in range(4):
			for j in range(4):
				x = B[i]
				y = B[j]
				t += A[i]*A[j]* exp(aprime*(2.*x - aprime) \
				     + (bprime*(2.*y - bprime)) + (2.*rho * (x - aprime)*(y-bprime)))

		p = (sqrt(1.-rho**2.)/pi) * t

	elif a * b * rho <= 0. :
		if a <= 0. and b >= 0. and rho >= 0. :
			p = normcdfM(a) - bivnormcdf(a,-b,-rho)
		elif a >= 0. and b <= 0. and rho >= 0. :
			p = normcdfM(b) - bivnormcdf(-a,b,-rho)
		elif a >= 0. and b >= 0. and rho <= 0. :
			p = normcdfM(a) + normcdfM(b) - 1. + bivnormcdf(-a,-b,rho)

	elif a*b*rho > 0. :
		if a >= 0. :
			asign = 1.
		else:
			asign = -1.

		if b >= 0.:
			bsign = 1.
		else:
			bsign = -1.

		rho1 = (rho*a - b)*asign/(sqrt(a**2. - (2.*rho*a*b) + b**2.))
		rho2 = (rho*b - a)*bsign/(sqrt(a**2. - (2.*rho*a*b) + b**2.))
		delta = (1. - (asign*bsign))/4.

		p = bivnormcdf(a,0,rho1) + bivnormcdf(b,0,rho2) - delta
	return p


##########################################################################################
# Option priceing formulas
##########################################################################################

def blsprice(S0, K, r, T, sigma, optionType='call'):
	''' Valuation of European option in BSM model Analytical formula.

	Parameters
	==========
	S0 : float
		initial stock/index level
	K : float
		strike price
	T : float
		maturity data (in year fractions)
	r : float
		constant risk-free short rate
	sigma : float
		volatility factor in diffusion term
	optionType : string
		type of the option to be valued ('call', 'put')

	Returns
	=======
	value : float
		present value of the European option
	'''
	# assert type(S0) is float and type(K) is float and type(r) is float and type(T) is float and type(sigma) is float, "Unrecongnized input, check type"
	d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	if optionType == 'call':
		value = (S0 * stats.norm.cdf(d1, 0.0, 1.0) - K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
	elif optionType == 'put':
		value = (K * exp(-r * T) * stats.norm.cdf(-d2, 0.0, 1.0) - S0 * stats.norm.cdf(-d1, 0.0, 1.0))
	return value

def blsprice2(S0, K, r, T, sigma):
	''' Valuation of European option in BSM model Analytical formula.
	'''
	# assert type(S0) is float and type(K) is float and type(r) is float and type(T) is float and type(sigma) is float, "Unrecongnized input, check type"
	d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	#if optionType == 'call':
	C_value = (S0 * stats.norm.cdf(d1, 0.0, 1.0) - K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
	#elif optionType == 'put':
	P_value = (K * exp(-r * T) * stats.norm.cdf(-d2, 0.0, 1.0) - S0 * stats.norm.cdf(-d1, 0.0, 1.0))
	return (C_value,P_value)

def test_blsprice():
	C0 = blsprice(100.,110.,0.03,1.,0.5); P0 = blsprice(100.,110.,0.03,1.,0.5,'put')
	print "Analytical: European call option (S0=100,K=110,r=0.03,T=1,sigma=0.5)"
	print "[call=17.2031, put=23.9521] <= MATLAB blsprice"
	print "[call=%7.4f, put=%7.4f] <= this python implementation" % (C0,P0)
	assert abs(C0-17.2031) < 1e-4 and abs(P0-23.9521) < 1e-4, \
		"Disagree from MATLAB results."


def blsprice_mcs(S0, K, r, T, sigma, optionType='call', M=1, I=10000):
	''' Valuation of European options in Black-Scholes-Merton by Monte Carlo simulation 
	(of index level paths)

	Parameters
	==========
	S0 : float
		initial stock/index level
	K : float
		strike price
	T : float
		maturity data (in year fractions)
	r : float
		constant risk-free short rate
	sigma : float
		volatility factor in diffusion term
	optionType : string
		type of the option to be valued ('call', 'put')
	M : integer
		number of time steps
	I : integer
		number of risk-neutral paths

	Returns
	=======
	value : float
		estimated present value of European option
	'''
	# assert type(S0) is float and type(K) is float and type(r) is float and type(T) is float and type(sigma) is float, "Unrecongnized input, check type"

	dt = T / M
	# simulation of index level paths
	S = np.zeros((M + 1, I))
	S[0] = S0
	sn = gen_sn(M, I, anti_paths=False, mo_match=False)
	for t in range(1, M + 1):
		S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * sn[t])
	# case-based calculation of payoff
	if optionType == 'call':
		hT = np.maximum(S[-1] - K, 0)
	elif optionType == 'put':
		hT = np.maximum(K - S[-1], 0)
	# calculation of MCS estimator
	value = np.exp(-r * T) * 1 / I * np.sum(hT)
	return value

def blsprice2_mcs(S0, K, r, T, sigma, M=1, I=10000):
	''' Valuation of European options in Black-Scholes-Merton by Monte Carlo simulation 
	(of index level paths)
	'''
	# assert type(S0) is float and type(K) is float and type(r) is float and type(T) is float and type(sigma) is float, "Unrecongnized input, check type"

	dt = T / M
	# simulation of index level paths
	S = np.zeros((M + 1, I))
	S[0] = S0
	sn = gen_sn(M, I, anti_paths=False, mo_match=False)
	for t in range(1, M + 1):
		S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * sn[t])
	# case-based calculation of payoff
	#if optionType == 'call':
	C_hT = np.maximum(S[-1] - K, 0)
	C_value = np.exp(-r * T) * 1 / I * np.sum(C_hT)
	#elif optionType == 'put':
	P_hT = np.maximum(K - S[-1], 0)
	P_value = np.exp(-r * T) * 1 / I * np.sum(P_hT)
	return (C_value,P_value)

def test_blsprice_mcs():
	C0 = blsprice_mcs(100.,110.,0.03,1.,0.5); P0 = blsprice_mcs(100.,110.,0.03,1.,0.5,'put')
	print "Monte Carlo: European call option (S0=100,K=110,r=0.03,T=1,sigma=0.5)"
	print "[call=17.2031, put=23.9521] <= MATLAB blsprice"
	print "[call=%7.4f, put=%7.4f] <= this python implementation" % (C0,P0)
	assert abs(C0-17.2031) < 1 and abs(P0-23.9521) < 1, \
		"Disagree from MATLAB results."

def dnOutPut(S0, K, r, T, sigma, Sb):
	''' Valuation of European down-and-out barrier put option in BSM model.
	Analytical formula.

	Parameters
	==========
	S0 : float
		initial stock/index level
	K : float
		strike price
	T : float
		maturity data (in year fractions)
	r : float
		constant risk-free short rate
	sigma : float
		volatility factor in diffusion term
	Sb : float
		barrirer level

	Returns
	=======
	value : float
		present value of the European option
	'''

	# assert type(S0) is float and type(Sb) is float and type(K) is float and type(r) is float and type(T) is float and type(sigma) is float, "Unrecongnized input, check type"
	a = (Sb / S0) ** (-1. + 2. * r / sigma **2)
	b = (Sb / S0) ** (1. + 2. * r / sigma **2)
	d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	d3 = (log(S0 / Sb) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	d4 = (log(S0 / Sb) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	d5 = (log(S0 / Sb) - (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	d6 = (log(S0 / Sb) - (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	d7 = (log(S0 * K / Sb ** 2) - (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	d8 = (log(S0 * K / Sb ** 2) - (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	value = K * exp(-r * T) * (stats.norm.cdf(d4, 0.0, 1.0) - stats.norm.cdf(d2, 0.0, 1.0) \
		- a * (stats.norm.cdf(d7, 0.0, 1.0) - stats.norm.cdf(d5, 0.0, 1.0))) \
		- S0 * (stats.norm.cdf(d3, 0.0, 1.0) - stats.norm.cdf(d1, 0.0, 1.0) \
		- b * (stats.norm.cdf(d8, 0.0, 1.0) - stats.norm.cdf(d6, 0.0, 1.0)))
	value = value.clip(0)
	return value

def dnOutCall(S0, K, r, T, sigma, Sb):
	''' Valuation of European down-and-out barrier call option in BSM model.
	Analytical formula.

	Parameters
	==========
	S0 : float
		initial stock/index level
	K : float
		strike price
	T : float
		maturity data (in year fractions)
	r : float
		constant risk-free short rate
	sigma : float
		volatility factor in diffusion term
	Sb : float
		barrirer level

	Returns
	=======
	value : float
		present value of the European option
	'''

	# assert type(S0) is float and type(Sb) is float and type(K) is float and type(r) is float and type(T) is float and type(sigma) is float, "Unrecongnized input, check type"
	a = (Sb / S0) ** (-1. + 2. * r / sigma **2)
	b = (Sb / S0) ** (1. + 2. * r / sigma **2)
	d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	d3 = (log(S0 / Sb) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	d4 = (log(S0 / Sb) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	d5 = (log(S0 / Sb) - (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	d6 = (log(S0 / Sb) - (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	d7 = (log(S0 * K / Sb ** 2) - (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	d8 = (log(S0 * K / Sb ** 2) - (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
	if K >= Sb:
		value = S0*(stats.norm.cdf(d1,0.,1.) - b*(1.-stats.norm.cdf(d8,0.,1.)))\
				- K*exp(-r*T)*(stats.norm.cdf(d2,0.,1.) - a*(1.-stats.norm.cdf(d7,0.,1.)))
	else:
		value = S0*(stats.norm.cdf(d3,0.,1.) - b*(1.-stats.norm.cdf(d6,0.,1.)))\
				- K*exp(-r*T)*(stats.norm.cdf(d4,0.,1.) - a*(1.-stats.norm.cdf(d5,0.,1.)))
	value = value.clip(0)
	return value

def test_dnOutPut():
	P0 = dnOutPut(S0=100.,K=101.,r=0.03,T=1./12.,sigma=0.2,Sb=91.)
	print "Analytical: European down-out put option (S0=100,K=101,r=0.03,T=1/12,sigma=0.2,Sb=91)"
	print "1.7088 <= MATLAB blsprice"
	print "%6.4f <= this python implementation" % P0
	assert abs(P0-1.7088) < 1e-4, "Disagree from MATLAB results."

def rear_end_dnOutPut(S, K, r, T, sig, H, tau, q):
    ''' Valuation of partial end European down-and-out barrier put option in BSM model.
	Analytical formula.

	C. H. Hui, Time-Dependent Barrier Option Values, Journal of Futures Markets, 17(6), 776-688 (1997)
	implemented by L. Wang

	Tested.

	Parameters
	==========
	S0 : float
		initial stock/index level
	K : float
		strike price
	T : float
		maturity data (in year fractions)
	r : float
		constant risk-free short rate
	sigma : float
		volatility factor in diffusion term
	Sb : float
		barrirer level
	tau : float
	    barrier starting time

	Returns
	=======
	value : float
		present value of the Barrier option

	'''
    from math import log, sqrt, exp
    
    a = lambda t: (log(S/K) + (r-q)*t) / (sig*sqrt(t)) - sig*sqrt(t)/2.
    a1 = lambda t: (log(S/K) + (r-q)*t) / (sig*sqrt(t)) + sig*sqrt(t)/2.

    b = lambda t: (log(S/H) + (r-q)*t) / (sig*sqrt(t)) - sig*sqrt(t)/2.
    b1 = lambda t: (log(S/H) + (r-q)*t) / (sig*sqrt(t)) + sig*sqrt(t)/2.

    c = lambda t: (log(H/S) + (r-q)*t) / (sig*sqrt(t)) - sig*sqrt(t)/2.
    c1 = lambda t: (log(H/S) + (r-q)*t) / (sig*sqrt(t)) + sig*sqrt(t)/2.

    d = lambda t: (log(H**2./(S*K)) + (r-q)*t) / (sig*sqrt(t)) - sig*sqrt(t)/2.
    d1 = lambda t: (log(H**2./(S*K)) + (r-q)*t) / (sig*sqrt(t)) + sig*sqrt(t)/2.

    rho = sqrt(tau/T)
    k1 = 2.*(r-q)/(sig**2.)


    P2do = K*exp(-r*T) * (bivnormcdf(b(tau),-a(T),-rho) - bivnormcdf(b(tau),-b(T),-rho)) \
	       - S*exp(-q*T) * (bivnormcdf(b1(tau),-a1(T),-rho) - bivnormcdf(b1(tau),-b1(T),-rho)) \
	       - (H/S)**(k1-1.) * K * exp(-r*T) * (bivnormcdf(-c(tau),-d(T),rho) - bivnormcdf(-c(tau),-c(T),rho)) \
	       + (H/S)**(k1+1.) * S * exp(-q*T) * (bivnormcdf(-c1(tau),-d1(T),rho) - bivnormcdf(-c1(tau),-c1(T),rho))

    return P2do

def rear_end_dnOutCall(S, K, r, T, sig, H, tau, q):
    ''' Valuation of partial end European down-and-out barrier call option in BSM model.
	Analytical formula.

	C. H. Hui, Time-Dependent Barrier Option Values, Journal of Futures Markets, 17(6), 776-688 (1997)
	implemented by L. Wang

	Tested.

	Parameters
	==========
	S0 : float
		initial stock/index level
	K : float
		strike price
	T : float
		maturity data (in year fractions)
	r : float
		constant risk-free short rate
	sigma : float
		volatility factor in diffusion term
	Sb : float
		barrirer level
	tau : float
	    barrier starting time

	Returns
	=======
	value : float
		present value of the Barrier option

	'''
    from math import log, sqrt, exp
    
    a = lambda t: (log(S/K) + (r-q)*t) / (sig*sqrt(t)) - sig*sqrt(t)/2.
    a1 = lambda t: (log(S/K) + (r-q)*t) / (sig*sqrt(t)) + sig*sqrt(t)/2.

    b = lambda t: (log(S/H) + (r-q)*t) / (sig*sqrt(t)) - sig*sqrt(t)/2.
    b1 = lambda t: (log(S/H) + (r-q)*t) / (sig*sqrt(t)) + sig*sqrt(t)/2.

    c = lambda t: (log(H/S) + (r-q)*t) / (sig*sqrt(t)) - sig*sqrt(t)/2.
    c1 = lambda t: (log(H/S) + (r-q)*t) / (sig*sqrt(t)) + sig*sqrt(t)/2.

    d = lambda t: (log(H**2./(S*K)) + (r-q)*t) / (sig*sqrt(t)) - sig*sqrt(t)/2.
    d1 = lambda t: (log(H**2./(S*K)) + (r-q)*t) / (sig*sqrt(t)) + sig*sqrt(t)/2.

    rho = sqrt(tau/T)
    k1 = 2.*(r-q)/(sig**2.)

    C2do = S*exp(-q*T) * bivnormcdf(b1(tau),a1(T),rho) - K*exp(-r*T) * bivnormcdf(b(tau),a(T),rho) \
    	   - (H/S)**(k1+1.) * S * exp(-q*T) * bivnormcdf(-c1(tau),d1(T),-rho) \
    	   + (H/S)**(k1-1.) * K * exp(-r*T) * bivnormcdf(-c(tau),d(T),-rho)

    return C2do

##########################################################################################
# Option priceing: Monte Carlo
##########################################################################################

def dnOutPut_mcs(S0, K, r, T, sigma, Sb, M=50, I=10000):
	''' Valuation of European down-and-out barrier put option in BSM model.
	Monte Carlo simulations.

	by L. Wang

	Tested: has hitting error, resulting a bias making the option price higher than true price.

	Parameters
	==========
	S0 : float
		initial stock/index level
	K : float
		strike price
	T : float
		maturity data (in year fractions)
	r : float
		constant risk-free short rate
	sigma : float
		volatility factor in diffusion term
	Sb : float
		barrirer level
	M : int
	    number of time steps
	I : int
	    number of Monte Carlo iterations

	Returns
	=======
	value : float
		present value of the Barrier option

	'''

	# assert type(S0) is float and type(Sb) is float and type(K) is float and type(r) is float and type(T) is float and type(sigma) is float, "Unrecongnized input, check type"

	dt = T / float(M)
	Y = np.zeros(M+1); Y[:] = np.log(S0)
	multi_a = (r - 0.5 * sigma ** 2) * dt; multi_b = sigma * np.sqrt(dt)
	value = np.zeros(I)
	for j in range(0, I):
		for i in range(0, M):
			sn = npr.standard_normal()
			Y[i+1] = Y[i] + multi_a + multi_b * sn
		if np.amin(Y) > np.log(Sb):
			value[j] = np.maximum(K - np.exp(Y[-1]), 0)
		else:
			value[j] = 0.
	value = np.exp(-r * T) * np.mean(value)
	return value

def dnOutCall_mcs(S0, K, r, T, sigma, Sb, M=50, I=10000):
	''' Valuation of European down-and-out barrier call option in BSM model.
	Monte Carlo simulations.

	by L. Wang

	Tested: has hitting error, resulting a bias making the option price higher than true price.

	Parameters
	==========
	S0 : float
		initial stock/index level
	K : float
		strike price
	T : float
		maturity data (in year fractions)
	r : float
		constant risk-free short rate
	sigma : float
		volatility factor in diffusion term
	Sb : float
		barrirer level
	M : int
	    number of time steps
	I : int
	    number of Monte Carlo iterations

	Returns
	=======
	value : float
		present value of the Barrier option

	'''

	# assert type(S0) is float and type(Sb) is float and type(K) is float and type(r) is float and type(T) is float and type(sigma) is float, "Unrecongnized input, check type"

	dt = T / float(M)
	Y = np.zeros(M+1); Y[:] = np.log(S0)
	multi_a = (r - 0.5 * sigma ** 2) * dt; multi_b = sigma * np.sqrt(dt)
	value = np.zeros(I)
	for j in range(0, I):
		for i in range(0, M):
			sn = npr.standard_normal()
			Y[i+1] = Y[i] + multi_a + multi_b * sn
		if np.amin(Y) > np.log(Sb):
			value[j] = np.maximum(np.exp(Y[-1]) - K, 0)
		else:
			value[j] = 0.
	value = np.exp(-r * T) * np.mean(value)
	return value


def dnOutPut_nmcs(S0, K, r, T, sigma, Sb, M=50, I=10000):
	''' Valuation of European down-and-out barrier put option in BSM model.
	New Monte Carlo simulations.

	from CS676 Asst 1, Winter 2015
	by L. Wang

	Tested: using Brownian bidge to compute the hitting probability. Lower bias is achieved.

	Parameters
	==========
	S0 : float
		initial stock/index level
	K : float
		strike price
	T : float
		maturity data (in year fractions)
	r : float
		constant risk-free short rate
	sigma : float
		volatility factor in diffusion term
	Sb : float
		barrirer level
	M : int
	    number of time steps
	I : int
	    number of Monte Carlo iterations

	Returns
	=======
	value : float
		present value of the Barrier option

	'''
	# assert type(S0) is float and type(Sb) is float and type(K) is float and type(r) is float and type(T) is float and type(sigma) is float, "Unrecongnized input, check type"

	dt = T / float(M)
	S = np.zeros(M+1); S[:] = S0
	P = np.zeros(M)
	multi_a = (r - 0.5 * sigma ** 2.) * dt; multi_b = sigma * np.sqrt(dt)
	value = np.zeros(I)
	for j in range(I):
		flag = False     # Brownin bridge prob hitting flag
		for i in range(M):
			sn = npr.standard_normal()
			S[i+1] = S[i] * np.exp(multi_a + multi_b * sn)
			#P[i] = -2. * (Sb-S[i]) * (Sb-S[i+1]) / ( S[i]**2. * multi_b**2. )
			P[i] = np.exp( -2. * (Sb-S[i]) * (Sb-S[i+1]) / ( S[i]**2. * multi_b**2. ) )
			#print P[i]
			if P[i] >= npr.uniform():
				flag = True
		if np.amin(S) > Sb and flag == False:
			value[j] = np.maximum(K - S[-1], 0)
		else:
			value[j] = 0.

	value = np.exp(-r * T) * np.mean(value)
	return value

def dnOutCall_nmcs(S0, K, r, T, sigma, Sb, M=100, I=1000):
	''' Valuation of European down-and-out barrier put option in BSM model.
	New Monte Carlo simulations.

	from CS676 Asst 1, Winter 2015
	by L. Wang

	Tested: using Brownian bidge to compute the hitting probability. Lower bias is achieved.

	Parameters
	==========
	S0 : float
		initial stock/index level
	K : float
		strike price
	T : float
		maturity data (in year fractions)
	r : float
		constant risk-free short rate
	sigma : float
		volatility factor in diffusion term
	Sb : float
		barrirer level
	M : int
	    number of time steps
	I : int
	    number of Monte Carlo iterations

	Returns
	=======
	value : float
		present value of the Barrier option

	'''
	# assert type(S0) is float and type(Sb) is float and type(K) is float and type(r) is float and type(T) is float and type(sigma) is float, "Unrecongnized input, check type"

	dt = T / float(M)
	S = np.zeros(M+1); S[:] = S0
	P = np.zeros(M)
	multi_a = (r - 0.5 * sigma ** 2.) * dt; multi_b = sigma * np.sqrt(dt)
	value = np.zeros(I)
	for j in range(I):
		flag = False     # Brownin bridge prob hitting flag
		for i in range(M):
			sn = npr.standard_normal()
			S[i+1] = S[i] * np.exp(multi_a + multi_b * sn)
			#P[i] = -2. * (Sb-S[i]) * (Sb-S[i+1]) / ( S[i]**2. * multi_b**2. )
			P[i] = np.exp( -2. * (Sb-S[i]) * (Sb-S[i+1]) / ( S[i]**2. * multi_b**2. ) )
			#print P[i]
			if P[i] >= npr.uniform():
				flag = True
		if np.amin(S) > Sb and flag == False:
			value[j] = np.maximum(S[-1] - K, 0)
		else:
			value[j] = 0.

	value = np.exp(-r * T) * np.mean(value)
	return value

def test_dnOutPut_mcs():
	P0 = dnOutPut_mcs(S0=100.,K=101.,r=0.03,T=1./12.,sigma=0.2,Sb=91.)
	print "Monte Carlo: European down-out put option (S0=100,K=101,r=0.03,T=1/12,sigma=0.2,Sb=91)"
	print "1.7088 <= MATLAB blsprice"
	print "%6.4f <= this python implementation" % P0
	assert abs(P0-1.7088) < 0.5, "Disagree from MATLAB results."


def partial_end_dnOutPut_mcs(S0, K, r, T, sigma, Sb, tau, M=1000, I=10000):
	''' Valuation of partial end European down-and-out barrier put option in BSM model.
	Monte Carlo simulations.

	implemented by L. Wang

	Tested.

	Parameters
	==========
	S0 : float
		initial stock/index level
	K : float
		strike price
	T : float
		maturity data (in year fractions)
	r : float
		constant risk-free short rate
	sigma : float
		volatility factor in diffusion term
	Sb : float
		barrirer level
	tau : float
	    barrier starting time
	M : int
	    number of time steps
	I : int
	    number of Monte Carlo iterations

	Returns
	=======
	value : float
		present value of the Barrier option

	'''

	import math
	# assert type(S0) is float and type(Sb) is float and type(K) is float and type(r) is float and type(T) is float and type(sigma) is float, "Unrecongnized input, check type"

	dt = T / float(M)
	tau_idx = math.ceil(tau*float(M)/T)
	Y = np.zeros(M+1); Y[:] = np.log(S0)
	multi_a = (r - 0.5 * sigma ** 2) * dt; multi_b = sigma * np.sqrt(dt)
	value = np.zeros(I)
	for j in range(0, I):
		for i in range(0, M):
			sn = npr.standard_normal()
			Y[i+1] = Y[i] + multi_a + multi_b * sn
		if np.amin(Y[tau_idx:]) > np.log(Sb):
			value[j] = np.maximum(K - np.exp(Y[-1]), 0)
	value = np.exp(-r * T) * np.mean(value)
	return value

def partial_end_dnOutCall_mcs(S0, K, r, T, sigma, Sb, tau, M=1000, I=10000):
	''' Valuation of partial end European down-and-out barrier call option in BSM model.
	Monte Carlo simulations.

	implemented by L. Wang

	Tested.

	Parameters
	==========
	S0 : float
		initial stock/index level
	K : float
		strike price
	T : float
		maturity data (in year fractions)
	r : float
		constant risk-free short rate
	sigma : float
		volatility factor in diffusion term
	Sb : float
		barrirer level
	tau : float
	    barrier starting time
	M : int
	    number of time steps
	I : int
	    number of Monte Carlo iterations

	Returns
	=======
	value : float
		present value of the Barrier option

	'''

	import math
	# assert type(S0) is float and type(Sb) is float and type(K) is float and type(r) is float and type(T) is float and type(sigma) is float, "Unrecongnized input, check type"

	dt = T / float(M)
	tau_idx = math.ceil(tau*float(M)/T)
	Y = np.zeros(M+1); Y[:] = np.log(S0)
	multi_a = (r - 0.5 * sigma ** 2) * dt; multi_b = sigma * np.sqrt(dt)
	value = np.zeros(I)
	for j in range(0, I):
		for i in range(0, M):
			sn = npr.standard_normal()
			Y[i+1] = Y[i] + multi_a + multi_b * sn
		if np.amin(Y[tau_idx:]) > np.log(Sb):
			value[j] = np.maximum(np.exp(Y[-1]) - K, 0)
	value = np.exp(-r * T) * np.mean(value)
	return value


def partial_end_dnOutPut_nmcs(S0, K, r, T, sigma, Sb, tau, M=1000, I=10000):
	''' Valuation of partial end European down-and-out barrier put option in BSM model.
	New Monte Carlo simulations.

	implemented by L. Wang

	Tested. Lower-bias.

	Parameters
	==========
	S0 : float
		initial stock/index level
	K : float
		strike price
	T : float
		maturity data (in year fractions)
	r : float
		constant risk-free short rate
	sigma : float
		volatility factor in diffusion term
	Sb : float
		barrirer level
	tau : float
	    barrier starting time
	M : int
	    number of time steps
	I : int
	    number of Monte Carlo iterations

	Returns
	=======
	value : float
		present value of the Barrier option

	'''
	import math
	# assert type(S0) is float and type(Sb) is float and type(K) is float and type(r) is float and type(T) is float and type(sigma) is float, "Unrecongnized input, check type"

	dt = T / float(M)
	tau_idx = math.ceil(tau*float(M)/T)
	S = np.zeros(M+1); S[:] = S0
	P = np.zeros(M)
	multi_a = (r - 0.5 * sigma ** 2.) * dt; multi_b = sigma * np.sqrt(dt)
	value = np.zeros(I)
	for j in range(I):
		flag = False     # Brownin bridge prob hitting flag
		for i in range(M):
			sn = npr.standard_normal()
			S[i+1] = S[i] * np.exp(multi_a + multi_b * sn)
			#P[i] = -2. * (Sb-S[i]) * (Sb-S[i+1]) / ( S[i]**2. * multi_b**2. )
			P[i] = np.exp( -2. * (Sb-S[i]) * (Sb-S[i+1]) / ( S[i]**2. * multi_b**2. ) )
			#print P[i]
			if P[i] >= npr.uniform() and i>= tau_idx:
				flag = True
		if np.amin(S[tau_idx:]) > Sb and flag == False:
			value[j] = np.maximum(K - S[-1], 0)
		else:
			value[j] = 0.

	value = np.exp(-r * T) * np.mean(value)
	return value

def partial_end_dnOutCall_nmcs(S0, K, r, T, sigma, Sb, tau, M=1000, I=10000):
	''' Valuation of partial end European down-and-out barrier call option in BSM model.
	New Monte Carlo simulations.

	implemented by L. Wang

	Tested. Lower-bias.

	Parameters
	==========
	S0 : float
		initial stock/index level
	K : float
		strike price
	T : float
		maturity data (in year fractions)
	r : float
		constant risk-free short rate
	sigma : float
		volatility factor in diffusion term
	Sb : float
		barrirer level
	tau : float
	    barrier starting time
	M : int
	    number of time steps
	I : int
	    number of Monte Carlo iterations

	Returns
	=======
	value : float
		present value of the Barrier option

	'''
	import math
	# assert type(S0) is float and type(Sb) is float and type(K) is float and type(r) is float and type(T) is float and type(sigma) is float, "Unrecongnized input, check type"

	dt = T / float(M)
	tau_idx = math.ceil(tau*float(M)/T)
	S = np.zeros(M+1); S[:] = S0
	P = np.zeros(M)
	multi_a = (r - 0.5 * sigma ** 2.) * dt; multi_b = sigma * np.sqrt(dt)
	value = np.zeros(I)
	for j in range(I):
		flag = False     # Brownin bridge prob hitting flag
		for i in range(M):
			sn = npr.standard_normal()
			S[i+1] = S[i] * np.exp(multi_a + multi_b * sn)
			#P[i] = -2. * (Sb-S[i]) * (Sb-S[i+1]) / ( S[i]**2. * multi_b**2. )
			P[i] = np.exp( -2. * (Sb-S[i]) * (Sb-S[i+1]) / ( S[i]**2. * multi_b**2. ) )
			#print P[i]
			if P[i] >= npr.uniform() and i>= tau_idx:
				flag = True
		if np.amin(S[tau_idx:]) > Sb and flag == False:
			value[j] = np.maximum(S[-1] - K, 0)
		else:
			value[j] = 0.

	value = np.exp(-r * T) * np.mean(value)
	return value


##########################################################################################
# Other functions
##########################################################################################

def nestedSimulationEX1(Ndata, K1=100, tn=50, n_repeat=1, sampling='mc'):
	''' Generation of nested simulation samples at the time horizon.
            Single asset example in Broadie et al (2014)

    by L. Wang

    Tested.

	Parameters
	==========
	Ndata : integer
		no. of training samples
	K1 : integer
		no. of inner loops
	tn : integer
		no. of time steps in between t = \tau and T
	n_repeat : integer
		no. of iterations for computing expectations
	sigma : float
		volatility factor in diffusion term

	Returns
	=======
	Stau : Ndata x 1 nparray
		realizations of underlying price at t = \tau
        Loss : Ndata x n_repeat nparray
                portfolio loss of corresponding Stau
        t_prep : float
                time spent in this procedure
	'''
	import time
	#from doe_lhs import lhs
	from scipy.stats.distributions import norm

    	# --- Parameters ---
	
	S0 = 100.; mu = 0.08; sigma = 0.2
	rfr = 0.03; T = 1./12.; tau = 1./52.
	K = np.array([101., 110., 114.5])
	H = np.array([91., 100., 104.5])
	pos = np.array([1., 1., -1.])
	n = len(K)    # number of options
	
	t0 = time.time()    # timer starts

	# --- portfolio price @ t = 0 ---
	V0 = np.zeros(n)
	for i in range(n):
	    V0[i] = rear_end_dnOutPut(S0, K[i], rfr, T, sigma, H[i], tau, q=0.)
	Value0 = np.sum(pos * V0)
	
	# --- portfolio loss distribution @ t = \tau ---
	
	# draw samples and generates real-world scenarios
	if sampling == 'mc':
	    sn = npr.standard_normal((Ndata, 1))    # be careful of the vector size
	elif sampling == 'lhs':
		sn = lhs(1,samples=Ndata); sn = norm(loc=0,scale=1).ppf(sn)

	
	Stau = np.zeros((Ndata, 1))
	Stau[:] = S0
	Stau = Stau * np.exp((mu - 0.5 * sigma ** 2) * tau + sigma * np.sqrt(tau) * sn)

	if n_repeat == 1:
		Vtau = np.zeros((Ndata, n))
		ValueTau = np.zeros(Ndata)   # be careful of the vector size
		for i in range(Ndata):
			for j in range(n):
				Vtau[i][j] = dnOutPut_nmcs(Stau[i], K[j], rfr, T-tau, sigma, H[j], M=tn, I=K1)
			ValueTau[i] = np.sum(pos * Vtau[i])
	else:
	    ValueTau = np.zeros((Ndata,n_repeat))   # be careful of the vector size
	    for i in range(n_repeat):
	    	Vtau = np.zeros((Ndata, n))
	    	for j in range(Ndata):
	    		for k in range(n):
	    			Vtau[j][k] = dnOutPut_nmcs(Stau[j], K[k], rfr, T-tau, sigma, H[k], M=tn, I=K1)
	    		ValueTau[j][i] = np.sum(pos * Vtau[j])
	Loss = Value0 - ValueTau
    
    
    
	t_prep = time.time() - t0    # timer off

	return Stau, Loss, t_prep


def analyticalEX1(Stau):
	''' Compute analytically the portfolio loss at t = \tau
            Single asset example in Broadie et al (2014)

    by L. Wang

    Tested.

	Parameters
	==========
	Stau : Float
		a realization of underlying price at t = \tau

	Returns
	=======
        Loss : Ndata x 1 nparray
                portfolio loss of corresponding Stau
	'''


    	# --- Parameters ---
	
	S0 = 100.; mu = 0.08; sigma = 0.2
	rfr = 0.03; T = 1./12.; tau = 1./52.
	K = np.array([101., 110., 114.5])
	H = np.array([91., 100., 104.5])
	pos = np.array([1., 1., -1.])
	n = len(K)    # number of options
	
  
    	# --- portfolio price @ t = 0 ---
    	
	Ndata = Stau.shape[0]
	V0 = np.zeros(n)
	for i in range(n):
	    V0[i] = rear_end_dnOutPut(S0, K[i], rfr, T, sigma, H[i], tau, q=0.)
	Value0 = np.sum(pos * V0)
	
	# --- portfolio loss distribution @ t = \tau ---
	
	Vtau = np.zeros((Ndata, n))
	ValueTau = np.zeros(Ndata)   # be careful of the vector size

	for i in range(Ndata):
		for j in range(n):
			Vtau[i][j] = dnOutPut(Stau[i], K[j], rfr, T-tau, sigma, H[j])
		ValueTau[i] = np.sum(pos * Vtau[i])
	Loss = Value0 - ValueTau
    

	return Loss


def opt_val(S,T,sig,IL):
    ''' Compute analytically the portfolio loss at t = \tau
            100-asset example in Broadie et al (2014)
        Monte Carlo method.

    by L. Wang


	Parameters
	==========

	Returns
	=======
        value : float
                portfolio value
    '''

    rfr = 0.05; K = 100.
    value = -10. * (blsprice_mcs(S, K, rfr, T, sig, optionType='call', M=1, I=IL)
                    + blsprice_mcs(S, K, rfr, T, sig, optionType='put', M=1, I=IL))
    return value

def opt_val_true(S,T,sig):
    ''' Compute analytically the portfolio loss at t = \tau
            100-asset example in Broadie et al (2014)
        Analytical formulas.

    by L. Wang


	Parameters
	==========

	Returns
	=======
        value : float
                portfolio value
    '''

    rfr = 0.05; K = 100.
    value = -10. * (blsprice(S, K, rfr, T, sig, optionType='call')
                    + blsprice(S, K, rfr, T, sig, optionType='put'))
    return value






if __name__ == "__main__":
	print "Running tests... "
	test_blsprice()
	test_blsprice_mcs()
	test_dnOutPut()
	test_dnOutPut_mcs()
