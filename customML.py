"""
Machine learning module for Risk Management research at the Scientific Computing group 
at University of Waterloo

Developed by Luyu Wang
2015-2016
"""
from __future__ import division
import numpy as np
from sklearn import linear_model
from sklearn.base import BaseEstimator
from sklearn.utils import gen_even_slices
from sklearn.base import BaseEstimator


def naivePolyFeature(X,deg,norm=False):
    S0 = 100.
    I = X.shape[0]
    D = X.shape[1]
    transMat = np.zeros((I,deg*D+1))
    transMat[:,0] = 1.
    if deg != 0 and norm == False:
        for i in range(deg):
            transMat[:,i*D+1:(i+1)*D+1] = X**(i+1)
    elif deg != 0 and norm == True:
        for i in range(deg):
            transMat[:,i*D+1:(i+1)*D+1] = (X/S0)**(i+1)
    return transMat


def specifiedFeature(X,deg,norm=False):
    S0 = 100.
    H = np.array([104.5,100.,91.])
    I = X.shape[0]
    D = X.shape[1]
    transMat = np.zeros((I,1+deg*D+2*deg*D))
    transMat[:,0] = 1.
    if deg != 0 and norm == False:
        for i in range(deg):
            transMat[:,i*D+1:(i+1)*D+1] = X**(i+1)
        for i in range(deg,2*deg):
            transMat[:,i*D+1:(i+1)*D+1] = np.maximum(X-H[0],0.)**(i-1)
        for i in range(2*deg,3*deg):
            transMat[:,i*D+1:(i+1)*D+1] = np.maximum(X-H[1],0.)**(i-3)
        #for i in range(3*deg,4*deg):
            #transMat[:,i*D+1:(i+1)*D+1] = np.maximum(X-H[2],0.)**(i-5)
    elif deg != 0 and norm == True:
        for i in range(deg):
            transMat[:,i*D+1:(i+1)*D+1] = (X/S0)**(i+1)
        for i in range(deg,2*deg):
            transMat[:,i*D+1:(i+1)*D+1] = (np.maximum(X-H[0],0.)/S0)**(i-1)
        for i in range(2*deg,3*deg):
            transMat[:,i*D+1:(i+1)*D+1] = (np.maximum(X-H[1],0.)/S0)**(i-3)
        #for i in range(3*deg,4*deg):
            #transMat[:,i*D+1:(i+1)*D+1] = (np.maximum(X-H[2],0.)/S0)**(i-5)
    return transMat


def normalizeX(Xtrain,Xpred):
    Xbar = np.mean(Xtrain,axis=0)
    Xstd = np.std(Xtrain,axis=0)
    Xtrain[:,1:] = (Xtrain[:,1:]-Xbar[1:]) / Xstd[1:]
    Xpred[:,1:] = (Xpred[:,1:]-Xbar[1:]) / Xstd[1:]
    return (Xtrain,Xpred)
    

class PolynomialRegression(BaseEstimator):
    def __init__(self, deg, norm=False, fit_intercept=True):
        self.deg = deg
        self.fit_intercept = fit_intercept
        self.norm = norm

    def fit(self, X, y):
        self.model = linear_model.LinearRegression(fit_intercept=self.fit_intercept)
        self.model.fit(naivePolyFeature(X,deg=self.deg,norm=self.norm), y)
    
    def predict(self, X_pred):
        return self.model.predict(naivePolyFeature(X_pred,deg=self.deg,norm=self.norm))
    
    @property
    def coef_(self):
        return self.model.coef_

class PolynomialRidge(BaseEstimator):
    def __init__(self, deg, alpha, norm=False):
        self.deg = deg
        self.alpha = alpha
        self.norm = norm

    def fit(self, X, y):
        self.model = linear_model.Ridge(alpha=self.alpha)
        self.model.fit(naivePolyFeature(X,deg=self.deg,norm=self.norm), y)
    
    def predict(self, X_pred):
        return self.model.predict(naivePolyFeature(X_pred,deg=self.deg,norm=self.norm))
    
    @property
    def coef_(self):
        return self.model.coef_


class FeatureRegression(BaseEstimator):
    def __init__(self, deg):
        self.deg = deg

    def fit(self, X, y):
        self.model = linear_model.LinearRegression(fit_intercept=True)
        self.model.fit(specifiedFeature(X,deg=self.deg), y)
    
    def predict(self, X_pred):
        return self.model.predict(specifiedFeature(X_pred,deg=self.deg))
    
    @property
    def coef_(self):
        return self.model.coef_

class FeatureRidge(BaseEstimator):
    def __init__(self, deg, alpha):
        self.deg = deg
        self.alpha = alpha

    def fit(self, X, y):
        self.model = linear_model.Ridge(alpha=self.alpha)
        self.model.fit(specifiedFeature(X,deg=self.deg), y)
    
    def predict(self, X_pred):
        return self.model.predict(specifiedFeature(X_pred,deg=self.deg))
    
    @property
    def coef_(self):
        return self.model.coef_


def LinearSpline(x,y,Lb=0.):
    """
    Kernel Generating Linear Splines
    """
    N=x.shape[0]
    N2=y.shape[0]
    XMatrix=np.tile(x,(N2,1)).T
    YMatrix=np.tile(y,(N,1))
    MinMat=np.minimum(XMatrix,YMatrix)
    K=1.+np.multiply(XMatrix,YMatrix)+1./2.*np.multiply(np.absolute(XMatrix-YMatrix),np.square(MinMat+Lb))+1./3.*np.power(MinMat+Lb,3)
    return K

def linSplKer(x,y,lb=0.):
    """
    Kernel Generating Linear Splines given two vectors of the same dimansion
    """
    N1 = x.shape[0]
    N2 = y.shape[0]
    minTerm = np.minimum(x,y)
    Kvec = 1. + x*y + 0.5*np.absolute(x-y)*minTerm**2 + 1./3.*minTerm**3
    K = 1.
    for i in range(N1):
        K *= Kvec[i]
    return K

def linSpl(x,y,Lb=0.):
    """
    Kernel Generating Linear Splines for 1D
    """
    temp = np.minimum(x,y)
    K= 1. + x*y + 1./2.*np.absolute(x-y)*temp**2 + 1./3.*temp**3
    return K

def splGram(X1, X2):
    """
    Construct the gram matrix for 1D
    """
    N1=X1.shape[0]
    N2=X2.shape[0]
    gram = np.zeros((N1, N2))
    for i in range(N1):
        gram[i] = linSpl(X1[i],X2).reshape(N2)
    #gram = Parallel(n_jobs=num_cores)(delayed(linSpl)(x,X2) for x in X1)
    return gram

def splKerGram(X,Y):
    N1 = X.shape[0]
    N2 = Y.shape[0]
    gram = np.zeros((N1, N2))
    for i in range(N1):
        # gram[i,:] = map(linSplKer, )
        for j in range(N2):
            gram[i, j] = linSplKer(X[i,:], Y[j,:])
    return gram