# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:40:30 2013
Holds all of the code for the bellman equation
@author: dgevans
"""
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.integrate import quad
from numpy import * 
from mpi4py import MPI
from Spline import Spline




def approximateValueFunction(V,Para):
    '''
    Approximates the value function over the grid defined by Para.domain.  Uses
    mpi.
    '''
    comm = MPI.COMM_WORLD
    #first split up domain for each 
    s = comm.Get_size()
    rank = comm.Get_rank()
    n = len(Para.domain)
    m = n/s
    r = n%s
    
    mydomain = Para.domain[rank*m+min(rank,r):(rank+1)*m+min(rank+1,r)]#split up the domain

    #get the value at each point in my domaain
    myV[0:len(mydomain)] = hstack(map(V,mydomain))
    
    Vs = comm.gather(myV)
    if rank == 0:
        Vs = hstack(Vs)
        Vf = Spline(Para.domain,Vs,Para.deg)
    else:
        Vf = None
    return comm.bcast(Vf)
    
    
    
    

class BellmanMap(object):
    '''
    Bellman map object.  Once constructed will take a value function and return
    a new value function
    '''
    
    def __init__(self,Para,r):
        '''
        Initializes the Bellman Map function.  Need parameters and r
        '''
        self.Para = Para
        self.r = r
        alpha = Para.alpha
        lrat = ( (Para.delta+r)/(1-alpha))**(1./alpha)
        self.w = alpha*lrat**(alpha-1)
        
    def __call__(self,Vf):
        '''
        Given a current value function return new value function
        '''
        self.Vf = Vf
        
        #create new value function by inegrating over new one
        def EVnew(state):
            a,z_ = state
            #get probability distributions
            Pz_min,Pz_max,z_pdf = self.get_z_distribution(z_)
            #new value function
            Vnew = lambda z: self.maximizeObjective(a,z)[0]
            z_min = self.Para.z_min
            z_max = self.Para.z_max
            return Pz_min*Vnew(z_min) + Pz_max*Vnew(z_max) + quad(lambda z: z_pdf(z)*Vnew(z),z_min,z_max)[0]
        return EVnew
        
        
    def maximizeObjective(self,a,z):
        '''
        Maximize the objective function Vf given the states a and productivity z.
        Note the state for this problem are assets and previous periods productivity
        return tuple (V,c_policy,a_policy)
        '''
        
        #total welath for agent is easy
        W = (1+self.r)*a + exp(z)*self.w
        beta = self.Para.beta
        Vf = self.Vf
        
        
        #define the derivative of the objective function
        def Df(c):
            aprime = W - c
            return self.Para.Uc(c) - beta*Vf([aprime,z],[1,0])
    
        c_min = max(0.00001,W-self.Para.a_max)
        c_max = W-self.Para.a_min
        
        if Df(c_min) <= 0:
            aprime = W-c_min
            return self.Para.U(c_min) + beta*Vf([aprime,z]),c_min,aprime
        elif Df(c_max) >= 0:
            aprime = W-c_max
            return self.Para.U(c_max) + beta*Vf([aprime,z]),c_max,aprime
        else:
            c = brentq(Df,c_min,c_max)
            aprime = W-c
            return self.Para.U(c) + beta*Vf([aprime,z]),c,aprime
            
            
    def get_z_distribution(self,z_):
        '''
        Returns three objects: the probabiliy that z == z_min, the probability
        that z = z_max and the pdf of z in between
        '''
        mu = self.Para.rho*z_
        rv = norm(mu,self.Para.sigma_e)
        Pz_min = rv.cdf(self.Para.z_min)
        Pz_max = 1-rv.cdf(self.Para.z_max)
        
        return Pz_min,Pz_max,lambda z: rv.pdf(z)/(1-Pz_min-Pz_max)
        
        
        
            
