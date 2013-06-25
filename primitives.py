# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:29:54 2013

@author: dgevans
"""
from numpy import *

class primitives(object):
    
    beta = 0.95
    
    rho = 0.9
    
    delta = 0.1
    
    sigma_e = 1.
    
    alpha = 0.3
    
    a_min = 0.0
    
    a_max = 50.0
    
    z_min = -5.
    
    z_max = 5.
    
    
    
class primitives_CRRA(primitives):
    
    sigma = 2.0
    
    def U(self,c):
        """
        CRRA utility function
        """
        sigma = self.sigma
        if sigma == 1.:
            return log(c)
        else:
            return (c)**(1-sigma)/(1-sigma)
            
    def Uc(self,c):
        """
        Derivative of the CRRA utility function
        """
        sigma = self.sigma
        return c**(-sigma)
        
        