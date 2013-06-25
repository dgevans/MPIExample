# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:50:32 2013

@author: dgevans
"""

from numpy import *
from primitives import primitives_CRRA
from Spline import Spline
import bellman

Para = primitives_CRRA()

agrid = hstack((linspace(Para.a_min,Para.a_min+1,10),linspace(Para.a_min+2,Para.a_max,5)))
zgrid = linspace(Para.z_min,Para.z_max,10)

Para.domain = Spline.makeGrid((agrid,zgrid))
r = 0.95*(1/Para.beta-1)

Para.deg = [2,2]

def V0(state,d=[0,0]):
    return 0.
    
T = bellman.BellmanMap(Para,r)

Vf = bellman.approximateValueFunction(T(V0),Para)


for t in range(0,100):  
    Vfnew = bellman.approximateValueFunction(T(Vf),Para)
    print max(abs(Vfnew.getCoeffs()-Vf.getCoeffs()))
    Vf = Vfnew
