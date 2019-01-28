import numpy as np
from scipy.optimize import fsolve

def sectionCalc(x):
    if x < 5:
        return 1. + 1.5*(1-x/5.)**2
    else:
        return 1. + 0.5*(1-x/5.)**2

def getMach(init_M, rel_S, gamma):
    data = rel_S, gamma
    M = fsolve(areaFcn, init_M, args=data)

def areaFcn(M, *data):
    rel_S, gamma = data
    return 1./M*(2./(gamma+1.)*(1. + (gamma+1.)/2. * M**2))**((gamma + 1.)/(2.*(gamma - 1.))) - rel_S

def (x, x_shock, S_star, p_01, T_01, init_M, gamma):
    n = len(x)
    M = np.zeros(n)
    p = np.zeros(n)

    for i in range(0,n): #march from inlet to outlet
        rel_S = sectionCalc(x[i])/S_star
        M[i] = getMach(init_M, rel_S, gamma)
        init_M = M[i] #update initial guess to start with adjacent point (except at inlet and after shock



