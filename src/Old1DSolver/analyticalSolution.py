# Tristan Montoya - Analytical Solution to the Euler Equations

import numpy as np
from scipy.optimize import fsolve

def sectionCalc(x):
    if x < 5:
        return 1. + 1.5*(1-x/5.)**2
    else:
        return 1. + 0.5*(1-x/5.)**2

def sectionCalcCinf(x):
    return -1./250.*x**3 + 0.1*x**2 - 0.7*x + 2.5

def getMach(init_M, rel_S, gamma):
    data = rel_S, gamma
    M = fsolve(areaFcn, init_M, args=data)
    return M

def areaFcn(M, *data):
    rel_S, gamma = data
    return 1./M*((2./(gamma+1.)*(1. + (gamma-1.)/2. * M**2))**((gamma + 1.)/(2.*(gamma - 1.)))) - rel_S

def getP(init_P, p_L, p_R, a_L, a_R, gamma):
    data = p_L, p_R, a_L, a_R, gamma
    P = fsolve(pressureFcn, init_P, args=data)
    return P

def pressureFcn(P, *data):
    p_L, p_R, a_L, a_R, gamma = data
    alpha = (gamma + 1.)/(gamma - 1.)
    return np.sqrt(2./(gamma*(gamma - 1.)))*(P-1.)/np.sqrt(1.+alpha*P) - 2./(gamma - 1.)*a_L/a_R*(1. - (p_R/p_L*P)**((gamma - 1.)/(2.*gamma)))

def quasi1D(x, S_star_L, p_01, T_01, gamma, R, x_shock=100, s_fun=sectionCalc):
    # assume subsonic inlet, throat (not necessarily sonic) at x=5
    # if no shock, x_shock=100 so shock position not reached in nozzle

    eps = 1.e-6

    n = len(x) #number of steps

    M = np.zeros(n)
    p = np.zeros(n)
    T = np.zeros(n)
    rho = np.zeros(n)
    u = np.zeros(n)
    a = np.zeros(n)
    e = np.zeros(n)
    Q = np.zeros(3*n)

    rho_01 = p_01/(R*T_01)
    a_01 = np.sqrt(gamma*p_01/rho_01)

    shockReached = False
    p_0 = p_01
    S_star = S_star_L

    for i in range(0, n):
        if shockReached == False and x[i] > x_shock - eps:
            shockReached = True

            #apply rankine-hugionot for stagnation pressure change
            p_0 = p_01 * ( ((gamma + 1.)/2.)*M[i-1]**2/(1 + ((gamma - 1)/2.)*M[i-1]**2))**(gamma/(gamma - 1.)) / (((2*gamma/(gamma + 1.))*M[i-1]**2 - (gamma - 1.)/(gamma + 1.))**(1/(gamma-1.)))

            #update critical area
            rho_0R = p_0/(R*T_01)
            a_0R = np.sqrt(gamma*p_0/rho_0R)

            # conversion factors between critical and stagnation values cancel
            S_star = S_star_L*(rho_01*a_01)/(rho_0R*a_0R)

        # if choked, look for supersonic solution between throat and shock
        if S_star_L > 1 - eps and shockReached == False and x[i] > 5:
            init_M = 10.
        else:  # otherwise subsonic
            init_M = 0.1

        rel_S = s_fun(x[i]) / S_star

        # calculate Mach number, temperature, and pressure
        M[i] = getMach(init_M, rel_S, gamma)
        T[i] = T_01/(1. + (gamma-1)/2.*M[i]**2)
        p[i] = p_0*(1. + (gamma-1)/2.*M[i]**2)**(-1.*(gamma)/(gamma-1.))
        rho[i] = p[i]/(R*T[i])
        a[i] = np.sqrt(gamma*p[i]/rho[i])
        u[i] = a[i]*M[i]
        e[i] = rho[i]*(R/(gamma-1.)*T[i] + 0.5*u[i]**2)
        Q[i*3] = rho[i]*s_fun(x[i])
        Q[i*3+1] = rho[i]*u[i]*s_fun(x[i])

        Q[i*3+2] = e[i]*s_fun(x[i])

    return M, T, p, Q

def shockTube(x, x_0, p_L, p_R, rho_L, rho_R, t_f, gamma):
    n = len(x)  # number of steps

    M = np.zeros(n)
    rho = np.zeros(n)
    p = np.zeros(n)

    alpha = (gamma + 1.)/(gamma - 1.)

    a_L = np.sqrt(gamma * p_L/rho_L)
    a_R = np.sqrt(gamma * p_R / rho_R)

    #pressure ratio across shock
    P = getP(1.0, p_L, p_R, a_L, a_R, gamma)

    #state 2 (right of contact surface, left of shock)
    p_2 = p_R*P
    rho_2 = rho_R*(1. + alpha*P)/(alpha + P)

    #state 3 (right of expansion wave, left of contact surface)
    p_3 = p_2
    rho_3 = rho_L * (p_3 / p_L) ** (1. / gamma)

    #fluid speed on each side of contact surface (u_3 = u_2 = V)
    V = 2./(gamma - 1.)*a_L*(1 - (p_3/p_L)**((gamma-1.)/(2.*gamma)))

    #Shock speed
    C = (P-1.)*a_R**2/(gamma*V)

    for i in range(0, n):
        #left side undisturbed
        if x[i] <= x_0 - a_L*t_f:
            M[i] = 0.
            rho[i] = rho_L
            p[i] = p_L

        #in expansion fan (5)
        elif x[i] > x_0 - a_L*t_f and x[i] <= x_0 + (V*(gamma+1.)/2. - a_L)*t_f:
            u = 2./(gamma + 1.)*((x[i] - x_0)/t_f + a_L)
            a = u - (x[i]-x_0)/t_f
            M[i] = u/a
            p[i] = p_L*(a/a_L)**((2.*gamma)/(gamma-1.))
            rho[i] = gamma*p[i]/a**2

        #left of contact surface (3)
        elif  x[i] > x_0 + (V*(gamma+1.)/2. - a_L)*t_f and x[i] <= x_0 + V*t_f:
            rho[i] = rho_3
            p[i] = p_3
            M[i] = V/np.sqrt(gamma*p_3/rho_3)

        #right of contact surface (2)
        elif x[i] > x_0 + V*t_f and x[i] <= x_0 + C * t_f:
            rho[i] = rho_2
            p[i] = p_2
            M[i] = V / np.sqrt(gamma * p_2 / rho_2)

        #right side undisturbed
        elif x[i] > x_0 + C*t_f:
            M[i] = 0.
            rho[i] = rho_R
            p[i] = p_R

    return M, rho, p