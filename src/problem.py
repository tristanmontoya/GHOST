# Tristan Montoya - Euler 1D - Problem Setup

# Defines all *physical* (not numerical) aspects of the problem

import numpy as np

class Problem:

    def __init__(self, problemType, L, gamma, R):
        self.problemType = problemType
        self.length = L #metres
        self.gamma = gamma #Cp/Cv
        self.R = R #N m/(kg K)

    def setBCs_subsonicRiemann(self, R1_in, R3_in, R2_out):
        self.R1_in = R1_in # u + 2a/(gamma-1)
        self.R3_in = R3_in # ln(p/(rho^gamma)
        self.R2_out = R2_out # u - 2a/(gamma-1)
        self.bcMode = 0

    def setBCs_allDirichlet(self, Q_in, Q_out):
        self.Q_in = Q_in
        self.Q_out = Q_out
        self.bcMode = 1


    def riemannToFlowVariables(self, R, x):
        u = 0.5*(R[0] + R[1])
        a = 0.25*(self.gamma - 1.)*(R[0] - R[1])
        rho = (a**2/(self.gamma*np.exp(R[2])))**(1./(self.gamma - 1.))
        p = rho*a**2/self.gamma
        e = p/(self.gamma - 1.) + 0.5*rho*u**2
        S = self.S(x)

        return np.array([rho*S, rho*u*S, e*S])

    def flowVariablesToRiemann(self, Q, x):
        S = self.S(x)
        rho = Q[0]/S
        u = Q[1]/Q[0]
        e = Q[2]/S
        p = (self.gamma - 1.)*(e - 0.5*rho*u**2)
        a = np.sqrt(self.gamma * p/rho)

        return np.array([u + 2*a/(self.gamma - 1.), u - 2*a/(self.gamma - 1.), np.log(p/(rho**self.gamma))])

    # section area
    def S(self, x):
        if self.problemType == 0: #quasi 1D
            return self.nozzle1(x)
        if self.problemType == 1: #shock tube
            return 1.

    # section area derivative
    def dSdx(self, x):
        if self.problemType == 0 : #quasi 1D
            return self.nozzle1_dSdx(x)
        if self.problemType == 1: #shock tube
            return 0.

    def nozzle1(self, x):
        if x < 5:
            return 1. + 1.5*(1-x/5.)**2
        else:
            return 1. + 0.5*(1-x/5.)**2

    def nozzle1_dSdx(self, x):
        if x < 5:
            return 0.12*x - 0.6
        else:
            return 0.04*x - 0.2

    #evaluate the initial condition (3-vector) at point x

    def evaluateInitialCondition(self, x):
        if self.problemType == 0: # quasi 1D
            return self.S(x)*np.array([self.rho_0, self.rhou_0, self.e_0])
        if self.problemType == 1:  # shock tube
            return self.shockTubeInitialCondition(x)

    def setUinformInitialCondition(self, rho_0, rhou_0, e_0):
        self.rho_0 = rho_0
        self.rhou_0 = rhou_0
        self.e_0 = e_0

    def setShockTubeInitialCondition(self, p_L, p_R, rho_L, rho_R, x_0):
        self.p_L = p_L
        self.p_R = p_R
        self.rho_L = rho_L
        self.rho_R = rho_R
        self.x_0 = x_0

    def shockTubeInitialCondition(self, x):
        if x < self.x_0:
            return np.array([self.rho_L, 0., self.p_L/(self.gamma - 1.)])
        else:
            return np.array([self.rho_R, 0., self.p_R / (self.gamma - 1.)])

    # Local flux Jacobian dE/dQ - note that E_j(Q_j) = A(Q_j) Q_j due to homogeneous property
    def A_j(self, Q_j):
        return np.array([[0.,1.,0.],
            [0.5*(self.gamma-3.)*(Q_j[1]/Q_j[0])**2, (3. - self.gamma)*(Q_j[1]/Q_j[0]), self.gamma-1.],
            [(self.gamma - 1.)*(Q_j[1]/Q_j[0])**3 - self.gamma*(Q_j[2]/Q_j[0])*(Q_j[1]/Q_j[0]),
             self.gamma*(Q_j[2]/Q_j[0]) - 1.5*(self.gamma-1.)*(Q_j[1]/Q_j[0])**2, self.gamma*(Q_j[1]/Q_j[0])]])

    # Source term Jacobian dH/dQ
    def B_j(self, Q_j, x):
        return (self.gamma - 1.)/self.S(x)*self.dSdx(x)*np.array([[0., 0., 0.],
            [0.5*Q_j[1]**2/Q_j[0]**2, -Q_j[1]/Q_j[0], 1.],
                         [0., 0., 0.]])

    #Local flux vector
    def E_j(self, Q_j):
        return np.array([
            Q_j[1],
            (self.gamma - 1.)*Q_j[2] + (3. - self.gamma)/2.*(Q_j[1]**2)/Q_j[0],
            self.gamma*Q_j[2]*Q_j[1]/Q_j[0] - 0.5*(self.gamma - 1.)*(Q_j[1]**3/Q_j[0]**2)])

    # source term H = p dS/dx for momentum equation
    def H_j(self, Q_j, x):
        p = (self.gamma - 1.)/self.S(x) * (Q_j[2] - 0.5*Q_j[1]**2/Q_j[0])
        return np.array([0., p*self.dSdx(x), 0.])



    # diagonalization of the flux Jacobian (use for diagonal form, must then return lambda and X)
    def eigsA_j(self, Q_j):
        A = self.A_j(Q_j)

        lamda, X = np.linalg.eig(A)

        #order u-a, u, u+a
        idx = np.argsort(lamda)
        lamda = lamda[idx]
        X = X[:,idx]

        X_inv = np.linalg.inv(X)
        X_inv_plus = np.zeros(shape=[3,3])
        X_inv_minus = np.zeros(shape=[3, 3])

        #split left and right running characteristics
        for m in range(0, 3):
            if lamda[m] > 0:
                X_inv_plus[m, :] = X_inv[m, :]
            else:
                X_inv_minus[m, :] = X_inv[m, :]

        w_plus = X_inv_plus @ Q_j
        w_minus = X_inv_minus @ Q_j

        return w_plus, w_minus, X_inv_plus, X_inv_minus

    #spectral radius of A_j
    def specrA_j(self, Q_j, x):
        q_j = Q_j/self.S(x)
        p_j = (self.gamma - 1.) * (q_j[2] - 0.5 * q_j[1] ** 2 / q_j[0])
        a_j = np.sqrt(self.gamma*(p_j / q_j[0]))
        return np.abs(q_j[1] / q_j[0]) + a_j
