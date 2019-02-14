# Tristan Montoya - Euler 1D - Problem Setup
# Defines all *physical* (not numerical) aspects of the problem

import numpy as np

class Problem:

    def __init__(self, problemType):
        self.problemType = problemType
        self.length = 10.0 #m
        self.gamma = 1.4
        self.R = 287.0 #N m/(kg K)

        self.inletIsSupersonic = False
        self.exitIsSupersonic = False

    def setSubsonicInlet(self, rho, rhou):
        self.rho_inlet = rho
        self.rhou_inlet = rhou
        self.inletIsSupersonic = False

    def setSubsonicExit(self, p):
        self.p_exit=p
        self.exitIsSupersonic = False

    def S(self, x):
        if self.problemType == 0 or self.problemType == 1:
            return self.nozzle1(x)
        if self.problemType == 2:
            return 1.
    def dSdx(self, x):
        if self.problemType == 0 or self.problemType == 1:
            return self.nozzle1_dSdx(x)
        if self.problemType == 2:
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

    # Local flux Jacobian - note that E_j(Q_j) = A(Q_j) Q_j due to homogeneous property
    def A_j(self, Q_j):
        return np.array([[0.,1.,0.],
            [0.5*(self.gamma-3.)*(Q_j[1]/Q_j[0])**2, (3. - self.gamma)*(Q_j[1]/Q_j[0]), self.gamma-1.],
            [(self.gamma - 1.)*(Q_j[1]/Q_j[0])**3 - self.gamma*(Q_j[2]/Q_j[0])*(Q_j[1]/Q_j[0]),
             self.gamma*(Q_j[2]/Q_j[0]) - 1.5*(self.gamma-1.)*(Q_j[1]/Q_j[0])**2, self.gamma*(Q_j[1]/Q_j[0])]])

    #Local flux vector
    def E_j(self, Q_j):
        return np.array([
            Q_j[1],
            (self.gamma - 1.)*Q_j[2] + (3. - self.gamma)/2.*(Q_j[1]**2)/Q_j[0],
            self.gamma*Q_j[2]*Q_j[1]/Q_j[0] - 0.5*(self.gamma - 1.)*(Q_j[1]**3/Q_j[0]**2)])

    # source term H = p dS/dx for momentum equation
    def H_j(self, Q_j, x):
        p = (self.problem.gamma - 1.)/self.problem.S(x) * (Q_j[2] - 0.5*Q_j[1]**2/Q_j[0])
        return np.array([0., p*self.problem.dSdx(x), 0.])

    # diagonalization of the flux Jacobian
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
