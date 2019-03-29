# Tristan Montoya - Euler 1D - Problem Setup

import numpy as np

class Problem:

    def __init__(self, problemType, L, gamma, R, fluxFunction='roe'):
        self.problemType = problemType
        self.length = L #metres
        self.gamma = gamma #Cp/Cv
        self.R = R #N m/(kg K)
        self.n_eq = 3
        self.fluxFunction = fluxFunction

    def setBCs_subsonicRiemann(self, R1_in, R3_in, R2_out, useLinearExtrapolation=False):
        self.R1_in = R1_in # u + 2a/(gamma-1)
        self.R3_in = R3_in # ln(p/(rho^gamma)
        self.R2_out = R2_out # u - 2a/(gamma-1)
        if useLinearExtrapolation:
            self.bcMode = 2
        else:
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

    # diagonalization of the flux Jacobian (use for diagonal form)
    def eigsA_j(self, Q_j, x):
        S = self.S(x)
        rho = Q_j[0]/S
        u = Q_j[1]/Q_j[0]
        e = Q_j[2]/S
        p = (self.gamma - 1.)*(e - 0.5*rho*u**2)
        a = np.sqrt(self.gamma * p/rho)
        alpha = rho/(np.sqrt(2.)*a)

        Diag = np.diag([u, u+a, u-a])

        T = [[1., alpha, alpha],
             [u, alpha*(u+a), alpha*(u-a)],
             [0.5*u**2, alpha*(0.5*u**2 + u*a + a**2/(self.gamma - 1)), alpha*(0.5*u**2 - u*a + a**2/(self.gamma - 1))]]

        T_inv = np.linalg.inv(T)

        return T, T_inv, Diag

    #spectral radius of A_j
    def specrA_j(self, Q_j, x):
        q_j = Q_j/self.S(x)
        p_j = (self.gamma - 1.) * (q_j[2] - 0.5 * q_j[1] ** 2 / q_j[0])
        a_j = np.sqrt(self.gamma*(p_j / q_j[0]))
        return np.abs(q_j[1] / q_j[0]) + a_j

    #roe average of states Q_L and Q_R
    def eigsA_roe(self, Q_L, Q_R, x):
        S = self.S(x)

        #left state
        rho_L = Q_L[0] / S
        u_L = Q_L[1] / Q_L[0]
        e_L = Q_L[2] / S
        p_L = (self.gamma - 1.) * (e_L - 0.5 * rho_L * u_L ** 2)
        H_L = (e_L + p_L)/rho_L

        #right state
        rho_R = Q_R[0] / S
        u_R = Q_R[1] / Q_R[0]
        e_R = Q_R[2] / S
        p_R = (self.gamma - 1.) * (e_R - 0.5 * rho_R * u_R ** 2)
        H_R = (e_R + p_R) / rho_R

        #roe average
        rho = np.sqrt(rho_L*rho_R)
        u = (np.sqrt(rho_L)*u_L + np.sqrt(rho_R)*u_R)/(np.sqrt(rho_L) + np.sqrt(rho_R))
        H = (np.sqrt(rho_L)*H_L + np.sqrt(rho_R)*H_R)/(np.sqrt(rho_L) + np.sqrt(rho_R))
        a = np.sqrt((self.gamma - 1) * (H - 0.5*u**2))
        alpha = rho / (np.sqrt(2.) * a)

        #diagonalize
        Diag = np.diag([u, u+a, u-a])
        T = [[1., alpha, alpha],
             [u, alpha * (u + a), alpha * (u - a)],
             [0.5 * u ** 2, alpha * (0.5 * u ** 2 + u * a + a ** 2 / (self.gamma - 1)),
              alpha * (0.5 * u ** 2 - u * a + a ** 2 / (self.gamma - 1))]]
        T_inv = np.linalg.inv(T)

        return T, T_inv, Diag

    def absA_roe(self, Q_L, Q_R, x):
        T, T_inv, Diag = self.eigsA_roe(Q_L, Q_R, x)
        return T @ np.absolute(Diag) @ T_inv

    def absA_roe_entopyFix(self, Q_L, Q_R, x):
        epsilon = 1.e-2
        T, T_inv, Diag = self.eigsA_roe(Q_L, Q_R, x)
        if Diag[2,2] <= epsilon:
            Diag[2,2] = 0.5*((Diag[2,2]**2)/epsilon + epsilon)
        if Diag[1,1] <= epsilon:
            Diag[1,1] = 0.5*((Diag[1,1])**2/epsilon + epsilon)
        return T @ np.absolute(Diag) @ T_inv

    def numericalFlux(self, Q_L, Q_R, x):
        if self.fluxFunction =='roe-ef':
            return self.roeFlux_entropy_Fix(Q_L, Q_R, x)
        if self.fluxFunction == 'lf':
            return self.laxFriedrichs(Q_L, Q_R, x)
        else: #roe
            return self.roeFlux(Q_L, Q_R, x)

    def roeFlux_entropyFix(self, Q_L, Q_R, x):
            return 0.5*(self.E_j(Q_L) + self.E_j(Q_R)) - 0.5*self.absA_roe_entropyFix(Q_L, Q_R, x) @ (Q_R - Q_L)

    def roeFlux(self, Q_L, Q_R, x):
            return 0.5*(self.E_j(Q_L) + self.E_j(Q_R)) - 0.5*self.absA_roe(Q_L, Q_R, x) @ (Q_R - Q_L)

    def laxFriedrichs(self, Q_L, Q_R, x):
            return 0.5*(self.E_j(Q_L) + self.E_j(Q_R)) - 0.5*max(self.specr_j(Q_L, x), self.specr_j(Q_R, x)) * (Q_R - Q_L)
