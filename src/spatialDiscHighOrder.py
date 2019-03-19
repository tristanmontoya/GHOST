# High Order Spatial Discretization

import numpy as np
from element import *

class SpatialDiscHighOrder:
    def __init__(self, problem, element, K):
        # Problem defines flux and source term
        # Element defines H, Q, t_L, t_R
        # K number of elements

        # physics setup
        self.n_eq = problem.n_eq
        self.F = problem.E_j
        self.Qfun = problem.H_j
        self.dFdu = problem.A_j
        self.dQdu = problem.B_j
        self.u_L = problem.Q_in
        self.u_R = problem.Q_out
        self.L = problem.length
        self.u_0 = problem.evaluateInitialCondition #function of x

        #flux jacobian
        self.A = problem.A_j

        #source term jacobian
        self.B = problem.B_j

        #roe flux function
        self.absA_roe = problem.absA_roe
        self.numFlux = problem.numericalFlux

        # element numerics
        self.p = element.p #element degree
        self.Np = element.Np #nodes per element
        self.H = np.copy(element.H) #mass/norm
        self.Hinv = np.linalg.inv(self.H)
        self.D = np.copy(element.D) #differentiation operator
        self.t_L = np.copy(element.t_L) #left projection
        self.t_R = np.copy(element.t_R) #right projection

        # grid
        self.K = K #number of elements
        self.M = self.Np * self.K #total DOF
        self.referenceGrid = element.referenceGrid
        self.referenceDx = element.dx
        self.createEqualGrid()
        self.interpolateInitialConditionOnMesh()

        # vectorized element operators
        self.H_n = np.kron(element.H, np.eye(self.n_eq))
        self.Hinv_n = np.kron(self.Hinv, np.eye(self.n_eq))
        self.D_n = np.kron(element.D, np.eye(self.n_eq))
        self.t_Ln = np.kron(element.t_L, np.eye(self.n_eq))
        self.t_Rn = np.kron(element.t_R, np.eye(self.n_eq))
        self.E_n = self.t_Rn @ self.t_Rn.T - self.t_Ln @ self.t_Ln.T

    def createEqualGrid(self):
        self.h = self.L/(self.K)
        self.mesh = np.zeros(self.M)
        for k in range(0, self.K):
            self.mesh[k * self.Np:(k + 1) * self.Np] = np.ones(self.Np) * (k * self.h) + \
                0.5 * (self.referenceGrid + np.ones(self.Np)) * self.h  # mapping each node of element k
        self.dx =0.5*self.referenceDx*self.h
        self.J = 0.5*self.dx*np.ones(self.K)
        self.Jinv = 2.0/self.dx * np.ones(self.K)

    def interpolateInitialConditionOnMesh(self):
        self.u_0_interp = np.zeros(self.M*self.n_eq)
        for i in range(0,self.M):
            self.u_0_interp[i*self.n_eq:(i+1)*self.n_eq] = self.u_0(self.mesh[i])

    def LCRq(self, u, k):
        u_km1 = u[(k-1) * self.Np * self.n_eq: k * self.Np * self.n_eq]
        u_k = u[k * self.Np * self.n_eq : (k + 1) * self.Np * self.n_eq]
        u_kp1 =  u[(k+1) * self.Np * self.n_eq: (k+2) * self.Np * self.n_eq]

        #(n_eq x Np) x (n_eq x Np)
        A_k = np.zeros([self.n_eq * self.Np,self.n_eq * self.Np ])
        A_km1 = np.zeros([self.n_eq * self.Np,self.n_eq * self.Np ])
        A_kp1 = np.zeros([self.n_eq * self.Np,self.n_eq * self.Np ])

        absA_roe_L = self.absA_roe(u_km1, u_k, k*self.dx)
        absA_roe_R = self.absA_roe(u_k, u_kp1, (k+1)*self.dx)

        q_k = np.zeros(self.Np * self.n_eq)
        for i in range(0, self.Np):
            q_k[i * self.n_eq:(i + 1) * self.n_eq] = self.Qfun(u_k[i * self.n_eq:(i + 1) * self.n_eq],
                                                               self.mesh[k * self.Np + i])
            A_k[i * self.n_eq:(i + 1) * self.n_eq, i * self.n_eq:(i + 1) * self.n_eq]\
                = self.A(u_k[i * self.n_eq:(i + 1) * self.n_eq])
            A_kp1[i * self.n_eq:(i + 1) * self.n_eq, i * self.n_eq:(i + 1) * self.n_eq] \
                = self.A(u_kp1[i * self.n_eq:(i + 1) * self.n_eq])
            A_km1[i * self.n_eq:(i + 1) * self.n_eq, i * self.n_eq:(i + 1) * self.n_eq] \
                = self.A(u_km1[i * self.n_eq:(i + 1) * self.n_eq])

        L = 0.5*self.Jinv[k] * self.Hinv_n @ (self.t_Ln @ self.t_Rn.T @ A_km1 + self.t_Ln @  absA_roe_L @ self.t_Rn.T)
        R = -0.5*self.Jinv[k] * self.Hinv_n @ (self.t_Rn @ self.t_Ln.T @ A_kp1 - self.t_Rn @ absA_roe_R @ self.t_Ln.T)
        C = -1.0*self.Jinv[k] * self.D_n @ A_k + \
            0.5*self.Jinv[k] * self.Hinv_n @ (self.E_n @ A_k - self.t_Rn @ absA_roe_R @ self.t_Rn.T \
                                          - self.t_Ln @ absA_roe_R @ self.t_Ln.T )
        return L, C, R, q_k

    def localResidualInterior(self, u, k):
        u_km1 = u[(k-1) * self.Np * self.n_eq: k * self.Np * self.n_eq]
        u_k = u[k * self.Np * self.n_eq : (k + 1) * self.Np * self.n_eq]
        u_kp1 =  u[(k+1) * self.Np * self.n_eq: (k+2) * self.Np * self.n_eq]
        L, C, R, q = self.LCRq(u, k)
        return L @ u_km1 + C @ u_k + q + R @ u_kp1

    def localResidualExplicitForm(self, u, k):
        #Left boundary
        if k == 0:
            u_k = u[k * self.Np * self.n_eq: (k + 1) * self.Np * self.n_eq]
            u_kp1 = u[(k + 1) * self.Np * self.n_eq: (k + 2) * self.Np * self.n_eq]
            q_k = np.zeros(self.Np * self.n_eq)
            f_k = np.zeros(self.Np * self.n_eq)
            f_kp1 = np.zeros(self.Np * self.n_eq)

            # nodal flux and source term
            for i in range(0, self.Np):
                q_k[i * self.n_eq:(i + 1) * self.n_eq] = self.Qfun(u_k[i * self.n_eq:(i + 1) * self.n_eq], \
                                                                   self.mesh[k * self.Np + i])
                f_k[i * self.n_eq:(i + 1) * self.n_eq] = self.F(u_k[i * self.n_eq:(i + 1) * self.n_eq])
                f_kp1[i * self.n_eq:(i + 1) * self.n_eq] = self.F(u_kp1[i * self.n_eq:(i + 1) * self.n_eq])

            absA_roe_L = self.absA_roe(self.u_L, u_k, 0.0)
            absA_roe_R = self.absA_roe(u_k, u_kp1, (k+1)*self.dx)

            f_L = self.F(self.u_L)

            F_L = 0.5 * (f_L + self.t_Ln.T @ f_k) + \
                  0.5 * absA_roe_L @ (self.u_L - self.t_Ln.T @ u_k)
            F_R = 0.5 * (self.t_Rn.T @ f_k + self.t_Ln.T @ f_kp1) + \
                  0.5 * absA_roe_R @ (self.t_Rn.T @ u_k - self.t_Ln.T @ u_kp1)

        #right boundary
        elif k == self.K-1:
            u_km1 = u[(k - 1) * self.Np * self.n_eq: k * self.Np * self.n_eq]
            u_k = u[k * self.Np * self.n_eq: (k + 1) * self.Np * self.n_eq]
            q_k = np.zeros(self.Np * self.n_eq)
            f_k = np.zeros(self.Np * self.n_eq)
            f_km1 = np.zeros(self.Np * self.n_eq)

            # nodal flux and source term
            for i in range(0, self.Np):
                q_k[i * self.n_eq:(i + 1) * self.n_eq] = self.Qfun(u_k[i * self.n_eq:(i + 1) * self.n_eq], \
                                                                   self.mesh[k * self.Np + i])
                f_km1[i * self.n_eq:(i + 1) * self.n_eq] = self.F(u_km1[i * self.n_eq:(i + 1) * self.n_eq])
                f_k[i * self.n_eq:(i + 1) * self.n_eq] = self.F(u_k[i * self.n_eq:(i + 1) * self.n_eq])

            # numerical flux
            absA_roe_L = self.absA_roe(u_km1, u_k, k*self.dx)
            absA_roe_R = self.absA_roe(u_k, self.u_R, self.L)

            f_R = self.F(self.u_R)

            F_L = 0.5 * (self.t_Rn.T @ f_km1 + self.t_Ln.T @ f_k) + \
                  0.5 * absA_roe_L @ (self.t_Rn.T @ u_km1 - self.t_Ln.T @ u_k)

            F_R = 0.5 * (self.t_Rn.T @ f_k + f_R) + \
                  0.5*absA_roe_R @ (self.t_Rn.T @ u_k - self.u_R)

        #interior
        else:
            u_km1 = u[(k - 1) * self.Np * self.n_eq: k * self.Np * self.n_eq]
            u_k = u[k * self.Np * self.n_eq: (k + 1) * self.Np * self.n_eq]
            u_kp1 = u[(k + 1) * self.Np * self.n_eq: (k + 2) * self.Np * self.n_eq]
            q_k = np.zeros(self.Np * self.n_eq)
            f_k = np.zeros(self.Np * self.n_eq)
            f_km1 = np.zeros(self.Np * self.n_eq)
            f_kp1 = np.zeros(self.Np * self.n_eq)

            # nodal flux and source term
            for i in range(0, self.Np):
                q_k[i * self.n_eq:(i + 1) * self.n_eq] = self.Qfun(u_k[i * self.n_eq:(i + 1) * self.n_eq],\
                                                                   self.mesh[k * self.Np + i])
                f_km1[i * self.n_eq:(i + 1) * self.n_eq] = self.F(u_km1[i * self.n_eq:(i + 1) * self.n_eq])
                f_k[i * self.n_eq:(i + 1) * self.n_eq] = self.F(u_k[i * self.n_eq:(i + 1) * self.n_eq])
                f_kp1[i * self.n_eq:(i + 1) * self.n_eq] = self.F(u_kp1[i * self.n_eq:(i + 1) * self.n_eq])

            # numerical flux
            absA_roe_L = self.absA_roe(u_km1, u_k, k*self.dx)
            absA_roe_R = self.absA_roe(u_k, u_kp1, (k+1)*self.dx)

            F_L = 0.5 * (self.t_Rn.T @ f_km1 + self.t_Ln.T @ f_k) + \
                  0.5 * absA_roe_L @ (self.t_Rn.T @ u_km1 - self.t_Ln.T @ u_k)
            F_R = 0.5 * (self.t_Rn.T @ f_k + self.t_Ln.T @ f_kp1) + \
                  0.5 * absA_roe_R @ (self.t_Rn.T @ u_k - self.t_Ln.T @ u_kp1)

        #return residual
        return -1.0*self.Jinv[k] *self.D_n @ f_k + q_k \
                + self.t_Rn @ (self.t_Rn.T @ f_k - F_R) - self.t_Ln @ (self.t_Ln.T @ f_k - F_L)

    def flowResidual(self, u): #time discretization calls buildFlowResidual
        res = np.zeros(self.M * self. n_eq)
        for k in range(0, self.K):
            res[k * self.Np * self.n_eq : (k + 1) * self.Np * self.n_eq] = self.localResidualExplicitForm(u, k)
        return res

    def dRdu(self, u): #time discretization calls buildFlowJacobian
        dRdu = np.zeros([self.M, self.M])

        return dRdu
