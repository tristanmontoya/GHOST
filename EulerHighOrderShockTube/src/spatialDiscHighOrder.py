# Tristan Montoya - High Order Spatial Discretization

import numpy as np
from element import *
import scipy as sp

class SpatialDiscHighOrder:
    def __init__(self, problem, element, K, limiterType = 'muscl',
                 limiterVars = 'char', minmodfun='tvb'):
        # Problem defines flux and source term
        # Element defines H, Q, t_L, t_R
        # K number of elements

        # physics setup
        self.problem = problem
        self.n_eq = problem.n_eq
        self.F = problem.E_j
        self.Qfun = problem.H_j
        self.dFdu = problem.A_j
        self.dQdu = problem.B_j
        self.S = problem.S
        self.u_L = problem.Q_in
        self.u_R = problem.Q_out
        self.L = problem.length
        self.u_0 = problem.evaluateInitialCondition #function of x
        self.maxWave = problem.specrA_j

        #flux jacobian
        self.A = problem.A_j

        #source term jacobian
        self.B = problem.B_j

        #flux function (anything can be used for explicit methods,
        #but linearization only implemented with Roe scheme for now)
        self.absA_roe = problem.absA_roe
        self.numFlux = problem.numericalFlux
        self.limiterType = limiterType
        if minmodfun == 'tvb':
            self.minmod = self.TVBminmod
        else:
            self.minmod = self.TVDminmod

        self.limiterVars = limiterVars
        if self.limiterVars == 'char':
            self.generalizedLimiter = self.averagedCharacteristicLimiter2
        else:
            self.generalizedLimiter = self.conservativeLimiter

        # element numerics
        self.p = element.p #element degree
        self.Np = element.Np #nodes per element
        self.H = np.copy(element.H) #mass/norm
        self.V = np.copy(element.V) #Legendre Vandermonde (not implemented for CSBP)
        if self.Np == self.p + 1:
            self.Vinv = np.linalg.inv(self.V) #inverses
        self.Hinv = np.linalg.inv(self.H)
        self.D = np.copy(element.D) #differentiation operator
        self.t_L = np.copy(element.t_L) #left projection
        self.t_R = np.copy(element.t_R) #right projection

        # grid
        self.K = K #number of elements
        self.M = self.Np * self.K #total DOF per equation
        self.referenceGrid = element.referenceGrid
        self.referenceDx = element.dx
        self.createEqualGrid()
        self.interpolateInitialConditionOnMesh()

        # vectorized element operators
        self.H_n = np.kron(element.H, np.eye(self.n_eq))
        self.Hinv_n = np.kron(self.Hinv, np.eye(self.n_eq))
        self.D_n = np.kron(element.D, np.eye(self.n_eq))
        self.one = np.kron(np.ones(self.Np), np.eye(self.n_eq))
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
        self.J = 0.5*self.h*np.ones(self.K)
        self.Jinv = 2.0/self.h * np.ones(self.K)

    def interpolateInitialConditionOnMesh(self):
        self.u_0_interp = np.zeros(self.M*self.n_eq)
        for i in range(0,self.M):
            self.u_0_interp[i*self.n_eq:(i+1)*self.n_eq] = self.u_0(self.mesh[i])

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

            F_L = self.numFlux(self.u_L, self.t_Ln.T @ u_k, 0.0)
            F_R = self.numFlux(self.t_Rn.T @ u_k, self.t_Ln.T @ u_kp1,(k+1)*self.h)

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

            F_L = self.numFlux(self.t_Rn.T @ u_km1, self.t_Ln.T @ u_k, k*self.h)
            F_R = self.numFlux(self.t_Rn.T @ u_k, self.u_R,  self.L)

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

            F_L = self.numFlux(self.t_Rn.T @ u_km1, self.t_Ln.T @ u_k, (k)*self.h)
            F_R = self.numFlux(self.t_Rn.T @ u_k, self.t_Ln.T @ u_kp1,(k+1)*self.h)

        #return residual
        return -1.0*self.Jinv[k] *self.D_n @ f_k + q_k \
                + self.Jinv[k] * self.Hinv_n @ (self.t_Rn @ (self.t_Rn.T @ f_k - F_R) - self.t_Ln @ (self.t_Ln.T @ f_k - F_L))

    def flowResidual(self, u): #time discretization calls buildFlowResidual
        res = np.zeros(self.M * self. n_eq)
        for k in range(0, self.K):
            res[k * self.Np * self.n_eq : (k + 1) * self.Np * self.n_eq] = self.localResidualExplicitForm(u, k)
        return res

    # LIMITER CODE STARTS HERE

    def TVDminmod(self, a):
        eps0 = 1.e-8
        m = len(a)
        s = np.sum(np.sign(a)) / m
        if np.absolute(np.absolute(s) - 1) < eps0:
            return s * np.amin(np.absolute(a))
        return 0.0

    def TVBminmod(self, a):
        if np.abs(a[0]) <= 20.0*self.h**2:
            return a[0]
        else:
            return self.TVDminmod(a)

    def applyLimiter(self, k, u_k, ubar_km1, ubar_k, ubar_kp1, returnActive=False):
        # make linear (project down to linear, need to add a Vandermonde approximation for general SBP)
        if self.Np > 2:
            u_hat = self.Vinv @ u_k
            u_hat[2:self.Np] = np.zeros(self.Np - 2)
            u_linear = self.V @ u_hat

        slope = (self.t_R.T @ u_linear - self.t_L.T @ u_linear) / self.h
        x_c = self.h * k + 0.5 * self.h
        x = self.h * k + 0.5 * self.h * (self.referenceGrid + 1.0)

        if self.limiterType == 'cs': #cockburn shu
            limitedSlope = self.minmod(np.array([slope, 2.0*(ubar_kp1 - ubar_k) / self.h, 2.0*(ubar_k - ubar_km1) / self.h]))
        elif self.limiter == 'muscl':
            limitedSlope = self.minmod(np.array([slope, (ubar_kp1 - ubar_k) / self.h, (ubar_k - ubar_km1) / self.h]))

        # compute limited value

        if returnActive:
            if np.abs(slope-limitedSlope) <= 1.e-8:
                active=False
            else:
                active=True
            return ubar_k + (x - x_c) * limitedSlope, active

        return ubar_k + (x - x_c) * limitedSlope

    def getAverages(self, u):
        ubar = np.zeros(self.n_eq * self.K)
        # print('ubar length')
        for k in range(0, self.K):
            ubar[self.n_eq * k: self.n_eq * (k + 1)] = (1. / self.h) * self.J[k] * self.one @ \
                                                       self.H_n @ u[k * self.Np * self.n_eq: (k + 1) * self.Np * self.n_eq]
        return ubar

    def getEigsAvg(self,ubar):
        X = np.zeros(shape=(self.K, self.n_eq,self.n_eq))

        Xinv = np.zeros(shape=(self.K, self.n_eq,self.n_eq))
        for k in range(0, self.K):
            X[k,:,:], Xinv[k,:,:], Diag = self.problem.eigsA_j(ubar[self.n_eq * k: self.n_eq * (k + 1)], (k+0.5)*self.h)

        return X,Xinv

    #Conservative-variable limiter (Hesthaven and Warburton)
    def conservativeLimiter(self, u):
        eps0 = 1.e-8
        ubar = self.getAverages(u)
        isLimited = np.zeros(self.K)
        for k in range(1, self.K - 1):
            for j in range(0, self.n_eq):
                ubar_km1 = ubar[self.n_eq * (k - 1) + j]
                ubar_k = ubar[self.n_eq * (k) + j]
                ubar_kp1 = ubar[self.n_eq * (k + 1) + j]
                u_k = u[k * self.Np * self.n_eq + j: (k + 1) * self.Np * self.n_eq: self.n_eq]

                # check if limiting needed
                v_L = ubar_k - self.minmod(np.array([ubar_k - self.t_L.T @ u_k, ubar_k - ubar_km1, ubar_kp1 - ubar_k]))
                v_R = ubar_k + self.minmod(np.array([self.t_R.T @ u_k - ubar_k, ubar_k - ubar_km1, ubar_kp1 - ubar_k]))

                if np.abs(v_L - self.t_L.T @ u_k) < eps0 and np.abs(v_R - self.t_R.T @ u_k) < eps0:
                    continue  # no limiting needed

                # apply the limiter to equation j of element k
                isLimited[k] = 1
                u[k * self.Np * self.n_eq + j: (k + 1) * self.Np * self.n_eq: self.n_eq] = self.applyLimiter(k, u_k, ubar_km1,
                                                                                                             ubar_k,
                                                                                                             ubar_kp1)
        return u, isLimited

    #Apply limiter to characteristic variables at each node (expensive and doesn't work very well)
    def characteristicLimiterEachNode(self, u):
        X = np.zeros(shape=(self.M, self.n_eq, self.n_eq))
        w = np.zeros(self.M*self.n_eq)
        for i in range(0,self.M):
            #transform every DOF to local characteristic variables
            X[i,:,:],Xinv,Diag = self.problem.eigsA_j(u[i*self.n_eq:(i+1)*self.n_eq],self.mesh[i])
            w[i * self.n_eq:(i + 1) * self.n_eq] = Xinv @ u[i * self.n_eq:(i + 1) * self.n_eq]

        w, isLimited = self.conservativeLimiter(w)

        for i in range(0,self.M):
            #transform every DOF back
            u[i * self.n_eq:(i + 1 )* self.n_eq] = X[i,:,:]@ w[i * self.n_eq:(i + 1) * self.n_eq]

        return u, isLimited

    #best one so far - characteristic variables limited by linearizing about A(ubar_k)
    def averagedCharacteristicLimiter2(self, u):
        ubar = self.getAverages(u)
        X,Xinv = self.getEigsAvg(ubar)
        isLimited = np.zeros(self.K)
        for k in range(1, self.K - 1):
            ubar_km1 = ubar[self.n_eq * (k - 1): self.n_eq * (k)]
            ubar_k = ubar[self.n_eq * (k) : self.n_eq * (k + 1) ]
            ubar_kp1 = ubar[self.n_eq * (k + 1)  : self.n_eq * (k + 2) ]
            u_k = u[k * self.Np * self.n_eq: (k + 1) * self.Np * self.n_eq]

            wbar_km1 = Xinv[k,:,:] @ ubar_km1
            wbar_k = Xinv[k,:,:] @ ubar_k
            wbar_kp1 = Xinv[k,:,:] @ ubar_kp1

            w_k = np.zeros(self.Np*self.n_eq)
            for i in range(0,self.Np):
                w_k[i*self.n_eq:(i+1)*self.n_eq] = Xinv[k,:,:] @ u_k[i*self.n_eq:(i+1)*self.n_eq]

            for j in range(0, self.n_eq):
                # check if limiting needed
                limitedVars, isActive = self.applyLimiter(k, w_k[j::self.n_eq], wbar_km1[j], wbar_k[j], wbar_kp1[j], returnActive=True)

                if isActive == False:
                    continue  # no limiting needed

                # apply the limiter to equation j of element k
                isLimited[k] = 1
                w_k[j::self.n_eq] = limitedVars

            #transform back
            for i in range(0,self.Np):
                u[k * self.Np * self.n_eq + i*self.n_eq : k * self.Np * self.n_eq + \
                                    (i+1)*self.n_eq] = X[k,:,:] @ w_k[i*self.n_eq:(i+1)*self.n_eq]

        return u, isLimited

    #Dont need this stuff for A4 ---------------------------------------------------------------------------------------

    def LCRq(self, u, k, returnQ=False):
        u_km1 = u[(k - 1) * self.Np * self.n_eq: k * self.Np * self.n_eq]
        u_k = u[k * self.Np * self.n_eq: (k + 1) * self.Np * self.n_eq]
        u_kp1 = u[(k + 1) * self.Np * self.n_eq: (k + 2) * self.Np * self.n_eq]

        # (n_eq x Np) x (n_eq x Np)
        A_k = np.zeros([self.n_eq * self.Np, self.n_eq * self.Np])
        A_km1 = np.zeros([self.n_eq * self.Np, self.n_eq * self.Np])
        A_kp1 = np.zeros([self.n_eq * self.Np, self.n_eq * self.Np])

        absA_roe_L = self.absA_roe(u_km1, u_k, k * self.h)
        absA_roe_R = self.absA_roe(u_k, u_kp1, (k + 1) * self.h)

        q_k = np.zeros(self.Np * self.n_eq)
        for i in range(0, self.Np):
            if returnQ:
                q_k[i * self.n_eq:(i + 1) * self.n_eq] = self.Qfun(u_k[i * self.n_eq:(i + 1) * self.n_eq],
                                                                   self.mesh[k * self.Np + i])
            A_k[i * self.n_eq:(i + 1) * self.n_eq, i * self.n_eq:(i + 1) * self.n_eq] \
                = self.A(u_k[i * self.n_eq:(i + 1) * self.n_eq])
            A_kp1[i * self.n_eq:(i + 1) * self.n_eq, i * self.n_eq:(i + 1) * self.n_eq] \
                = self.A(u_kp1[i * self.n_eq:(i + 1) * self.n_eq])
            A_km1[i * self.n_eq:(i + 1) * self.n_eq, i * self.n_eq:(i + 1) * self.n_eq] \
                = self.A(u_km1[i * self.n_eq:(i + 1) * self.n_eq])

        L = 0.5 * self.Jinv[k] * self.Hinv_n @ (self.t_Ln @ self.t_Rn.T @ A_km1 + self.t_Ln @ absA_roe_L @ self.t_Rn.T)
        R = -0.5 * self.Jinv[k] * self.Hinv_n @ (self.t_Rn @ self.t_Ln.T @ A_kp1 - self.t_Rn @ absA_roe_R @ self.t_Ln.T)
        C = -1.0 * self.Jinv[k] * self.D_n @ A_k + \
            0.5 * self.Jinv[k] * self.Hinv_n @ (self.E_n @ A_k - self.t_Rn @ absA_roe_R @ self.t_Rn.T \
                                                - self.t_Ln @ absA_roe_L @ self.t_Ln.T)
        if returnQ:
            return L, C, R, q_k

        return L, C, R

    def left_old(self, u, k):
        u_km1 = u[(k - 1) * self.Np * self.n_eq: k * self.Np * self.n_eq]
        u_k = u[k * self.Np * self.n_eq: (k + 1) * self.Np * self.n_eq]

        A_km1 = np.zeros([self.n_eq * self.Np, self.n_eq * self.Np])
        absA_roe_L = self.absA_roe(u_km1, u_k, k * self.h)

        for i in range(0, self.Np):
            A_km1[i * self.n_eq:(i + 1) * self.n_eq, i * self.n_eq:(i + 1) * self.n_eq] \
                = self.A(u_km1[i * self.n_eq:(i + 1) * self.n_eq])

        return 0.5 * self.Jinv[k] * self.Hinv_n @ (self.t_Ln @ self.t_Rn.T @ A_km1 + \
                                                   self.t_Ln @ absA_roe_L @ self.t_Rn.T)

    def centre_old(self, u, k):

        u_k = u[k * self.Np * self.n_eq: (k + 1) * self.Np * self.n_eq]

        if k != 0:
            u_km1 = u[(k - 1) * self.Np * self.n_eq: k * self.Np * self.n_eq]
        if k != self.K - 1:
            u_kp1 = u[(k + 1) * self.Np * self.n_eq: (k + 2) * self.Np * self.n_eq]

        A_k = np.zeros([self.n_eq * self.Np, self.n_eq * self.Np])

        if k == 0:
            absA_roe_L = self.absA_roe(self.u_L, u_k, 0.0)
        else:
            absA_roe_L = self.absA_roe(u_km1, u_k, k * self.h)
        if k == self.K - 1:
            absA_roe_R = self.absA_roe(u_k, self.u_R, self.L)
        else:
            absA_roe_R = self.absA_roe(u_k, u_kp1, (k + 1) * self.h)

        for i in range(0, self.Np):
            A_k[i * self.n_eq:(i + 1) * self.n_eq, i * self.n_eq:(i + 1) * self.n_eq] \
                = self.A(u_k[i * self.n_eq:(i + 1) * self.n_eq])

        return -1.0 * self.Jinv[k] * self.D_n @ A_k + \
               0.5 * self.Jinv[k] * self.Hinv_n @ (self.E_n @ A_k - self.t_Rn @ absA_roe_R @ self.t_Rn.T \
                                                   - self.t_Ln @ absA_roe_L @ self.t_Ln.T)


    def right_old(self, u, k):
        u_k = u[k * self.Np * self.n_eq: (k + 1) * self.Np * self.n_eq]
        u_kp1 = u[(k + 1) * self.Np * self.n_eq: (k + 2) * self.Np * self.n_eq]
        A_kp1 = np.zeros([self.n_eq * self.Np, self.n_eq * self.Np])
        absA_roe_R = self.absA_roe(u_k, u_kp1, (k + 1) * self.h)

        for i in range(0, self.Np):
            A_kp1[i * self.n_eq:(i + 1) * self.n_eq, i * self.n_eq:(i + 1) * self.n_eq] \
                = self.A(u_kp1[i * self.n_eq:(i + 1) * self.n_eq])

        return -0.5 * self.Jinv[k] * self.Hinv_n @ (
                    self.t_Rn @ self.t_Ln.T @ A_kp1 - self.t_Rn @ absA_roe_R @ self.t_Ln.T)


    def sourceTermJacobian(self, u, k):
        u_k = u[k * self.Np * self.n_eq: (k + 1) * self.Np * self.n_eq]
        B_k = np.zeros([self.n_eq * self.Np, self.n_eq * self.Np])
        for i in range(0, self.Np):
            B_k[i * self.n_eq:(i + 1) * self.n_eq, i * self.n_eq:(i + 1) * self.n_eq] = \
                self.B(u_k[i * self.n_eq:(i + 1) * self.n_eq], self.mesh[k * self.Np + i])
        return B_k

    def localResidualInterior(self, u, k):
        u_km1 = u[(k - 1) * self.Np * self.n_eq: k * self.Np * self.n_eq]
        u_k = u[k * self.Np * self.n_eq: (k + 1) * self.Np * self.n_eq]
        u_kp1 = u[(k + 1) * self.Np * self.n_eq: (k + 2) * self.Np * self.n_eq]
        L, C, R, q = self.LCRq(u, k, returnQ=True)
        return L @ u_km1 + C @ u_k + q + R @ u_kp1

    def dRdu(self, u): #time discretization calls buildFlowJacobian
        jacobianData = np.zeros([3 * self.K - 2, self.Np *self.n_eq, self.Np *self.n_eq])

        #interior elements
        for k in range(1,self.K-1):
            jacobianData[k*3 - 1], jacobianData[k*3], jacobianData[k*3 + 1] = self.LCRq(u, k)
            jacobianData[k*3] = jacobianData[k*3] + self.sourceTermJacobian(u,k)


        #top left
        jacobianData[0] = self.centre_old(u,0) + self.sourceTermJacobian(u,0)
        jacobianData[1] = self.right_old(u,0)

        #bottom right
        jacobianData[(self.K - 1)*3 - 1] = self.left_old(u, self.K-1)
        jacobianData[(self.K - 1) * 3] = self.centre_old(u, self.K - 1) + self.sourceTermJacobian(u,0)

        # print(jacobianData)

        indptr = np.empty(self.K + 1, dtype=int)
        index = np.empty(3 * self.K - 2)

        indptr[0] = 0
        indptr[1] = 2

        for i in range(2, self.K):
            indptr[i] = indptr[i - 1] + 3
            # print(i, self.K)
            # print(indptr[i])

        indptr[self.K] = indptr[self.K - 1] + 2

        # print("total nonzero blocks: ", indptr[self.K])
        # print("should be: ", 3*self.K - 2)

        # top left
        index[0] = 0
        index[1] = 1

        # bottom right
        index[(self.K - 1) * 3 - 1] = self.K - 2
        index[(self.K - 1) * 3] = self.K - 1

        # interior elements
        for i in range(1, self.K - 1):
            index[i * 3 - 1] = i - 1
            index[i * 3] = i
            index[i * 3 + 1] = i + 1

        dRdu = sp.sparse.bsr_matrix((jacobianData, index, indptr), shape=(self.K * self.Np *self.n_eq, self.K * self.Np *self.n_eq))
        # print(dRdu.toarray())

        return dRdu

    def testFlowJacobianFiniteDifference(self, u):
        residual = lambda u, row : self.flowResidual(u)[row]
        jacobian = np.zeros(shape=[self.M*self.n_eq, self.M*self.n_eq])
        for i in range(0, self.M*3):
            jacobian[i,:] = sp.optimize.approx_fprime(u, residual, np.sqrt(np.finfo(np.float).eps), i)
        return jacobian

    def calculateError(self,u, u_exact):
        normsquared = 0.0
        for k in range(0,self.K):
            error = u[k * self.Np : (k + 1) * self.Np] - u_exact[k * self.Np : (k + 1) * self.Np]
            normsquared = normsquared + error.T @ (self.J[k]*self.H) @ error

        return np.sqrt(normsquared)
