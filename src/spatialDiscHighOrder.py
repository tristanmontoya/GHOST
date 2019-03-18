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
        self.Q = problem.H_j
        self.dFdu = problem.A_j
        self.dQdu = problem.B_j
        self.u_L = problem.Q_in
        self.u_R = problem.Q_out
        self.L = problem.length

        #roe flux function
        self.absA_roe = problem.absA_roe

        # element numerics
        self.p = element.p #element degree
        self.Np = element.Np #nodes per element
        self.H = np.copy(element.H) #mass/norm
        self.D = np.copy(element.D) #differentiation operator
        self.Q = np.copy(element.Q) #stiffness
        self.t_L = np.copy(element.t_L) #left projection
        self.t_R = np.copy(element.t_R) #right projection

        # grid
        self.K = K #number of elements
        self.M = self.Np * self.K #total DOF
        self.referenceGrid = element.referenceGrid
        self.referenceDx = element.dx
        self.createEqualGrid()

    def createEqualGrid(self):
        self.h = self.L/(self.K)
        self.mesh = np.zeros(self.M)
        for k in range(0, self.K):
            self.mesh[k * self.Np:(k + 1) * self.Np] = np.ones(self.Np) * (k * self.h) + \
                0.5 * (self.referenceGrid + np.ones(self.Np)) * self.h  # mapping each node of element k
        self.dx =0.5*self.referenceDx*self.h

    def R(self, u): #time discretization calls buildFlowResidual
        R = np.zeros(self.M)

        return R

    def dRdu(self, u): #time discretization calls buildFlowJacobian
        dRdu = np.zeros([self.M, self.M])

        return dRdu
