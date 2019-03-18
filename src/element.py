# High Order Spatial Discretization - Elemental Properties

import numpy as np
import quadpy as qp
from scipy import special as sp_special

class Element:
    def __init__(self, elementType, p, gridType="uniform", Np=10):
        self.p = p  # operator degree
        if elementType == "dg_dense":
            self.type = 0
            self.Np = p+1
            self.gridType = gridType
            self.generateGrid()
            self.constructDG(diag=False)
        elif elementType == "dg_diag":
            self.type = 1
            self.Np = p+1
            self.gridType = gridType
            self.generateGrid()
            self.constructDG(diag=True)

        elif elementType == "csbp":
            self.type = 2
            self.Np = Np
            self.gridType = "uniform"

    def generateGrid(self):

        print("Np: ", self.Np)
        if self.gridType == "lgl":
            quad = qp.line_segment.GaussLobatto(self.Np)
            self.referenceGrid = quad.points
            self.quadratureWeights = quad.weights
        elif self.gridType == "lg":
            quad = qp.line_segment.GaussLegendre(self.Np)
            self.referenceGrid = quad.points
            self.quadratureWeights = quad.weights
        else: #uniform
            self.referenceGrid= np.linspace(-1., 1., self.Np)

        dx = np.zeros(self.Np - 1)
        for i in range(0, self.Np - 1):
            dx[i] = (self.referenceGrid[i + 1] - self.referenceGrid[i])
        self.dx = np.amin(dx)

    def constructDG(self, diag=True):
        V = np.polynomial.legendre.legvander(self.referenceGrid, self.p)
        modesL = np.zeros(self.Np)
        modesR = np.zeros(self.Np)
        Vr = np.zeros([self.Np, self.Np])
        for j in range(0, self.Np):
            normalizationFactor = np.sqrt(2. / (2 * j + 1))
            V[:, j] /= normalizationFactor  # normalization factor
            modesL[j] = sp_special.legendre(j)(-1.) / normalizationFactor
            modesR[j] = sp_special.legendre(j)(1.) / normalizationFactor
            dPdxi = np.polyder(sp_special.legendre(j))
            Vr[:, j] = dPdxi(self.referenceGrid) / normalizationFactor  # derivatives of legendre basis


        if diag: # collocated quadrature
            self.H = np.diag(self.quadratureWeights)
        else:  # H = (V*V')^-1
            self.H = np.linalg.inv(np.matmul(V, np.transpose(V)))

        # D = Vr*(V)^-1
        self.D = np.matmul(Vr, np.linalg.inv(V))
        self.Q = self.H @ self.D

        # Projection operators V'*proj(x) = modes(x), reshape to make them column vectors
        self.t_L = np.linalg.solve(np.transpose(V), modesL).reshape(self.Np, 1)
        self.t_R = np.linalg.solve(np.transpose(V), modesR).reshape(self.Np, 1)


