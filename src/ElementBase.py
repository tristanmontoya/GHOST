#GHOST - Element Base Class

import numpy as np
from scipy import special

class ElementBase:
    """
    # Properties
        Basis functions are orthonormal Legendre polynomials
        Coordinates refer to reference element

        dim     # int, dimension (1, 2, 3)
        p       # int, degree of accuracy for which discrete residual is zero
        Nq      # int, number of volume quadrature points

        wq      # volume quadrature weights  (length Nq)

        W       # Diagonal quadrature matrix containing weights wq (size Nq x Nq)
                # nodal values of functions v and u are v_n and u_n
                # v_n.T @ W @ u_n = (v,w)_W

        V       # Volume Vandermonde matrix for basis (size Nq x Np)
                V_{ij} = phi_j(xq_i)

        M       # Modal mass matrix (size Np x Np)
                # M_{ij} = (phi_i, phi_j)_W

    # Methods
        setQuadrature
        setVandermonde

    """

    def __init__(self, dim, p):
        self.dim = dim
        self.p = p
        self.Np = p

    def setQuadrature(self, wq, xq):
        self.Nq = len(wq)
        self.wq = wq
        self.xq = xq
        self.W = np.diag(wq)

    def setVandermonde(self,wq):
        V = np.polynomial.legendre.legvander(self.xq, self.p)
        modesL = np.zeros(self.Np)
        modesR = np.zeros(self.Np)
        Vr = np.zeros([self.Np, self.Np])
        for j in range(0, self.Np):
            normalizationFactor = np.sqrt(2. / (2 * j + 1))
            V[:, j] /= normalizationFactor  # normalization factor
            modesL[j] = special.legendre(j)(-1.) / normalizationFactor
            modesR[j] = special.legendre(j)(1.) / normalizationFactor
            dPdxi = np.polyder(special.legendre(j))
            Vr[:, j] = dPdxi(self.xq / normalizationFactor  # derivatives of normalized legendre basis
