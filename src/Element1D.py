# GHOST - 1D Simplex Element

from ElementBase import ElementBase
import numpy as np
import quadpy as qp
from scipy import special

class Element1D(ElementBase):

    def __init__(self, p, Nq, basis='orthonormal', quadratureType='LGL'):

        # Volume quadrature
        if quadratureType == 'LGL':
            quad = qp.line_segment.GaussLobatto(Nq)
        else: #LGL by default
            quad = qp.line_segment.GaussLobatto(Nq)

        wq = quad.weights
        xq = quad.points.reshape([Nq, 1]) # make column vector for dimension-agnostic implementation

        # Facet quadrature
        xqf = None #not needed here (dimension zero)
        xqfe = np.array([[[-1.0]], [[1.0]]])
        wqf = np.array([1.0])

        ElementBase.__init__(self, 1, basis, 'simplex', p, wq, xq, wqf, xqf, xqfe)

    def setVolumeVandermonde(self):
        # only orthonormal basis defined for now

        self.V = np.polynomial.legendre.legvander(self.xq[:,0], self.p)
        self.Vx = np.zeros([self.Nq, self.Np])

        for j in range(0, self.Np):
            normalizationFactor = np.sqrt(2. / (2 * j + 1))
            self.V[:, j] /= normalizationFactor  # normalization factor
            dPdxi = np.polyder(special.legendre(j))
            self.Vx[:, j] = dPdxi(self.xq[:,0]) / normalizationFactor  # derivatives of normalized legendre basis

    def setFacetVandermonde(self):
        # only orthonormal basis defined for now

        self.Vf = np.zeros([self.Nf, self.Nqf, self.Np])

        for gamma in range(0, self.Nf):
            self.Vf[gamma, :, :] = np.polynomial.legendre.legvander(self.xqfe[gamma,:,0], self.p)
            for j in range(0, self.Np):
                self.Vf[gamma, :, j] /= np.sqrt(2. / (2 * j + 1))

testEl = Element1D(p=2, Nq = 3, basis='orthonormal', quadratureType='LGL')
