# GHOST - 1D Simplex Element

from LocalDiscretizationBase import LocalDiscretizationBase
import numpy as np
import quadpy as qp
from scipy import special


class Simplex1D(LocalDiscretizationBase):

    def __init__(self, p, Nq, basis='orthonormal', quadrature_type='LGL'):

        # Volume quadrature
        if quadrature_type == 'LGL':
            quad = qp.line_segment.GaussLobatto(Nq)
        else: # LGL by default
            quad = qp.line_segment.GaussLobatto(Nq)

        wq = quad.weights
        xq = quad.points.reshape([Nq, 1]) # make column vector for dimension-agnostic implementation

        # Facet quadrature
        xqf = None # not needed here (dimension zero)
        xqfe = np.array([[[-1.0]], [[1.0]]])
        wqf = np.array([1.0])

        LocalDiscretizationBase.__init__(self, 1, basis, 'simplex', p, wq, xq, wqf, xqf, xqfe)

    def set_volume_vandermonde(self):
        # only orthonormal basis defined for now

        self.V = np.polynomial.legendre.legvander(self.xq[:,0], self.p)
        self.Vx = np.zeros([self.Nq, self.Np])

        for j in range(0, self.Np):
            normalization_factor = np.sqrt(2. / (2 * j + 1))
            self.V[:, j] /=normalization_factor  # normalization factor
            dPdxi = np.polyder(special.legendre(j))
            self.Vx[:, j] = dPdxi(self.xq[:,0]) /normalization_factor  # derivatives of normalized legendre basis

    def set_facet_vandermonde(self):
        # only orthonormal basis defined for now

        self.Vf = np.zeros([self.Nf, self.Nqf, self.Np])

        for gamma in range(0, self.Nf):
            self.Vf[gamma, :, :] = np.polynomial.legendre.legvander(self.xqfe[gamma,:,0], self.p)
            for j in range(0, self.Np):
                self.Vf[gamma, :, j] /= np.sqrt(2. / (2 * j + 1))
