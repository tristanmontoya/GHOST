import numpy as np
from local_discretization.LocalDiscretization import LocalDiscretization
from scipy import special
import utils.quadUtils


class DGQuadratureSimplex(LocalDiscretization):
    """Quadrature-based DG scheme of Cockburn and Shu"""

    def __init__(self, d, basis, p, volume_quadrature_type, facet_quadrature_type, Nq, Nqf):

        xq, wq = utils.quadUtils.volume_quadrature(d, volume_quadrature_type, Nq)
        xqf, xqfe, wqf = utils.quadUtils.facet_quadrature(d, facet_quadrature_type, Nqf)

        LocalDiscretization.__init__(self, d=d,
                                     basis=basis,
                                     elementType='simplex',
                                     p=p,
                                     Ns=special.comb(p + d, d, exact=True),
                                     xq=xq,
                                     xqf=xqf,
                                     xqfe=xqfe)

        self.wq = wq
        self.wqf = wqf
        self.W = np.diag(wq.flatten())
        self.Wf = np.diag(wqf.flatten())
        self.M = self.Vq.T @ self.W @ self.Vq
        self.Minv = np.linalg.inv(self.M)

        self.create_operators()

    def set_solution_to_volume_quadrature(self):
        self.Psq = self.Vq

    def set_volume_quadrature_to_solution(self):
        self.Pqs = self.Minv @ self.Vq.T @ self.W

    def set_derivative_operator(self):
        for m in range(0, self.d):
            self.D[m, :, :] = self.Vqx[m, :, :] @ self.Pqs

    def set_lifting_operator(self):
        for gamma in range(0, self.Nf):
            self.L[gamma, :, :] = self.Minv @ self.Vqf[gamma, :, :].T @ self.Wf

