# GHOST - 1D Simplex Element

from ElementBase import ElementBase
import numpy as np
import quadpy as qp

class Element1D(ElementBase):
    def __init__(self, p, Nq, quadratureType='LGL'):

        if quadratureType == 'LGL':
            quad = qp.line_segment.GaussLobatto(Nq)
            wq = quad.weights
            xq = quad.points
        else: #LGL by default
            quad = qp.line_segment.GaussLobatto(Nq)
            wq = quad.weights
            xq = quad.points

        xqf = np.array([[-1.0], [1.0]])
        wqf = np.array([[1.0], [1.0]])
        n_hat = np.array([-1.0, 1.0])
        ElementBase.__init__(self, 1, p, 2, n_hat, wq, xq, wqf, xqf)

lgl = Element1D(2,3)