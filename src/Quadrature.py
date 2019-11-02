# Partially Deprecated

"""
Numerical quadrature tools
"""

import numpy as np
import quadpy as qp
from Operator import DenseLinearOperator


def volume_quadrature(d, quadratureType, N):
    if d == 1:
        xq, wq = quadrature_1d(quadratureType, N)
    else:
        raise NotImplementedError

    return xq, DenseLinearOperator(np.diag(wq))


def facet_quadrature(d, gamma, N=1, quadratureType=None):
    if d == 1:
        wqf = np.array([1.0])  # unity, just pointwise evaluation
        if gamma == 0:
            xqfe = np.array([[-1.0]])
        if gamma == 1:
            xqfe = np.array([[1.0]])
        else:
            raise ValueError("1D line segment must be called with gamma=0 (left boundary) or gamma=1 (right boundary)")
    else:
        raise NotImplementedError

    return xqfe, DenseLinearOperator(np.diag(wqf))


def quadrature_1d(quadratureType, N):
    """
    Quadrature rules for the reference line segment (-1, 1)
    """

    if quadratureType == 'LG':
        quad = qp.line_segment.GaussLegendre(N)
    elif quadratureType == 'LGL':
        quad = qp.line_segment.GaussLobatto(N)
    else:
        raise NotImplementedError

    return quad.points.reshape((N, 1)), quad.weights
