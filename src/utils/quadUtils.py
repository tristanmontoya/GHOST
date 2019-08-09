"""
Numerical quadrature tools
"""

import numpy as np
import quadpy as qp


def volume_quadrature(d, quadratureType, Nq, return_weights=True):
    if d == 1:
        xq, wq = quadrature_1d(quadratureType, Nq)
    elif d == 2:
        raise NotImplementedError

    if return_weights:
        return xq, wq
    return xq


def facet_quadrature(d, quadratureType, Nqf, return_weights=True):
    if d == 1:
        xqf = None
        wqf = np.array([[1.0]])
        xqfe = np.array([[[-1.0]], [[1.0]]])
    elif d == 2:
        raise NotImplementedError

    if return_weights:
        return xqf, xqfe, wqf
    return xqf, xqfe


def quadrature_1d(quadratureType, Nq):
    """
    Quadrature rules for the reference line segment (-1, 1)
    """

    if quadratureType == 'LG':
        quad = qp.line_segment.GaussLegendre(Nq)
    elif quadratureType == 'LGL':
        quad = qp.line_segment.GaussLobatto(Nq)

    return quad.points.reshape((Nq, 1)), quad.weights.reshape((Nq, 1)),
