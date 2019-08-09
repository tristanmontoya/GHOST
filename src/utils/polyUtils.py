"""
Polynomial interpolation and projection tools
"""

import numpy as np
from scipy import special


def vandermonde(d, basis, x,p):
    if d == 1:
        if basis == 'legendre-normalized':
            return orthonormal_vandermonde_1d(x, p)
        elif basis == 'lagrange':
            raise NotImplementedError
    elif d == 2:
        raise NotImplementedError


def grad_vandermonde(d, basis, x,p):
    if d == 1:
        if basis == 'legendre-normalized':
            return orthonormal_grad_vandermonde_1d(x, p)
        elif basis == 'lagrange':
            raise NotImplementedError
    elif d == 2:
        raise NotImplementedError


def orthonormal_vandermonde_1d(x, p):
    V = np.polynomial.legendre.legvander(x[:, 0], p)

    for j in range(0, p+1):
        normalization_factor = np.sqrt(2. / (2 * j + 1))
        V[:, j] /= normalization_factor

    return V


def orthonormal_grad_vandermonde_1d(x, p):
    Vx = np.zeros([len(x), p + 1])

    for j in range(0, p + 1):
        normalization_factor = np.sqrt(2. / (2 * j + 1))
        dPdx = np.polyder(special.legendre(j))
        Vx[:, j] = dPdx(x[:, 0]) / normalization_factor

    return Vx