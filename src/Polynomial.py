# In development

from Operator import DenseLinearOperator, Identity
import numpy as np
from scipy import special
import quadpy as qp


def cardinality(d: int, p: int):
    return special.comb(p + d, d, exact=True)


def vandermonde(d: int, p: int, basis: str, xv: np.ndarray) -> DenseLinearOperator:

    phi_leg = legendre_basis(d,p)
    V_leg = np.empty([xv.shape[0], p + 1])

    for i in range(0, xv.shape[0]):
        for j in range(0, p + 1):
            V_leg[i, j] = phi_leg[j](xv[i, 0])

    T = change_polynomial_basis(d, p, 'legendre', basis)
    return DenseLinearOperator(V_leg) * T


def legendre_basis(d: int, p:int) -> list:
    if d == 1:
        return list(map(lambda k: special.legendre(k), range(0, p+1)))
    else:
        raise NotImplementedError


def change_polynomial_basis(d: int, p: int, basis1: str, basis2: str) -> DenseLinearOperator:
    Np = cardinality(d,p)

    if basis1 == basis2:
        return Identity(Np)
    if d == 1:

        # Nodal
        if basis1 == 'lagrange-lg':
            xs = qp.line_segment.GaussLegendre(Np).points.reshape([Np,1])
            return vandermonde(1, p, basis2, xs)
        if basis1 == 'lagrange-lgl':
            xs = qp.line_segment.GaussLobatto(Np).points.reshape([Np,1])
            return vandermonde(1, p, basis2, xs)
        if basis2 == 'lagrange-lg' or basis2 == 'lagrange-lgl':
            return change_polynomial_basis(d,p,basis2, basis1).inv

        # Modal
        if basis1 == 'orthonormal':
            if basis2 == 'legendre':
                return DenseLinearOperator(np.diag(np.sqrt(2.0/(2.0*np.arange(Np) + 1))))
        if basis1 == 'legendre':
            if basis2 == 'orthonormal':
                return change_polynomial_basis(d,p,'orthonormal', 'legendre').inv
    else:
        raise NotImplementedError


def reference_mass_matrix_exact(d: int, p: int, basis: str) -> DenseLinearOperator:
    Np = cardinality(d,p)

    if d == 1:
        M_orth = Identity(Np)
        T = change_polynomial_basis(d, p, 'orthonormal', basis)
        return T.T * M_orth * T
    else:
        raise NotImplementedError


def poly_deriv(d: int, p: int, der, basis: str) -> DenseLinearOperator:
    Np = cardinality(d,p)
    if d == 1:
        D_leg = DenseLinearOperator(
            np.triu(np.diag(2*np.arange(Np) + 1)
                    @ np.array([[(i + j) % 2 for i in range(Np)] for j in range(Np)])))
        T = change_polynomial_basis(d, p, 'legendre', basis)
        return T.inv * D_leg**der * T
    else: #if d > 1, der should be array with how many times to diff. each variable
        raise NotImplementedError
