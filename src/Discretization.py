# GHOST - Spatial Discretization

from Operator import DenseLinearOperator, DiagonalOperator, Identity
import numpy as np
from collections import namedtuple
from scipy import special
import quadpy as qp

Discretization = namedtuple('Discretization', 'H D P_v R_v L R')


def construct_reference_dg_fr(d, p, Nv, basis='lagrange-lgl',
                                          volume_nodes='lg',
                                          facet_nodes='endpoints',
                                          c=0.0):

    # maybe encapsulate these vandermonde functions
    if d == 1:
        if volume_nodes == 'lg':
            V = vandermonde(d,p, basis,
                            qp.line_segment.GaussLegendre(Nv).points.reshape([Nv, 1]))
        elif volume_nodes == 'lgl':
            V = vandermonde(d, p, basis,
                            qp.line_segment.GaussLobatto(Nv).points.reshape([Nv, 1]))
        Vf = [vandermonde(d,p,basis,np.array([[-1.0]])),
              vandermonde(d,p,basis,np.array([[1.0]]))]

    else:
        raise NotImplementedError

    return Discretization(H=fr_filter(d,p,c,basis,mass_matrix=True),
                          D=poly_deriv(d,p,1,basis),
                          P_v=volume_project(d,p,Nv,basis, quadrature=volume_nodes),
                          R_v=V,
                          L=lift(d,p,basis,scheme=c),
                          R=Vf)


# These assume the same local discretization is used everywhere (no adaptivity)

def project_to_solution(disc, mesh, u_v):
    return [disc.P_v(u_v[k]) for k in range(0, mesh.K)]


def evaluate_at_volume_nodes(disc, mesh, u_s):
    return [disc.R_v(u_s[k]) for k in range(0, mesh.K)]


def evaluate_at_facet_nodes(disc, mesh, u_s):
    return [[disc.R[gamma](u_s[k]) for
             gamma in range(0, mesh.Nf[k])] for k in range(0, mesh.K)]


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


def change_polynomial_basis(d: int, p: int, basis1: str,
                            basis2: str) -> DenseLinearOperator:
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
                return change_polynomial_basis(d, p, 'orthonormal', 'legendre').inv
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
                    @ np.array([[(i + j) % 2 for i in range(Np)]
                                for j in range(Np)])))
        T = change_polynomial_basis(d, p, 'legendre', basis)
        return T.inv * D_leg**der * T
    else:  # if d > 1, der should be array with how many times to diff. each variable
        raise NotImplementedError


def fr_filter(d: int, p: int, scheme, basis: str,
              mass_matrix=False) -> DenseLinearOperator:

    Np = cardinality(d,p)

    if d == 1:
        M = reference_mass_matrix_exact(d, p, 'legendre')
        Dp = poly_deriv(d, p, p, 'legendre')
        a_p = special.legendre(p)[p]
        if scheme == 'huynh':
            c = 2.0 * (p + 1.0) / ((2.0 * p + 1.0) * p *(np.math.factorial(p) * a_p) ** 2.0)
        elif scheme == 'dg':
            c = 0.0
        elif scheme == 'sd':
            c = 2.0 * p / ((2.0 * p + 1.0) * (p+1.0) * (np.math.factorial(p) * a_p) ** 2.0)
        else:
            c = scheme
        Finv = (Identity(Np) + c*M.inv*Dp.T*Dp)
        T = change_polynomial_basis(d, p, 'legendre', basis)
        if mass_matrix:
            return T.T * M * Finv * T
        return T.inv * Finv.inv * T
    else:
        raise NotImplementedError


def volume_project(d: int, p: int, Nv: int, basis: str,
                        quadrature='lg', scheme=0.0) -> DenseLinearOperator:
    if d == 1:
        if quadrature == 'lg':
            xv = qp.line_segment.GaussLegendre(Nv).points.reshape([Nv, 1])
            W = DiagonalOperator(qp.line_segment.GaussLegendre(Nv).weights)
        elif quadrature == 'lgl':
            xv = qp.line_segment.GaussLobatto(Nv).points.reshape([Nv, 1])
            W = DiagonalOperator(qp.line_segment.GaussLobatto(Nv).weights)
        else:
            raise NotImplementedError
        M = fr_filter(d,p,scheme,basis,mass_matrix=True)
        V = vandermonde(d, p, basis, xv)

    else:
        raise NotImplementedError

    return M.inv * V.T * W


def lift(d: int, p: int, basis: str, elem_type='simplex',
         cubature='endpoints', N_gamma=1.0, scheme=0.0):

    if elem_type == 'simplex':
        Nf = d + 1
    else:
        raise NotImplementedError

    if d == 1:
        Vf = [vandermonde(d,p,basis,np.array([[-1.0]])), vandermonde(d,p,basis,np.array([[1.0]]))]
        M = fr_filter(d,p,scheme,basis,mass_matrix=True)
        Wf = DenseLinearOperator(np.array([[1.0]]))
    else:
        raise NotImplementedError

    return [M.inv * Vf[i].T * Wf for i in range(0, Nf)]


# should be diagonal (it is the DGSEM)
#M_hu = fr_filter(1, 4, 'huynh', 'lagrange-lgl', mass_matrix=True)