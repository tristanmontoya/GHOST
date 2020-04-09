import numpy as np
from scipy import signal
import quadpy as qp
from Mesh import make_mesh_1d, plot_mesh
from Operator import DiagonalOperator
from Discretization import volume_project, poly_deriv, \
    construct_reference_dg_fr, fr_filter, vandermonde, reference_mass_matrix_exact

N_TEST = 6
P_TEST = 3
K_TEST = 3


def test_mappings_1d():

    Nv = N_TEST
    p = P_TEST

    poly = np.polyval(np.poly1d(signal.unit_impulse(p+1)), np.poly1d([1.0, 1.0]))
    mesh = make_mesh_1d('test_mesh1', -1.0, 1.0,
                               1, Nv, 'lg', 'uniform', 'ordered',
                        transform=lambda x: np.polyval(poly, x))

    P = volume_project(1, p, Nv, 'legendre', 'lg')
    P_fr = volume_project(1, p, Nv, 'legendre', 'lg', scheme='sd')
    D = poly_deriv(1, p, 1, 'legendre')
    V = vandermonde(1, p, 'legendre',
                qp.line_segment.GaussLegendre(Nv).points.reshape([Nv, 1]))
    Dproj = D*P
    M_dg = reference_mass_matrix_exact(1,p,'legendre')

   #print("PV_fr: \n", PV_fr)

    J = [Dproj(mesh.xv[k].reshape(Nv)) for k in range(0, mesh.K)]
    Jn = DiagonalOperator(V(J[0]))
    Jh = P * Jn * V
    print("J", J)
    print("Jhat:\n", Jh )
    print("MJh:\n", M_dg * Jh)

    Jpoly = [np.polynomial.legendre.leg2poly(J[k]) for k in range(0, mesh.K)]
    xpoly = [np.polynomial.legendre.leg2poly(P(mesh.xv[k].reshape(Nv)))
             for k in range(0, mesh.K)]

    print("Initial:\n", poly)
    print("Final:\n", np.poly1d(xpoly[0][::-1]))
    print("Initial dx/dxi:\n", np.polyder(poly))
    print("Final dx/dxi:\n", np.poly1d(Jpoly[0][::-1]))

    assert np.allclose(poly, np.poly1d(xpoly[0][::-1])), 'Mesh projection failed'
    assert np.allclose(np.polyder(poly), np.poly1d(Jpoly[0][::-1])), 'Metric calculation failed'


def test_plot_mesh_1d():
    K = K_TEST
    Nv = N_TEST
    p = P_TEST

    mesh = make_mesh_1d('test_mesh1', 0.0, 1.0,
                        K, Nv, 'lg', 'uniform',
                        'random', transform=lambda x: x**p)

    P = volume_project(1,p,Nv,'legendre','lg')
    D = poly_deriv(1,p,1,'legendre')
    Dproj = D*P

    J = [Dproj(mesh.xv[k].reshape(Nv)) for k in range(0, mesh.K)]

    Jpoly = [np.polynomial.legendre.leg2poly(J[k]) for k in range(0, mesh.K)]
    xpoly = [np.polynomial.legendre.leg2poly(P(mesh.xv[k].reshape(Nv)))
             for k in range(0, mesh.K)]

    plot_mesh(mesh)
