import numpy as np
from scipy import signal
from Mesh import make_mesh_1d, plot_mesh
from Discretization import volume_project, poly_deriv, construct_reference_collocation_dg_fr

N_TEST = 10
P_TEST = 2
K_TEST = 3


def test_mappings_1d():

    Nv = N_TEST
    p = P_TEST

    poly = np.polyval(np.poly1d(signal.unit_impulse(p+1)), np.poly1d([1.0, 1.0]))
    mesh = make_mesh_1d('test_mesh1', -1.0, 1.0,
                               1, Nv, 'lg', 'uniform', 'ordered',
                        transform=lambda x: np.polyval(poly, x))

    P = volume_project(1, p, Nv, 'legendre', 'lg')
    D = poly_deriv(1, p, 1, 'legendre')
    Dproj = D*P
    J = [Dproj(mesh.xv[k].reshape(Nv)) for k in range(0, mesh.K)]
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
