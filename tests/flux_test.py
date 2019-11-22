import numpy as np
import Problem
import Mesh
import Discretization

K_TEST = 3
NV_TEST = 10
P_TEST = 2


def flux_test():

    problem = Problem.AdvectionEquation(d=1, a=1.0)

    mesh = Mesh.make_mesh_1d('test_mesh1', 0.0, 1.0,
                        K_TEST, NV_TEST, 'lg', 'uniform',
                        'random', transform=lambda x: x ** 2)

    disc = Discretization.construct_reference_dg_fr(1, P_TEST, NV_TEST, basis='legendre',
                                                 volume_nodes='lg',
                                                 surface_nodes='endpoints',
                                                 c = 0.0)

    f = lambda x: np.sin(2*np.pi*x[:,0])

    u_v = Mesh.eval_grid_function(mesh, f)
    u_gamma = Mesh.eval_facet_function(mesh, f)

    Mesh.plot_mesh(mesh)

    u_s = Discretization.project_to_solution(disc, mesh, u_v)
    u_v_h = Discretization.evaluate_at_volume_nodes(disc, mesh, u_s)
    u_gamma_h = Discretization.evaluate_at_facet_nodes(disc, mesh, u_s)

    f_inter = Problem.interface_flux('advection', 'average', problem,
                                     mesh, u_gamma_h)

    Mesh.plot_on_volume_and_facet_nodes(mesh,u_v, u_gamma, 'exact')
    Mesh.plot_on_volume_and_facet_nodes(mesh, u_v_h, f_inter, 'numerical')


flux_test()