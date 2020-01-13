import numpy as np
import Problem
import Solver
import Mesh
import Discretization

K_TEST = 3
NV_TEST = 10
P_TEST = 2


def flux_test():
    problem = Problem.const_advection_init(d=1, a=1.0)

    spatial_solver = Solver.SpatialSolver(form='strong', metric_method='project')

    time_solver = Solver.TimeSolver(method='rk4')

    mesh = Mesh.make_mesh_1d('test_mesh1', 0.0, 1.0,
                        K_TEST, NV_TEST, 'lg', 'uniform',
                        'random', transform=lambda x: x ** 2)

    disc = Discretization.construct_reference_dg_fr(1, P_TEST, NV_TEST, basis='legendre',
                                                 volume_nodes='lg',
                                                 facet_nodes='endpoints',
                                                 c = 0.0)

    f = lambda x: np.sin(2*np.pi*x[:,0])
    u_v = Mesh.eval_grid_function(mesh, f)
    u_s = Discretization.project_to_solution(disc, mesh, u_v)
    res = Solver.global_residual(problem,mesh,spatial_solver,disc,u_s)

    u_gamma = Mesh.eval_facet_function(mesh, f)

    u_v_h = Discretization.evaluate_at_volume_nodes(disc, mesh, u_s)
    u_gamma_h = Discretization.evaluate_at_facet_nodes(disc, mesh, u_s)

    f_inter = Solver.facet_numerical_flux(problem, mesh, u_gamma_h)

    # output test data
    print("solution DOF: ", u_s)
    print("residual: ", res)
    print("volume solution: ", u_v_h)
    print("facet solution: ", u_gamma_h)
    print("interface flux: ", f_inter)
    print("facet coords: ", mesh.x_gamma)
    print("facet normals: ", mesh.n_gamma)

    # plot the mesh
    Mesh.plot_mesh(mesh)
    Mesh.plot_on_volume_and_facet_nodes(mesh,u_v, u_gamma, 'exact')
    Mesh.plot_on_volume_and_facet_nodes(mesh, u_v_h, f_inter, 'numerical')


flux_test()