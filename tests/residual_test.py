import Problem
import Solver
import Mesh
import Discretization
import numpy as np

P_MAP = 1
K_TEST = 3
P_TEST = 5
NV_TEST = 6  # >= cardinality


def test_res_freestream_1d_advection():

    # set up problem
    problem = Problem.const_advection_init(d=1, a=1.0)

    # generate mesh
    ref_vol_nodes = 'lgl'
    ref_fac_nodes = 'endpoints'
    L = 10.0
    mesh = Mesh.make_mesh_1d('test_mesh_uniform', -1.0, 1.0,
                             K_TEST, NV_TEST, ref_vol_nodes, 'uniform', 'ordered',
                             transform=(lambda x: Mesh.nonsingular_map_1d(P_MAP,x,L)))

    # set solver parameters
    spatial_solver = Solver.SpatialSolver(form='strong', metric_method='project')

    # set up discretization operators
    disc = Discretization.construct_reference_dg_fr(1, P_TEST, NV_TEST, basis='lagrange-lgl',
                                                    volume_nodes=ref_vol_nodes,
                                                    facet_nodes=ref_fac_nodes,
                                                    c=0.0)

    # compute grid metrics
    metric_data = Solver.compute_metrics(mesh, ref_vol_nodes, ref_fac_nodes, disc, P_MAP)

    # make spatial residual du/dt = R(u)
    R = Solver.make_residual(problem,mesh,spatial_solver,disc,metric_data,print_output=True)

    # evaluate initial condition
    u_v_t0 = Mesh.eval_grid_function(mesh, lambda x: np.ones(len(x)))
    u_s = Discretization.project_to_solution(disc, mesh, u_v_t0)

    # evaluate residual
    res = R(u_s)
    print("res:", res)

    # evaluate final solution and plot result
    u_v_tf = Discretization.evaluate_at_volume_nodes(disc, mesh, u_s)
    Mesh.plot_mesh(mesh)
    Mesh.plot_on_volume_nodes(mesh, u_v_tf, 'uniform_test')

    # Check that for uniform flow, residual is zero (free-stream preservation)
    for k in range(0, mesh.K):
        err = 'error in residual for element ' + str(k)
        assert(np.allclose(res[k], np.zeros(P_TEST+1))), err


test_res_freestream_1d_advection()