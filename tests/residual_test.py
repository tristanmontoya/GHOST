import Problem
import Solver
import Mesh
import Discretization
import numpy as np

P_MAP = 2
K_TEST = 3
NV_TEST = 5
P_TEST = 2


def res_test_1d():

    ref_vol_nodes = 'lgl'
    ref_fac_nodes = 'endpoints'

    # Problem Setup
    problem = Problem.const_advection_init(d=1, a=1.0)
    mesh = Mesh.make_mesh_1d('test_mesh_uniform', -1.0, 1.0,
                             K_TEST, NV_TEST, ref_vol_nodes, 'uniform', 'ordered',
                             transform=(lambda x: Mesh.nonsingular_map_1d(P_MAP,x)))
    spatial_solver = Solver.SpatialSolver(form='strong', metric_method='project')
    disc = Discretization.construct_reference_dg_fr(1, P_TEST, NV_TEST, basis='lagrange-lgl',
                                                    volume_nodes=ref_vol_nodes,
                                                    facet_nodes=ref_fac_nodes,
                                                    c=0.0)
    metric_data = Solver.compute_metrics(mesh, ref_vol_nodes, ref_fac_nodes, disc, P_MAP)

    print("nodal metric: ", metric_data.nodal_dxdxi)

    print("projected metric: ", metric_data.inv_proj_jac)

    u_v_t0 = Mesh.eval_grid_function(mesh, lambda x: np.ones(len(x)))

    u_s = Discretization.project_to_solution(disc, mesh, u_v_t0)

    u_v_tf = Discretization.evaluate_at_volume_nodes(disc, mesh, u_s)

    Mesh.plot_mesh(mesh)
    Mesh.plot_on_volume_nodes(mesh, u_v_tf, 'uniform_test')

    res = Solver.global_residual(problem, mesh, spatial_solver,
                                 metric_data, disc, u_s)
    print("res:", res)

    return


res_test_1d()