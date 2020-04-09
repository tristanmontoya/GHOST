import Problem
import Solver
import Mesh
import Discretization
import numpy as np
from matplotlib import pyplot as plt

P_MAP = 1
K_TEST = 2
P_TEST = 4
NV_TEST = 10  # >= cardinality


def solver_1d_advection():
    
    """ Periodic linear advection test problem """

    basis = 'legendre'
    ref_vol_nodes = 'lgl'
    ref_fac_nodes = 'endpoints'
    L = 1.0
    C = 0.1
    a = 1.0
    t_f = 1.0
    beta = 0.0  # 1.0 central, 0.0 upwind
    scheme = 'dg'

    # set up problem
    problem = Problem.const_advection_init(d=1, a=a, t_f=t_f, beta=beta)

    # generate mesh
    mesh = Mesh.make_mesh_1d('test_mesh_uniform', -1.0, 1.0,
                             K_TEST, NV_TEST, ref_vol_nodes, 'uniform', 'ordered',
                             transform=(lambda x: Mesh.nonsingular_map_1d(P_MAP,x,L)))

    # set solver parameters
    spatial_solver = Solver.SpatialSolver(form='strong', metric_method='project')
    time_solver = Solver.TimeSolver(method='rk4', cfl=C)

    # set up discretization operators
    disc = Discretization.construct_reference_dg_fr(1, P_TEST, NV_TEST, basis=basis,
                                                    volume_nodes=ref_vol_nodes,
                                                    facet_nodes=ref_fac_nodes,
                                                    c=scheme)
    M = Discretization.reference_mass_matrix_exact(1,P_TEST,basis)
    F = Discretization.fr_filter(1,P_TEST,scheme,basis)

    # compute grid metrics
    metric_data = Solver.compute_metrics(mesh, ref_vol_nodes, ref_fac_nodes, disc, P_MAP)

    # set up solver
    R = Solver.make_residual(problem,mesh,spatial_solver,disc,metric_data,print_output=False)
    solve = Solver.make_solver(problem,mesh,time_solver,R,print_output=False)

    # evaluate initial condition
    u_v_t0 = Mesh.eval_grid_function(mesh, lambda x: np.sin(2*np.pi*x[:,0]))
    u_0 = Discretization.project_to_solution(disc, mesh, u_v_t0)

    # get initial energy
    e_0 = Solver.calc_energy(M,metric_data, u_0)

    # solve the PDE
    u_f = solve(u_0)

    # get final energy and output energy loss/gain
    e_f = Solver.calc_energy(M,metric_data, u_f)
    print('e_0: ', e_0)
    print('e_f: ', e_f)
    print('e_f/e_0: ', e_f/e_0)

    # evaluate final solution and plot result
    u_v_tf = Discretization.evaluate_at_volume_nodes(disc, mesh, u_f)
    Mesh.plot_on_volume_nodes(mesh, u_v_tf, 'sin_final_'+str(scheme))
    Mesh.plot_mesh(mesh)


def solver_1d_eig():
    """ Plot how maximum real part of the spectrum varies with VCJH
    parameter c, to examine stability for linear advection """

    basis = 'legendre'
    ref_vol_nodes = 'lgl'
    ref_fac_nodes = 'endpoints'
    L = 1.0
    a = 1.0
    t_f = 1.0
    beta = 1.0
    scheme = 0.0
    N_step = 100

    # set up problem
    problem = Problem.const_advection_init(d=1, a=a, t_f=t_f, beta=beta)

    # generate mesh
    mesh = Mesh.make_mesh_1d('test_mesh_uniform', -1.0, 1.0,
                             K_TEST, NV_TEST, ref_vol_nodes, 'uniform', 'ordered',
                             transform=(lambda x: Mesh.nonsingular_map_1d(P_MAP,x,L)))

    # set solver parameters
    spatial_solver = Solver.SpatialSolver(form='strong', metric_method='project')

    # set up discretization operators
    disc = Discretization.construct_reference_dg_fr(1, P_TEST, NV_TEST, basis=basis,
                                                    volume_nodes=ref_vol_nodes,
                                                    facet_nodes=ref_fac_nodes,
                                                    c=scheme)

    # compute grid metrics
    metric_data = Solver.compute_metrics(mesh, ref_vol_nodes, ref_fac_nodes, disc, P_MAP)

    c_sd = Discretization.fr_c(1,P_TEST,'sd',basis)
    print("c_sd = ", c_sd)

    # set up solver
    c_min = 0.0*c_sd
    c_max = 10*c_sd
    c = np.linspace(c_min, c_max, N_step)
    emax = np.zeros(N_step)

    for i in range(0,N_step):
        disc = Discretization.construct_reference_dg_fr(1, P_TEST, NV_TEST, basis=basis,
                                                    volume_nodes=ref_vol_nodes,
                                                    facet_nodes=ref_fac_nodes,
                                                    c=c[i])
        R = Solver.make_residual(problem, mesh, spatial_solver, disc, metric_data, print_output=False)
        w, v = Solver.get_eigenvalues(R, K_TEST, P_TEST + 1)
        emax[i] = np.amax(w.real)
        print('c: ', c[i], 'emax: ', emax[i])

    eigplt = plt.figure()
    plt.plot(c, emax, '-')
    eigplt.savefig("./eigs.pdf", bbox_inches=0, pad_inches=0)
    plt.grid()
    plt.show()
    return


#solver_1d_advection()
solver_1d_eig()