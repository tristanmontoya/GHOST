# GHOST - Solver Components

from collections import namedtuple
import Operator
import Mesh
import Discretization
import numpy as np
from scipy import signal

SpatialSolver = namedtuple('SpatialSolver', 'form metric_method')
TimeSolver = namedtuple('TimeSolver', 'method cfl')
MetricData = namedtuple('MetricData', 'inv_proj_jac nodal_dxdxi FJinv')


def compute_metrics(mesh: namedtuple,
                    ref_vol_nodes: str,
                    ref_fac_nodes: str,
                    disc: namedtuple,
                    p_mapping: int):

    inv_proj_jac = []
    nodal_dxdxi = []
    FJinv = []

    if mesh.d == 1:
        for k in range(0, mesh.K):
            metric_disc = Discretization.construct_reference_dg_fr(mesh.d,
                                          p_mapping, mesh.Nv[k], basis='legendre',
                                          volume_nodes=ref_vol_nodes,
                                          facet_nodes=ref_fac_nodes,
                                          c=0.0)

            J = Operator.DiagonalOperator(
                (metric_disc.R_v*metric_disc.D*metric_disc.P_v)(
                    mesh.xv[k].reshape(mesh.Nv[k])))
            Jmodal = disc.P_v * J * disc.R_v
            inv_proj_jac = inv_proj_jac + [Jmodal.inv]
            nodal_dxdxi = nodal_dxdxi + [J]
            Dp = (disc.D)**disc.p
            K = disc.c*(Dp.T)*(disc.M*Jmodal)*Dp
            I = Operator.Identity(disc.p + 1)
            FJinv = FJinv + [(I + (disc.M * Jmodal).inv * K ).inv * inv_proj_jac[k]]

    return MetricData(inv_proj_jac=inv_proj_jac,
                      nodal_dxdxi=nodal_dxdxi,
                      FJinv=FJinv)


def calc_energy(M_ref, metric_data, u):
    e = 0.0
    for k in range(0, len(u)):
        e = e + u[k] @ (M_ref * metric_data.inv_proj_jac[k].inv).mat @ u[k]
    return e


def make_residual(problem: namedtuple,
                  mesh: namedtuple,
                  spatial_solver: namedtuple,
                  disc: namedtuple,
                  metric_data: namedtuple,
                  print_output=False):

    return lambda u: global_residual(problem,mesh,spatial_solver,disc,
                                     metric_data,u,print_output)


def make_solver(problem: namedtuple, mesh: namedtuple, time_solver: namedtuple,
                R: callable, print_output=False):

    dt_desired = calculate_time_step(problem, mesh, time_solver, print_output)
    N_dt = np.int(np.ceil(problem.t_f / dt_desired))
    dt = problem.t_f / N_dt
    return lambda u: ode_solve(time_solver, R, u, N_dt, dt, print_output)


def get_eigenvalues(R: callable, K, Np):

    N = K*Np
    cols = []
    for i in range(0, N):
        ei = np.split(signal.unit_impulse(N,i), K)
        cols = cols + [np.concatenate(R(ei))]

    A = np.column_stack(cols)
    return np.linalg.eig(A)


def ode_solve(time_solver: namedtuple, R: callable, u, N_dt, dt,
              print_output=False):

    if time_solver.method == 'rk4':
        for n in range(0, N_dt):
            if print_output:
                print('n: ', n, 't: ', n*dt)
            r_u = R(u)

            u_hat_nphalf = [u[k] + 0.5 * dt * r_u[k] for k in range(0, len(u))]
            r_u_hat_nphalf = R(u_hat_nphalf)

            u_tilde_nphalf = [u[k] + 0.5 * dt * r_u_hat_nphalf[k] for k in range(0, len(u))]
            r_u_tilde_nphalf = R(u_tilde_nphalf)

            u_bar_np1 = [u[k] + dt * r_u_tilde_nphalf[k] for k in range(0, len(u))]
            r_u_bar_np1 = R(u_bar_np1)

            u = [u[k] + 1. / 6. * dt * (r_u[k] + 2. * (r_u_hat_nphalf[k] + r_u_tilde_nphalf[k]) + r_u_bar_np1[k])
                 for k in range(0, len(u))]
    else:
        raise NotImplementedError

    return u


def calculate_time_step(problem: namedtuple, mesh: namedtuple,
                        time_solver: namedtuple, print_output=False):
    # This uses the nodal points for the CFL. Could define modally too

    if mesh.d == 1:
        dx = []
        for k in range(0,mesh.K):
            dx = dx + [mesh.xv[k][i+1,0]-mesh.xv[k][i,0]
                       for i in range(0, mesh.Nv[k] - 1)]
        dxmin = min(dx)
        if print_output:
            print("dxmin = ", dxmin)

        if problem.problem_type == 'const_advection':
            dt = time_solver.cfl/(np.abs(problem.a))*dxmin
            if print_output:
                print("dt = ", dt)

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return dt


def facet_numerical_flux(problem: namedtuple,
                         mesh: namedtuple,
                         u_gamma: np.ndarray):

    if problem.d == 1:
        if problem.problem_type == 'const_advection':
            # gather the extrapolated solution
            u_gathered = Mesh.gather_field(mesh, u_gamma)

            # compute the numerical flux
            f = [problem.numerical_flux(u_gathered[i][0],
                                               u_gathered[i][1],
                                               mesh.n_gathered[i][0])
                              for i in range(0, mesh.Nf_total)]

            # Re-distribute the numerical flux
            return [[f[mesh.FtoE[k, 0]]*mesh.f_side[k][0],
                     f[mesh.FtoE[k, 1]]*mesh.f_side[k][1]]
                    for k in range(0, mesh.K)]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def global_residual(problem: namedtuple,
                    mesh: namedtuple,
                    spatial_solver: namedtuple,
                    disc: namedtuple,
                    metric_data: namedtuple,
                    u: np.ndarray,
                    print_output=False):

    # Extrapolate solution to facets
    u_f = Discretization.evaluate_at_facet_nodes(disc, mesh, u)

    # Evaluate numerical flux functions
    f_n = facet_numerical_flux(problem, mesh, u_f)

    return [local_residual(problem, mesh, spatial_solver, disc, metric_data,
                           u, u_f, f_n, k, print_output)
            for k in range(0, mesh.K)]


def local_residual(problem: namedtuple, mesh: namedtuple,
                   spatial_solver: namedtuple, disc: namedtuple,
                   metric_data: namedtuple, u: np.ndarray,
                   u_f, f_n, k, print_output=False):

    if problem.d == 1:
        if problem.problem_type == 'const_advection':
            if spatial_solver.form == 'strong':

                # volume and surface residuals
                I = Operator.Identity(len(u[k]))
                vol = -1.0*disc.D(u[k])
                surf = [disc.L[gamma](mesh.n_gamma[k][gamma].reshape(1)*u_f[k][gamma]
                                      - f_n[k][gamma]) for gamma in range(0, 2)]

                if print_output:
                    print("k: ", k, " vol: ", vol)
                    print("normals: ",mesh.n_gamma[k])
                    print("fluxes: ", f_n[k])
                    print("surf: ", surf)
                return metric_data.inv_proj_jac[k](vol + sum(surf))
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

