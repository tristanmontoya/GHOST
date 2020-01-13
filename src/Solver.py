# GHOST - Solver Components

from collections import namedtuple
import Operator
import Mesh
import Discretization
import numpy as np
import quadpy as qp

SpatialSolver = namedtuple('SpatialSolver', 'form metric_method')
TimeSolver = namedtuple('TimeSolver', 'method')
MetricData = namedtuple('MetricData', 'inv_proj_jac nodal_dxdxi')


def compute_metrics(mesh: namedtuple, ref_vol_nodes: str, ref_fac_nodes: str,
                    disc: namedtuple, p_mapping: int):
    inv_proj_jac = []
    nodal_dxdxi = []

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

            inv_proj_jac = inv_proj_jac + [(disc.P_v*J*disc.R_v).inv]
            nodal_dxdxi = nodal_dxdxi + [J]

    return MetricData(inv_proj_jac=inv_proj_jac,
                      nodal_dxdxi=nodal_dxdxi)


def facet_numerical_flux(problem: namedtuple, mesh: namedtuple,
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



            # Re-distribute the numerical flux (this is wrong).
            # Need to determine whether facet gamma of element k is a "minus" or "plus"
            # facet and choose sign accordingly
            return [[f[mesh.FtoE[k, 0]]*mesh.f_side[k][0], f[mesh.FtoE[k, 1]]*mesh.f_side[k][1]]
                     for k in range(0, mesh.K)]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


# define as partial application for passing to time marching
def global_residual(problem: namedtuple, mesh: namedtuple,
                    spatial_solver: namedtuple,
                    metric_data: namedtuple,
                    disc: namedtuple, u: np.ndarray):

    # Extrapolate solution to facets
    u_f = Discretization.evaluate_at_facet_nodes(disc, mesh, u)

    # Evaluate numerical flux functions
    f_n = facet_numerical_flux(problem, mesh, u_f)

    return [local_residual(problem, mesh, spatial_solver, disc, metric_data,
                           u, u_f, f_n, k)
            for k in range(0, mesh.K)]


def local_residual(problem: namedtuple, mesh: namedtuple,
                   spatial_solver: namedtuple, disc: namedtuple,
                   metric_data: namedtuple, u: np.ndarray,
                   u_f, f_n, k):

    if problem.d == 1:
        if problem.problem_type == 'const_advection':
            if spatial_solver.form == 'strong':
                vol = -1.0*disc.D(u[k])
                print("k: ", k, " vol: ", vol)
                print("normals: ",mesh.n_gamma[k])
                print("fluxes: ", f_n[k])
                surf = [disc.L[gamma](mesh.n_gamma[k][gamma].reshape(1)*u_f[k][gamma]
                                      - f_n[k][gamma]) for gamma in range(0, 2)]

                print("surf: ", surf)
                return metric_data.inv_proj_jac[k](vol + sum(surf))
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

