# GHOST - Physical Problem Definition

from Operator import Operator, DenseLinearOperator, DiagonalOperator, Identity
import numpy as np
from collections import namedtuple

AdvectionEquation = namedtuple('AdvectionEquation', 'd a')


def physical_flux(problem_type: str, problem_data: namedtuple,
                  mesh: namedtuple):

    if problem_type == 'advection':
        return [advection_physical_flux(problem_data.d, problem_data.a,
                    mesh.xv[k]) for k in range(0, mesh.K)]
    else:
        raise NotImplementedError


def interface_flux(problem_type: str, flux_type: str,
                   problem_data: namedtuple,
                   mesh: namedtuple, u_gamma):
    if problem_data.d == 1:
        if problem_type == 'advection':
            if flux_type == 'average':
                u_gathered = gather_extrapolated_solution(mesh, problem_type, u_gamma)
                f_gathered = [0.5 * problem_data.a*(u_gathered[i][0] + u_gathered[i][1])
                              for i in range(0, mesh.Nf_total)]
                return [[f_gathered[mesh.FtoE[k,0]], f_gathered[mesh.FtoE[k,1]]]
                        for k in range(0, mesh.K)]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def advection_physical_flux(d: int, a, xv, constant=True):

    # returns flux operator, that takes in nodal solution and returns nodal flux
    # in each direction (at volume flux/quadrature nodes)

    # xv Nv by d array of volume node positions in physical space

    # a is either a constant, function, or d-tuple of arrays
    # if it's a function, need to specify xv to evaluate at

    # afterwards can differentiate then project, or project then differentiate

    if d == 1:
        if constant:
            return DiagonalOperator(a*np.ones(xv.shape[0]))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def gather_extrapolated_solution(mesh, problem_type, u_gamma):

    if problem_type == 'advection':
        return [(u_gamma[mesh.EtoF[f][0][0]][mesh.EtoF[f][1][0]],
                 u_gamma[mesh.EtoF[f][0][1]][mesh.EtoF[f][1][1]])
                for f in range(0, mesh.Nf_total)]

