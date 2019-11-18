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


def advection_physical_flux(d: int, a, xv, constant=True):

    # returns flux operator, that takes in nodal solution and returns nodal flux
    # in each direction (at volume flux/quadrature nodes)

    # xv Nv by d array of volume node positions

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


def sc_flux(problem_type: str, problem_data: namedtuple, x_gamma: np.ndarray):

    return None


def ss_dissipation(problem_type: str, problem_data: namedtuple, x_gamma: np.ndarray):

    return None