# Mostly Deprecated

import numpy as np
from typing import Callable


class Flux:

    def __init__(self):
        self.d = None
        self.N_e = None
        self.pointWiseFlux = None # function, R^{N_e x 1} -> R^{N_e x d}

    def __call__(self, u):
        if len(u.shape) == 1:  # not intended for this, but it works
            return self.pointWiseFlux(u.reshape(len(u),1))
        elif len(u.shape) == 2:
            return self.pointWiseFlux(u)
        elif len(u.shape) == 3:
            return np.apply_along_axis(self.pointWiseFlux, 1, u)
        else:
            raise TypeError("Flux must act on Ne x 1 matrix, or an array of those.")


class LinearAdvectionFlux(Flux):
    pass


class ConservationLaw:

    def __init__(self, d: int, N_e: int) -> None:
        self.d = d # spatial dimension
        self.N_e = N_e # number of equations
        self.f = None  # fluxFunction
        self.u_0 = None  # initial condition. function, R^{d x 1} -> R^{N_e x 1}
        self.t_f = None # final time (assume time starts at 0)
        self.u_exact = None # callable, R^{d x 1} -> R^{N_e x 1}


class AdvectionEquation(ConservationLaw):

    def __init__(self, d: int, a: np.ndarray, u_0: Callable) -> None:
        super().__init__(d, 1)
        assert a.shape == (self.d, 1), "Advection velocity must be d x 1"
        self.u_0 = u_0
        self.f = advectionFlux(d, a)
        self.u_exact = exactSolutionAdvection(i_c, t_f)


class Problem:

    def __init__(self, eqn : ConservationLaw, mesh, spatialDiscretizationParams,
                 temporalDiscretizationParams) -> None:
       pass
