# GHOST - Physical Problem Definition

import numpy as np
from collections import namedtuple

ConstantAdvectionEquation = namedtuple('ConstantAdvectionEquation', 'problem_type t_f N_e physical_flux numerical_flux d a')


def const_advection_init(d, a, t_f, beta):
    return ConstantAdvectionEquation(problem_type='const_advection',
                                     N_e=1,
                                     t_f = t_f,
                                     physical_flux=lambda u,x: a*u,
                                     numerical_flux=(lambda u_1,u_2,n_1: 0.5*a*n_1[0]*(u_1 + u_2) + 0.5*np.abs(a)*(1.0-beta)*(u_1 - u_2)),
                                     d=d,
                                     a=a)


