# GHOST - Physical Problem Definition

import numpy as np
from collections import namedtuple

ConstantAdvectionEquation = namedtuple('ConstantAdvectionEquation', 'problem_type N_e physical_flux numerical_flux d a')


def const_advection_init(d, a):
    return ConstantAdvectionEquation(problem_type='const_advection',
                                     N_e=1,
                                     physical_flux=lambda u,x: a*u,
                                     numerical_flux=lambda u_1,u_2,n_1: 0.5*n_1[0]*(u_1 + u_2),
                                     d=d,
                                     a=a)


