# GHOST - Solver Components

import Mesh
import Problem
import Discretization

import numpy as np

class Solver:
    
    def __init__(self, params, mesh):
        
        # set spatial dimension since this is used everywhere
        self.d = mesh.d
        
        # physical problem
        if params.problem == "constant_advection":
            
            f = Problem.ConstantAdvectionPhysicalFlux(params.a)
            f_star = Problem.ConstantAdvectionNumericalFlux(params.a, 
                                                            params.alpha)
            
        else:
            raise NotImplementedError
        
        # initial condition
        if params.ic == "sine":
            if wavelengths in params:
                self.u_0 = sine_wave(params.wavelengths) 
            else:
                self.u_0 = sine_wave(np.ones(self.d)) 
            
        else:
            raise NotImplementedError
        
        # spatial discretization
        if params.integration_type == "quadrature":
            discretization = D
                
            if facet_integration_rule in params:
                raise NotImplementedError
        
        elif params.integration_type == "collocation":
            raise NotImplementedError
            
        else:
            raise NotImplementedError
            
        
        
    @staticmethod
    def sine_wave(wavelengths):
        # numpy array of length d wavelengths
        def g(x):
            return np.prod(np.sin(2.0*np.pi*x/wavelengths))
        
        return g
        
        
    def project_function(g):
        
        
