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
        if params["problem"] == "constant_advection":
            
            f = Problem.ConstantAdvectionPhysicalFlux(params["wave_speed"])
            f_star = Problem.ConstantAdvectionNumericalFlux(
                params["wave_speed"], 
                params["upwind_parameter"])
            
        elif params["problem"] == "projection":
            pass
        
        else:
            raise NotImplementedError
        
        # initial condition
        if params["initial_condition"] == "sine":
            if "wavelength" in params:
                self.u_0 = Solver.sine_wave(params["wavelength"]) 
            else:
                self.u_0 = Solver.sine_wave(np.ones(self.d)) 
            
        else:
            raise NotImplementedError
            
        # exact solution
        if params["problem"] == "projection":
            self.u = self.u_0
        
        # spatial discretization
        if params["integration_type"] == "quadrature":
            self.discretization = Discretization.SimplexQuadratureDiscretization(
                mesh,
                params["solution_degree"], 
                params["volume_quadrature_degree"], 
                params["facet_quadrature_degree"])
                
            if "facet_integration_rule" in params:
                raise NotImplementedError
        
        elif params["integration_type"] == "collocation":
            raise NotImplementedError
            
        else:
            raise NotImplementedError
            
        # save params
        self.params = params
        
    @staticmethod
    def sine_wave(wavelength):
        # numpy array of length d wavelengths
        def g(x):
            return np.prod(np.sin(2.0*np.pi*x/wavelength))
        
        return g
    
    #def evaluate_on_volume_nodes(g,k):
        
        
    def project_function(self,g):
        raise NotImplementedError
        
    def run(self):
        if self.params["problem"] == "projection":
            self.uhat = self.project_function(self.u_0)
        
        else:
            raise NotImplementedError
        
