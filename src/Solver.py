# GHOST - Solver Components

import Problem
import SpatialDiscretization
import TimeIntegrator
import numpy as np
import modepy as mp
import matplotlib.pyplot as plt
from math import ceil
import os
import pickle

class Solver:
    
    def __init__(self, params, mesh, h=None):
        
        self.project_title = params["project_title"]
        
        # physical problem
        self.d = mesh.d
        
        if params["problem"] == "constant_advection":
            
            if "wave_speed" not in params:
                params["wave_speed"] = np.ones(self.d)
            
            if "upwind_parameter" not in params:

                if params["numerical_flux"] == "upwind":
                    params["upwind_parameter"] = 1.0

                elif (params["numerical_flux"] == "central" 
                      or params["numerical_flux"] == "symmetric"):
                    params["upwind_parameter"] = 0.0

                else:
                    raise NotImplementedError
                
            self.pde = Problem.ConstantAdvection(params["wave_speed"],
                params["upwind_parameter"])
            self.f = self.pde.build_physical_flux()
            self.f_star = self.pde.build_numerical_flux()
            self.is_unsteady = True
            self.cfl_speed = np.linalg.norm(self.pde.a)
            self.N_eq = self.pde.N_eq
            
        elif params["problem"] == "projection":
            
            self.is_unsteady = False
            self.N_eq = 1
        
        elif params["problem"] == "compressible_euler":
            
            if "specific_heat_ratio" not in params:
                params["specific_heat_ratio"] = 1.4

            if "numerical_flux" not in params:
                params["numerical_flux"] = "roe"

            self.pde = Problem.Euler(self.d,
                                     zeta=params["specific_heat_ratio"],
                                     numerical_flux=params["numerical_flux"])
            self.f = self.pde.build_physical_flux()
            self.f_star = self.pde.build_numerical_flux()
            self.is_unsteady = True
            self.N_eq = self.pde.N_eq
        
        else:
            raise NotImplementedError
            
        # initial conditions
        if params["initial_condition"] == "sine":
            
            if self.N_eq != 1:
                raise ValueError(
                    "Sine initial condition implemented only for scalar problems")
            
            if "wavelength" not in params:
                params["wavelength"] = np.ones(self.d)
          
            self.u_0 = Solver.sine_wave(np.ones(self.d))
                
        elif params["initial_condition"] == "constant":
            
            if params["problem"] == "compressible_euler":
                self.u_0 = Solver.euler_freestream(self.d, params["specific_heat_ratio"])
                self.cfl_speed = 1.0
                
            else:
                self.u_0 = lambda x: np.array([np.ones(x.shape[1]) for e in range(
                    0,self.N_eq)])
                self.cfl_speed = 1.0
            
        elif params["initial_condition"] == "isentropic_vortex":
            
            if params["problem"] != "compressible_euler":
                raise ValueError(
                    "Isentropic vortex only applicable to Euler equations")
            if self.d != 2:
                raise NotImplementedError

            if "initial_vortex_centre" not in params:
                params["vortex_centre"] =  np.array([5.0,5.0])

            if "background_velocity" not in params:
                params["background_velocity"] =  np.array([1.0,0.0])

            if "background_temperature" not in params:
                params["background_temperature"] = 1

            if "vortex_strength" not in params:
                params["vortex_strength"] = 1.0

            if "mach_number" not in params:
                params["mach_number"] = None

            if "angle" not in params:
                params["angle"] = None
            
            
            self.u_0 = Solver.isentropic_vortex(eps=params["vortex_strength"], 
                                                zeta=params["specific_heat_ratio"],
                                                x_0=params["initial_vortex_centre"],
                                                T_iN_facty=params["background_temperature"],
                                                v_iN_facty=params["background_velocity"],
                                                M_iN_facty=params["mach_number"],
                                                theta=params["angle"])
            
            if "cfl_speed" not in params and "mach_number" in params:
                self.cfl_speed = params["mach_number"]

            else:
                self.cfl_speed = np.linalg.norm(params["background_velocity"])
            
        elif params["initial_condition"] == "entropy_wave":

            if params["problem"] != "compressible_euler":
                raise ValueError(
                    "Entropy wave only applicable to Euler equations")

            if self.d != 1:
                raise NotImplementedError
                
            self.u_0 = Solver.entropy_wave_1d(params["specific_heat_ratio"])
            
            self.cfl_speed = 1.0
            
        else:
            raise NotImplementedError    
            
        if "initial_projection" not in params:
            params["initial_projection"] = "weighted"
      
        # boundary conditions
        self.bcs = {} # initially set homogeneous
        for bc_index in mesh.local_to_bc_index.values():
            self.bcs[bc_index] = [lambda x,t: 0.0 
                                  for e in range(0,self.N_eq)]
 
        # exact solution (assume equal to initial in the cases here)
        self.u = self.u_0
 
        # spatial discretization
        if "form" not in params:
            params["form"] = "weak"
            
        if "solution_representation" not in params:
            params["solution_representation"] = "modal"
            
        if "correction" not in params:
            params["correction"] = "c_dg"
        
        if params["integration_type"] == "quadrature":
            
            if "volume_quadrature_degree" not in params:
                params["volume_quadrature_degree"] = None

            if "facet_quadrature_degree" not in params:
                params["facet_quadrature_degree"] = None

            if "facet_rule" not in params:
                params["facet_rule"] = "lg"
                
            self.discretization = SpatialDiscretization.SimplexQuadratureDiscretization(
                mesh,
                params["solution_degree"], 
                params["volume_quadrature_degree"], 
                params["facet_quadrature_degree"],
                facet_rule=params["facet_rule"],
                form=params["form"],
                solution_representation=params["solution_representation"],
                correction=params["correction"])
                
            if "facet_integration_rule" in params:
                raise NotImplementedError
        
        elif params["integration_type"] == "collocation":
                
            if "volume_collocation_degree" not in params:
                params["volume_collocation_degree"] = None

            if "facet_collocation_degree" not in params:
                params["facet_collocation_degree"] = None

            if "use_lumping" not in params:
                params["use_lumping"] = False
                
            self.discretization = SpatialDiscretization.SimplexCollocationDiscretization(
                mesh,
                params["solution_degree"], 
                params["volume_collocation_degree"], 
                params["facet_collocation_degree"],
                form=params["form"],
                solution_representation=params["solution_representation"],
                use_lumping=params["use_lumping"],
                correction=params["correction"])
            
        else:
            raise NotImplementedError
             
        # temporal discretization
        if self.is_unsteady:

            # build global residual function such that du/dt = R(u)
            self.R = self.discretization.build_global_residual(
                self.f, self.f_star, self.bcs, self.N_eq)

            # choose time marching method
            if "time_integrator" in params:
                self.time_marching_method = params["time_integrator"]

            else:
                self.time_marching_method = "rk44"
                
            if "time_step_scale" in params:
                self.beta = params["time_step_scale"]

            else:
                self.beta = 0.1
                
            self.time_integrator = TimeIntegrator.TimeIntegrator(
                self.R, TimeIntegrator.TimeIntegrator.calculate_time_step(
                    self.discretization, self.cfl_speed, self.beta, h),
                self.time_marching_method)
            self.T = params["final_time"]
        
        # save params
        self.params = params
        
        # LaTeX plotting preamble
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble']=\
            r"\usepackage{amsmath}\usepackage{bm}\usepackage[cm]{sfmath}"
        

    @staticmethod
    def sine_wave(wavelength):
        
        # numpy array of length d wavelengths
        def g(x):
            
            return np.array([np.apply_along_axis(
                lambda xi: 
                    np.prod(
                        np.sin(2.0*np.pi*xi/wavelength)),
                    0,x)])
        return g
    
    @staticmethod
    def isentropic_vortex(eps, zeta, x_0, T_iN_facty=1.0, 
                          v_iN_facty = np.array([1.0,1.0]),
                          M_iN_facty=0.4, theta=np.pi/4.):

        def g(x):
            
            beta = M_iN_facty*eps
            
            delta_x = x - x_0
    
            delta_T = -0.5*(zeta-1.0)*beta**2*np.exp(
                1-np.linalg.norm(delta_x)**2)
            delta_v = beta*np.exp((1-np.linalg.norm(x-x_0)**2)/2.0)*np.array(
                [-delta_x[1], delta_x[0]])
            rho = (1 + delta_T)**(1.0/(zeta-1.0))
            p = (1.0/zeta)*rho**zeta
            v = M_iN_facty*(np.array([np.cos(theta), np.sin(theta)])) + delta_v
        
            return np.concatenate(
                ([rho],rho*v, 
                 [p/(zeta-1) + 0.5*rho*np.linalg.norm(v)**2]))
                
        return lambda x: np.apply_along_axis(g, 0, x)
            
    
    @staticmethod
    def euler_freestream(d,zeta):
        
        def g(x):
            
            q = np.ones([d+2])
            return np.concatenate((q[0:d+1], [q[d+1]/(zeta-1)+ 
                              0.5*q[0]*(np.linalg.norm(q[1:d-1]))**2]))
        
        return lambda x: np.apply_along_axis(g, 0, x)
    
    @staticmethod
    def entropy_wave_1d(zeta):
        
        def g(x):
            
            rho = 2.0 + np.sin(2*np.pi*x)
            u = 1.0
            p = 1.0
            
            return np.array([rho, rho*u,  
                             p/(zeta-1) + 0.5*rho*u**2])
        
        return lambda x: np.apply_along_axis(g, 0, x[0])
        
    def project_function(self,g):
        
        if self.params["initial_projection"] == "unweighted":
            
            return [np.array([self.discretization.P[
                self.discretization.element_to_discretization[k]] @
                    g(self.discretization.x[k])[e] 
                    for e in range(0,self.N_eq)])
                    for k in range(0,self.discretization.mesh.N_el)]
                
        elif self.params["initial_projection"] == "weighted":
            
            return [np.array([self.discretization.P_phys[k] @
                    g(self.discretization.x[k])[e] 
                    for e in range(0,self.N_eq)])
                    for k in range(0,self.discretization.mesh.N_el)]
            
        raise NotImplementedError
        
        
    def run(self,
            results_path=None, 
            write_interval=None,
            print_interval=None,
            prefix="",
            clear_write_dir=False,
            restart=True):
        
        if results_path is None:
            results_path = "../results/" + self.project_title + "/"
            
        if not os.path.exists(results_path):
            os.makedirs(results_path)
            
        # run problem
        if self.params["problem"] == "projection":
            self.u_tilde = self.project_function(self.u_0)
            pickle.dump(self.u_tilde, open(results_path+"res_" 
                                         + str(0) + ".dat", "wb" ))
            
        elif self.params["problem"] == "constant_advection" \
            or self.params["problem"] == "compressible_euler":
        
            # if existing solution exists in data file and user has selected
            # option to restart, begin from that solution.
            if restart and os.path.isfile(results_path+"times.dat"):
                
                    times = pickle.load(open(results_path+"times.dat", "rb"))
                    
                    print("Loaded from time step ", times[-1][0])
                
                    self.load_solution(results_path, times[-1][0])
                    
                    self.u_tilde = self.time_integrator.run(self.u_tilde, self.T,
                                                     results_path,
                                                     write_interval,
                                                     print_interval,
                                                     restart=True,
                                                     prefix=prefix)
                   
            # otherwise, clear write directory and start a new simulation
            else:
                
                os.system("rm -rf "+results_path+"*")
            
                # evaluate initial condition by projection
                self.u_tilde = self.project_function(self.u_0)
                
                pickle.dump(self.u_tilde, open(results_path+"res_" 
                                             + str(0) + ".dat", "wb" ))
            
                self.I_0 = self.calculate_conserved_integral() 
                self.E_0 = self.calculate_energy()
           
                pickle.dump(self.I_0, 
                            open(results_path+"conservation_initial.dat", "wb" ))
                pickle.dump(self.E_0, 
                            open(results_path+"energy_initial.dat", "wb" ))
                
                self.u_tilde = self.time_integrator.run(self.u_tilde, self.T,
                                                      results_path,
                                                      write_interval,
                                                      print_interval,
                                                      restart=False,
                                                      prefix=prefix)
            
            if self.time_integrator.is_done:
                
                self.I_0 = pickle.load(open(results_path+"conservation_initial.dat", "rb" ))
                self.E_0 = pickle.load(open(results_path+"energy_initial.dat", "rb" ))
                self.I_f = self.calculate_conserved_integral()
                self.E_f = self.calculate_energy()
            
        else:
            
            raise NotImplementedError
   
    
    def post_process(self, visualization_resolution=10, 
                     error_quadrature_degree=10,
                     process_visualization=True,
                     process_exact_solution=True):
        
        if process_visualization:
            
            # reconstruct nodal values at integration points
            self.u_h = []
            self.u_h_fac = []
            for k in range(0, self.discretization.mesh.N_el):
                
                self.u_h.append([])
                self.u_h_fac.append([]) 
                for e in range(0, self.N_eq):
                    self.u_h[k].append(self.discretization.V[
                     self.discretization.element_to_discretization[k]]
                        @ self.u_tilde[k][e])
                 
                for zeta in range(0, self.discretization.mesh.N_fac[k]):
                    self.u_h_fac[k].append([])
                    for e in range(0, self.N_eq):
                        self.u_h_fac[k][zeta].append((
                            self.discretization.R[
                     self.discretization.element_to_discretization[k]][zeta] 
                            @ self.u_tilde[k][e]))
                 
            # max and min values at integration points
            self.u_h_min = [min([np.amin(self.u_h[k][e]) 
                               for k in range(0,self.discretization.mesh.N_el)]) 
                          for e in range(0,self.N_eq)] 
            self.u_h_max = [max([np.amax(self.u_h[k][e]) 
                               for k in range(0,self.discretization.mesh.N_el)]) 
                          for e in range(0,self.N_eq)]
            
            # reconstruct nodal values at visualization points
            if self.discretization.basis is None:
                
                self.x_v = self.discretization.x
                self.u_h_v = self.u_h
                self.u_h_v_min = self.u_h_min
                self.u_h_v_max = self.u_h_max
                
            else:
                
                if self.d== 1:
                    
                    ref_volume_points =  np.array(mp.equidistant_nodes(
                        self.d, visualization_resolution))
                    V_map_to_plot = mp.vandermonde(self.discretization.mesh.basis_map,
                                            ref_volume_points[0])
                else:
                    
                    ref_volume_points =  mp.XiaoGimbutasSimplexQuadrature(
                        visualization_resolution, self.d).nodes 
                    V_map_to_plot = mp.vandermonde(self.discretization.mesh.basis_map,
                                            ref_volume_points)
            
                self.u_h_v = []
                self.x_v = []
                for k in range(0, self.discretization.mesh.N_el):
                    
                    # get x at visualization points
                    self.x_v.append((V_map_to_plot 
                                     @ self.discretization.mesh.x_tilde_map[k]).T)
                        
                    self.u_h_v.append([])
                    
                    if self.discretization.solution_representation == "modal":
                        
                        V_plot = mp.vandermonde(self.discretization.basis[
                                    self.discretization.element_to_discretization[k]],
                                    ref_volume_points)
                    elif self.discretization.solution_representation == "nodal":
                        
                        V_plot = mp.vandermonde(self.discretization.basis[
                                    self.discretization.element_to_discretization[k]],
                                    ref_volume_points) @ self.discretization.Vp_inv[
                                    self.discretization.element_to_discretization[k]]
                                        
                    else:
                        
                        raise NotImplementedError
                    
                    for e in range(0, self.N_eq):
                         self.u_h_v[k].append(V_plot @ self.u_tilde[k][e])
                         
            # max and min values at visualization points
            self.u_h_v_max = [min([np.amin(self.u_h_v[k][e]) 
                               for k in range(0,self.discretization.mesh.N_el)]) 
                          for e in range(0,self.N_eq)] 
            self.u_h_v_max = [max([np.amax(self.u_h_v[k][e]) 
                               for k in range(0,self.discretization.mesh.N_el)]) 
                          for e in range(0,self.N_eq)] 
            
            self.x_v_global = np.concatenate([self.x_v[k] 
                                              for k in range(0,self.discretization.mesh.N_el)],
                                             axis=1)
            
            self.u_h_v_global = [np.concatenate([self.u_h_v[k][e]
                                                for k in range(0,self.discretization.mesh.N_el)]) 
                                for e in range(0,self.N_eq)]
            
        
        # evaluate numerical solution on error evaluation points, also
        # evaluate exact solution at error evaluation and visualization points
        if process_exact_solution and self.u is not None:
    
            # reconstruct nodal values at error evaluation points
            if self.discretization.basis is None:
                
                self.x_e = self.discretization.x
                self.u_h_error = self.u_h
                self.J_error = self.discretization.J
                
            else:
                
                if self.d==1:
                    
                    self.ref_error_quadrature = mp.LegendreGaussQuadrature(
                        ceil((error_quadrature_degree-1)/2))
                    self.N_error_pts = self.ref_error_quadrature.nodes.shape[0]
                    
                    V_grad_map_to_error = [mp.vandermonde(
                        self.discretization.mesh.grad_basis_map,
                        self.ref_error_quadrature.nodes)]
                    
                else:
                    
                    self.ref_error_quadrature = mp.XiaoGimbutasSimplexQuadrature(
                        error_quadrature_degree, self.d)
                    self.N_error_pts = self.ref_error_quadrature.nodes.shape[1]
                    
                    V_grad_map_to_error = list(mp.vandermonde(
                        self.discretization.mesh.grad_basis_map,
                        self.ref_error_quadrature.nodes))
                    
                V_map_to_error = mp.vandermonde(self.discretization.mesh.basis_map,
                                        self.ref_error_quadrature.nodes)
                
                self.u_h_error = []
                self.x_e = []
                self.J_error = []
                for k in range(0, self.discretization.mesh.N_el):
                    
                    self.x_e.append((V_map_to_error
                                     @ self.discretization.mesh.x_tilde_map[k]).T)
                    
                    self.u_h_error.append([])
                    
                    if self.discretization.solution_representation == "modal":
                        
                        V_error = mp.vandermonde(self.discretization.basis[
                                    self.discretization.element_to_discretization[k]],
                                    self.ref_error_quadrature.nodes)
                        
                    elif self.discretization.solution_representation == "nodal":
                        
                        V_error = mp.vandermonde(self.discretization.basis[
                                    self.discretization.element_to_discretization[k]],
                                    self.ref_error_quadrature.nodes) @ self.discretization.Vp_inv[
                                    self.discretization.element_to_discretization[k]]
                                        
                    else:
                        
                        raise NotImplementedError
                    
                    for e in range(0, self.N_eq):
                        
                         self.u_h_error[k].append(V_error @ self.u_tilde[k][e])
                        
                    G_error = np.zeros([self.N_error_pts,
                                            self.d, self.d])
                    
                    for m in range(0, self.d):
                        G_error[:,:,m] = V_grad_map_to_error[m] \
                            @ self.discretization.mesh.x_tilde_map[k]
                        
                    self.J_error.append(
                        np.array([np.linalg.det(G_error[j,:,:]) 
                                  for j in range(0,self.N_error_pts)]))
            
            # evaluate exact solution at error evaluation points
            self.u_e = [[self.u(self.x_e[k])[e]
                         for e in range(0, self.N_eq)]
                        for k in range(0,self.discretization.mesh.N_el)]
                   
            
            if process_visualization:
                
                # evaluate exact solution at visualization points
                self.u_v = [[self.u(self.x_v[k])[e] 
                                   for e in range(0, self.N_eq)] 
                                  for k in range(0,self.discretization.mesh.N_el)]
                self.u_v_max = [min([np.amin(self.u_v[k][e]) 
                               for k in range(0,self.discretization.mesh.N_el)]) 
                          for e in range(0,self.N_eq)] 
                self.u_v_max = [max([np.amax(self.u_v[k][e]) 
                               for k in range(0,self.discretization.mesh.N_el)]) 
                          for e in range(0,self.N_eq)] 
                
                self.u_v_global = [np.concatenate(
                [self.u_v[k][e] for k in range(0,self.discretization.mesh.N_el)]) 
                    for e in range(0, self.N_eq)]
               
                
    def calculate_error(self, norm="L2"):
        
        if norm == "L2":
            
            return np.array([np.sqrt(sum([np.dot((self.u_e[k][e] - self.u_h_error[k][e])**2,
                                        self.J_error[k]*self.ref_error_quadrature.weights)
                                for k in range(0,self.discretization.mesh.N_el)]))
                                for e in range(0,self.N_eq)])
        
        else:
            raise NotImplementedError
            
            
    def calculate_difference(self, other_solver, norm="L2"):
        
        if norm == "L2":
            
             # must have the same quadrature points for u_h_error
             return np.array([np.sqrt(sum([np.dot((self.u_h_error[k][e] - other_solver.u_h_error[k][e])**2,
                                            self.J_error[k]*self.ref_error_quadrature.weights)
                                    for k in range(0,self.discretization.mesh.N_el)]))
                                    for e in range(0,self.N_eq)])
        
        
    def calculate_energy(self):
            
            return np.array([sum([self.u_tilde[k][e].T 
                                          @ self.discretization.M_phys[k]
                                          @ self.u_tilde[k][e]
                                    for k in range(0,self.discretization.mesh.N_el)])
                                    for e in range(0,self.N_eq)])
        
        
    def calculate_conserved_integral(self):
        
            # must have the same quadrature points for u_h_error
            return np.array([sum([np.ones(self.discretization.N_omega[
                self.discretization.element_to_discretization[k]]).T @
                                      self.discretization.W[
                                          self.discretization.element_to_discretization[k]] 
                                      @ np.diag(self.discretization.J[k]) @ 
                                      self.discretization.V[
                                          self.discretization.element_to_discretization[k]] 
                                      @ self.u_tilde[k][e]
                                    for k in range(0,self.discretization.mesh.N_el)])
                                    for e in range(0,self.N_eq)])
            
        
    def plot(self,
             filename=None,
             title=None,
             equation_index=0,
             plot_exact=True, 
             plot_numerical=True, 
             plot_curves=True,
             plot_nodes=False,
             markersize=4, 
             geometry_resolution=10,
             u_range = [-1.0,1.0],
             show_fig=True,
             transparent=True):
        
        u_diff = u_range[1] - u_range[0]
        self.color = iter(plt.cm.rainbow(
            np.linspace(0, 1, self.discretization.mesh.N_el)))
        
        # asking for exact solution that doesn't exist
        if plot_exact and self.u is None:
            return ValueError
        
        if self.d== 1:
        
            solution_plot = plt.figure()
            ax = plt.axes()
            plt.xlim([self.discretization.mesh.x_min[0] 
                      - 0.025 * self.discretization.mesh.extent[0], 
                      self.discretization.mesh.x_max[0] 
                      + 0.025 * self.discretization.mesh.extent[0]])
            plt.xlabel("$x$")
        
            # loop through all elemeents
            for k in range(0, self.discretization.mesh.N_el):
                current_color = next(self.color)
                
                # plot exact solution on visualization nodes
                if plot_exact:
                        exact, = ax.plot(self.x_v[k][0,:], 
                           self.u_v[k][equation_index],
                            "-k") 
                        
                # plot numerical solution on visualization nodes
                if plot_numerical:
                    numerical, = ax.plot(self.x_v[k][0,:], 
                           self.u_h_v[k][equation_index],
                            "-", color = current_color) 
                    
                    #plot node positions
                    if plot_nodes:
                        ax.plot(self.discretization.x[k][0,:], 
                           self.u_h[k][equation_index], "o",
                          markersize=markersize,
                          color = current_color)
                        
                        ax.plot(self.discretization.x_fac[k][0][0,0], 
                                self.u_h_fac[k][0][equation_index][0], 
                                    "s", 
                                    markersize=markersize, 
                                    color="black")
                                
                        ax.plot(self.discretization.x_fac[k][1][0,0], 
                                self.u_h_fac[k][1][equation_index][0],
                                    "s", 
                                    markersize=markersize, 
                                    color="black")               
            
            # make title
            if title is not None:
                plt.title(title)
                        
            # make legend labels
            if plot_numerical:
                
                if self.N_eq == 1:
                    numerical.set_label("$U^h(x,t)$")
                else:
                    numerical.set_label("$U_{" 
                                                + str(equation_index+1) 
                                                +"}^h(x,t)$")
            if plot_exact:
                
                if self.N_eq == 1:
                    exact.set_label("$U(x,t)$")
                else:
                    exact.set_label("$U_{" 
                                                + str(equation_index+1) 
                                                +"}(x,t)$")
            ax.legend()
            
            if show_fig:
                plt.show()
            plt.close()
            
            if transparent: 
                ax.set_rasterized(True)
            if filename is None:
                solution_plot.savefig("../plots/"
                                      + self.params["project_title"] 
                                      + "_solution.pdf", 
                                      facecolor="white", 
                                      transparent=transparent)
                
            else:
                ax.set_rasterized(True)
                solution_plot.savefig(filename, facecolor="white", 
                                      transparent=transparent, dpi=300)
                 
        elif self.d == 2:
 
            # place contours
            contours = np.linspace(u_range[0]-0.1*u_diff, 
                                   u_range[1]+0.1*u_diff,25)
            
            # set up plots
            if plot_numerical:
                
                numerical = plt.figure(1)
                ax = plt.axes()
                ax.set_xlim([self.discretization.mesh.x_min[0] 
                             - 0.025 * self.discretization.mesh.extent[0],
                              self.discretization.mesh.x_max[0] 
                              + 0.025 * self.discretization.mesh.extent[0]])
                
                ax.set_ylim([self.discretization.mesh.x_min[1] 
                             - 0.025 * self.discretization.mesh.extent[1],
                              self.discretization.mesh.x_max[1] 
                              + 0.025 * self.discretization.mesh.extent[1]]) 
                ax.set_aspect('equal')
                plt.xlabel("$x_1$")
                plt.ylabel("$x_2$")
                
            if plot_exact: 
                
                exact = plt.figure(2)
                ax2 = plt.axes()
                ax2.set_xlim([self.discretization.mesh.x_min[0] 
                              - 0.025 * self.discretization.mesh.extent[0],
                              self.discretization.mesh.x_max[0] 
                              + 0.025 * self.discretization.mesh.extent[0]])
                
                ax2.set_ylim([self.discretization.mesh.x_min[1] 
                              - 0.025 * self.discretization.mesh.extent[1],
                              self.discretization.mesh.x_max[1] 
                              + 0.025 * self.discretization.mesh.extent[1]])
                ax2.set_aspect('equal')
                plt.xlabel("$x_1$")
                plt.ylabel("$x_2$")
            
            # only works for triangles
            ref_edge_points = SpatialDiscretization.SpatialDiscretization.map_unit_to_facets(
                np.linspace(-1.0,1.0,geometry_resolution))
            V_edge_map = [mp.vandermonde(
                self.discretization.mesh.basis_map, ref_edge_points[zeta])
                          for zeta in range(0,3)]

            # loop through all elements
            for k in range(0, self.discretization.mesh.N_el):
                current_color = next(self.color)
                
                if plot_nodes and plot_numerical:
                    
                    ax.plot(self.discretization.x[k][0,:], 
                            self.discretization.x[k][1,:], "o",
                          markersize=markersize,
                          markeredgecolor='black',
                          color = current_color)
                        
                if plot_nodes and plot_exact:
                    
                    ax2.plot(self.discretization.x[k][0,:],
                             self.discretization.x[k][1,:], "o",
                             markersize=markersize,
                             markeredgecolor='black',
                             color = current_color)
                    
                for zeta in range(0, self.discretization.mesh.N_fac[k]):
                    
                    # plot facet edge curves
                    edge_points = (V_edge_map[zeta] 
                                   @ self.discretization.mesh.x_tilde_map[k]).T
                    
                    if plot_numerical:
                        
                        if plot_curves:
                            ax.plot(edge_points[0,:], 
                                        edge_points[1,:], 
                                        '-', 
                                        color="black")
                        
                        if plot_nodes:
                           
                            # plot facet nodes
                            ax.plot(self.discretization.x_fac[k][zeta][0,:], 
                                    self.discretization.x_fac[k][zeta][1,:],
                                    "s", 
                                    markersize=markersize, 
                                    color="black")
                            
                    if plot_exact:
                        
                        if plot_curves:
                            ax2.plot(edge_points[0,:], 
                                        edge_points[1,:], 
                                        '-', 
                                        color="black")
                        
                        if plot_nodes:
                           
                            # plot facet nodes
                            ax2.plot(
                                self.discretization.x_fac[k][zeta][0,:],
                                self.discretization.x_fac[k][zeta][1,:],
                                    "s", 
                                    markersize=markersize, 
                                    color="black")
                        
            if plot_numerical:
                
                contour_numerical = ax.tricontourf(
                        self.x_v_global[0,:], self.x_v_global[1,:],
                        self.u_h_v_global[equation_index],
                                   levels=contours,
                                   cmap="viridis")
                cbar = numerical.colorbar(contour_numerical)
                if self.N_eq == 1:
                    cbar.ax.set_ylabel("$U^h(\\bm{x},t)$")  
                else:
                    cbar.ax.set_ylabel("$U_{" 
                                        + str(equation_index+1) 
                                        +"}^h(\\bm{x},t)$")
                    
                cbar.set_ticks(np.linspace(u_range[0],u_range[1],11))

                # make title
                if title is not None:
                    plt.title(title)
                
                if filename is None:
                    numerical.savefig("../plots/" 
                                      + self.params["project_title"]
                        + "_numerical.pdf",
                        transparent=transparent,
                        bbox_inches="tight", 
                        facecolor='none', 
                        edgecolor='none', 
                        pad_inches=0)
                        
                else:
                    numerical.savefig(filename, 
                    transparent=transparent, 
                    facecolor='none', 
                    edgecolor='none', 
                    dpi=300)
               
            if plot_exact:
                
                contour_exact = ax2.tricontourf(
                        self.x_v_global[0,:], self.x_v_global[1,:],
                        self.u_v_global[equation_index],
                                   levels=contours,
                                   cmap="viridis")
                cbar_ex = exact.colorbar(contour_exact)
                
                if self.N_eq == 1:
                    cbar_ex.ax.set_ylabel("$U(\\bm{x},t)$")
                else:
                    cbar_ex.ax.set_ylabel("$U_{" 
                                          + str(equation_index+1) 
                                          +"}(\\bm{x},t)$")
                    
                cbar_ex.set_ticks(np.linspace(u_range[0],u_range[1],11))
                exact.savefig(
                    "../plots/" + self.params["project_title"]
                    + "_exact.pdf", bbox_inches="tight", pad_inches=0)
                
            if show_fig:
                plt.show()
            plt.close() 
            
        else:
            raise NotImplementedError

            
    def load_solution(self, results_path=None, time_step=0):
        
        if results_path is None:
              results_path = "../results/" + self.project_title + "/"
        
        self.u_tilde = pickle.load(open(results_path 
                                      + "res_" 
                                      + str(time_step) + ".dat", "rb"))
        
        
    def plot_time_steps(self, results_path=None, 
                        plots_path=None,
                        u_range = [0.0,1.0], 
                        equation_index=0,
                        clear_write_dir=True,
                        make_video=True,
                        framerate=2):
        
        if results_path is None:
              results_path = "../results/" + self.project_title + "/"
              
        if plots_path is None:
              plots_path = "../plots/" + self.project_title + "/"
        
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        elif clear_write_dir:
            os.system("rm -rf "+plots_path+"*")
        
        times = pickle.load(open(results_path + "times.dat", "rb"))
        
        for i in range(0,len(times)):

            self.load_solution(results_path=results_path,
                               time_step=times[i][0])
            self.post_process(visualization_resolution=20,
                              process_exact_solution=False)
            self.plot(filename=plots_path+"frame_"+str(i)+".png",
                           title="$t = " 
                           + str(np.round(times[i][1], decimals=2)) + "$",
                           equation_index=equation_index, plot_numerical=True,
                           plot_exact=False, u_range=u_range, show_fig=False)
            
        if make_video:

            ff_call = "ffmpeg -framerate "+ str(framerate)+ \
            " -i "+plots_path+"frame_%d.png " +plots_path+"video.mp4"
            print(ff_call)
            os.system(ff_call)