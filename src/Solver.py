# GHOST - Solver Components

import Problem
import Discretization

import numpy as np
import modepy as mp
import matplotlib.pyplot as plt


class Solver:
    
    def __init__(self, params, mesh):
        
        # set spatial dimension since this is used everywhere
        self.d = mesh.d
        
        # physical problem
        if params["problem"] == "constant_advection":
            
            self.f = Problem.ConstantAdvectionPhysicalFlux(params["wave_speed"])
            self.f_star = Problem.ConstantAdvectionNumericalFlux(
                params["wave_speed"], 
                params["upwind_parameter"])
            
            self.is_unsteady = True
            self.N_eq = 1
            
        elif params["problem"] == "projection":
            self.is_unsteady = False
            self.N_eq = 1
        
        elif params["problem"] == "compressible_euler":
            self.is_unsteady = True
            self.N_eq = self.d + 2
        
        else:
            raise NotImplementedError
        
        # initial conditions
        if params["initial_condition"] == "sine":
            if "wavelength" in params:
                self.u_0 = [Solver.sine_wave(params["wavelength"])]
            else:
                self.u_0 = [Solver.sine_wave(np.ones(self.d))]
            
        else:
            raise NotImplementedError
            
        # exact solution
        if params["problem"] == "projection":
            self.u = self.u_0
        
        # spatial discretization
        if params["integration_type"] == "quadrature":
            
            if "volume_quadrature_degree" not in params:
                params["volume_quadrature_degree"] = None
            if "facet_quadrature_degree" not in params:
                params["facet_quadrature_degree"] = None
                
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
            
        # time discretization
        if self.is_unsteady:
             pass
             # self.time_integrator = Discretization.TimeIntegrator(self.dt)
      
        # boundary conditions
        self.bcs = {} # initially set homogeneous
        for bc_index in self.discretization.mesh.local_to_bc_index.values():
            self.bcs[bc_index] = lambda x,t: np.zeros(self.N_eq)
         
        
        # save params
        self.params = params
        
        
    @staticmethod
    def sine_wave(wavelength):
        # numpy array of length d wavelengths
        def g(x):
            return np.apply_along_axis(
                lambda xi: 
                    np.prod(
                        np.sin(2.0*np.pi*xi/wavelength)),
                    0,x)
        
        return g
        
    
    def project_function(self,g):
        # takes list of functions as input
        
        return [[self.discretization.P[
            self.discretization.element_to_discretization[k]] @
                g[e](self.discretization.x_omega[k]) 
                for e in range(0,self.N_eq)]
                for k in range(0,self.discretization.mesh.K)]
            
            
        raise NotImplementedError
        
        
    def get_time_step(self, CFL, L):
        pass
        
    
    def run(self):
        
        # run problem
        if self.params["problem"] == "projection":
            self.u_hat = self.project_function(self.u_0)
        
        else:
            raise NotImplementedError
        
   
    def post_process(self, solution_resolution=10):
        
        # reconstruct nodal values at integration points
        self.u_h = []
        self.u_h_gamma = []
        for k in range(0, self.discretization.mesh.K):
            self.u_h.append([])
            self.u_h_gamma.append([]) 
            for e in range(0, self.N_eq):
                self.u_h[k].append(self.discretization.V[
                 self.discretization.element_to_discretization[k]] @ self.u_hat[k][e])
             
            for gamma in range(0, self.discretization.mesh.Nf[k]):
                self.u_h_gamma[k].append([])
                for e in range(0, self.N_eq):
                    self.u_h_gamma[k][gamma].append((self.discretization.V_gamma[
                 self.discretization.element_to_discretization[k]][gamma] 
                        @ self.u_hat[k][e]))
             
        # max and min values at integration points
        self.u_hmin = [min([np.amin(self.u_h[k][e]) 
                           for k in range(0,self.discretization.mesh.K)]) 
                      for e in range(0,self.N_eq)] 
        self.u_hmax = [max([np.amax(self.u_h[k][e]) 
                           for k in range(0,self.discretization.mesh.K)]) 
                      for e in range(0,self.N_eq)] 

        # reconstruct nodal values at visualization points
        if self.discretization.basis is None:
            self.x_v = self.discretization.x_omega
            self.u_hv = self.u_h
            self.u_hvmin = self.u_hmin
            self.u_hmax = self.u_hmax
        else:
            if self.d == 1:
                ref_volume_points =  np.array(mp.equidistant_nodes(
                    self.d, solution_resolution))
                V_geo_to_plot = mp.vandermonde(self.discretization.mesh.basis_geo,
                                        ref_volume_points[0])
            else:
                ref_volume_points =  mp.XiaoGimbutasSimplexQuadrature(
                    solution_resolution, self.d).nodes 
                V_geo_to_plot = mp.vandermonde(self.discretization.mesh.basis_geo,
                                        ref_volume_points)
        
            self.u_hv = []
            self.x_v = []
            for k in range(0, self.discretization.mesh.K):
                
                # get x at visualization points
                self.x_v.append((V_geo_to_plot 
                                 @ self.discretization.mesh.xhat_geo[k]).T)
                    
                self.u_hv.append([])
                V_plot = mp.vandermonde(self.discretization.basis[
                            self.discretization.element_to_discretization[k]],
                            ref_volume_points)
                for e in range(0, self.N_eq):
                     self.u_hv[k].append(V_plot @ self.u_hat[k][e])
                     
        # max and min values at visualization points
        self.u_hvmin = [min([np.amin(self.u_hv[k][e]) 
                           for k in range(0,self.discretization.mesh.K)]) 
                      for e in range(0,self.N_eq)] 
        self.u_hvmax = [max([np.amax(self.u_hv[k][e]) 
                           for k in range(0,self.discretization.mesh.K)]) 
                      for e in range(0,self.N_eq)] 
        
        if self.u is not None:
            
            # evaluate exact solution at visualization points
            self.u_v = [[self.u[e](self.x_v[k]) 
                               for e in range(0, self.N_eq)] 
                              for k in range(0,self.discretization.mesh.K)]
            self.u_vmin = [min([np.amin(self.u_v[k][e]) 
                           for k in range(0,self.discretization.mesh.K)]) 
                      for e in range(0,self.N_eq)] 
            self.u_vmax = [max([np.amax(self.u_v[k][e]) 
                           for k in range(0,self.discretization.mesh.K)]) 
                      for e in range(0,self.N_eq)] 
        
      
        # global visualization (concatenated)
        self.x_v_global = np.concatenate(
            [self.x_v[k] for k in range(0,self.discretization.mesh.K)], axis=1)
        self.u_v_global = [np.concatenate(
            [self.u_v[k][e] for k in range(0,self.discretization.mesh.K)]) 
            for e in range(0, self.N_eq)]
        self.u_hv_global = [np.concatenate(
            [self.u_hv[k][e] for k in range(0,self.discretization.mesh.K)]) 
            for e in range(0, self.N_eq)]
        
    
    def plot(self, 
             equation_index=0,
             plot_exact=True, 
             plot_numerical=True, 
             plot_curves=True,
             plot_nodes=False,
             markersize=4, 
             geometry_resolution=10,
             u_range = [-1.0,1.0]):
        
        # asking for exact solution that doesn't exist
        if plot_exact and self.u is None:
            return ValueError
        
        if self.d == 1:
        
            solution_plot = plt.figure()
            ax = plt.axes()
            plt.xlim([self.discretization.mesh.xmin[0] 
                      - 0.025 * self.discretization.mesh.extent[0], 
                      self.discretization.mesh.xmax[0] 
                      + 0.025 * self.discretization.mesh.extent[0]])
            plt.xlabel("$x$")
        
            # loop through all elemeents
            for k in range(0, self.discretization.mesh.K):
                current_color = next(self.discretization.color)
                
                # plot exact solution on visualization nodes
                if plot_exact:
                        exact, = ax.plot(self.x_v[k][0,:], 
                           self.u_v[k][equation_index],
                            "-k") 
                        
                # plot numerical solution on visualization nodes
                if plot_numerical:
                    numerical, = ax.plot(self.x_v[k][0,:], 
                           self.u_hv[k][equation_index],
                            "-", color = current_color) 
                    
                    #plot node positions
                    if plot_nodes:
                        ax.plot(self.discretization.x_omega[k][0,:], 
                           self.u_h[k][equation_index], "o",
                          markersize=markersize,
                          color = current_color)
                        
                        ax.plot(self.discretization.x_gamma[k][0][0,0], 
                                self.u_h_gamma[k][0][equation_index][0], 
                                    "s", 
                                    markersize=markersize, 
                                    color="black")
                                
                        ax.plot(self.discretization.x_gamma[k][1][0,0], 
                                self.u_h_gamma[k][1][equation_index][0],
                                    "s", 
                                    markersize=markersize, 
                                    color="black")                     
                    
            # make legend labels
            if plot_numerical:
                
                if self.N_eq == 1:
                    numerical.set_label("$\mathcal{U}^h(x,t)$")
                else:
                    numerical.set_label("$\mathcal{U}_{" 
                                                + str(equation_index) 
                                                +"^h(x,t)$")
            if plot_exact:
                
                if self.N_eq == 1:
                    exact.set_label("$\mathcal{U}(x,t)$")
                else:
                    exact.set_label("$\mathcal{U}_{" 
                                                + str(equation_index) 
                                                +"(x,t)$")
            ax.legend()
            plt.show()
            
            solution_plot.savefig("../plots/" + self.params["project_title"] + 
                            "_exact.pdf")
            
        elif self.d == 2:
            
            # place contours
            contours = np.linspace(u_range[0], 
                                   u_range[1],100)
            
            # set up plots
            if plot_numerical:
                
                numerical = plt.figure(1)
                ax = plt.axes()
                ax.set_xlim([self.discretization.mesh.xmin[0] 
                             - 0.025 * self.discretization.mesh.extent[0],
                              self.discretization.mesh.xmax[0] 
                              + 0.025 * self.discretization.mesh.extent[0]])
                
                ax.set_ylim([self.discretization.mesh.xmin[1] 
                             - 0.025 * self.discretization.mesh.extent[1],
                              self.discretization.mesh.xmax[1] 
                              + 0.025 * self.discretization.mesh.extent[1]]) 
                ax.set_aspect('equal')
                plt.xlabel("$x_1$")
                plt.ylabel("$x_2$")
                
            if plot_exact: 
                
                exact = plt.figure(2)
                ax2 = plt.axes()
                ax2.set_xlim([self.discretization.mesh.xmin[0] 
                              - 0.025 * self.discretization.mesh.extent[0],
                              self.discretization.mesh.xmax[0] 
                              + 0.025 * self.discretization.mesh.extent[0]])
                
                ax2.set_ylim([self.discretization.mesh.xmin[1] 
                              - 0.025 * self.discretization.mesh.extent[1],
                              self.discretization.mesh.xmax[1] 
                              + 0.025 * self.discretization.mesh.extent[1]])
                ax2.set_aspect('equal')
                plt.xlabel("$x_1$")
                plt.ylabel("$x_2$")
            
            # only works for triangles, otherwise need to do this 
            # for each discretization type and put in loop over k
            ref_edge_points = Discretization.SpatialDiscretization.map_unit_to_facets(
                np.linspace(-1.0,1.0,geometry_resolution))
            V_edge_geo = [mp.vandermonde(
                self.discretization.mesh.basis_geo, ref_edge_points[gamma])
                          for gamma in range(0,3)]

            # loop through all elements
            for k in range(0, self.discretization.mesh.K):
                current_color = next(self.discretization.color)
                
                if plot_nodes and plot_numerical:
                    
                    ax.plot(self.discretization.x_omega[k][0,:], 
                            self.discretization.x_omega[k][1,:], "o",
                          markersize=markersize,
                          markeredgecolor='black',
                          color = current_color)
                        
                if plot_nodes and plot_exact:
                    
                    ax2.plot(self.discretization.x_omega[k][0,:],
                             self.discretization.x_omega[k][1,:], "o",
                             markersize=markersize,
                             markeredgecolor='black',
                             color = current_color)
                    
                for gamma in range(0, self.discretization.mesh.Nf[k]):
                    
                    # plot facet edge curves
                    edge_points = (V_edge_geo[gamma] 
                                   @ self.discretization.mesh.xhat_geo[k]).T
                    
                    if plot_numerical:
                        
                        if plot_curves:
                            ax.plot(edge_points[0,:], 
                                        edge_points[1,:], 
                                        '-', 
                                        color="black")
                        
                        if plot_nodes:
                           
                            # plot facet nodes
                            ax.plot(self.discretization.x_gamma[k][gamma][0,:], 
                                    self.discretization.x_gamma[k][gamma][1,:],
                                    "s", 
                                    markersize=0.75*markersize, 
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
                                self.discretization.x_gamma[k][gamma][0,:],
                                self.discretization.x_gamma[k][gamma][1,:],
                                    "s", 
                                    markersize=0.75*markersize, 
                                    color="black")
                        
            if plot_numerical:
                contour_numerical = ax.tricontourf(
                        self.x_v_global[0,:], self.x_v_global[1,:],
                        self.u_hv_global[equation_index],
                                   levels=contours,
                                   cmap="jet")
                cbar = numerical.colorbar(contour_numerical)
                if self.N_eq == 1:
                    cbar.ax.set_ylabel("$\mathcal{U}^h(\mathbf{x},t)$")  
                else:
                    cbar.ax.set_ylabel("$\mathcal{U}_{" 
                                       + str(equation_index) 
                                       +"}^h(\mathbf{x},t)$")
                cbar.set_ticks(np.linspace(u_range[0],u_range[1],10))
                numerical.savefig(
                    "../plots/" + self.params["project_title"]
                    + "_numerical.pdf", bbox_inches="tight", pad_inches=0)
            
            if plot_exact:
                contour_exact = ax2.tricontourf(
                        self.x_v_global[0,:], self.x_v_global[1,:],
                        self.u_v_global[equation_index],
                                   levels=contours,
                                   cmap="jet")
                cbar_ex = exact.colorbar(contour_exact)
                if self.N_eq == 1:
                    cbar_ex.ax.set_ylabel("$\mathcal{U}(\mathbf{x},t)$")
                else:
                    cbar_ex.ax.set_ylabel("$\mathcal{U}_{" 
                                          + str(equation_index) 
                                          +"}(\mathbf{x},t)$")
                cbar_ex.set_ticks(np.linspace(u_range[0],u_range[1],10))
                exact.savefig(
                    "../plots/" + self.params["project_title"]
                    + "_exact.pdf", bbox_inches="tight", pad_inches=0)
            
            plt.show()
            
        else:
            raise NotImplementedError