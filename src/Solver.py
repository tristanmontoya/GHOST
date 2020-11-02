# GHOST - Solver Components

import Mesh
import Problem
import Discretization

import numpy as np
import matplotlib.pyplot as plt


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
            return np.apply_along_axis(lambda xi: np.prod(np.sin(2.0*np.pi*xi/wavelength)), 0,x)
        
        return g
        
    def project_function(self,g):
        
        return [self.discretization.P[self.discretization.element_to_discretization[k]] @
                g(self.discretization.x_omega[k]) for k in range(0,self.discretization.mesh.K)]
            
            
        raise NotImplementedError
        
    def run(self):
        if self.params["problem"] == "projection":
            self.uhat = self.project_function(self.u_0)
        
        else:
            raise NotImplementedError
    
    def plot_exact_solution(self, markersize=4, resolution=20):
        
        if self.d == 1:
            
            x_L = np.amin(self.discretization.mesh.v[0,:])
            x_R = np.amax(self.discretization.mesh.v[0,:])
            L = x_R - x_L
            
            meshplt = plt.figure()
            ax = plt.axes()
            plt.xlim([x_L - 0.1 * L, x_R + 0.1 * L])
            #plt.ylim([-0.1 * L, 0.1 * L])
            #ax.get_xaxis().set_visible(False)  
            #ax.get_yaxis().set_visible(False)  
            #ax.set_aspect('equal')
            #plt.axis('off')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$\mathcal{U}(x,T)$')
        
            color = iter(plt.cm.rainbow(
                np.linspace(0, 1, self.discretization.mesh.K)))
            
            # loop through all elemeents
            for k in range(0, self.discretization.mesh.K):
                
                # plot on volume nodes
                ax.plot(self.discretization.x_omega[k][0,:], 
                        self.u(self.discretization.x_omega[k]),
                        "-", 
                        markersize=markersize, 
                        color = next(color))
    
            plt.show()
            meshplt.savefig("../plots/" + self.params["project_title"] + "_exact.pdf")
            
        elif self.d == 2:
            
            x_L = np.amin(self.discretization.mesh.v[0,:])
            x_H = np.amax(self.discretization.mesh.v[0,:])
            y_L = np.amin(self.discretization.mesh.v[1,:])
            y_H = np.amax(self.discretization.mesh.v[1,:])
            W = x_H - x_L
            H = y_H - y_L
            
            meshplt = plt.figure()
            ax = plt.axes()
            plt.xlim([x_L - 0.1 * W, x_H + 0.1 * W])
            plt.ylim([y_L - 0.1 * H, y_H + 0.1 * H])
            ax.get_xaxis().set_visible(False)  
            ax.get_yaxis().set_visible(False)  
            ax.set_aspect('equal')
            plt.axis('off')
        
            color = iter(plt.cm.rainbow(np.linspace(0, 1, self.discretization.mesh.K)))
            
            # only works for triangles, otherwise need to do this 
            # for each discretization type and put in loop over k
            ref_edge_points = Discretization.SpatialDiscretization.map_unit_to_facets(
                np.linspace(-1.0,1.0,resolution))

            # loop through all elemeents
            for k in range(0, self.discretization.mesh.K):
         
                # plot volume nodes
                x = self.discretization.x_omega[k][0,:]
                y = self.discretization.x_omega[k][1,:]
                
                ax.plot(x, y, "o",
                      markersize=markersize,
                      color = next(color))
                
                ax.tricontourf(x,y,
                              self.u(np.array([x,y])),
                              vmin=-1.0, vmax=1.0, levels=30, cmap="RdBu_r")
                
                
                for gamma in range(0, self.discretization.mesh.Nf[k]):
                    
                    # plot facet edge curves
                    edge_points = np.array([
                    self.discretization.mesh.X[k](ref_edge_points[gamma][:,i]) 
                                   for i in range(0,resolution)]).T  
                    
                    ax.plot(edge_points[0,:], 
                            edge_points[1,:], 
                            '-', 
                            color="black")
                    
                    # plot facet nodes
                    ax.plot(self.discretization.x_gamma[k][gamma][0,:], 
                            self.discretization.x_gamma[k][gamma][1,:],
                            "o", 
                            markersize=markersize, 
                            color = "black")
                                 
            plt.show()
            
            meshplt.savefig(
                "../plots/" + self.params["project_title"] + "_numerical.pdf",
                            bbox_inches="tight", pad_inches=0)
        
        else:
            raise NotImplementedError

    def plot_numerical_solution(self, markersize=4, resolution=20):
        
        if self.d == 1:
            
            x_L = np.amin(self.discretization.mesh.v[0,:])
            x_R = np.amax(self.discretization.mesh.v[0,:])
            L = x_R - x_L
            
            meshplt = plt.figure()
            ax = plt.axes()
            plt.xlim([x_L - 0.1 * L, x_R + 0.1 * L])
            #plt.ylim([-0.1 * L, 0.1 * L])
            #ax.get_xaxis().set_visible(False)  
            #ax.get_yaxis().set_visible(False)  
            #ax.set_aspect('equal')
            #plt.axis('off')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$\mathcal{U}(x,T)$')
        
            color = iter(plt.cm.rainbow(
                np.linspace(0, 1, self.discretization.mesh.K)))
            
            # loop through all elemeents
            for k in range(0, self.discretization.mesh.K):
                
                # plot on volume nodes
                ax.plot(self.discretization.x_omega[k][0,:], 
                        self.discretization.V[self.discretization.element_to_discretization[k]] @ self.uhat[k],
                        "-", 
                        markersize=markersize, 
                        color = next(color))
    
            plt.show()
            meshplt.savefig("../plots/" + self.params["project_title"] + "_exact.pdf")
            
        elif self.d == 2:
            
            x_L = np.amin(self.discretization.mesh.v[0,:])
            x_H = np.amax(self.discretization.mesh.v[0,:])
            y_L = np.amin(self.discretization.mesh.v[1,:])
            y_H = np.amax(self.discretization.mesh.v[1,:])
            W = x_H - x_L
            H = y_H - y_L
            
            meshplt = plt.figure()
            ax = plt.axes()
            plt.xlim([x_L - 0.1 * W, x_H + 0.1 * W])
            plt.ylim([y_L - 0.1 * H, y_H + 0.1 * H])
            ax.get_xaxis().set_visible(False)  
            ax.get_yaxis().set_visible(False)  
            ax.set_aspect('equal')
            plt.axis('off')
        
            color = iter(plt.cm.rainbow(np.linspace(0, 1, self.discretization.mesh.K)))
            
            # only works for triangles, otherwise need to do this 
            # for each discretization type and put in loop over k
            ref_edge_points = Discretization.SpatialDiscretization.map_unit_to_facets(
                np.linspace(-1.0,1.0,resolution))

            # loop through all elemeents
            for k in range(0, self.discretization.mesh.K):
         
                # plot volume nodes
                x = self.discretization.x_omega[k][0,:]
                y = self.discretization.x_omega[k][1,:]
                
                ax.plot(x, y, "o",
                      markersize=markersize,
                      color = next(color))
                
                ax.tricontourf(x,y,
                               self.discretization.V[self.discretization.element_to_discretization[k]] @ self.uhat[k],
                               self.u(np.array([x,y])),
                               vmin=-1.0, vmax=1.0, levels=30, cmap="RdBu_r")
                
                
                for gamma in range(0, self.discretization.mesh.Nf[k]):
                    
                    # plot facet edge curves
                    edge_points = np.array([
                    self.discretization.mesh.X[k](ref_edge_points[gamma][:,i]) 
                                   for i in range(0,resolution)]).T  
                    
                    ax.plot(edge_points[0,:], 
                            edge_points[1,:], 
                            '-', 
                            color="black")
                    
                    # plot facet nodes
                    ax.plot(self.discretization.x_gamma[k][gamma][0,:], 
                            self.discretization.x_gamma[k][gamma][1,:],
                            "o", 
                            markersize=markersize, 
                            color = "black")
                                 
                
            plt.show()
            
            meshplt.savefig(
                "../plots/" + self.params["project_title"] + "_exact.pdf",
                            bbox_inches="tight", pad_inches=0)
        
        else:
            raise NotImplementedError