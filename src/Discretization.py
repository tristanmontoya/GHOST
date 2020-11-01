# GHOST - Spatial and Temporal Discretization

#from Operator import DenseLinearOperator, DiagonalOperator, Identity
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import quadpy as qp
import modepy as mp


class SpatialDiscretization:
    
    def __init__(self, mesh, element_to_discretization, p,
                 xi_omega, xi_gamma, W, W_gamma):
        
        # mesh
        self.mesh = mesh
        
        # spatial dimension
        self.d = mesh.d  
        
        # map from element index to discretization type index
        self.element_to_discretization = element_to_discretization  
        
        # polynomial degree
        self.p = p
        
        # number of discretization types
        self.Nd = len(self.p) 
        
        # dimension of polynomial space (assume total-degree for now)
        self.Np = [special.comb(self.p[i] + self.d, self.d, 
                                exact=True) for i in range(0,self.Nd)]
        
        # volume nodes (probably should check dimension compatibility)
        self.xi_omega = xi_omega
        self.W = W
        self.N_omega = [xi_omega[i].shape[1] for i in range(0,self.Nd)]
        
        # facet nodes (probably should check dimension compatibility)
        self.xi_gamma = xi_gamma
        self.W_gamma = W_gamma
        self.Nf = [len(xi_gamma[i]) for i in range(0,self.Nd)]
        
        # maybe check if this is compatible w/ mesh.N_gamma[k]
        self.N_gamma = [[xi_gamma[i][gamma].shape[1] 
                         for gamma in range(0,self.Nf[i])] 
                        for i in range(0,self.Nd)]
    
        self.V = None
        self.V_gamma = None
        self.M = None
        self.Minv = None
        self.L = None
    
        self.get_facet_permutation()
        self.set_interpolation()
        self.build_projection()
        self.build_lift()
        
        
    @staticmethod
    def map_unit_to_facets(xi_ref, element_type="triangle"):

        # send (d-1)-dimensional reference facet to facets bounding ref. elem.
        
        N_gamma = xi_ref.shape[0]
        
        if element_type == "triangle":
            bottom = np.reshape(np.array([xi_ref,-np.ones(N_gamma)]),(2,N_gamma))
            left = np.reshape(np.array([-np.ones(N_gamma),np.flip(xi_ref)]),(2,N_gamma))
            hypotenuse = np.array([[1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)],
                       [-1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)]]) @ np.array(
                           [np.sqrt(2.0)*np.flip(xi_ref), np.zeros(N_gamma)])

            # counter-clockwise ordering of all nodes, so neighbouring edges always
            # have reversed order (i.e. permutation is just a flipping)
            return [bottom,hypotenuse,left] 
        
        else:
            raise NotImplementedError
            
        
    def get_facet_permutation(self):
        
        # indexed using mesh local indices
        
        if self.d == 1:
         
            self.facet_permutation = [[np.eye(1), np.eye(1)] 
                                      for k in range(0,self.mesh.K)]
            
        elif self.d==2:
            # assume all nodes are counterclockwise ordered
            self.facet_permutation = [
                [np.eye(self.N_gamma[
                    self.element_to_discretization[k]][gamma])[::-1]
                 for gamma in range(0,self.mesh.Nf[k])] 
                                      for k in range(0,self.mesh.K)]
            
        else:
            raise NotImplementedError
        
        
    def set_interpolation(self):
        # currently only simplex
        
        basis = [mp.simplex_onb(self.d,self.p[i]) for i in range(0,self.Nd)]
        
        if self.d == 1:  
            
            self.V = [mp.vandermonde(basis[i],self.xi_omega[i][0,:]) 
                      for i in range(0,self.Nd)]
              
            self.V_gamma = [[mp.vandermonde(basis[i],self.xi_gamma[i][gamma][0,:]) 
                        for gamma in range(0,self.Nf[i])] for i in range(0,self.Nd)]
        
        else:
            
            self.V = [mp.vandermonde(basis[i],self.xi_omega[i]) 
                      for i in range(0,self.Nd)]
        
        self.V_gamma = [[mp.vandermonde(basis[i],self.xi_gamma[i][gamma]) 
                        for gamma in range(0,self.Nf[i])]
                        for i in range(0,self.Nd)]
        
        
    def build_local_mass(self):
        
        self.M = [self.V[i].T @ self.W[i] @ self.V[i] for i in range(0,self.Nd)]
       
        
    def invert_local_mass(self):
        
        if self.M is None:
            
            self.build_local_mass()
            
        self.Minv = [self.M[i] for i in range(0, self.Nd)]
        
    
    def build_projection(self):
        
        if self.Minv is None:
            
            self.invert_local_mass()
        
        self.P = [self.Minv[i] @ self.V[i].T @ self.W[i] 
                  for i in range(0,self.Nd)]
        
        
    def build_lift(self):
        
        if self.Minv is None:
            
            self.invert_local_mass()
        
        self.L = [[self.Minv @ self.V_gamma[i][gamma].T @ self.W_gamma[i][gamma] 
                   for gamma in range(0,self.Nf[i])]
                  for i in range(0,self.Nd)]
        
    
    def set_differentiation(self, D):
        
        self.D = D
        
    
    def build_weak_residual(self, f, f_star):
        
        raise NotImplementedError


    def build_strong_residual(self, f, f_star):
        
        raise NotImplementedError
        
        
    def plot(self, markersize=5, resolution=20):
        
        if self.d == 1:
            
            x_L = np.amin(self.mesh.v[0,:])
            x_R = np.amax(self.mesh.v[0,:])
            L = x_R - x_L
            
            meshplt = plt.figure()
            ax = plt.axes()
            plt.xlim([x_L - 0.1 * L, x_R + 0.1 * L])
            plt.ylim([-0.1 * L, 0.1 * L])
            ax.get_xaxis().set_visible(False)  
            ax.get_yaxis().set_visible(False)  
            ax.set_aspect('equal')
            plt.axis('off')
        
            color = iter(plt.cm.rainbow(np.linspace(0, 1, self.mesh.K)))
            
            # loop through all elemeents
            for k in range(0, self.mesh.K):
                
                # plot flux nodes
                x_omega = np.array([[self.mesh.X[k](self.xi_omega[self.element_to_discretization[k]][0,i]) 
                                     for i in range(0,self.N_omega[self.element_to_discretization[k]])]])
                
                ax.plot(x_omega[0,:], 
                        np.zeros(self.N_omega[self.element_to_discretization[k]]),
                        "o", markersize=markersize, color = next(color))
                
                # plot vertices
                for gamma in range(0, self.mesh.Nv_local[k]):
                    ax.plot(self.mesh.v[0,self.mesh.local_to_vertex[k][gamma][0]],0.0,
                            's', markersize=markersize, color="black")
    
            plt.show()
            meshplt.savefig("../plots/" + self.mesh.name + "_nodes.pdf",
                            bbox_inches=0, pad_inches=0)
               
        elif self.d==2:
            
            x_L = np.amin(self.mesh.v[0,:])
            x_H = np.amax(self.mesh.v[0,:])
            y_L = np.amin(self.mesh.v[1,:])
            y_H = np.amax(self.mesh.v[1,:])
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
        
            color = iter(plt.cm.rainbow(np.linspace(0, 1, self.mesh.K)))
            
            # only works for triangles, otherwise need to do this for each discretization 
            # type and put in loop over k
            ref_edge_points = SpatialDiscretization.map_unit_to_facets(
                np.linspace(-1.0,1.0,resolution))

            # loop through all elemeents
            for k in range(0, self.mesh.K):
         
                # plot flux nodes
                x_omega = np.array([self.mesh.X[k](self.xi_omega[self.element_to_discretization[k]][:,i]) 
                                   for i in range(0,self.N_omega[self.element_to_discretization[k]])]).T  
                ax.plot(x_omega[0,:], x_omega[1,:],"o", markersize=markersize, color = next(color))
                
                
                # plot facets
                for gamma in range(0, 
                                   self.Nf[self.element_to_discretization[k]]):
                    
                    # facet edges
                    edge_points = np.array([
                    self.mesh.X[k](ref_edge_points[gamma][:,i]) 
                                   for i in range(0,resolution)]).T  
                    
                    ax.plot(edge_points[0,:], edge_points[1,:], '-', color="black")
                    
                    # facet nodes
                    x_gamma = np.array([
                    self.mesh.X[k](self.xi_gamma[self.element_to_discretization[k]][gamma][:,i]) 
                                   for i in range(0,self.N_gamma[
                                           self.element_to_discretization[k]][gamma])]).T  
                    
                    ax.plot(x_gamma[0,:], x_gamma[1,:], "o", markersize=markersize, color = "black")
                                 
            plt.show()
            meshplt.savefig("../plots/" + self.mesh.name + "_discretization.pdf",
                            bbox_inches="tight", pad_inches=0)
            

class TimeIntegrator:
    
    def __init__(self, residual, disc_type="rk45"):
        
        raise NotImplementedError
            