# GHOST - Spatial and Temporal Discretization

import numpy as np
from scipy import special
from math import floor, ceil
import modepy as mp
import matplotlib.pyplot as plt

NORMAL_TOL = 1.0e-8

class SpatialDiscretization:
    
    def __init__(self, mesh, element_to_discretization, p,
                 xi_omega, xi_gamma, W, W_gamma,
                 n_hat, form = "weak", name=None):
        
        # mesh
        self.mesh = mesh
        
        if name is None:
            self.name = self.mesh.name
        else:
            self.name=name
        
        self.form=form
        
        # spatial dimension
        self.d = mesh.d  
        
        # map from element index to discretization type index
        self.element_to_discretization = element_to_discretization  
        
        # polynomial degree
        self.p = p
        
        # reference element normal vectors
        self.n_hat = n_hat
        
        # number of discretization types
        self.Nd = len(self.p) 
        
        # dimension of polynomial space (assume total-degree for now)
        self.Np = [special.comb(self.p[i] + self.d, self.d, 
                                exact=True) for i in range(0,self.Nd)]
        
        # volume nodes/weights (probably should check dimension compatibility)
        self.xi_omega = xi_omega
        self.W = W
        self.N_omega = [xi_omega[i].shape[1] for i in range(0,self.Nd)]
        
        # facet nodes/weights (probably should check dimension compatibility)
        self.xi_gamma = xi_gamma
        self.W_gamma = W_gamma
        self.Nf = [len(xi_gamma[i]) for i in range(0,self.Nd)]
        self.N_gamma = [[xi_gamma[i][gamma].shape[1] 
                         for gamma in range(0,self.Nf[i])] 
                        for i in range(0,self.Nd)]
        
        # initialize and build operators
        self.V = None
        self.V_gamma = None
        self.M = None
        self.Minv = None
        self.L = None
        self.facet_permutation = None
    
        self.build_facet_permutation()
        self.build_interpolation()
        self.build_projection()
        self.build_lift()
        self.build_differentiation()
        
        # evaluate grid nodes, normals and metric
        self.x_omega = []
        self.x_gamma = []
        self.x_prime_omega = []
        self.x_prime_gamma = []
        self.x_prime_inv_omega = []
        self.x_prime_inv_gamma = []
        self.J_omega = [] 
        self.J_gamma = []
        self.Jf_gamma = []
        self.n_gamma = [] 
        self.map_nodes()
        self.build_physical_mass_matrix()
        
        # init residual function
        self.residual = None
        
        # assign a colour to each element
        self.color = iter(plt.cm.rainbow(
            np.linspace(0, 1, self.mesh.K)))
        
        
    @staticmethod
    def map_unit_to_facets(xi_ref, element_type="triangle"):

        # send (d-1)-dimensional reference facet to facets bounding ref. elem.
        
        N_gamma = xi_ref.shape[0]
        
        if element_type == "triangle":
            bottom = np.reshape(np.array([xi_ref,-np.ones(N_gamma)]),
                                (2,N_gamma))
            left = np.reshape(np.array([-np.ones(N_gamma),
                                        np.flip(xi_ref)]),(2,N_gamma))
            hypotenuse = np.array([[1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)],
                       [-1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)]]) @ np.array(
                           [np.sqrt(2.0)*np.flip(xi_ref), np.zeros(N_gamma)])

            # counter-clockwise ordering of all nodes, so neighbouring 
            # edges always have reversed order 
            # (i.e. permutation is just a flipping)
            return [bottom,hypotenuse,left] 
        
        else:
            raise NotImplementedError
            
        
    def build_facet_permutation(self):
        
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
        
        
    def build_interpolation(self):
        
        # currently only simplex total-degree space
        self.basis = [mp.simplex_onb(self.d,self.p[i]) 
                      for i in range(0,self.Nd)]
        
        if self.d == 1:  
            
            self.V = [mp.vandermonde(self.basis[i], self.xi_omega[i][0,:]) 
                      for i in range(0,self.Nd)]
              
            self.V_gamma = [[mp.vandermonde(self.basis[i],
                                            self.xi_gamma[i][gamma][0,:]) 
                        for gamma in range(0,self.Nf[i])] 
                            for i in range(0,self.Nd)]
        
        else:
            
            self.V = [mp.vandermonde(self.basis[i], self.xi_omega[i]) 
                      for i in range(0,self.Nd)]
        
            self.V_gamma = [[mp.vandermonde(self.basis[i],
                                            self.xi_gamma[i][gamma]) 
                            for gamma in range(0,self.Nf[i])]
                            for i in range(0,self.Nd)]
        
        
    def build_local_mass(self):
        
        self.M = [self.V[i].T @ self.W[i] @ self.V[i] 
                  for i in range(0,self.Nd)]
       
        
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
        
        self.L = [[self.Minv[i] @ self.V_gamma[i][gamma].T 
                   @ self.W_gamma[i][gamma]
                   for gamma in range(0,self.Nf[i])]
                  for i in range(0,self.Nd)]
        
    
    def build_differentiation(self):
    
        self.grad_basis = [mp.grad_simplex_onb(self.d,self.p[i])
                           for i in range(0, self.Nd)]
        
        if self.d==1:
            self.V_xi = [[mp.vandermonde(self.grad_basis[i], self.xi_omega[i])]
                         for i in range(0,self.Nd)]
        else:
            self.V_xi =[list(mp.vandermonde(self.grad_basis[i], 
                                            self.xi_omega[i]))
                        for i in range(0,self.Nd)]
            
        if self.P is None:
            self.build_projection()
        
        self.Dhat = [[self.P[i] @ self.V_xi[i][m] for m in range(0, self.d)]
                     for i in range(0, self.Nd)]
                              
        
    def build_physical_mass_matrix(self):
        
        self.M_J = [self.V[self.element_to_discretization[k]].T \
            @ self.W[self.element_to_discretization[k]] @ np.diag(self.J_omega[k]) \
                @ self.V[self.element_to_discretization[k]] for k in range(0,self.mesh.K)]
        
        self.M_J_inv = [np.linalg.inv(self.M_J[k]) for k in range(0, self.mesh.K)]
    
    
    def map_nodes(self):
        
        # go through each element (note: this could be done in parallel)
        for k in range(0, self.mesh.K):
            i = self.element_to_discretization[k]
            
            # volume node positions
            self.x_omega.append((mp.vandermonde(
                self.mesh.basis_geo,
                self.xi_omega[i]) 
                @ self.mesh.xhat_geo[k]).T)
            
            # facet node positions
            self.x_gamma.append([(mp.vandermonde(self.mesh.basis_geo,
            self.xi_gamma[self.element_to_discretization[k]][gamma]) 
                                  @ self.mesh.xhat_geo[k]).T
                                 for gamma in range(0,self.mesh.Nf[k])])
        
            # jacobian at volume and facet nodes
            self.x_prime_omega.append(
                np.zeros([self.N_omega[i], self.d, self.d]))
            self.x_prime_gamma.append([
                np.zeros([self.N_gamma[i][gamma], self.d, self.d])
                for gamma in range(0,self.mesh.Nf[k])])
            
            # vandermonde derivatives of geometry mapping
            if self.d==1:
                V_geo_xi = [mp.vandermonde(
                    self.mesh.grad_basis_geo, self.xi_omega[i])]
                V_geo_gamma_xi = [[mp.vandermonde(
                    self.mesh.grad_basis_geo, self.xi_gamma[i][gamma])] 
                    for gamma in range(0,self.mesh.Nf[k])]
            else:
                V_geo_xi = list(mp.vandermonde(
                    self.mesh.grad_basis_geo, self.xi_omega[i]))      
                V_geo_gamma_xi = [mp.vandermonde(
                    self.mesh.grad_basis_geo, self.xi_gamma[i][gamma]) 
                    for gamma in range(0,self.mesh.Nf[k])]
                
            for m in range(0, self.d):
                # jacobian at volume nodes
                self.x_prime_omega[k][:,:,m] = V_geo_xi[m] @ self.mesh.xhat_geo[k]
                
                # jacobian at facet nodes
                for gamma in range(0,self.mesh.Nf[k]):
                    self.x_prime_gamma[k][gamma][:,:,m] = V_geo_gamma_xi[gamma][m] \
                    @ self.mesh.xhat_geo[k]
            
            # inverse Jacobian
            self.x_prime_inv_omega.append(np.array([ 
                np.linalg.inv(self.x_prime_omega[k][j,:,:])
                for j in range(0,self.N_omega[i])]))
            
            self.x_prime_inv_gamma.append([np.array([ 
                np.linalg.inv(self.x_prime_gamma[k][gamma][j,:,:])
                for j in range(0,self.N_gamma[i][gamma])]) 
                for gamma in range(0,self.mesh.Nf[k])]) 
            
            # Jacobian determinant
            self.J_omega.append(
                np.array([np.linalg.det(self.x_prime_omega[k][j,:,:]) 
                          for j in range(0,self.N_omega[i])]))
            
            self.J_gamma.append([np.array([np.linalg.det(
                self.x_prime_gamma[k][gamma][j,:,:]) 
                          for j in range(0,self.N_gamma[i][gamma])]) 
                                 for gamma in range(0, self.mesh.Nf[k])])
            
            # Unscaled normal vectors
            n_gamma_unscl = [np.array([
                self.J_gamma[i][gamma][j]*self.x_prime_inv_gamma[k][gamma][j,:,:].T @ 
                self.n_hat[i][gamma]
                for j in range(0,self.N_gamma[i][gamma])])
                for gamma in range(0,self.mesh.Nf[k])]
            
            # facet jacobian
            self.Jf_gamma.append([np.array([np.linalg.norm(
                n_gamma_unscl[gamma][j,:]) 
                for j in range(0,self.N_gamma[i][gamma])])
                for gamma in range(0,self.mesh.Nf[k])])
            
            # unit normal vectors
            self.n_gamma.append([np.array(
                [n_gamma_unscl[gamma][j,:]/self.Jf_gamma[k][gamma][j]
                for j in range(0,self.N_gamma[i][gamma])])
                for gamma in range(0,self.mesh.Nf[k])])
            
        self.test_normals()
        
     
    def build_local_operators(self):
        
        if self.form == "weak":
            
            self.vol = [[self.M_J_inv[k] @ self.Dhat[self.element_to_discretization[k]][m].T
                        @ self.V[self.element_to_discretization[k]].T 
                        @ self.W[self.element_to_discretization[k]]
                        for m in range(0,self.d)]
                        for k in range(0,self.mesh.K)]
            
            self.fac = [[-1.0*self.M_J_inv[k] @ self.V_gamma[self.element_to_discretization[k]][gamma].T
                         @ self.W_gamma[self.element_to_discretization[k]][gamma]
                         for gamma in range(0, self.mesh.Nf[k])]
                         for k in range(0,self.mesh.K)]
            
        elif self.form == "strong":
            
            raise NotImplementedError
            
        else:
            
            raise NotImplementedError

        
    def build_global_residual(self, f, f_star, bc, N_eq):
        
        self.build_local_operators()
        
        def global_residual(self, f, f_star, bc, N_eq, u_hat, t):
            f_trans_omega = []
            f_trans_gamma = []
            for k in range(0,self.mesh.K):
                
                i = self.element_to_discretization[k]
                f_omega = f((self.V[i] @ u_hat[k].T).T, self.x_omega[k])
                
                f_trans_omega.append([sum(
                    [self.J_omega[k]*self.x_prime_inv_omega[k][:,m,n] * f_omega[n] 
                     for n in range(0,self.d)]) for m in range(0, self.d)])
                
                f_trans_gamma.append([])
                for gamma in range(0, self.mesh.Nf[k]):  
                    
                    if self.mesh.local_to_local[(k,gamma)] is not None:     
                        nu, rho = self.mesh.local_to_local[(k,gamma)]
                        u_plus = (self.facet_permutation[k][gamma].T @ self.V_gamma[
                            self.element_to_discretization[nu]][rho] @ u_hat[nu].T).T
                    else:
                        u_plus = [bc[self.mesh.local_to_bc_index[(k,gamma)]][e](
                            self.x_gamma[k][gamma], t) 
                            for e in range(0, N_eq)]
                
                    f_trans_gamma[k].append(self.Jf_gamma[k][gamma] * f_star(
                        (self.V_gamma[i][gamma] @ u_hat[k].T).T, u_plus,
                        self.x_gamma[k][gamma], self.n_gamma[k][gamma]))
                    

            return [np.array([sum([self.vol[k][m] @ f_trans_omega[k][m][e,:] 
                                      for m in range(0,self.d)])
                     + sum([self.fac[k][gamma] @ 
                               f_trans_gamma[k][gamma][e,:]
                               for gamma in range(0, self.mesh.Nf[k])])
                    for e in range(0, N_eq)])
                    for k in range(0, self.mesh.K)]
                
            
        return lambda u_hat, t: global_residual(self, f, f_star,
                                                bc, N_eq, u_hat, t)
        

    def test_normals(self, print_output=True):
        
        self.normals_good = []
        
        for k in range(0,self.mesh.K):
            self.normals_good.append([True]*self.mesh.Nf[k])
            for gamma in range(0,self.mesh.Nf[k]):
                if self.mesh.local_to_local[(k,gamma)] is not None:
                    nu, rho = self.mesh.local_to_local[(k,gamma)]
                    if np.amax(np.abs(self.n_gamma[nu][rho] 
                                      + self.facet_permutation[k][gamma] 
                                      @ self.n_gamma[k][gamma])) > NORMAL_TOL:
                        self.normals_good[k][gamma] = False
                        if print_output:
                            print("Watertightness violated at (k,gamma) = ",
                                  (k,gamma))
            
            
    def plot(self, plot_nodes=True, markersize=4, geometry_resolution=10):
        
        if self.d == 1:
        
            mesh_plot = plt.figure()
            ax = plt.axes()
            ax.axes.yaxis.set_visible(False)
            plt.xlim([self.mesh.xmin[0] 
                      - 0.025 * self.mesh.extent[0], 
                      self.mesh.xmax[0] 
                      + 0.025 * self.mesh.extent[0]])
            plt.xlabel("$x$")
            
            for k in range(0, self.mesh.K):
                current_color = next(self.color)
                
                #plot node positions
                if plot_nodes:
                    ax.plot(self.x_omega[k][0,:], 
                      np.zeros(self.N_omega[
                          self.element_to_discretization[k]]), "o",
                      markersize=markersize,
                      color = current_color)
                    
                ax.plot(np.array(
                    [self.mesh.v[0,self.mesh.element_to_vertex[k][0]],
                     self.mesh.v[0,self.mesh.element_to_vertex[k][1]]]),
                        np.zeros(2), "-s", markersize=markersize,
                        color = current_color,
                        markerfacecolor="black")          
                   
            plt.show()
            
            mesh_plot.savefig("../plots/" + self.name + 
                            "_discretization.pdf")
        elif self.d == 2:
            
            mesh_plot = plt.figure()
            ax = plt.axes()
            ax.set_xlim([self.mesh.xmin[0] 
                         - 0.025 * self.mesh.extent[0],
                          self.mesh.xmax[0] 
                          + 0.025 * self.mesh.extent[0]])
            
            ax.set_ylim([self.mesh.xmin[1] 
                         - 0.025 * self.mesh.extent[1],
                          self.mesh.xmax[1] 
                          + 0.025 * self.mesh.extent[1]]) 
            ax.set_aspect('equal')
            plt.xlabel("$x_1$")
            plt.ylabel("$x_2$")
                
            # only works for triangles, otherwise need to do this 
            # for each discretization type and put in loop over k
            ref_edge_points = SpatialDiscretization.map_unit_to_facets(
                np.linspace(-1.0,1.0,geometry_resolution))
            V_edge_geo = [mp.vandermonde(
                self.mesh.basis_geo, ref_edge_points[gamma])
                          for gamma in range(0,3)]

            # loop through all elements
            for k in range(0, self.mesh.K):
                current_color = next(self.color)
                
                if plot_nodes:
                    
                    ax.plot(self.x_omega[k][0,:], 
                            self.x_omega[k][1,:], "o",
                          markersize=markersize,
                          color = current_color)
                        
                for gamma in range(0, self.mesh.Nf[k]):
                    
                    # plot facet edge curves
                    edge_points = (V_edge_geo[gamma] 
                                   @ self.mesh.xhat_geo[k]).T
                    
                    ax.plot(edge_points[0,:], 
                                edge_points[1,:], 
                                '-', 
                                color="black")
                        
                    if plot_nodes:
                       
                        # plot facet nodes
                        ax.plot(self.x_gamma[k][gamma][0,:], 
                                self.x_gamma[k][gamma][1,:],
                                "s", 
                                markersize=markersize, 
                                color="black")
                   
            mesh_plot.savefig("../plots/" + self.name + 
                            "_discretization.pdf")
            
            plt.show()
        else: 
            raise NotImplementedError
        
        
class SimplexQuadratureDiscretization(SpatialDiscretization):
    
    def __init__(self, mesh, p, tau=None, mu=None, form="weak"):
        
        if tau is None:
            tau = 2*p
            
        if mu is None:
            mu = 2*p + 1
        
        if mesh.d == 1:
            volume_quadrature = mp.LegendreGaussQuadrature(floor((tau-1)/2))
            volume_nodes = np.array([volume_quadrature.nodes])
            W = np.diag(volume_quadrature.weights)
            
            facet_nodes = [np.array([[-1.0]]),np.array([[1.0]])]
            W_gamma = [np.array([[1.0]]),np.array([[1.0]])]
            n_hat = [np.array([-1.0]), np.array([1.0])]
        
        elif mesh.d == 2:
            
            volume_quadrature = mp.XiaoGimbutasSimplexQuadrature(tau,2)
            volume_nodes = volume_quadrature.nodes
            W = np.diag(volume_quadrature.weights)
            
            facet_quadrature = mp.LegendreGaussQuadrature(floor((mu-1)/2))
            facet_nodes = SpatialDiscretization.map_unit_to_facets(
                facet_quadrature.nodes,
                element_type="triangle") 
            W_gamma = [np.diag(facet_quadrature.weights) for gamma in range(0,3)]
            n_hat = [np.array([0.0,-1.0]), 
                     np.array([1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)]),
                     np.array([-1.0, 0.0])]
            
        else: 
            raise NotImplementedError
    
        super().__init__(mesh, [0]*mesh.K, [p],
                 [volume_nodes], [facet_nodes], [W], [W_gamma], 
                 [n_hat], form=form)
    
    
class SimplexCollocationDiscretization(SpatialDiscretization):
    
    def __init__(self, mesh, p, q=None, r=None):
        
        raise NotImplementedError


class TimeIntegrator:
    
    def __init__(self, residual, dt, discretization_type="rk44"):
        
        self.dt_target = dt
        self.type = discretization_type
        self.R = residual
    
    @staticmethod
    def calculate_time_step(spatial_discretization, wave_speed, beta):
        
        h = np.amin(
            spatial_discretization.mesh.extent)/(
                spatial_discretization.mesh.K ** (
                    1.0/spatial_discretization.d))
        return beta/(2*max(spatial_discretization.p) + 1.0)*h/wave_speed
        
    def run(self, u_0, T):
        N_t = ceil(T/self.dt_target) 
        dt = T/N_t
        u = np.copy(u_0)
        t = 0
        print("dt = ", dt)
        for n in range(0,N_t):
            
            u = np.copy(self.time_step(u,t,dt))
            t = t + dt
        
        return u
    

    def time_step(self, u, t, dt):
        
        if self.type == "rk44":
        
            r_u = self.R(u,t)

            u_hat_nphalf = [np.array([u[k][e,:] + 0.5 * dt * r_u[k][e,:] 
                            for e in range(0, u[k].shape[0])])
                            for k in range(0, len(u))]
            
            r_u_hat_nphalf = self.R(u_hat_nphalf, t + 0.5*dt)

            u_tilde_nphalf = [np.array([u[k][e,:] + 0.5 * dt * r_u_hat_nphalf[k][e,:]
                            for e in range(0, u[k].shape[0])])
                            for k in range(0, len(u))]
                    
            r_u_tilde_nphalf = self.R(u_tilde_nphalf, t + 0.5*dt)

            u_bar_np1 = [np.array([u[k][e,:] + dt * r_u_tilde_nphalf[k][e,:] 
                          for e in range(0, u[k].shape[0])])
                          for k in range(0, len(u))]
                         
            r_u_bar_np1 = self.R(u_bar_np1, t + 1.0*dt)

            return [np.array([u[k][e,:] + (1. / 6.) * dt * (r_u[k][e,:] + 2. * (r_u_hat_nphalf[k][e,:] + r_u_tilde_nphalf[k][e,:])
                                               + r_u_bar_np1[k][e,:])
                        for e in range(0, u[k].shape[0])])
                        for k in range(0, len(u))]
            
        elif self.type == "explicit_euler":
            r = self.R(u,t)
            return [np.array([u[k][e,:] + dt *r[k][e,:] for e in range(u[k].shape[0])]) for k in range(0, len(u))]
        
        
        else:
            raise NotImplementedError
