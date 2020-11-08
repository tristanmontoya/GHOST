# GHOST - Spatial and Temporal Discretization

import numpy as np
from scipy import special
from math import floor, ceil
import modepy as mp


class SpatialDiscretization:
    
    def __init__(self, mesh, element_to_discretization, p,
                 xi_omega, xi_gamma, W, W_gamma, n_hat, form = "weak"):
        
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
        
        # facet normals
        self.n_hat = n_hat
        
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
        
        # evaluate grid nodes, normals and metric
        self.x_omega = None
        self.x_gamma = None
        self.x_prime_omega = None
        self.J_omega = None
        self.J_gamma = None
        self.n_gamma = None
        self.map_nodes()
        
        # init residual function
        self.residual = None
        
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
        
        self.V_gamma = [[mp.vandermonde(self.basis[i],self.xi_gamma[i][gamma]) 
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
                              
        
        raise NotImplementedError
        
    
    def map_nodes(self):
        
        self.x_omega = []
        self.x_gamma = []
        self.x_prime_omega = []
        self.x_prime_gamma = []
        self.J_omega = [] # TODO
        self.J_gamma = [] # TODO
        self.n_gamma = [] # TODO
        
        for k in range(0, self.mesh.K):
            i = self.element_to_discretization[k]
            # volume node positions
            self.x_omega.append((mp.vandermonde(
                self.mesh.basis_geo,
                self.xi_omega[i]) 
                @ self.mesh.xhat_geo[k]).T)
        
            # jacobian at volume nodes
            self.x_prime_omega.append(
                np.zeros([self.N_omega[i], self.d, self.d]))
            if self.d==1:
                V_geo_xi = [mp.vandermonde(
                    self.mesh.grad_basis_geo, self.xi_omega[i])]
            else:
                V_geo_xi = list(mp.vandermonde(
                    self.mesh.grad_basis_geo, self.xi_omega[i]))
                
            for m in range(0, self.d):
                self.x_prime_omega[k][:,:,m] = V_geo_xi[m] @ self.mesh.xhat_geo[k]
                
            # jacobian at facet nodes
            
            # Jacobian determinant at volume nodes
            self.J_omega.append(
                np.array([np.linalg.det(self.x_prime_omega[k][i,:,:]) 
                          for i in range(
                                  0,self.N_omega[
                                      self.element_to_discretization[k]])]))
                
            # facet node positions
            self.x_gamma.append([(mp.vandermonde(self.mesh.basis_geo,
            self.xi_gamma[self.element_to_discretization[k]][gamma]) 
                                  @ self.mesh.xhat_geo[k]).T
                                 for gamma in range(0,self.mesh.Nf[k])])
        
        
    def build_local_residual(self, f, f_star, N_eq):
        
        def local_residual(self, k, f_omega, f_star_gamma, N_eq):
            r = np.zeros([N_eq, self.N_p[self.element_to_discretization[k]]])
            
            for e in range(0, N_eq):
                r[e,:] = sum([self.vol[k][m] @ f_omega[m][e,:] 
                              for m in (0,self.d)]) \
                + sum([self.fac[k][gamma] @ f_star_gamma[gamma][e,:] 
                       for gamma in (0, self.d)])
            return r
        
        return [lambda f_omega, f_star_gamma, k=k: local_residual(
            self, k, f_omega, f_star_gamma, N_eq)
                for k in range(0, self.mesh.K)]
    
        raise NotImplementedError

        
    def build_global_residual(self):
        raise NotImplementedError
        
        
class SimplexQuadratureDiscretization(SpatialDiscretization):
    
    def __init__(self, mesh, p, tau=None, mu=None):
        
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
            n_hat = [np.array([-1.0]), np.array([-1.0])]
        
        elif mesh.d == 2:
            
            volume_quadrature = mp.XiaoGimbutasSimplexQuadrature(tau,2)
            volume_nodes = volume_quadrature.nodes
            W = np.diag(volume_quadrature.weights)
            
            facet_quadrature = mp.LegendreGaussQuadrature(floor((mu-1)/2))
            facet_nodes = SpatialDiscretization.map_unit_to_facets(
                facet_quadrature.nodes,
                element_type="triangle") 
            W_gamma = np.diag(facet_quadrature.weights)
            n_hat = [np.array([0.0,-1.0]), 
                     np.array([1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)]),
                     np.array([-1.0, 0.0])]
            
        else: 
            raise NotImplementedError
    
            
        super().__init__(mesh, [0]*mesh.K, [p],
                 [volume_nodes], [facet_nodes], [W], [W_gamma], [n_hat])
    
    
class SimplexCollocationDiscretization(SpatialDiscretization):
    
    def __init__(self, mesh, p, q=None, r=None):
        
        raise NotImplementedError


class TimeIntegrator:
    
    def __init__(self, residual, dt, discretization_type="rk44"):
        
        self.dt_target = dt
        self.discretization_type = discretization_type
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
        
        for n in range(0,N_t):
            
            u = self.time_step(u,t,dt)
            t = t + dt
        
        return u

    def time_step(self, u, t, dt):
        
        if self.discretization_type == "rk44":
        
            r_u = self.R(u)

            u_hat_nphalf = [[u[k][e] + 0.5 * dt * r_u[k][e] 
                            for e in range(0, len(u[k]))]
                            for k in range(0, len(u))]
            
            r_u_hat_nphalf = self.R(u_hat_nphalf, t + 0.5*dt)

            u_tilde_nphalf = [[u[k][e] + 0.5 * dt * r_u_hat_nphalf[k][e]
                            for e in range(0, len(u[k]))]
                            for k in range(0, len(u))]
                    
            r_u_tilde_nphalf = self.R(u_tilde_nphalf, t + 0.5*dt)

            u_bar_np1 = [[u[k][e] + dt * r_u_tilde_nphalf[k][e] 
                          for e in range(0, len(u[k]))]
                          for k in range(0, len(u))]
                         
            r_u_bar_np1 = self.R(u_bar_np1, t + 1.0*dt)

            return [[u[k][e] + 1. / 6. * dt * (r_u[k][e] + 2. * (r_u_hat_nphalf[k][e] + r_u_tilde_nphalf[k][e])
                                               + r_u_bar_np1[k][e])
                        for e in range(0, len(u[k]))]
                        for k in range(0, len(u))]
            
        else:
            raise NotImplementedError

    
    
            