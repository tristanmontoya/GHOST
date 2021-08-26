# GHOST - Spatial and Temporal Discretization

import numpy as np
from scipy import special
from math import floor, ceil, factorial
import modepy as mp
import quadpy as qp
import matplotlib.pyplot as plt
import pickle
import time
import os.path

NORMAL_TOL = 1.0e-8


class SpatialDiscretization:
    
    def __init__(self, mesh, element_to_discretization, p,
                 xi_omega, xi_gamma, W, W_gamma,
                 n_hat, solution_representation ="modal",
                 form = "weak", correction="c_dg", name=None):
        
        # mesh
        self.mesh = mesh
        
        if name is None:
            self.name = self.mesh.name
        else:
            self.name=name
        
        self.form=form
        self.solution_representation = solution_representation
        
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
        self.K = None
        self.Minv = None
        self.L = None
        self.facet_permutation = None
    
        self.build_facet_permutation()
        self.build_interpolation()
        
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
        
        # VCJH parameter
        self.correction=correction
        
        # build volume and facet operators
        self.build_local_operators()
        
        # init residual function (set based on problem)
        self.residual = None
        
        # LaTeX plotting preamble
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}\usepackage{bm}"
        
        
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
        # this is the modal orthogonal basis
        self.basis = [mp.simplex_onb(self.d,self.p[i]) 
                      for i in range(0,self.Nd)]
        
        self.grad_basis = [mp.grad_simplex_onb(self.d,self.p[i])
                           for i in range(0, self.Nd)]
        
        if self.d == 1:  
            
            if self.solution_representation == "modal":
                
                self.V = [mp.vandermonde(self.basis[i], self.xi_omega[i][0,:]) 
                          for i in range(0,self.Nd)]
                  
                self.V_gamma = [[mp.vandermonde(self.basis[i],
                                                self.xi_gamma[i][gamma][0,:]) 
                            for gamma in range(0,self.Nf[i])] 
                                for i in range(0,self.Nd)]
            
            elif self.solution_representation == "nodal":
                
                self.xi_p = [np.array(
                    [mp.quadrature.jacobi_gauss.legendre_gauss_lobatto_nodes(
                        self.p[i])]) 
                             for i in range(0,self.Nd)]
                
                self.Vp_inv = [np.linalg.inv(mp.vandermonde(self.basis[i],
                                                            self.xi_p[i][0,:])) 
                                for i in range(0,self.Nd)]
                
                self.V = [mp.vandermonde(self.basis[i], self.xi_omega[i][0,:]) @
                          self.Vp_inv[i] for i in range(0,self.Nd)]
                
                self.V_gamma = [[mp.vandermonde(self.basis[i],
                                                self.xi_gamma[i][gamma][0,:]) @
                                 self.Vp_inv[i] for gamma in range(0,self.Nf[i])] 
                                for i in range(0,self.Nd)]
                    
            else: 
                raise NotImplementedError
                
        else:
            
            if self.solution_representation == "modal":
                
                self.V = [mp.vandermonde(self.basis[i], self.xi_omega[i]) 
                          for i in range(0,self.Nd)]
            
                self.V_gamma = [[mp.vandermonde(self.basis[i],
                                                self.xi_gamma[i][gamma]) 
                                for gamma in range(0,self.Nf[i])]
                                for i in range(0,self.Nd)]
                
            elif self.solution_representation == "nodal":
                
                self.xi_p = [mp.warp_and_blend_nodes(self.d, self.p[i]) 
                             for i in range(0,self.Nd)]
                
                self.Vp_inv = [np.linalg.inv(mp.vandermonde(self.basis[i],
                                                            self.xi_p[i])) 
                                for i in range(0,self.Nd)]
                
                self.V = [mp.vandermonde(self.basis[i], self.xi_omega[i]) @ 
                          self.Vp_inv[i] for i in range(0,self.Nd)]
            
                self.V_gamma = [[mp.vandermonde(self.basis[i],
                                                self.xi_gamma[i][gamma]) @
                                 self.Vp_inv[i] 
                                 for gamma in range(0,self.Nf[i])]
                                for i in range(0,self.Nd)]
                
            else: 
                raise NotImplementedError
        
              
    def map_nodes(self):
        
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
                self.J_gamma[k][gamma][j]*self.x_prime_inv_gamma[k][gamma][j,:,:].T @ 
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
                for j in range(0,self.N_gamma[i][gamma])]).T
                for gamma in range(0,self.mesh.Nf[k])])
            
        self.test_normals()
        
     
    def build_local_operators(self):
        
        # mass matrix and inverse
        self.M = [self.V[i].T @ self.W[i] @ self.V[i] 
                  for i in range(0,self.Nd)]
        self.Minv = [np.linalg.inv(self.M[i]) for i in range(0, self.Nd)]
    
        # discrete orthogonal projection matrix
        self.P = [self.Minv[i] @ self.V[i].T @ self.W[i] 
                for i in range(0,self.Nd)]
        
        # weighted projection matrix
        self.P_J =  [np.linalg.inv(self.V[self.element_to_discretization[k]].T @ 
                                   self.W[self.element_to_discretization[k]] @ 
                                   np.diag(self.J_omega[k]) @ 
                                   self.V[self.element_to_discretization[k]]) @ 
        self.V[self.element_to_discretization[k]].T @ 
        self.W[self.element_to_discretization[k]] @ 
        np.diag(self.J_omega[k]) for k in range(0,self.mesh.K)]
        
        # derivative operator
        if self.solution_representation == "modal":
            
            if self.d==1:
                
                self.V_xi = [[mp.vandermonde(self.grad_basis[i],
                                             self.xi_omega[i])]
                             for i in range(0,self.Nd)]
            else:
                
                self.V_xi =[list(mp.vandermonde(self.grad_basis[i], 
                                                self.xi_omega[i]))
                            for i in range(0,self.Nd)]
                
            # self.Dhat = [[self.P[i] @ self.V_xi[i][m] 
            #               for m in range(0, self.d)]
            #              for i in range(0, self.Nd)]
            
            self.S = [[self.V[i].T @ self.W[i] @ self.V_xi[i][m] 
                       for m in range(0, self.d)]
                       for i in range(self.Nd)]
            
            self.Dhat = [[self.Minv[i] @ self.S[i][m]
                          for m in range(0, self.d)]
                         for i in range(0, self.Nd)]
            
        elif self.solution_representation == "nodal":
            
            if self.d==1:
                
                self.Vp_xi = [[mp.vandermonde(self.grad_basis[i],
                                             self.xi_p[i])]
                             for i in range(0,self.Nd)]
            else:
                
                self.Vp_xi = [list(mp.vandermonde(self.grad_basis[i],
                                             self.xi_p[i]))
                             for i in range(0,self.Nd)]
                
            self.Dhat = [[self.Vp_xi[i][m] @ self.Vp_inv[i] 
                          for m in range(0,self.d)]
                         for i in range(0, self.Nd)]
            
        else:
            
            raise NotImplementedError
        
        if self.correction == "c_dg":
    
            self.M_J = [self.V[self.element_to_discretization[k]].T 
                @ self.W[self.element_to_discretization[k]] 
                @ np.diag(self.J_omega[k]) 
                @ self.V[self.element_to_discretization[k]] 
                for k in range(0,self.mesh.K)]
            
            self.L = [[self.Minv[i] @ self.V_gamma[i][gamma].T
                       @ self.W_gamma[i][gamma] for gamma in range(0, self.Nf[i])]
                       for i in range(0,self.Nd)]
                    
        elif self.correction == "c_+":
            
            abs_omega = 2.0**self.d/(factorial(self.d))
            
            if self.d == 1:
                
                c_plus = {2: 0.183,
                          3: 3.6e-3,
                          4: 4.67e-3,
                          5: 4.28e-7}
                
                D_alpha = [np.linalg.matrix_power(self.Dhat[i][0],self.p[i]) 
                           for i in range(0,self.Nd)]
                
                self.K = [1.0/abs_omega*c_plus[self.p[i]]*D_alpha[i].T 
                          @ self.M[i] @ D_alpha[i] 
                          for i in range(0,self.Nd)] 
                
            elif self.d == 2:
                
                c_plus = {2: 4.3e-2,
                          3: 6.0e-4,
                          4: 5.6e-6}
                
                # alpha = (alpha_0, alpha_1) = (p-q, q) , q = 0, ..., p
                alpha = [np.array([[self.p[i] - q,q] for q in range(0,self.p[i] + 1)]) 
                         for i in range(0,self.Nd)]
        
                # c_alpha = p choose q, q = 0, ..., p
                c_alpha = [c_plus[self.p[i]]*np.array([special.comb(self.p[i], q, 
                                exact=True) for q in range(0,self.p[i] + 1)])
                           for i in range(0,self.Nd)]
                
                # d^{|alpha|}/dx^{alpha_0}dy^{alpha_1} 
                D_alpha = [[np.linalg.matrix_power(self.Dhat[i][0], alpha[i][q][0])
                           @ np.linalg.matrix_power(self.Dhat[i][1], alpha[i][q][1])
                           for q in range(0,self.p[i]+1)]
                           for i in range(0,self.Nd)]
                    
                self.K = [1.0/abs_omega*sum([c_alpha[i][q]*D_alpha[i][q].T 
                                             @ self.M[i] @ D_alpha[i][q]
                               for q in range(0,self.p[i]+1)])
                    for i in range(0,self.Nd)]
        
            self.M_J = [(self.M[self.element_to_discretization[k]]
                + self.K[self.element_to_discretization[k]]) @ 
                self.P[self.element_to_discretization[k]] @ np.diag(self.J_omega[k]) 
                @ self.V[self.element_to_discretization[k]]
                for k in range(0,self.mesh.K)]
            
            self.L = [[np.linalg.inv(self.M[i] + self.K[i]) 
                       @ self.V_gamma[i][gamma].T
                       @ self.W_gamma[i][gamma] for gamma in range(0, self.Nf[i])]
                       for i in range(0,self.Nd)]
            
        else:
            raise NotImplementedError
            
        self.M_J_inv = [np.linalg.inv(self.M_J[k]) for k in range(0, self.mesh.K)]
        
        # pre-compute full volume and facet operators
        if self.form == "weak":
            
            self.vol = [[self.M_J_inv[k] 
                         @ (self.Dhat[self.element_to_discretization[k]][m]).T
                         @ self.V[self.element_to_discretization[k]].T 
                         @ self.W[self.element_to_discretization[k]]
                        for m in range(0,self.d)]
                        for k in range(0,self.mesh.K)]
            
            self.fac = [[-1.0*self.M_J_inv[k] 
                         @ self.V_gamma[self.element_to_discretization[k]][gamma].T
                         @ self.W_gamma[self.element_to_discretization[k]][gamma]
                         for gamma in range(0, self.mesh.Nf[k])]
                         for k in range(0,self.mesh.K)]
            
        elif self.form == "strong":
                   
            self.vol = [[-1.0*np.linalg.inv(self.P[self.element_to_discretization[k]] 
                                            @ np.diag(self.J_omega[k]) 
                                            @ self.V[self.element_to_discretization[k]]) 
                         @ self.Dhat[self.element_to_discretization[k]][m] @
                          self.P[self.element_to_discretization[k]]
                        for m in range(0,self.d)] for k in range(0,self.mesh.K)]
            
            self.fac = [[-1.0*np.linalg.inv(self.P[self.element_to_discretization[k]] 
                                            @ np.diag(self.J_omega[k]) 
                                            @ self.V[self.element_to_discretization[k]])
                         @ self.L[self.element_to_discretization[k]][gamma]
                         for gamma in range(0, self.mesh.Nf[k])]
                         for k in range(0,self.mesh.K)]
            
        else:
            
            raise NotImplementedError

        
    def build_global_residual(self, f, f_star, bc, N_eq):
        
        def global_residual(self, f, f_star, bc, N_eq, u_hat, t,
                            print_output=False):
            
            f_trans_omega = []
            f_trans_gamma = []
            for k in range(0,self.mesh.K):
                
                i = self.element_to_discretization[k]
               
                f_omega = f((self.V[i] @ u_hat[k].T).T, self.x_omega[k])
                
                f_trans_omega.append([sum(
                    [self.J_omega[k]*self.x_prime_inv_omega[k][:,m,n]* f_omega[n] 
                     for n in range(0,self.d)]) for m in range(0, self.d)])
               
                f_trans_gamma.append([])
                for gamma in range(0, self.mesh.Nf[k]):  
                    
                    if self.mesh.local_to_local[(k,gamma)] is not None:  
                        
                        nu, eta = self.mesh.local_to_local[(k,gamma)]
                        u_plus = (self.facet_permutation[k][gamma].T @ self.V_gamma[
                            self.element_to_discretization[nu]][eta] @ u_hat[nu].T).T
                        
                    else:
                        
                        u_plus = [bc[self.mesh.local_to_bc_index[(k,gamma)]][e](
                            self.x_gamma[k][gamma], t) 
                            for e in range(0, N_eq)]
                    
                    if self.form == "weak":
                        
                        f_trans_gamma[k].append(
                            self.Jf_gamma[k][gamma] * f_star(
                            (self.V_gamma[i][gamma] @ u_hat[k].T).T, u_plus,
                            self.x_gamma[k][gamma], self.n_gamma[k][gamma]))
                        
                        
                    elif self.form == "strong":
               
                        f_trans_gamma[k].append(self.Jf_gamma[k][gamma] * f_star(
                            (self.V_gamma[i][gamma] @ u_hat[k].T).T, u_plus,
                            self.x_gamma[k][gamma], self.n_gamma[k][gamma])
                            - (self.V_gamma[i][gamma] @ self.P[i] 
                               @ sum([f_trans_omega[k][m].T*self.n_hat[i][gamma][m] 
                                      for m in range(0,self.d)])).T)
                  
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
                    
                    nu, eta = self.mesh.local_to_local[(k,gamma)]
                    if np.amax(np.abs(self.n_gamma[nu][eta].T 
                                      + self.facet_permutation[k][gamma] 
                                      @ self.n_gamma[k][gamma].T)) > NORMAL_TOL:
                        self.normals_good[k][gamma] = False
                        
                        if print_output:
                            print("Watertightness violated at (k,gamma) = ",
                                  (k,gamma))
                            print("T(n_k_gamma)", 
                                  self.facet_permutation[k][gamma] 
                                      @ self.n_gamma[k][gamma], "n_nu,eta: ", 
                                      self.n_gamma[nu][eta])
            
            
    def plot(self, plot_nodes=True, plot_geometry_nodes=False, axes=True,
             markersize=4, geometry_resolution=10, filename=None):
    
        # assign a colour to each element
        self.color = iter(plt.cm.rainbow(
            np.linspace(0, 1, self.mesh.K)))
        
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
                
                if plot_geometry_nodes:
                    [ax.plot(self.mesh.x_geo[k][0,:], 
                      np.zeros(self.mesh.p_geo+1), "ok",
                      markersize=markersize,
                      markerfacecolor=None)]
            
            plt.tight_layout()
            plt.show()
            
            if filename is None:
                mesh_plot.savefig("../plots/" + self.name + 
                                "_discretization.pdf",
                                bbox_inches='tight')
            else:
                mesh_plot.savefig(filename, 
                                  bbox_inches='tight')
                
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
            
            if axes == False:   
                ax.axis('off')
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
            
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
                    
                if plot_geometry_nodes:
                    ax.plot(self.mesh.x_geo[k][0,:], 
                            self.mesh.x_geo[k][1,:], "ok",
                          #fillstyle='solid',
                          markersize=markersize)
                        
                for gamma in range(0, self.mesh.Nf[k]):
                    
                    # plot facet edge curves
                    edge_points = (V_edge_geo[gamma] 
                                   @ self.mesh.xhat_geo[k]).T
                    
                    ax.plot(edge_points[0,:], 
                                edge_points[1,:], 
                                '-', 
                                linewidth=markersize*0.25,
                                color="black")
                         
                    if plot_nodes:
                         
                        # plot facet nodes
                        ax.plot(self.x_gamma[k][gamma][0,:], 
                                self.x_gamma[k][gamma][1,:],
                                "s", 
                                markersize=markersize, 
                                markeredgewidth=markersize*0.25,
                                color="black",
                                fillstyle='none')
            
            plt.show()
            
            if filename is None:
                mesh_plot.savefig("../plots/" + self.name + 
                                "_discretization.pdf", bbox_inches='tight')
                
            else:
                mesh_plot.savefig(filename, bbox_inches='tight')
            
        else: 
            raise NotImplementedError
        
        
    def plot_ref_el(self, filename=None, markersize=8):
        
        if self.d == 2:  # assume for now to be triangles
            
            elem_plot = plt.figure(frameon=False)
            ax = plt.axes()
            ax.axis('off')
            ax.set_aspect('equal')
            
            ax.plot(np.array([-1.0,1.0,-1.0,-1.0]),
                    np.array([-1.0,-1.0,1.0,-1.0]), "-k")
            
            ax.plot(self.xi_omega[0][0,:],
                    self.xi_omega[0][1,:], "o",
                    markersize=markersize,
                    color="black",
                    fillstyle='none')
            
            for gamma in range(0, self.Nf[0]):
                
                ax.plot(self.xi_gamma[0][gamma][0,:], 
                        self.xi_gamma[0][gamma][1,:],
                        "s", 
                        markersize=0.5*markersize, 
                        color="black")
                
            if filename is None:
                elem_plot.savefig("../plots/" + self.name + 
                                "_element.pdf",
                                bbox_inches='tight')
                
            else:
                elem_plot.savefig(filename, bbox_inches='tight')
            
        else:
            raise NotImplementedError
            

class SimplexQuadratureDiscretization(SpatialDiscretization):
    
    def __init__(self, mesh, p, tau=None, mu=None,
                 volume_rule=None, facet_rule="lg", 
                 form="weak", solution_representation="modal",
                 correction="c_dg"):
        
        if tau is None:
            tau = 2*p
            
        if mu is None:
            mu = 2*p + 1
        
        if mesh.d == 1:
            
            if volume_rule == None or volume_rule == "lg":
                
                volume_quadrature = mp.LegendreGaussQuadrature(ceil((tau-1)/2))
                volume_nodes = np.array([volume_quadrature.nodes])
                W = np.diag(volume_quadrature.weights)
                
            elif volume_rule == "lgl":
                
               raise NotImplementedError
            
            facet_nodes = [np.array([[-1.0]]),np.array([[1.0]])]
            W_gamma = [np.array([[1.0]]),np.array([[1.0]])]
            n_hat = [np.array([-1.0]), np.array([1.0])]
        
        elif mesh.d == 2:
            
            volume_quadrature = mp.XiaoGimbutasSimplexQuadrature(tau,2)
            volume_nodes = volume_quadrature.nodes
            W = np.diag(volume_quadrature.weights)
            
            if facet_rule == "lg":
                
                facet_quadrature = mp.LegendreGaussQuadrature(ceil((mu-1)/2))
                facet_nodes = SpatialDiscretization.map_unit_to_facets(
                    facet_quadrature.nodes,
                    element_type="triangle") 
                
            elif facet_rule == "lgl":
                
                facet_quadrature = qp.c1.gauss_lobatto(ceil((mu+3)/2))
                facet_nodes = SpatialDiscretization.map_unit_to_facets(
                    facet_quadrature.points,
                    element_type="triangle") 
                
            n_hat = [np.array([0.0,-1.0]), 
                     np.array([1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)]),
                     np.array([-1.0, 0.0])]
            
            W_gamma = [np.diag(facet_quadrature.weights),
                       np.sqrt(2.0)*np.diag(facet_quadrature.weights),
                       np.diag(facet_quadrature.weights)]
            
        else: 
            raise NotImplementedError
    
        super().__init__(mesh, [0]*mesh.K, [p], [volume_nodes],
                         [facet_nodes], [W], [W_gamma], [n_hat], form=form,
                         solution_representation=solution_representation,
                         correction=correction)
    
    
class SimplexCollocationDiscretization(SpatialDiscretization):
    
    def __init__(self, mesh, p, p_omega=None, p_gamma=None,
                 form="weak", solution_representation="modal",
                 correction="c_dg", use_lumping=False):
            
        if p_omega is None:
            p_omega = p
        
        if p_gamma is None:
            p_gamma = p
            
        # modal basis to construct nodal mass matrices
        volume_basis = mp.simplex_onb(mesh.d, p_omega)    
        
        if mesh.d == 1:
            
            volume_nodes = np.array(
                [mp.quadrature.jacobi_gauss.legendre_gauss_lobatto_nodes(p_omega)]) 
            
            V_p_omega = mp.vandermonde(volume_basis, volume_nodes[0,:]) 
            W = np.linalg.inv(V_p_omega @ V_p_omega.T)
            facet_nodes = [np.array([[-1.0]]),np.array([[1.0]])]
            W_gamma = [np.array([[1.0]]),np.array([[1.0]])]
            n_hat = [np.array([-1.0]), np.array([1.0])]
            
        elif mesh.d == 2:
            
            facet_basis = mp.simplex_onb(mesh.d-1, p_gamma)
            
            volume_nodes = mp.warp_and_blend_nodes(2, p_omega)
          
            V_p_omega = mp.vandermonde(volume_basis, volume_nodes) 
            W = np.linalg.inv(V_p_omega @ V_p_omega.T)
            
            facet_nodes_1D =  mp.quadrature.jacobi_gauss.legendre_gauss_lobatto_nodes(p_omega)
            V_p_gamma_1D = mp.vandermonde(facet_basis,facet_nodes_1D) 
            W_gamma_1D =  np.linalg.inv(V_p_gamma_1D @ V_p_gamma_1D.T)
            
            facet_nodes = SpatialDiscretization.map_unit_to_facets(
                facet_nodes_1D, element_type="triangle")
            
            if use_lumping:
                
                W = np.diag(np.sum(W,axis=1))
                W_gamma_1D = np.diag(np.sum(W_gamma_1D,axis=1))
             
            n_hat = [np.array([0.0,-1.0]), 
                     np.array([1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)]),
                     np.array([-1.0, 0.0])]
            
            W_gamma = [W_gamma_1D, np.sqrt(2.0)*W_gamma_1D, W_gamma_1D]

        
        super().__init__(mesh, [0]*mesh.K, [p], [volume_nodes],
                         [facet_nodes], [W], [W_gamma], [n_hat], form=form,
                         solution_representation=solution_representation,
                         correction=correction)
     
    
class TimeIntegrator:
    
    def __init__(self, residual, dt, discretization_type="rk44"):
        
        self.dt_target = dt
        self.type = discretization_type
        self.R = residual
    
    
    @staticmethod
    def calculate_time_step(spatial_discretization, wave_speed, beta, h=None):
        
        if h is None:
            
            h = np.amin(
                spatial_discretization.mesh.extent)/(
                    spatial_discretization.mesh.K ** (
                        1.0/spatial_discretization.d))
                    
        return beta/(2*max(spatial_discretization.p) + 1.0)*h/wave_speed
        
    
    def run(self, u_0, T, results_path,
            write_interval, 
            print_interval, 
            restart=True,
            prefix=""):
        
        if restart:
            
            if os.path.isfile(results_path+"times.dat"):
            
                times = None
                dt = None
                N_t = None
                N_write = None
                self.is_done = False
                
                times = pickle.load(open(results_path+"times.dat", "rb"))
                dt = pickle.load(open(results_path+"time_step_size.dat", "rb" ))
                N_t = pickle.load(open(results_path+"number_of_steps.dat", "rb" ))
         
                u = np.copy(u_0)
                n_0 = times[-1][0]
                t = times[-1][1]
                
                screen = open(results_path + "screen.txt", "a")
                print(prefix, "restarting from time step ", n_0, 
                     ", t=", t, file=screen)
                screen.close()
            
            else:
                
                screen = open(results_path + "screen.txt", "w")
                print(prefix, "No previous file found for restart. Starting new run.",
                      file=screen)
                screen.close()
                restart=False
    
        else:
            
            # calculate number of steps to take and actual time step
            N_t = floor(T/self.dt_target) 
            dt = T/N_t
            self.is_done = False
        
            u = np.copy(u_0)
            n_0 = 0
            t = 0
            times = [[n_0,t]]
               
            pickle.dump(dt, open(results_path+"time_step_size.dat", "wb" ))
            pickle.dump(N_t, open(results_path+"number_of_steps.dat", "wb" ))
        
        # interval between prints and writes to file
        if print_interval is None:
            N_print = N_t
        else:
            N_print = floor(print_interval/dt)
        if write_interval is None:
            N_write = N_t
        else:
            N_write = floor(write_interval/dt)    
              
        screen = open(results_path + "screen.txt", "w")
        print(prefix, " dt = ", dt, file=screen)
        print(prefix, "writing every ", 
              N_write, " time steps, total ", N_t, file=screen)
        screen.close()
        start = time.time()
        
        for n in range(n_0,N_t):
            
            u = np.copy(self.time_step(u,t,dt)) # update solution

            t = t + dt
            if ((n+1) % N_print == 0) or (n+1 == N_t):
                screen = open(results_path + "screen.txt", "a")
                print(prefix, "time step: ", n+1, "t: ", t, "wall time: ", 
                      time.time()-start, file=screen)
                screen.close()
                
            if ((n+1) % N_write == 0) or (n+1 == N_t):
                screen = open(results_path + "screen.txt", "a")
                print(prefix, "writing time step ", n+1, "t = ", t, file=screen)
                screen.close()
                times.append([n+1,t])
                
                pickle.dump(u, open(results_path+"res_" +
                                    str(n+1) + ".dat", "wb" ))
                
                pickle.dump(times, open(
                   results_path+"times.dat", "wb" ))
                
            if np.isnan(np.sum(np.array([[np.sum(u[k][e]) 
                                          for e in range(0, u[k].shape[0])] 
                                         for k in range(0,len(u))]))):
                
                pickle.dump(times, open(
                   results_path+"times.dat", "wb" ))
                return None
        
        pickle.dump(times, open(
                   results_path+"times.dat", "wb" ))
        
        
        screen = open(results_path + "screen.txt", "a")
        print(prefix, "Simulation complete.",file=screen)
        screen.close()
        
        self.is_done = True
        
        return u
    

    def time_step(self, u, t, dt):
        
        if self.type == "rk44":
        
            r_u = self.R(u,t)

            u_hat_nphalf = [np.array([u[k][e,:] + 0.5 * dt * r_u[k][e,:] 
                            for e in range(0, u[k].shape[0])])
                            for k in range(0, len(u))]
            
            r_u_hat_nphalf = self.R(u_hat_nphalf, t + 0.5*dt)

            u_tilde_nphalf = [np.array([u[k][e,:] 
                                        + 0.5 * dt * r_u_hat_nphalf[k][e,:]
                            for e in range(0, u[k].shape[0])])
                            for k in range(0, len(u))]
                    
            r_u_tilde_nphalf = self.R(u_tilde_nphalf, t + 0.5*dt)

            u_bar_np1 = [np.array([u[k][e,:] + dt * r_u_tilde_nphalf[k][e,:] 
                          for e in range(0, u[k].shape[0])])
                          for k in range(0, len(u))]
                         
            r_u_bar_np1 = self.R(u_bar_np1, t + 1.0*dt)

            return [np.array([u[k][e,:] + (1. / 6.) * dt * (
                r_u[k][e,:] + 2. * (r_u_hat_nphalf[k][e,:] 
                                    + r_u_tilde_nphalf[k][e,:]) 
                + r_u_bar_np1[k][e,:])
                        for e in range(0, u[k].shape[0])])
                        for k in range(0, len(u))]
            
        elif self.type == "explicit_euler":
            
            r = self.R(u,t)
            return [np.array([u[k][e,:] + dt *r[k][e,:] 
                              for e in range(u[k].shape[0])])
                    for k in range(0, len(u))]
        
        else:
            raise NotImplementedError
