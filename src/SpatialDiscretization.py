# GHOST - Spatial discretizations

import numpy as np
from scipy import special
from math import floor, ceil, factorial
import modepy as mp
import quadpy as qp
import matplotlib.pyplot as plt
import time
import os.path

class SpatialDiscretization:
    
    def __init__(self, mesh, element_to_discretization, p,
                 x_hat, x_hat_fac, W, B,
                 n_hat, solution_representation ="modal",
                 form = "weak", correction="c_dg", name=None):
        
        self.mesh = mesh
        
        if name is None:
            self.name = self.mesh.name
        else:
            self.name=name
        
        # "strong" or "weak"
        self.form=form

        # "modal" or "nodal"
        self.solution_representation = solution_representation
        
        # spatial dimension (1 or 2)
        self.d = mesh.d  
        
        # map from element index to discretization type index
        self.element_to_discretization = element_to_discretization  
        
        # polynomial degree
        self.p = p
        
        # reference element normal vectors
        self.n_hat = n_hat
        
        # number of discretization types
        self.N_d = len(self.p) 
        
        # dimension of total-degree polynomial space 
        self.N_p = [special.comb(self.p[i] + self.d, self.d, 
                                exact=True) for i in range(0,self.N_d)]
        
        # volume nodes/weights
        self.x_hat = x_hat
        self.W = W
        self.N_omega = [x_hat[i].shape[1] for i in range(0,self.N_d)]
        
        # number of facets per element
        self.N_fac = [len(x_hat_fac[i]) for i in range(0,self.N_d)]

        # facet nodes/weights
        self.x_hat_fac = x_hat_fac
        self.B = B
        self.N_gamma = [[x_hat_fac[i][zeta].shape[1] 
                         for zeta in range(0,self.N_fac[i])] 
                        for i in range(0,self.N_d)]
       
        # build permutation and interpolation operators
        self.build_facet_permutation()
        self.build_interpolation()
        
        # volume nodes in physical space
        self.x = []

        # facet nodes in physical space
        self.x_fac = []

        # metric Jacobian matrices/inverses at volume nodes
        self.G = []
        self.G_inv = []

        # metric Jacobian matrices/inverses at facet nodes
        self.G_fac = []
        self.G_inv_fac = []

        # metric Jacobian determinant at volume and facet nodes
        self.J = [] 
        self.J_fac = []

        # facet Jacobian at facet nodes
        self.Jf = []

        # outward unit normal in physical space computed via mapping
        self.n = []

        # evaluate metric factors and node positions
        self.map_nodes()
        
        # VCJH scheme "c_dg" or "c_+"
        self.correction=correction
        
        # build volume and facet operators
        self.build_local_operators()
        
        # LaTeX plotting preamble
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}\usepackage{bm}"
        
        
    @staticmethod
    def map_unit_to_facets(ref_nodes, element_type="triangle"):

        # send (d-1)-dimensional reference facet to facets bounding ref. elem.
        N_gamma = ref_nodes.shape[0]
        
        if element_type == "triangle":

            bottom = np.reshape(np.array([ref_nodes,-np.ones(N_gamma)]),
                                (2,N_gamma))
            left = np.reshape(np.array([-np.ones(N_gamma),
                                        np.flip(ref_nodes)]),(2,N_gamma))
            hypotenuse = np.array([[1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)],
                       [-1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)]]) @ np.array(
                           [np.sqrt(2.0)*np.flip(ref_nodes), np.zeros(N_gamma)])

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
                                      for k in range(0,self.mesh.N_el)]
            
        elif self.d==2:
            
            # assume all nodes are counterclockwise ordered
            self.facet_permutation = [
                [np.eye(self.N_gamma[
                    self.element_to_discretization[k]][zeta])[::-1]
                 for zeta in range(0,self.mesh.N_fac[k])] 
                                      for k in range(0,self.mesh.N_el)]
            
        else:
            raise NotImplementedError
        
        
    def build_interpolation(self):
        
        #modal orthonormal basis on simplex
        self.basis = [mp.simplex_onb(self.d,self.p[i]) 
                      for i in range(0,self.N_d)]
        
        #derivatives of modal orthonormal basis
        self.grad_basis = [mp.grad_simplex_onb(self.d,self.p[i])
                           for i in range(0, self.N_d)]
        
        if self.d == 1:  
            
            # directly evolve modal expansion coefficients
            if self.solution_representation == "modal":
                
                self.V = [mp.vandermonde(self.basis[i], self.x_hat[i][0,:]) 
                          for i in range(0,self.N_d)]
                  
                self.R = [[mp.vandermonde(self.basis[i],
                                                self.x_hat_fac[i][zeta][0,:]) 
                            for zeta in range(0,self.N_fac[i])] 
                                for i in range(0,self.N_d)]
            
            # evolve nodal values
            elif self.solution_representation == "nodal":
            
                self.x_hat_tilde = [np.array(
                    [mp.quadrature.jacobi_gauss.legendre_gauss_lobatto_nodes(
                        self.p[i])]) 
                             for i in range(0,self.N_d)]
                
                self.V_tilde_inv = [np.linalg.inv(mp.vandermonde(self.basis[i],
                                                            self.x_hat_tilde[i][0,:])) 
                                for i in range(0,self.N_d)]
                
                self.V = [mp.vandermonde(self.basis[i], self.x_hat[i][0,:]) @
                          self.V_tilde_inv[i] for i in range(0,self.N_d)]
                
                self.R = [[mp.vandermonde(self.basis[i],
                                                self.x_hat_fac[i][zeta][0,:]) @
                                 self.V_tilde_inv[i] for zeta in range(0,self.N_fac[i])] 
                                for i in range(0,self.N_d)]
                    
            else: 
                raise NotImplementedError
                
        else:
            
            if self.solution_representation == "modal":
                
                self.V = [mp.vandermonde(self.basis[i], self.x_hat[i]) 
                          for i in range(0,self.N_d)]
            
                self.R = [[mp.vandermonde(self.basis[i],
                                                self.x_hat_fac[i][zeta]) 
                                for zeta in range(0,self.N_fac[i])]
                                for i in range(0,self.N_d)]
                
            elif self.solution_representation == "nodal":
                
                self.x_hat_tilde = [mp.warp_and_blend_nodes(self.d, self.p[i]) 
                             for i in range(0,self.N_d)]
                
                self.V_tilde_inv = [np.linalg.inv(mp.vandermonde(self.basis[i],
                                                            self.x_hat_tilde[i])) 
                                for i in range(0,self.N_d)]
                
                self.V = [mp.vandermonde(self.basis[i], self.x_hat[i]) @ 
                          self.V_tilde_inv[i] for i in range(0,self.N_d)]
            
                self.R = [[mp.vandermonde(self.basis[i],
                                                self.x_hat_fac[i][zeta]) @
                                 self.V_tilde_inv[i] 
                                 for zeta in range(0,self.N_fac[i])]
                                for i in range(0,self.N_d)]
                
            else: 
                raise NotImplementedError
        
              
    def map_nodes(self):
        
        for k in range(0, self.mesh.N_el):
            i = self.element_to_discretization[k]
            
            # Lagrange interpolation to map volume nodes in physical space
            # uses modal basis for geometry mapping
            self.x.append((mp.vandermonde(
                self.mesh.basis_map,
                self.x_hat[i]) 
                @ self.mesh.x_tilde_map[k]).T)
            
            # Lagrange interpolation to map facet nodes in physical space
            self.x_fac.append([(mp.vandermonde(self.mesh.basis_map,
            self.x_hat_fac[self.element_to_discretization[k]][zeta]) 
                                  @ self.mesh.x_tilde_map[k]).T
                                 for zeta in range(0,self.mesh.N_fac[k])])
        
            # init jacobian at volume and facet nodes
            self.G.append(
                np.zeros([self.N_omega[i], self.d, self.d]))
            self.G_fac.append([
                np.zeros([self.N_gamma[i][zeta], self.d, self.d])
                for zeta in range(0,self.mesh.N_fac[k])])
            
            # Evaluate derivatives of modal basis for geometry mapping
            if self.d==1:
                V_grad_map = [mp.vandermonde(
                    self.mesh.grad_basis_map, self.x_hat[i])]
                V_grad_map_fac = [[mp.vandermonde(
                    self.mesh.grad_basis_map, self.x_hat_fac[i][zeta])] 
                    for zeta in range(0,self.mesh.N_fac[k])]
            else:
                V_grad_map = list(mp.vandermonde(
                    self.mesh.grad_basis_map, self.x_hat[i]))      
                V_grad_map_fac = [mp.vandermonde(
                    self.mesh.grad_basis_map, self.x_hat_fac[i][zeta]) 
                    for zeta in range(0,self.mesh.N_fac[k])]
                
            for m in range(0, self.d):
                
                # jacobian at volume nodes
                self.G[k][:,:,m] = V_grad_map[m] @ self.mesh.x_tilde_map[k]
                
                # jacobian at facet nodes
                for zeta in range(0,self.mesh.N_fac[k]):
                    self.G_fac[k][zeta][:,:,m] = V_grad_map_fac[zeta][m] \
                    @ self.mesh.x_tilde_map[k]
            
            # inverse Jacobian
            self.G_inv.append(np.array([ 
                np.linalg.inv(self.G[k][j,:,:])
                for j in range(0,self.N_omega[i])]))
            
            self.G_inv_fac.append([np.array([ 
                np.linalg.inv(self.G_fac[k][zeta][j,:,:])
                for j in range(0,self.N_gamma[i][zeta])]) 
                for zeta in range(0,self.mesh.N_fac[k])]) 
            
            # Jacobian determinant
            self.J.append(
                np.array([np.linalg.det(self.G[k][j,:,:]) 
                          for j in range(0,self.N_omega[i])]))
            
            self.J_fac.append([np.array([np.linalg.det(
                self.G_fac[k][zeta][j,:,:]) 
                          for j in range(0,self.N_gamma[i][zeta])]) 
                                 for zeta in range(0, self.mesh.N_fac[k])])
            
            # Unscaled normal vectors
            n_unscl = [np.array([
                self.J_fac[k][zeta][j]*self.G_inv_fac[k][zeta][j,:,:].T @ 
                self.n_hat[i][zeta]
                for j in range(0,self.N_gamma[i][zeta])])
                for zeta in range(0,self.mesh.N_fac[k])]
            
            # facet Jacobian
            self.Jf.append([np.array([np.linalg.norm(
                n_unscl[zeta][j,:]) 
                for j in range(0,self.N_gamma[i][zeta])])
                for zeta in range(0,self.mesh.N_fac[k])])
            
            # scale unit normal vectors (i.e. divide by facet Jacobian)
            self.n.append([np.array(
                [n_unscl[zeta][j,:]/self.Jf[k][zeta][j]
                for j in range(0,self.N_gamma[i][zeta])]).T
                for zeta in range(0,self.mesh.N_fac[k])])
            
        self.test_normals()
        
     
    def build_local_operators(self):
        
        # mass matrix and inverse
        self.M = [self.V[i].T @ self.W[i] @ self.V[i] 
                  for i in range(0,self.N_d)]
        self.M_inv = [np.linalg.inv(self.M[i]) for i in range(0, self.N_d)]
    
        # discrete orthogonal projection matrix
        self.P = [self.M_inv[i] @ self.V[i].T @ self.W[i] 
                for i in range(0,self.N_d)]
        
        # weighted (i.e. physical space) projection matrix
        self.P_phys =  [np.linalg.inv(self.V[self.element_to_discretization[k]].T @ 
                                   self.W[self.element_to_discretization[k]] @ 
                                   np.diag(self.J[k]) @ 
                                   self.V[self.element_to_discretization[k]]) @ 
        self.V[self.element_to_discretization[k]].T @ 
        self.W[self.element_to_discretization[k]] @ 
        np.diag(self.J[k]) for k in range(0,self.mesh.N_el)]
        
        # derivative operator
        if self.solution_representation == "modal":
            
            if self.d==1:
                
                self.V_grad = [[mp.vandermonde(self.grad_basis[i],
                                             self.x_hat[i])]
                             for i in range(0,self.N_d)]
            else:
                
                self.V_grad =[list(mp.vandermonde(self.grad_basis[i], 
                                                self.x_hat[i]))
                            for i in range(0,self.N_d)]
               
            # stiffness matrix
            self.S = [[self.V[i].T @ self.W[i] @ self.V_grad[i][m] 
                       for m in range(0, self.d)]
                       for i in range(self.N_d)]
            
            self.D = [[self.M_inv[i] @ self.S[i][m]
                          for m in range(0, self.d)]
                         for i in range(0, self.N_d)]
            
        elif self.solution_representation == "nodal":
            
            if self.d==1:
                
                self.V_tilde_grad = [[mp.vandermonde(self.grad_basis[i],
                                             self.x_hat_tilde[i])]
                             for i in range(0,self.N_d)]
            else:
                
                self.V_tilde_grad = [list(mp.vandermonde(self.grad_basis[i],
                                             self.x_hat_tilde[i]))
                             for i in range(0,self.N_d)]
                
            self.D = [[self.V_tilde_grad[i][m] @ self.V_tilde_inv[i] 
                          for m in range(0,self.d)]
                         for i in range(0, self.N_d)]
            
        else:
            
            raise NotImplementedError
        
        if self.correction == "c_dg":
    
            self.M_phys = [self.V[self.element_to_discretization[k]].T 
                @ self.W[self.element_to_discretization[k]] 
                @ np.diag(self.J[k]) 
                @ self.V[self.element_to_discretization[k]] 
                for k in range(0,self.mesh.N_el)]
            
            self.L = [[self.M_inv[i] @ self.R[i][zeta].T
                       @ self.B[i][zeta] for zeta in range(0, self.N_fac[i])]
                       for i in range(0,self.N_d)]
                    
        elif self.correction == "c_+":
            
            # volume of reference element
            abs_omega = 2.0**self.d/(factorial(self.d))
            
            if self.d == 1:
                
                c_plus = {2: 0.183,
                          3: 3.6e-3,
                          4: 4.67e-3,
                          5: 4.28e-7}
                
                D_alpha = [np.linalg.matrix_power(self.D[i][0],self.p[i]) 
                           for i in range(0,self.N_d)]
                
                self.K = [1.0/abs_omega*c_plus[self.p[i]]*D_alpha[i].T 
                          @ self.M[i] @ D_alpha[i] 
                          for i in range(0,self.N_d)] 
                
            elif self.d == 2:
                
                c_plus = {2: 4.3e-2,
                          3: 6.0e-4,
                          4: 5.6e-6}
                
                # alpha = (alpha_0, alpha_1) = (p-q, q) , q = 0, ..., p
                alpha = [np.array([[self.p[i] - q,q] for q in range(0,self.p[i] + 1)]) 
                         for i in range(0,self.N_d)]
        
                # c_alpha = p choose q, q = 0, ..., p
                c_alpha = [c_plus[self.p[i]]*np.array([special.comb(self.p[i], q, 
                                exact=True) for q in range(0,self.p[i] + 1)])
                           for i in range(0,self.N_d)]
                
                # d^{|alpha|}/dx^{alpha_0}dy^{alpha_1} 
                D_alpha = [[np.linalg.matrix_power(self.D[i][0], alpha[i][q][0])
                           @ np.linalg.matrix_power(self.D[i][1], alpha[i][q][1])
                           for q in range(0,self.p[i]+1)]
                           for i in range(0,self.N_d)]
                    
                self.K = [1.0/abs_omega*sum([c_alpha[i][q]*D_alpha[i][q].T 
                                             @ self.M[i] @ D_alpha[i][q]
                               for q in range(0,self.p[i]+1)])
                    for i in range(0,self.N_d)]
        
            # physical mass matrix
            self.M_phys = [(self.M[self.element_to_discretization[k]]
                + self.K[self.element_to_discretization[k]]) @ 
                self.P[self.element_to_discretization[k]] @ np.diag(self.J[k]) 
                @ self.V[self.element_to_discretization[k]]
                for k in range(0,self.mesh.N_el)]
            
            # lifting operator
            self.L = [[np.linalg.inv(self.M[i] + self.K[i]) 
                       @ self.R[i][zeta].T
                       @ self.B[i][zeta] for zeta in range(0, self.N_fac[i])]
                       for i in range(0,self.N_d)]
            
        else:
            raise NotImplementedError
            
        self.M_phys_inv = [np.linalg.inv(self.M_phys[k]) for k in range(0, self.mesh.N_el)]
        
        # pre-compute full volume and facet operators
        if self.form == "weak":
            
            self.vol = [[self.M_phys_inv[k] 
                         @ (self.D[self.element_to_discretization[k]][m]).T
                         @ self.V[self.element_to_discretization[k]].T 
                         @ self.W[self.element_to_discretization[k]]
                        for m in range(0,self.d)]
                        for k in range(0,self.mesh.N_el)]
            
            self.fac = [[-1.0*self.M_phys_inv[k] 
                         @ self.R[self.element_to_discretization[k]][zeta].T
                         @ self.B[self.element_to_discretization[k]][zeta]
                         for zeta in range(0, self.mesh.N_fac[k])]
                         for k in range(0,self.mesh.N_el)]
            
        elif self.form == "strong":
                   
            self.vol = [[-1.0*np.linalg.inv(self.P[self.element_to_discretization[k]] 
                                            @ np.diag(self.J[k]) 
                                            @ self.V[self.element_to_discretization[k]]) 
                         @ self.D[self.element_to_discretization[k]][m] @
                          self.P[self.element_to_discretization[k]]
                        for m in range(0,self.d)] for k in range(0,self.mesh.N_el)]
            
            self.fac = [[-1.0*np.linalg.inv(self.P[self.element_to_discretization[k]] 
                                            @ np.diag(self.J[k]) 
                                            @ self.V[self.element_to_discretization[k]])
                         @ self.L[self.element_to_discretization[k]][zeta]
                         for zeta in range(0, self.mesh.N_fac[k])]
                         for k in range(0,self.mesh.N_el)]
            
        else:
            
            raise NotImplementedError

        
    def build_global_residual(self, f, f_star, bc, N_eq):
        
        def global_residual(self, f, f_star, bc, N_eq, u_tilde, t,
                            print_output=False):
            
            f_contravariant = []
            f_facet = []
            for k in range(0,self.mesh.N_el):
                
                i = self.element_to_discretization[k]
               
                # compute volume-weighted contravariant fluxes
                f_contravariant.append([sum([self.J[k]*self.G_inv[k][:,m,n]*f((self.V[i] @ u_tilde[k].T).T, self.x[k])[n] 
                     for n in range(0,self.d)]) for m in range(0, self.d)])
               
                f_facet.append([])
                for zeta in range(0, self.mesh.N_fac[k]):  
                    
                    # get external state for interior facet
                    if self.mesh.local_to_local[(k,zeta)] is not None:  
                        
                        nu, eta = self.mesh.local_to_local[(k,zeta)]
                        u_plus = (self.facet_permutation[k][zeta].T @ self.R[
                            self.element_to_discretization[nu]][eta] @ u_tilde[nu].T).T
                        
                    # get external state for boundary facet
                    else:
                        
                        u_plus = [bc[self.mesh.local_to_bc_index[(k,zeta)]][e](
                            self.x_fac[k][zeta], t) 
                            for e in range(0, N_eq)]
                    
                    # weak form - facet flux is just transformed numerical flux
                    if self.form == "weak":
                        
                        f_facet[k].append(
                            self.Jf[k][zeta] * f_star(
                            (self.R[i][zeta] @ u_tilde[k].T).T, u_plus,
                            self.x_fac[k][zeta], self.n[k][zeta]))
                        
                    # weak form - facet flux is the difference between
                    # transformed numerical and physical flux
                    elif self.form == "strong":
                        
                        f_facet[k].append(self.Jf[k][zeta] * f_star(
                            (self.R[i][zeta] @ u_tilde[k].T).T, u_plus,
                            self.x_fac[k][zeta], self.n[k][zeta])
                            - (self.R[i][zeta] @ self.P[i] 
                               @ sum([f_contravariant[k][m].T*self.n_hat[i][zeta][m] 
                                      for m in range(0,self.d)])).T)
                  
            # apply pre-computeds strong or weak volume and facet matrices
            return [np.array([sum([self.vol[k][m] @ f_contravariant[k][m][e,:] 
                                      for m in range(0,self.d)])
                     + sum([self.fac[k][zeta] @ 
                               f_facet[k][zeta][e,:]
                               for zeta in range(0, self.mesh.N_fac[k])])
                    for e in range(0, N_eq)])
                    for k in range(0, self.mesh.N_el)]
                
        # return global residual
        return lambda u_tilde, t: global_residual(self, f, f_star,
                                                bc, N_eq, u_tilde, t)
        

    def test_normals(self, print_output=True, tol=1.0e-8):
        # check each interface to see if the normals computed for
        # each coincident face by mapping are opposite one another
        
        self.normals_good = []
        
        for k in range(0,self.mesh.N_el):
            
            self.normals_good.append([True]*self.mesh.N_fac[k])
            for zeta in range(0,self.mesh.N_fac[k]):
                
                if self.mesh.local_to_local[(k,zeta)] is not None:
                    
                    nu, eta = self.mesh.local_to_local[(k,zeta)]
                    if np.amax(np.abs(self.n[nu][eta].T 
                                      + self.facet_permutation[k][zeta] 
                                      @ self.n[k][zeta].T)) > tol:
                        self.normals_good[k][zeta] = False
                        
                        if print_output:
                            print("Watertightness violated at (k,zeta) = ",
                                  (k,zeta))
                            print("T @ n_k,zeta", 
                                  self.facet_permutation[k][zeta] 
                                      @ self.n[k][zeta], "n_nu,eta: ", 
                                      self.n[nu][eta])
            
            
    def plot(self, plot_nodes=True, plot_geometry_nodes=False, axes=True,
             markersize=4, geometry_resolution=10, filename=None):
    
        # assign a colour to each element
        self.color = iter(plt.cm.rainbow(
            np.linspace(0, 1, self.mesh.N_el)))
        
        if self.d == 1:
        
            mesh_plot = plt.figure()
            ax = plt.axes()
            ax.axes.yaxis.set_visible(False)
            plt.xlim([self.mesh.x_min[0] 
                      - 0.025 * self.mesh.extent[0], 
                      self.mesh.x_max[0] 
                      + 0.025 * self.mesh.extent[0]])
            plt.xlabel("$x$")
            
            for k in range(0, self.mesh.N_el):
                current_color = next(self.color)
                
                #plot node positions
                if plot_nodes:
                    ax.plot(self.x[k][0,:], 
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
                    [ax.plot(self.mesh.x_map[k][0,:], 
                      np.zeros(self.mesh.p_map+1), "ok",
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
            ax.set_xlim([self.mesh.x_min[0] 
                         - 0.025 * self.mesh.extent[0],
                          self.mesh.x_max[0] 
                          + 0.025 * self.mesh.extent[0]])
            
            ax.set_ylim([self.mesh.x_min[1] 
                         - 0.025 * self.mesh.extent[1],
                          self.mesh.x_max[1] 
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
            
            V_edge_map = [mp.vandermonde(
                self.mesh.basis_map, ref_edge_points[zeta])
                          for zeta in range(0,3)]

            # loop through all elements
            for k in range(0, self.mesh.N_el):
                current_color = next(self.color)
                
                if plot_nodes:
                    
                    ax.plot(self.x[k][0,:], 
                            self.x[k][1,:], "o",
                          markersize=markersize,
                          color = current_color)
                    
                if plot_geometry_nodes:
                    ax.plot(self.mesh.x_map[k][0,:], 
                            self.mesh.x_map[k][1,:], "ok",
                          #fillstyle='solid',
                          markersize=markersize)
                        
                for zeta in range(0, self.mesh.N_fac[k]):
                    
                    # plot facet edge curves
                    edge_points = (V_edge_map[zeta] 
                                   @ self.mesh.x_tilde_map[k]).T
                    
                    ax.plot(edge_points[0,:], 
                                edge_points[1,:], 
                                '-', 
                                linewidth=markersize*0.25,
                                color="black")
                         
                    if plot_nodes:
                         
                        # plot facet nodes
                        ax.plot(self.x_fac[k][zeta][0,:], 
                                self.x_fac[k][zeta][1,:],
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
            
            ax.plot(self.x_hat[0][0,:],
                    self.x_hat[0][1,:], "o",
                    markersize=markersize,
                    color="black",
                    fillstyle='none')
            
            for zeta in range(0, self.N_fac[0]):
                
                ax.plot(self.x_hat_fac[0][zeta][0,:], 
                        self.x_hat_fac[0][zeta][1,:],
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
    
    def __init__(self, mesh, p, volume_quadrature_degree=None, 
                 facet_quadrature_degree=None,
                 volume_rule=None, facet_rule="lg", 
                 form="weak", solution_representation="modal",
                 correction="c_dg"):
        
        if volume_quadrature_degree is None:
            volume_quadrature_degree = 2*p
            
        if facet_quadrature_degree is None:
            facet_quadrature_degree = 2*p + 1
        
        if mesh.d == 1:
            
            if volume_rule == None or volume_rule == "lg":
                
                volume_quadrature = mp.LegendreGaussQuadrature(
                    ceil((volume_quadrature_degree-1)/2))
                volume_nodes = np.array([volume_quadrature.nodes])
                W = np.diag(volume_quadrature.weights)
                
            elif volume_rule == "lgl":
                
               raise NotImplementedError
            
            facet_nodes = [np.array([[-1.0]]),np.array([[1.0]])]
            B = [np.array([[1.0]]),np.array([[1.0]])]
            n_hat = [np.array([-1.0]), np.array([1.0])]
        
        elif mesh.d == 2:
            
            volume_quadrature = mp.XiaoGimbutasSimplexQuadrature(
                volume_quadrature_degree,2)
            volume_nodes = volume_quadrature.nodes
            W = np.diag(volume_quadrature.weights)
            
            if facet_rule == "lg":
                
                facet_quadrature = mp.LegendreGaussQuadrature(
                    ceil((facet_quadrature_degree-1)/2))
                facet_nodes = SpatialDiscretization.map_unit_to_facets(
                    facet_quadrature.nodes,
                    element_type="triangle") 
                
            elif facet_rule == "lgl":
                
                facet_quadrature = qp.c1.gauss_lobatto(
                    ceil((facet_quadrature_degree+3)/2))
                facet_nodes = SpatialDiscretization.map_unit_to_facets(
                    facet_quadrature.points,
                    element_type="triangle") 
                
            n_hat = [np.array([0.0,-1.0]), 
                     np.array([1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)]),
                     np.array([-1.0, 0.0])]
            
            B = [np.diag(facet_quadrature.weights),
                       np.sqrt(2.0)*np.diag(facet_quadrature.weights),
                       np.diag(facet_quadrature.weights)]
            
        else: 
            raise NotImplementedError
    
        super().__init__(mesh, [0]*mesh.N_el, [p], [volume_nodes],
                         [facet_nodes], [W], [B], [n_hat], form=form,
                         solution_representation=solution_representation,
                         correction=correction)
    
    
class SimplexCollocationDiscretization(SpatialDiscretization):
    
    def __init__(self, mesh, p, volume_collocation_degree=None, 
                 facet_collocation_degree=None,
                 form="weak", solution_representation="modal",
                 correction="c_dg", use_lumping=False):
            
        if volume_collocation_degree is None:
            volume_collocation_degree = p
        
        if facet_collocation_degree is None:
            facet_collocation_degree = p
            
        # modal basis to construct nodal mass matrices
        volume_basis = mp.simplex_onb(mesh.d, volume_collocation_degree)    
        
        if mesh.d == 1:
            
            volume_nodes = np.array(
                [mp.quadrature.jacobi_gauss.legendre_gauss_lobatto_nodes(
                    volume_collocation_degree)]) 
            
            V_vol = mp.vandermonde(volume_basis, volume_nodes[0,:]) 
            W = np.linalg.inv(V_vol @ V_vol.T)
            facet_nodes = [np.array([[-1.0]]),np.array([[1.0]])]
            B = [np.array([[1.0]]),np.array([[1.0]])]
            n_hat = [np.array([-1.0]), np.array([1.0])]
            
        elif mesh.d == 2:
            
            facet_basis = mp.simplex_onb(mesh.d-1, facet_collocation_degree)
            
            volume_nodes = mp.warp_and_blend_nodes(2, volume_collocation_degree)
          
            V_vol = mp.vandermonde(volume_basis, volume_nodes) 
            W = np.linalg.inv(V_vol @ V_vol.T)
            
            facet_nodes_1D =  mp.quadrature.jacobi_gauss.legendre_gauss_lobatto_nodes(
                volume_collocation_degree)
            V_fac_1D = mp.vandermonde(facet_basis,facet_nodes_1D) 
            B_1D =  np.linalg.inv(V_fac_1D @ V_fac_1D.T)
            
            facet_nodes = SpatialDiscretization.map_unit_to_facets(
                facet_nodes_1D, element_type="triangle")
            
            if use_lumping:
                
                W = np.diag(np.sum(W,axis=1))
                B_1D = np.diag(np.sum(B_1D,axis=1))
             
            n_hat = [np.array([0.0,-1.0]), 
                     np.array([1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)]),
                     np.array([-1.0, 0.0])]
            
            B = [B_1D, np.sqrt(2.0)*B_1D, B_1D]

        
        super().__init__(mesh, [0]*mesh.N_el, [p], [volume_nodes],
                         [facet_nodes], [W], [B], [n_hat], form=form,
                         solution_representation=solution_representation,
                         correction=correction)