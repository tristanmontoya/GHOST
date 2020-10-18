# GHOST - Mesh Data Structure (Unstructured)

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from functools import partial
import meshio

BC_TOL = 1.0e-8

class Mesh(ABC):
    
    def __init__(self, name, d):
        
        # name and dimension
        self.name = name
        self.d = d
        
        # evaluate the mapping and metric terms from affine mesh
        self.compute_affine_mapping()
        self.map_mesh()
        
    @abstractmethod 
    def compute_affine_mapping(self):
        pass
    
    
    @abstractmethod
    def plot_mesh(self, fontsize=8):
        pass
    
    def map_mesh(self, f_map=lambda x: x, J_map = None):
        
        # default is to keep original affine mapping, do not curve
        if J_map is None:
            J_map = lambda x: np.eye(self.d)
        
        self.X = [(lambda xi, k=k: f_map(self.X_affine[k](xi))) 
                  for k in range(0,self.K)]
        self.J = [(lambda xi, k=k: 
                   J_map(self.X_affine[k](xi)) @ self.J_affine[k](xi))
                  for k in range(0,self.K)]
        self.detJ = [(lambda xi, k=k: np.linalg.det(self.J(xi))) 
                     for k in range(0,self.K)]
    
    
    def add_bc_at_facet(self, local_index, bc_index):
        
        self.local_to_bc_index[local_index] = bc_index
        
        
    def add_bcs_by_indicators(self, indicator, bc_index):
        
        # indicator function is defined such that it is zero everywhere except
        # for the boundary where this bc is being imposed (lines/faces 
        # should have thickness for tolerance)
        
        # have to loop through all facets and see which ones have midpoints 
        # on which indicated boundary
        for k in range(0,self.K):
            for gamma in range(0,self.Nf_local[k]): 
                
                # get physical vertex locations for this local facet
                facet_vertices = [self.v[self.local_to_vertex[k][gamma][i]] 
                                  for i in range(0,len(
                                          self.local_to_vertex[k][gamma]))]
               
                # add bc if lies on midpoint of facet
                midpoint = sum(facet_vertices)/len(facet_vertices)
                for i in range(0,len(indicator)):
                    if indicator[i](midpoint) != 0:
                        self.add_bc_at_facet((k,gamma), bc_index[i])


    def add_bc_on_hyperplanes(self, coeffs, bc_index, tol=BC_TOL):
        
        hyperplanes = [partial(Mesh.hyperplane_indicator,coeffs[i],tol) 
                       for i in range(0,len(coeffs))]
        
        self.add_bcs_by_indicators(hyperplanes, bc_index)
        
        
    def make_periodic(self, bc_index, hyperplane_axes=[0], tol=BC_TOL):
        
        # use local-to-local conectivity to model periodicity
        
        for local_index in self.local_to_bc_index:

            if self.local_to_bc_index[local_index] == bc_index[0]:
                k,gamma = local_index[0], local_index[1]
                
                # get physical vertex locations for this local facet
                facet_vertices = [self.v[self.local_to_vertex[k][gamma][i]]
                                  for i in range(0,len(
                                          self.local_to_vertex[k][gamma]))]
                
                midpoint = sum(facet_vertices)/len(facet_vertices)
                
                for other_index in self.local_to_bc_index:
                    if self.local_to_bc_index[other_index] == bc_index[1]:
                        nu,rho = other_index[0], other_index[1]
                        
                        other_facet_vertices = [self.v[self.local_to_vertex[nu][rho][i]]
                                      for i in range(0,len(
                                              self.local_to_vertex[nu][rho]))]
                        other_midpoint = sum(other_facet_vertices)/len(
                            other_facet_vertices)   
                        
                        # match up if corresponds to translation along a coordinate axis
                        if max([midpoint[hyperplane_axes[axis]] - other_midpoint[hyperplane_axes[axis]] 
                                for axis in range(0,len(hyperplane_axes))]) < BC_TOL:
                            
                            
                            # add local-to-local connectivity to dictionary
                            self.local_to_local[k,gamma] = (nu,rho)
                            self.local_to_local[nu,rho] = (k,gamma)             
                            break
                    
        
    @staticmethod
    def hyperplane_indicator(coeffs, tol, x):
        if np.abs(np.dot(coeffs[0:-1], x) - coeffs[-1]) < tol:
            return 1
        else:
            return 0
        
        
class Mesh1D(Mesh):
    
    def __init__(self, name, x_L, x_R, K, 
                 spacing='uniform', 
                 periodic=True,
                 transform=None):
        
        self.K = K
        self.Nv_global = self.K+1
        self.Nv_local = [2 for k in range(0,self.K)]
        self.Nf_local = self.Nv_local.copy()
        

        # generate vertices
        if spacing == 'uniform':
            self.v = np.linspace(x_L, x_R, self.K+1).reshape([self.K+1, 1])
        else:
            raise NotImplementedError
            
        # generate element to vertex connectivity
        self.element_to_vertex = [[k,k+1] for k in range(0,self.K)]
        
        # generate local facet to global vertex connectivity
        self.local_to_vertex = [[(k,),(k+1,)] for k in range(0,self.K)]
        
        # local to local is None for boundary facets
        self.local_to_local = {}
        
        # when evaluating fluxes look for this when local to local is None
        self.local_to_bc_index = {}
        
        if periodic:
            global_to_local = [[(k,1),(k+1,0)] for k in range(0,self.K-1)]
            global_to_local.insert(0,[(self.K-1,1),(0,0)])
            
        else:
           global_to_local = [[(k,1),(k+1,0)] for k in range(0,self.K)]
    
        # generate local to local connectivity (kappa,gamma to rho,nu)

        # get interior facet to facet connectivity
        for k in range(0,self.K):
            
            for gamma in range(0,2):
                
                # initially assume face has no neignbours before searching
                self.local_to_local[k,gamma] = None
                
                # find (nu,rho) matching (k,gamma)
                for nu in range(0,self.K):
                    if nu == k:
                        continue
                    try:
                        rho = self.local_to_vertex[nu].index(
                            (self.local_to_vertex[k][gamma][0],))
                    except ValueError:
                        # this element is not a neighbour of (k,gamma)
                        continue 
    
                    # add to dictionaries
                    self.local_to_local[k,gamma] = (nu,rho)
                    break
    
        super().__init__(name,1)
        
        
    def compute_affine_mapping(self):
        
        # map from (-1, 1) to physical element
        self.X_affine = [(lambda xi, k=k: self.v[
            self.local_to_vertex[k][0][0]] + 0.5*(
                self.v[self.local_to_vertex[k][1][0]] -self.v[
                                  self.local_to_vertex[k][0][0]])*(xi+1)) 
                         for k in range(0,self.K)]
        
        self.J_affine = [(lambda xi,k=k: 
                          np.array([0.5*(
                              self.v[self.local_to_vertex[k][1][0]] -self.v[
                                  self.local_to_vertex[k][0][0]])]))
                         for k in range(0,self.K)]
    
        
    def plot_mesh(self, fontsize=8):
        
        x_L = np.amin(self.v[:,0])
        x_R = np.amax(self.v[:,0])
        L = x_R - x_L
        
        meshplt = plt.figure()
        ax = plt.axes()
        plt.xlim([x_L - 0.1 * L, x_R + 0.1 * L])
        plt.ylim([-0.1 * L, 0.1 * L])
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False)  
        ax.set_aspect('equal')
        plt.axis('off')
    
        color = iter(plt.cm.rainbow(np.linspace(0, 1, self.K)))
        for k in range(0, self.K):
            ax.plot([self.v[k,0], self.v[k+1,0]],[0.0,0.0], '-', 
                    color=next(color))
            plt.text((self.v[k,0] + self.v[k+1,0])*0.5, 0.05 * L,
                     str(k), color='black', fontsize=fontsize, ha='center')
    
        plt.show()
        meshplt.savefig("../plots/" + self.name + ".pdf", 
                        bbox_inches=0, pad_inches=0)
     

class Mesh2D(Mesh):
    
    def __init__(self, name, filename, transform=None):
        
        # assume gmsh format for now
        mesh_data = meshio.read(filename)
        
        self.v = mesh_data.points[:,0:2]
        self.Nv_global = self.v.shape[0]
        self.K = mesh_data.cells[0][1].shape[0]
        self.element_to_vertex = [list(mesh_data.cells[0][1][k]) 
                                  for k in range(0,self.K)]
        self.Nv_local = [len(self.element_to_vertex[k]) 
                         for k in range(0,self.K)]
        self.Nf_local = self.Nv_local.copy()
        
        # get facet to vertex connectivity
        self.local_to_vertex = [[(self.element_to_vertex[k][i],
                                  self.element_to_vertex[k][i+1]) 
                                 for i in range(0,self.Nv_local[k]-1)] + [(
                                    self.element_to_vertex[k][
                                        self.Nv_local[k]-1],
                                    self.element_to_vertex[k][0])] 
                                for k in range(0,self.K)]
        
        # local to local is None for boundary facets
        self.local_to_local = {}
        
        # when evaluating fluxes look for this when local to local is None
        self.local_to_bc_index = {}
        
        # get interior facet to facet connectivity
        for k in range(0,self.K):
            for gamma in range(0,self.Nf_local[k]):
                
                # initially assume face has no neignbours before searching
                self.local_to_local[k,gamma] = None
                
                # find (nu,rho) with edge containing matching vertices 
                # corresponding to (k,gamma) swapped (due to CCW ordering)
                for nu in range(0,self.K):
                    try:
                        rho = self.local_to_vertex[nu].index(
                            (self.local_to_vertex[k][gamma][1],
                             self.local_to_vertex[k][gamma][0]))
                    except ValueError:
                        continue # this element is not a neighbour of (k,gamma)
                        
                    # add to dictionaries
                    self.local_to_local[k,gamma] = (nu,rho)
                    break
                
        super().__init__(name,2)
        
                           
    def compute_affine_mapping(self):
        pass
        
    
    def plot_mesh(self, fontsize=8):
        
        x_L = np.amin(self.v[:,0])
        x_R = np.amax(self.v[:,0])
        y_L = np.amin(self.v[:,1])
        y_R = np.amax(self.v[:,1])
        
        W = x_R - x_L
        H = y_R - y_L
        
        plt.xlim([x_L - 0.1 * W, x_R + 0.1 * W])
        plt.ylim([y_L - 0.1 * H, y_R + 0.1 * H])

        
        meshplt = plt.figure()
        ax = plt.axes()
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False) 
        plt.axis('off')
        plt.show()
        meshplt.savefig("../plots/" + self.name + ".pdf",
                        bbox_inches=0, pad_inches=0)
    