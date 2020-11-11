# GHOST - Mesh Data Structure and Utilities

import numpy as np
import modepy as mp
from abc import ABC, abstractmethod
from functools import partial
import meshio

# tolerance for detecting points on boundaries
BC_TOL = 1.0e-8

class Mesh(ABC):
    
    def __init__(self, name, d):
        
        # name and dimension
        self.name = name
        self.d = d
        
        # evaluate the mapping and metric terms from affine mesh
        self.v_affine = np.copy(self.v)
        self.compute_affine_mapping()
        self.map_mesh()
    
    @abstractmethod 
    def compute_affine_mapping(self):
        pass

    def map_mesh(self, f_map=None, p_geo=3):
        
        # default is to keep original affine mapping, do not curve
        if f_map is None:
            f_map = lambda x: x
        
        # geometry degree
        self.p_geo = p_geo
        
        # for simplex mesh only (represent geometry in modal basis)
        self.basis_geo = mp.simplex_onb(self.d,self.p_geo) 
        self.grad_basis_geo = mp.grad_simplex_onb(self.d, self.p_geo)
    
        if self.d == 1:
            
            self.xi_geo =np.array([mp.LegendreGaussQuadrature(self.p_geo).nodes])
            self.Vinv_geo =np.linalg.inv(mp.vandermonde(self.basis_geo, self.xi_geo[0]))
            
        else:
            
            self.xi_geo = mp.warp_and_blend_nodes(self.d, self.p_geo)
            self.Vinv_geo =np.linalg.inv(mp.vandermonde(self.basis_geo, self.xi_geo))
        
        self.Np_geo = self.xi_geo.shape[1]
        self.x_geo = []
        self.xhat_geo = []
        isMoved = np.zeros(self.Nv_global,dtype=int)
        for k in range(0,self.K):
            
            # perturbed nodes
            self.x_geo.append(np.array([f_map(self.X_affine[k](self.xi_geo[:,i])) 
                                 for i in range(0,self.Np_geo)]).T)
            
            # modal coefficients in basis_geo for displacement field
            self.xhat_geo.append(self.Vinv_geo @ self.x_geo[k].T)
        
            # move each vertex (only once)
            for i in range(0,self.Nv_local[k]):
                if isMoved[self.element_to_vertex[k][i]] == 0:
                    self.v[:,self.element_to_vertex[k][i]]= f_map(
                        self.v_affine[:,self.element_to_vertex[k][i]])
                    isMoved[self.element_to_vertex[k][i]] = 1
                    
        # get geometry info for mapped grid
        self.xmin = np.amin(self.v, axis=1)
        self.xmax = np.amax(self.v, axis=1)
        self.extent = self.xmax - self.xmin
        
    def add_bc_at_facet(self, local_index, bc_index):
        
        self.local_to_bc_index[local_index] = bc_index
        
        
    def add_bcs_by_indicators(self, indicator, bc_index):
        
        # indicator function is defined such that it is zero everywhere except
        # for the boundary where this bc is being imposed (lines/faces 
        # should have thickness for tolerance)
        
        # have to loop through all facets and see which ones have midpoints 
        # on which indicated boundary
        for k in range(0,self.K):
            for gamma in range(0,self.Nf[k]): 
                
                # get physical vertex locations for this local facet
                facet_vertices = [self.v[:,self.local_to_vertex[k][gamma][i]] 
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
                facet_vertices = [self.v[:,self.local_to_vertex[k][gamma][i]]
                                  for i in range(0,len(
                                          self.local_to_vertex[k][gamma]))]
                
                midpoint = sum(facet_vertices)/len(facet_vertices)

                for other_index in self.local_to_bc_index:
                    
                    
                    if self.local_to_bc_index[other_index] == bc_index[1]:
                        nu,rho = other_index[0], other_index[1]
            
                        other_facet_vertices = [
                            self.v[:,self.local_to_vertex[nu][rho][i]]
                                      for i in range(0,len(
                                              self.local_to_vertex[nu][rho]))]
                        other_midpoint = sum(other_facet_vertices)/len(
                            other_facet_vertices)   
                        
                        # match up if corresponds to translation along 
                        # a coordinate axis
                        if (max(np.abs([midpoint[hyperplane_axes[axis]] 
                                        - other_midpoint[hyperplane_axes[axis]] 
                                for axis in range(
                                        0,len(hyperplane_axes))])) 
                            < BC_TOL) or self.d == 1:
                            # if d is 1, then any two facets
                            # on the periodic bcs are neighbours
                            
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
                 spacing='uniform'):
        
        self.K = K
        self.Nv_global = self.K+1
        self.Nv_local = [2 for k in range(0,self.K)]
        self.Nf = self.Nv_local.copy()
        

        # generate vertices
        if spacing == 'uniform':
            self.v = np.array([np.linspace(x_L, x_R, self.K+1)])
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
    
        # generate local to local connectivity (kappa,gamma to rho,nu)
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
        self.X_affine = [(lambda xi, k=k: self.v[0,
            self.local_to_vertex[k][0][0]] + 0.5*(
                self.v[0,self.local_to_vertex[k][1][0]] 
                - self.v[0,self.local_to_vertex[k][0][0]])*(xi+1))
                         for k in range(0,self.K)]
                
    @staticmethod
    def grid_transformation(warp_factor=0.2):
        
        return lambda xi: np.array([xi[0] + warp_factor*np.sin(np.pi*xi[0])])

class Mesh2D(Mesh):
    
    def __init__(self, name, filename, transform=None):
        
        # assume gmsh format for now
        mesh_data = meshio.read(filename)
        
        self.v = mesh_data.points[:,0:2].T
        self.Nv_global = self.v.shape[1]
        self.K = mesh_data.cells[0][1].shape[0]
        self.element_to_vertex = [list(mesh_data.cells[0][1][k]) 
                                  for k in range(0,self.K)]
        self.Nv_local = [len(self.element_to_vertex[k]) 
                         for k in range(0,self.K)]
        self.Nf = self.Nv_local.copy()
        
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
        
        # generate local to local connectivity (kappa,gamma to rho,nu)
        for k in range(0,self.K):
            for gamma in range(0,self.Nf[k]):
                
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
        
        self.X_affine = []
        
        for k in range(0,self.K):
            
            # affine triangle mapping
            if self.Nv_local[k] == 3:
                
                self.X_affine.append(lambda xi,k=k: 
                                     -0.5*(xi[0]+xi[1])*self.v_affine[:,self.element_to_vertex[k][0]] \
                                         + 0.5*(xi[0] + 1)*self.v_affine[:,self.element_to_vertex[k][1]] \
                                             + 0.5*(xi[1] + 1)*self.v_affine[:,self.element_to_vertex[k][2]])
            else: 
                raise NotImplementedError
        
            
    @staticmethod
    def grid_transformation(warp_factor=0.2):
        
        return lambda xi: np.array([xi[0] + warp_factor*np.sin(
                np.pi*xi[0])*np.sin(np.pi*xi[1]),
                xi[1] + warp_factor*np.exp(1-xi[1])*np.sin(
                np.pi*xi[0])*np.sin(np.pi*xi[1])])
        
    