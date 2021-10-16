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


    def map_mesh(self, curving_function=None, p_map=3):
        
        # default is to keep original affine mapping, do not curve
        if curving_function is None:
            curving_function = lambda x: x
        
        # degree of Lagrange interpolant used to represent curvilinear mapping
        self.p_map = p_map
        
        # for simplex mesh only (represent geometry in modal basis)
        self.basis_map = mp.simplex_onb(self.d,self.p_map) 
        self.grad_basis_map = mp.grad_simplex_onb(self.d, self.p_map)
    
        if self.d == 1:
            
            self.x_hat_map =np.array(
                [mp.quadrature.jacobi_gauss.legendre_gauss_lobatto_nodes(
                    self.p_map)])
            self.V_inv_map =np.linalg.inv(mp.vandermonde(
                self.basis_map, self.x_hat_map[0]))
            
        else:
            
            self.x_hat_map = mp.warp_and_blend_nodes(self.d, self.p_map)
            self.V_inv_map =np.linalg.inv(mp.vandermonde(self.basis_map,
                                                        self.x_hat_map))
        
        self.N_map = self.x_hat_map.shape[1]
        self.x_map = []
        self.x_tilde_map = []
       
        # move vertices
        for v in range(0,self.N_v_global):
            self.v[:,v] = curving_function(self.v_affine[:,v])
                    
        # generate displacement field
        for k in range(0,self.N_el):
            
            # perturbed nodes
            self.x_map.append(np.array([curving_function(self.X_affine[k](
                self.x_hat_map[:,i])) 
                for i in range(0,self.N_map)]).T)
            
            # modal coefficients in basis_map for displacement field
            self.x_tilde_map.append(self.V_inv_map @ self.x_map[k].T)
            
        # get geometry info for mapped grid
        self.x_min = np.amin(self.v, axis=1)
        self.x_max = np.amax(self.v, axis=1)
        self.extent = self.x_max - self.x_min
        
    def add_bc_at_facet(self, local_index, bc_index):
        
        self.local_to_bc_index[local_index] = bc_index
        
        
    def add_bcs_by_indicators(self, indicator, bc_index):
        
        # indicator function is defined such that it is zero everywhere except
        # for the boundary where this bc is being imposed (lines/faces 
        # should have thickness for tolerance)
        
        # have to loop through all facets and see which ones have midpoints 
        # on which indicated boundary
        for k in range(0,self.N_el):
            for zeta in range(0,self.N_fac[k]): 
                
                # get physical vertex locations for this local facet
                facet_vertices = [self.v[:,self.local_to_vertex[k][zeta][i]] 
                                  for i in range(0,len(
                                          self.local_to_vertex[k][zeta]))]
               
                # add bc if lies on midpoint of facet
                midpoint = sum(facet_vertices)/len(facet_vertices)
                for i in range(0,len(indicator)):
                    if indicator[i](midpoint) != 0:
                        self.add_bc_at_facet((k,zeta), bc_index[i])


    def add_bc_on_hyperplanes(self, coeffs, bc_index, tol=BC_TOL):
        
        hyperplanes = [partial(Mesh.hyperplane_indicator,coeffs[i],tol) 
                       for i in range(0,len(coeffs))]
        
        self.add_bcs_by_indicators(hyperplanes, bc_index)
        
        
    def make_periodic(self, bc_index, hyperplane_axes=[0], tol=BC_TOL):
        
        # use local-to-local conectivity to model periodicity
        
        for local_index in self.local_to_bc_index:

            if self.local_to_bc_index[local_index] == bc_index[0]:
                k,zeta = local_index[0], local_index[1]
                
                # get physical vertex locations for this local facet
                facet_vertices = [self.v[:,self.local_to_vertex[k][zeta][i]]
                                  for i in range(0,len(
                                          self.local_to_vertex[k][zeta]))]
                
                midpoint = sum(facet_vertices)/len(facet_vertices)

                for other_index in self.local_to_bc_index:
                    
                    
                    if self.local_to_bc_index[other_index] == bc_index[1]:
                        nu,eta = other_index[0], other_index[1]
            
                        other_facet_vertices = [
                            self.v[:,self.local_to_vertex[nu][eta][i]]
                                      for i in range(0,len(
                                              self.local_to_vertex[nu][eta]))]
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
                            self.local_to_local[k,zeta] = (nu,eta)
                            self.local_to_local[nu,eta] = (k,zeta)             
                            break
                        
        
    @staticmethod
    def hyperplane_indicator(coeffs, tol, x):
        if np.abs(np.dot(coeffs[0:-1], x) - coeffs[-1]) < tol:
            return 1
        else:
            return 0
        
        
class Mesh1D(Mesh):
    
    def __init__(self, name, x_L, x_R, N_el, 
                 spacing='uniform'):
        
        self.N_el = N_el
        self.N_v_global = self.N_el+1
        self.N_v_local = [2 for k in range(0,self.N_el)]
        self.N_fac = self.N_v_local.copy()
        

        # generate vertices
        if spacing == 'uniform':
            self.v = np.array([np.linspace(x_L, x_R, self.N_el+1)])
        else:
            raise NotImplementedError
            
        # generate element to vertex connectivity
        self.element_to_vertex = [[k,k+1] for k in range(0,self.N_el)]
        
        # generate local facet to global vertex connectivity
        self.local_to_vertex = [[(k,),(k+1,)] for k in range(0,self.N_el)]
        
        # local to local is None for boundary facets
        self.local_to_local = {}
        
        # when evaluating fluxes look for this when local to local is None
        self.local_to_bc_index = {}
    
        # generate local to local connectivity (kappa,zeta to eta,nu)
        for k in range(0,self.N_el):
            
            for zeta in range(0,2):
                
                # initially assume face has no neignbours before searching
                self.local_to_local[k,zeta] = None
                
                # find (nu,eta) matching (k,zeta)
                for nu in range(0,self.N_el):
                    if nu == k:
                        continue
                    
                    try:
                        eta = self.local_to_vertex[nu].index(
                            (self.local_to_vertex[k][zeta][0],))
                        
                    except ValueError:
                        # this element is not a neighbour of (k,zeta)
                        continue 
    
                    # add to dictionaries
                    self.local_to_local[k,zeta] = (nu,eta)
                    break
    
        super().__init__(name,1)
        
        
    def compute_affine_mapping(self):
        
        # map from (-1, 1) to physical element
        self.X_affine = [(lambda x_hat, k=k: self.v_affine[0,
            self.local_to_vertex[k][0][0]] + 0.5*(
                self.v_affine[0,self.local_to_vertex[k][1][0]] 
                - self.v_affine[0,self.local_to_vertex[k][0][0]])*(x_hat+1))
                         for k in range(0,self.N_el)]
                
    @staticmethod
    def grid_transformation(warp_factor=0.2):
        
        return lambda x_hat: np.array([x_hat[0] +
                                    warp_factor*np.sin(np.pi*x_hat[0])])


class Mesh2D(Mesh):
    
    def __init__(self, name, filename, transform=None):
        
        # assume gmsh format for now
        mesh_data = meshio.read(filename)
        
        self.v = mesh_data.points[:,0:2].T
        self.N_v_global = self.v.shape[1]
        self.N_el = mesh_data.cells[0][1].shape[0]
        self.element_to_vertex = [list(mesh_data.cells[0][1][k]) 
                                  for k in range(0,self.N_el)]
        self.N_v_local = [len(self.element_to_vertex[k]) 
                         for k in range(0,self.N_el)]
        self.N_fac = self.N_v_local.copy()
        
        # get facet to vertex connectivity
        self.local_to_vertex = [[(self.element_to_vertex[k][i],
                                  self.element_to_vertex[k][i+1]) 
                                 for i in range(0,self.N_v_local[k]-1)] + [(
                                    self.element_to_vertex[k][
                                        self.N_v_local[k]-1],
                                    self.element_to_vertex[k][0])] 
                                for k in range(0,self.N_el)]
        
        # local to local is None for boundary facets
        self.local_to_local = {}
        
        # when evaluating fluxes look for this when local to local is None
        self.local_to_bc_index = {}
        
        # generate local to local connectivity (kappa,zeta to nu,eta)
        for k in range(0,self.N_el):
            for zeta in range(0,self.N_fac[k]):
                
                # initially assume face has no neignbours before searching
                self.local_to_local[k,zeta] = None
                
                # find (nu,eta) with edge containing matching vertices 
                # corresponding to (k,zeta) swapped (due to CCW ordering)
                for nu in range(0,self.N_el):
                    try:
                        eta = self.local_to_vertex[nu].index(
                            (self.local_to_vertex[k][zeta][1],
                             self.local_to_vertex[k][zeta][0]))
                    except ValueError:
                        continue # this element is not a neighbour of (k,zeta)
                        
                    # add to dictionaries
                    self.local_to_local[k,zeta] = (nu,eta)
                    break
                
        super().__init__(name,2)
        
                           
    def compute_affine_mapping(self):
        
        self.X_affine = []
        
        for k in range(0,self.N_el):
            
            # affine triangle mapping
            if self.N_v_local[k] == 3:
                
                self.X_affine.append(lambda x_hat,k=k: 
                    -0.5*(x_hat[0]+x_hat[1])*self.v_affine[:,self.element_to_vertex[k][0]] \
                        + 0.5*(x_hat[0] + 1)*self.v_affine[:,self.element_to_vertex[k][1]] \
                            + 0.5*(x_hat[1] + 1)*self.v_affine[:,self.element_to_vertex[k][2]])
            else: 
                raise NotImplementedError
        
            
    @staticmethod
    def grid_transformation(warp_factor=0.2, L=1.0):
        
        return lambda x: np.array([x[0] + warp_factor*L*np.sin(
                np.pi*x[0]/L)*np.sin(np.pi*x[1]/L),
                x[1] + warp_factor*L*np.exp(1-x[1]/L)*np.sin(
                np.pi*x[0]/L)*np.sin(np.pi*x[1]/L)])