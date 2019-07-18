# GHOST - Affine Mesh Base Class

class AffineMesh:
    """

    # Notes

        For now, assume all elements topologically the same, and mesh is conforming

    # Properties

        type (string)
            # options: 'simplex'
            # what elements make up the mesh
            # may add mixed and non-conforming later

        Nef (int)
            # number of facets per element

        Nev (int)
            # number of vertices per element

        dim (int)
            # spatial dimension

        K (int)
            # number of elements in mesh

        Nv (int)
            # number of vertices in mesh (shared ones counted once)

        Nf (int)
            # number of facets in mesh (shared ones counted once)

        v (real, Nv x dim)
            # vertex physical coordinates

        VtoE (int, K x Nev)
            # vertex indices belonging to element
            # counter-clockwise in 2D (should be from mesh generator)

        facets (dictionary)
            # each key is a facet id, i.e. a tuple of vertices making up a facet (ordered low to high index)
            # returns list containing the two elements/BCs sharing that facet
                (order does not matter since numerical fluxes are symmetric)
                [(element_1, local_facet_id_1) (element_2, local_facet_id_2)]
            # if one side is a BC then will not be assigned until set_bc is called for that facet id
            # before simulation, should check that *something* is on either side of each facet (either an element or BC)
            # even if BC is only required on inflow side, that is dictated by numerical flux, not the user
            # first step of residual evaluation is to go through the facet dictionary and compute numerical fluxes,
              which can be stored in another dictionary for the solver and then accessed when solving
              for local residual on each element through dictionary lookup

        bc_table (dictionary)
        # each key is a string denoting a particular boundary condition
        # (Simulation object assigns BC data to given names from Problem class)
        # returns list containing three elements:
            bc_type (string, 'riemann' or 'periodic')
            facet_ids (list of tuples)
                if bc_type == 'riemann', this is just a list of facet ids
                if bc_type == 'periodic', this is a list of tuples (pairs) of facet_ids that match up
            bc_id (128-bit integer)
                automatically generated id for this bc, replaces id of "neighbouring element"

        xbar (real, K x dim)
            # element centroid coordinates

        J (real, K x dim x dim)
            # element Jacobian

        detJ (real, K)
            # element Jacobian determinant

        s (real, K x dim)
            # element translation such that
            # x = J x_hat + s
            # this is what the origin of the reference element gets mapped to
            # in 1D it is equal to xbar
            # for 2D triangles it is the midpoint of facet 1 (i.e. between vertices 2 and 3)

        n (real, K x Nef x dim)
            # element facet normal vectors

    # Methods

        plot_mesh

        compute_centroids

        compute_mapping

        compute_facets

        set_bc

    """

    def __init__(self, dim, v, VtoE, type='simplex'):
        self.dim = dim
        self.v = v
        self.VtoE = VtoE
        self.type = type

        self.Nv = v.shape[0]
        self.K = VtoE.shape[0]

        if type=='simplex':
            self.Nef = self.dim+1
            self.Nev = self.dim+1

        self.xbar = None
        self.J = None
        self.detJ = None
        self.s = None
        self.bc_table = dict()

        self.compute_centroids()
        self.compute_mapping()
        self.compute_facets()

    def compute_centroids(self):
        raise NotImplementedError

    def compute_mapping(self):
        raise NotImplementedError

    def compute_facets(self):
        raise NotImplementedError

    def set_bc(self,bc_name, facet_ids, bc_type):
        raise NotImplementedError

    def plot_mesh(self, figtitle):
        raise NotImplementedError
