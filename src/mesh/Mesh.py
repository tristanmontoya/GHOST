# GHOST - Mesh Base Class


class Mesh:
    """
    The geometric aspects of the mesh used for the spatial discretizationf

    Basic Mesh Properties
    ---------------------

    type : string
        what elements make up the mesh, currently must be 'simplex' but
        may add mixed, curvilinear, and non-conforming

    Nef : int
        number of facets per element

    Nev : int
        number of vertices per element

    d : int
        spatial dimension

    K : int
        number of elements in mesh

    Nv : int
        total number of vertices in mesh (shared ones counted once)

    Nf : int
        total number of facets in mesh (shared ones counted once)

    Mesh Data and Connectivity
    --------------------------

    v : ndarray
        dtype=float, shape =(Nv, d)

        vertex physical coordinates, e.g. in 2D
        [[x_1, y_1], [x_2, y_2]]

    VtoE : ndarray
        dtype=int, shape=(K, Nev)

        VtoE[i,j] is the index in v of the jth vertex belonging to element
        i, numbered counter-clockwise in 2D (should be from mesh generator)

    facets : dict
        each key is a facet id, i.e. a tuple of vertices making up a facet
        (ordered low to high index)

        returns list containing the two elements/BCs sharing that facet
        (order does not matter since numerical fluxes are symmetric)
        [(element_1, local_facet_id_1) (element_2, local_facet_id_2)]

        if one side is a BC then will not be assigned until set_bc is
        called for that facet id

        before simulation, should check that *something* is on either side
        of each facet (either an element or BC) -- even if BC is only
        required on inflow side, that is dictated by numerical flux, not
        the user.

        First step of residual evaluation is to go through the facet
        dictionary and compute numerical fluxes, which can be stored in
        another dictionary for the solver and then accessed when solving
        for local residual on each element through dictionary lookup

    bc_table : dict
        each key is a string denoting a particular boundary condition
        (Simulation object assigns BC data to given names from
        Problem class)

        returns list containing three elements:
            bc_type (string, 'riemann' or 'periodic')
                will add solid wall BCs later
            facet_ids (list of tuples)
                if bc_type == 'riemann', this is just a list of facet ids
                if bc_type == 'periodic', this is a list of tuples (pairs)
                of facet_ids that match up
            bc_id (128-bit integer)
                automatically generated id for this bc, replaces id of "neighbouring element"

    xbar : ndarray
        dtype=float, shape=(K,d)

        element centroid coordinates

    Methods
    -------

        plot_mesh

        compute_centroids

        compute_mapping

        compute_facets

        map

        jacobian

        normal

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

    def map(self, k, xhat):
        """Mapping from reference to physical coordinates

        x = X(k,x)

        Parameters
        ----------
        k : int
            What element to map to

        xhat : ndarray
            dtype=float, shape=(n, self.d)

            points to map from reference coordinates

            n is the number of points (this should be vectorized)

        Returns
        -------
        x : ndarray
            dtype=float, shape=(n, self.d)

            points in physical coordinates
        """

        raise NotImplementedError

    def jacobian(self, k, xhat):
        """Jacobian of mapping from reference to physical coordinates

        Parameters
        ----------
        k : int
            What element to map to

        xhat : ndarray
            dtype=float, shape=(n, self.d)

            points in reference coordinates to compute Jacobian at

            n is the number of points (this should be vectorized)

        Returns
        -------
        J : ndarray
            dtype=float, shape=(n, d, d)

            Jacobian matrix at each point

        detJ
            dtype-float, shape=(d)

            Jacobian determinant at each point

        """

        raise NotImplementedError

    def normal(self,xhat):
        raise NotImplementedError

    def plot_mesh(self, figtitle):
        raise NotImplementedError
