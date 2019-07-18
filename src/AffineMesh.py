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

        bc_table (dictionary)
            # each key is a string denoting a particular boundary condition
            # (Simulation object assigns BC data to given names from Problem class)
            # returns list of integers denoting nodes on that boundary

        VtoE (int, K x Nev)
            # vertex indices belonging to element
            # counter-clockwise in 2D (should be from mesh generator)

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

        plotMesh
        computeCentroids

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

        self.computeCentroids()
        self.computeMapping()

    def computeCentroids(self):
        raise NotImplementedError

    def computeMapping(self):
        raise NotImplementedError

    def computeAdjacency(self):
        raise NotImplementedError

    def plotMesh(self, figtitle):
        raise NotImplementedError
