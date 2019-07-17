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

        VtoE (int, K x dim)
            # vertex indices belonging to element
            # counter-clockwise in 2D (should be from mesh generator)

        localVtoF (int, K x ref.Nf x dim)
            # vertex indices belonging to element facet
            localVtoF[K,i,:] are the vertex indices (ordered low to high) for Facet i of element K

        J (real, K x dim x dim)
            # element Jacobian

        detJ (real, K)
            # element Jacobian determinant

        s (real, K x dim)
            # element translation such that
            # x = J x_hat + s

        n (real, K x Nef x dim)
            # element facet normal vectors

    # Methods

        plotMesh

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

    def plotMesh(self):
        raise NotImplementedError

