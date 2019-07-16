# GHOST - Affine Mesh Base Class

class Mesh:
    """
    # Notes

        For now, assume all elements based on the same reference element
        See Andreas Kloeckner's HEDGE solver for inspiration maybe
        See how Hesthaven and Warburton treat facet indexing
        *When computing fluxes for an element, we can find what facets those are
         but how do we know what other elements share that facet (search a priori)


    # Properties

        ref     # reference element, type ElementBase

        K       # int, number of elements in mesh

        Nv      # int, number of vertices in mesh (shared ones counted once)

        Nf      # int, number of facets in mesh (shared ones counted once)

        dim     # int, spatial dimension = ref.dim

        v       # vertex physical coordinates (size Nv x dim)

        VtoE    # vertex indices belonging to element (int, size K x dim)

        J       # element Jacobian (size K x dim x dim)

        detJ    # element Jacobian determinant (size K)

        s       # element translation such that
                # x = J x_hat + s

        n       # element facet normal vectors (size K x ref.Nf x ref.dim)


    # Methods

    """

    def __init__(self, referenceElement):
        self.referenceElement = referenceElement

