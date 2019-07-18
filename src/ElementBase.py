#GHOST - Element Base Class


import numpy as np
from scipy import special

class ElementBase:
    """
    # Notes

        This class defines the numerics of the problem on the reference element, i.e. it defines a type of element,
        rather than a particular geometric element in the mesh. Geometric information is included in the
        AffineMesh class

        Basis functions are normalized Legendre polynomials for now
        Coordinates refer to reference element (Omega_hat) and reference Facet (Gamma_Hat)

    # Properties

        dim (int)
            # dimension in space (1, 2, 3)

        basis (string)
            # options: 'orthonormal'
            # TODO 'lagrange' (with specified nodal distribution)

        elementShape (string)
            # options: 'simplex'
            # in 1D this means reference line segment given by:
                Omega_hat = (-1,1)
                Facets Gamma_hat_1 = v_hat_1 = {-1}, Gamma_hat_2 = v_hat_1 = {1}
            # in 2D this means a right triangle with reference vertices:
                v_hat_1 = [-1,-1]^T, v_hat_2 =[1,-1]^T, v_hat_3 = [-1,1]^T
                Facets Gamma_hat_gamma connect corresponding opposite vertices

        p (int)
            # degree of polynomial space

        Np (int)
            # dimension of polynomial space (p+dim choose dim unless tensor product space)

        Nf (int)
            # number of facets (dim + 1 for simplex)

        Nq (int)
            # number of volume quadrature points
        Nqf (int)
            # number of facet quadrature points per facet (1 in 1D for endpoints)

        xq (real, Nq x d)
            # volume quadrature points
        wq
            # volume quadrature weights (size Nq x 1)

        xqf (real, Nqf x (d-1))
            # facet quadrature points on reference facet
            # In 1D this is of size 1 x 0, not assigned

        xqfe (real, Nf x Nqf x d)
            # facet quadrature points on reference element
            # In 1D this is [ [[-1]], [[1]]]
            # In 2D we must apply a transformation to xqf to map it to the faces

        wqf (real, Nqf x 1)
            # facet quadrature weights
            # In 1D this is just the 1x1 matrix [[1]]

        W (real, Nq x Nq)
            # Diagonal quadrature matrix containing weights wq (
            # nodal values of functions v and u are v_n and u_n
            # v_n^T W u_n = (v,w)_omega,W

        Wf (real, Nqf x Nqf)
            # Diagonal facet quadrature matrix containing weights wqf on reference facet
            # v_n^T Wf u_n = (v,w)_gamma,Wf
            # In 1D this is just [[1]]

        V (real, Nq x Np)
            # Volume Vandermonde interpolation matrix for basis
            # V_{ij} = phi_j(xq_i)
            # Maps basis expansion coefficients to nodal values at quadrature points

        Vx (real, d x Nq x Np)
            # Volume Vandermonde derivative matrix for basis
            # V_{m,ij} = d/dx_m(phi_j)(xq_i)

        Vf (real, size Nf x Nqf x Np)
            # Facet Vandermonde interpolation/extrapolation matrix for basis
            # Vf_{gamma,ij} = phi_j(xf__{gamma,i})
            # Vf_{gamma,:,:} Maps basis expansion coefficients to quadrature points on facet Gamma_hat_gamma

        M (real, Np x Np)
            # Modal mass matrix
            # M_{ij} = (phi_i, phi_j)_omega,W
            # M = V^T W V

        P (real, Np x Nq)
            # Quadrature-based projection operator
            # P = M^{-1} V^T W
            # Maps nodal values at quadrature points to polynomial expansion coefficients


    # Methods

        __init__

        setVolumeVandermonde

        setFacetVandermonde

        setVolumeMassMatrix

    """

    def __init__(self, dim, basis, elementShape, p, wq, xq, wqf, xqf, xqfe):

        # Total degree p approximation space
        self.dim = dim
        self.basis = basis
        self.elementShape = elementShape
        if elementShape == 'simplex':
            self.Nf = dim + 1
        self.p = p
        self.Np = special.comb(p + dim, dim, exact=True)

        # Volume quadrature
        self.wq = wq
        self.xq = xq
        self.Nq = len(self.wq)
        self.W = np.diag(self.wq)

        # Facet quadrature
        self.wqf = wqf
        self.xqf = xqf
        self.xqfe = xqfe
        self.Nqf = len(self.wqf)
        self.Wf = np.diag(self.wqf)

        # Defined elsewhere
        self.V = None
        self.Vx = None
        self.Vf = None
        self.M = None

        self.setVolumeVandermonde()
        self.setFacetVandermonde()
        self.setMassMatrix()

    def setVolumeVandermonde(self):
        raise NotImplementedError

    def setFacetVandermonde(self):
        raise NotImplementedError

    def setMassMatrix(self):
        self.M = self.V.T @ self.W @ self.V
