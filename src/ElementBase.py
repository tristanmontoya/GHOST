#GHOST - Element Base Class

import numpy as np
from scipy import special

class ElementBase:
    """
    # Properties (all derived classes inherit these)

        Basis functions are normalized Legendre polynomials
        Coordinates refer to reference element
        Some things are hard-coded in 1D for now

        Note: facets will be each treated separately, like Chen/Shu, not Chan
        Each facet is denoted gamma (lowercase), their union is Gamma (partial omega)
        All facets assumed to have the same number of quadrature points

        dim     # int, dimension (1, 2, 3) - currently only 1D
        p       # int, degree of accuracy for which discrete residual is zero
        Np      # int, cardinality of polynomial space (p+dim choose dim)
        Nf      # int, number of facets (currently dim + 1 for simplex)

        Nq      # int, number of volume quadrature points
        Nqf     # int, number of facet quadrature points per facet (1 in 1D)

        xq      # volume quadrature points (length Nq in 1D)
        wq      # volume quadrature weights (length Nq)

        xqf     # facet quadrature points (size Nf x Nqf, [[-1,0],[1.0]] in 1D)
                # will generally be Nf x Nqf x dim
        wqf     # facet quadrature weights (size Nf x Nqf, [[1.0],[1.0]] in 1D)

        n_hat   # facet outward normals (length Nf, [-1.0, 1.0] in 1D)
                # will generally be Nf x dim, each containing x, y, z component of normal to that facet

        W       # Diagonal quadrature matrix containing weights wq (size Nq x Nq)
                # nodal values of functions v and u are v_n and u_n
                # v_n^T W u_n = (v,w)_omega,W
                # "M^kappa" in Chen/Shu review

        Wf      # Diagonal facet quadrature matrix containing weights wqf (size Nf x Nqf x Nqf)
                # v_n^T Wf(gamma,:,:) u_n = (v,w)_gamma,Wf
                # In 1D this is just [[[1]],[[1]]]
                # "Wf_{gamma,:,:} = B^gamma" in Chen/Shu review

        V       # Volume Vandermonde interpolation matrix for basis (size Nq x Np)
                # V_{ij} = phi_j(xq_i)
                # Maps basis expansion coefficients to nodal values at quadrature points

        Vx      # Volume Vandermonde derivative matrix for basis (size Nq x Np)
                # V_{ij} = d/dx(phi_j)(xq_i)
                # Equal to V Dp, where Dp is the modal differentiation matrix which maps polynomial expansion coeffs
                # to the expansion coeffs of that polynomial's derivative

        Vf      # Facet Vandermonde interpolation matrix for basis (size Nf x Nqf x Np)
                # Vf_{gamma,ij} = phi_j(xqf_{gamma,i})
                # Vf_{gamma,:,:} Maps basis expansion coefficients to quadrature points on facet gamma

        M       # Modal volume mass matrix (size Np x Np)
                # M_{ij} = (phi_i, phi_j)_omega,W
                # M = V^T W V

        Mf      # Modal facet mass matrix for facet gamma (size Nf x Np x Np)
                # Mf_{gamma,ij} = (phi_i, phi_j)_gamma,Wf

        P       # Quadrature-based projection operator (size Np x Nq)
                P = M^{-1} V^T W
                # Maps nodal values at quadrature points to polynomial expansion coefficients

        Pf      # Quadrature-based lifting operator

        D


    # Methods (defined here)
        __init__
        setVolumeVandermonde
        setFacetVandermonde
        setVolumeMassMatrix
        setFacetMassMatrix

    """

    def __init__(self, dim, p, Nf, n_hat, wq, xq, wqf, xqf):

        # Total degree p approximation space
        self.dim = dim
        self.Nf = Nf
        self.p = p
        self.Np = special.comb(p + dim, dim, exact=True)

        # Reference element normals
        self.n_hat = n_hat

        # Volume quadrature
        self.wq = wq
        self.xq = xq
        self.Nq = len(self.wq)
        self.W = np.diag(self.wq)

        # Facet quadrature
        self.wqf = wqf
        self.xqf = xqf
        self.Nqf = self.wqf.shape[1]
        self.Wf = np.zeros([self.Nf, self.Nqf, self.Nqf])
        for gamma in range(0,self.Nf):
            self.Wf[gamma,:,:] = np.diag(self.wqf[gamma, :])

        self.setVolumeVandermonde()
        self.setFacetVandermonde()
        self.setVolumeMassMatrix()

    def setVolumeVandermonde(self):
        self.V = np.polynomial.legendre.legvander(self.xq, self.p)
        self.Vx = np.zeros([self.Nq, self.Np])

        for j in range(0, self.Np):
            normalizationFactor = np.sqrt(2. / (2 * j + 1))
            self.V[:, j] /= normalizationFactor  # normalization factor
            dPdxi = np.polyder(special.legendre(j))
            self.Vx[:, j] = dPdxi(self.xq) / normalizationFactor  # derivatives of normalized legendre basis

    def setFacetVandermonde(self):
        self.Vf = np.zeros(self.Nf, self.Nqf, self.Np)

        for gamma in range(0, self.Nf):
            self.Vf[gamma, :, :] = np.polynomial.legendre.legvander(self.xq, self.p)
            for j in range(0, self.Np):
                self.Vf[gamma, :, j] /= np.sqrt(2. / (2 * j + 1))

    def setVolumeMassMatrix(self):
        self.M = self.V.T @ self.W @ self.V

    def setFacetMassMatrix(self):
        self.Mf = np.zeros(self.Nf, self.Np, self.Np)
        for gamma in range(0, self.Nf):
            self.Mf = self.Vf[gamma, :, :].T @ self.Wf[gamma, :, :] @ self.Vf[gamma, :, :]