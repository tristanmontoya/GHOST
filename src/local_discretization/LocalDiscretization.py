import numpy as np
from local_discretization.PolynomialSpace import PolynomialSpace


class LocalDiscretization:
    """
    Local spatial discretization on the reference element

    Basic Element Data
    ------------------

    space : PolynomialSpace
        The approximation space for the chosen method

    d : int
        Dimension in space (1, 2, 3)

    element_shape : str
        Geometry of reference element. May be 'simplex', 'quadrilateral,
        or 'hexahedron'

        In 1D, there is only 'simplex', which means reference line segment
        given by Omega_hat = (-1,1), and facets
        Gamma_hat_1 = v_hat_1 = {-1}, Gamma_hat_2 = v_hat_1 = {1}

        In 2D 'simplex' means a right triangle with reference vertices:
        v_hat_1 = [-1,-1]^T, v_hat_2 =[1,-1]^T, v_hat_3 = [-1,1]^T
        Facets Gamma_hat_gamma connect corresponding opposite vertices

        Quadrilateral to be implemented (e.g. with tensor-product elements)

    Np : int
        Dimension of polynomial space (p+dim choose dim for total degree)

    Ns : int
        Number of elemental degrees of freedom. May be Np for DG/FR, or
        Nq for collocated SBP, or neither for staggered SBP

    Nf : int
        Number of facets (dim + 1 for simplex, 2*dim for tensor product)

    Nq : int
        Number of volume quadrature/flux points

    Nqf : int
        Number of facet quadrature/flux points per facet (1 in 1D for endpoints)

    form_compatibility : str
        Is the discretization applied to the strong or weak form of the
        problem?

        'strong', 'weak', or 'both'

    Flux Evaluation
    ---------------

    isCollocated : bool
        Set to True to avoid multiplying by Psq = Pqs = I for collocated
        solution and quadrature nodes. This should be done for collocation
        based nodal DG/FR and DGSEM, as well as standard GSBP schemes,
        but not modal/overintegrated/staggered/decoupled schemes.

    volume_flux_evaluation : str
        How is the flux f(u) at the volume quadrature nodes obtained
        from the solution degrees of freedom?

        'evaluate-interpolate': if using a nodal solution representation,
        compute flux at each solution node and then interpolate that to
        quadrature nodes

        'interpolate-evaluate' (default): interpolate solution DOFs to
        volume quadrature nodes (e.g. using the continuous representation
        and simply evaluating) with Psq and then evaluate fluxes at volume
        quadrature nodes

    facet_flux_evaluation : str
        How is the flux f(u) at the facet quadrature nodes obtained
        from the solution degrees of freedom?

        'evaluate-extrapolate' (default): extrapolate flux to facet as
        f_f = R f( Psq u)

        'extrapolate-evaluate': extrapolate solution to facet and then
        evaluate f_f = f(R Psq u)


    Reference Node Positions
    ------------------------

    xp : ndarray
        dtype=float, shape=(N, d)

        Solution points. Not used for modal DG

    xq : ndarray
        dtype=float, shape=(Nq, d)

        Volume quadrature points, may or may not be equal to interpolation
        points in the case of a Lagrange basis with collocation projection.

        In the literature of staggered SBP operators, these are referred
        to as the flux points.

        The differencing operator is applied at the quadrature points and
        then projected or interpolated to the solution degrees of freedom

    wq : ndarray
        dtype=float, shape=(Nq, 1)

        Volume quadrature weights

    xqf : ndarray
        dtype=float, shape=(Nqf, d-1)

        Facet quadrature points on reference facet
        In 1D this is of shape 1 x 0, not assigned

    xqfe : ndarray
        dtype=float, shape=(Nf, Nqf, d)

        Facet quadrature points on reference element. In 1D this is
        [ [[-1]], [[1]] ]. In 2D we must apply a transformation to xqf to
        map it to the faces

    wqf : ndarray
        dtype=float, shape=(Nqf, 1)

        Facet quadrature weights. In 1D this is just the 1x1 matrix [[1]]

    Bilinear Operators
    ------------------

    W : ndarray
        dtype=float, shape=(Nq, Nq)

        Diagonal quadrature matrix containing weights wq. If nodal values
        of functions v and u are v_n and u_n, then

        v_n^T W u_n = (v,w)_omega,W

    Wf : ndarray
        dtype=float, shape=(Nqf, Nqf)

        Diagonal facet quadrature matrix containing weights wqf on
        reference facet. In 1D this is just [[1]].

        v_n^T Wf u_n = (v,w)_gamma,Wf

    M : ndarray
        dtype=float, shape=(Np, Np)

        Modal mass matrix, computed with quadrature as M = V^T W V,
        optionally could compute exactly. For the orthonormal basis with
        quadrature at least 2p, this is identity.

        M_{ij} = (phi_i, phi_j)_omega,W

    Linear Operators (Specific)
    ---------------------------

    Vq : ndarray
        dtype=float, shape=(Nq, Np)

        Volume Vandermonde interpolation matrix for basis. Maps basis
        expansion coefficients to nodal values at quadrature/flux points

        V_{ij} = phi_j(xq_i)

    Vqx : ndarray
        dtype=float, shape=(d, Nq, Np)

        Volume Vandermonde derivative matrix for basis. Maps polynomial
        expansion coefficients to their derivative values at
        quadrature/flux points

        Vx_{m,ij} = d/dx_m(phi_j)(xq_i)

    Vqf : ndarray
        dtype=float, shape=(Nf, Nqf, Np)

        Facet Vandermonde interpolation/extrapolation matrix for basis.
        Vf_{gamma,:,:} maps basis expansion coefficients to quadrature/flux
        points on facet Gamma_hat_gamma

        Vf_{gamma,ij} = phi_j(xqfe_{gamma,i})

    D : ndarray
        dtype=float, shape=(d, Nq, Nq)

        Nodal derivative operator at quadrature points. D_{m,:,:} maps
        nodal values of a function at volume quadrature/flux points to
        nodal values of its derivative. The "canonical" choice for DG/FR is
        D_m = V Dhat_m P = Vx P, and in that case if the basis is nodal
        with Np = Nq, then P = V = I, and thus Dhat_m = D_m = Vx_m.

    Dp : ndarray
        dtype=float, shape=(d, Np, Np)

        Modal derivative operator. Dhat_{m,:,:} maps polynomial expansion
        coefficients to the expansion coefficients of that polynomial's
        derivative in the m direction

        Dp_m = P Vx_m

    Pp : ndarray
        dtype=float, shape=(Np, Nq)

        Modal projection operator. Maps nodal values at volume
        quadrature/flux points to polynomial expansion coefficients.
        Identity for collocation projection on Lagrange nodes

        Pp = M^{-1} V^T W

    Lp : ndarray
        dtype=float, shape=(Nf, Np, Nqf)

        Modal lifting operator. Maps nodal values at facet quadrature/flux
        points (first index denotes which facet) to polynomial expansion
        coefficients.

    Linear Operators (Generalized)
    ------------------------------

    L : ndarray
        dtype=float, shape=(Nf, Ns, Nqf)

        Generalized lifting operator. L_{f,:,:} Maps nodal values at facet
        quadrature/flux points to solution DOF. Includes SBP extrapolation,
        DG lifting projection, or FR correction fields (divergence of
        correction functions, see Ranocha or Zwanenburg and compare to
        Chen/Shu or Chan)

    Pqs : ndarray
        dtype=float, shape=(Ns, Nq)

        Generalized projection/interpolation operator. Maps nodal values at
        quadrature/flux points to whatever the solution degrees of freedom
        are.

        Identity to solve directly for nodal values at quadrature points,
        Pp to solve for expansion coefficients, or an interpolation
        operator in the case of a staggered-grid method.

    Psq : ndarray
        dtype=float, shape=(Nq, Ns)

        Generalized projection/interpolation operator. Maps degrees of
        freedom for solution to nodal values at quadrature/flux points.
        Identity to solve directly for nodal values at quadrature points,
        Vq to solve for expansion coefficients, or an interpolation operator
        in the case of a staggered-grid method.

    Ddc : ndarray
        dtype=float, shape=(d, Nq + Nqf*Nf, Nq + Nqf*Nf)

        Decoupled SBP operator (Chan, 2018). Ddc_{m,:,:} maps nodal values
        of a function at volume and surface quadrature points to nodal
        values at volume and surface quadrature points. It is not actually
        a high-order approximation of the derivative. Must be combined with
        projection and lifting operators to map to expansion coefficients.

    Methods
    -------

    __init__

    create_operators

    print_operators

    set_xq_to_solution

    set_solution_to_xq

    set_derivative_operator

    set_lifting_operator

    Notes
    -----

    Not all of the above operators are defined for every type of element.
    This documentation serves as a "master list" of all the types of
    elemental operations that could be used.

    Generalized Nodal SBP can be applied directly at the quadrature points.
    FR, Modal DG, DGSEM, and Hesthaven/Warburton's nodal DG can be applied
    using "modal" operators for the orthonormal or Lagrange bases. For FR,
    the SpatialDiscretization object must substitute M^{-1} V_f Wf for the
    FR correction field (i.e. derivative of correction functions and the
    Lagrange basis should be used. Nodal DG from Hesthaven and Warburton
    must specify to compute the mass matrix exactly.

    """

    def __init__(self, space, element_shape, Ns, xq, xqf, xqfe):

        self.d = space.d
        self.space = space
        self.Ns = Ns

        self.element_shape = element_shape
        if self.element_shape == 'simplex':
            self.Nf = self.d + 1
        elif self.element_shape in ['quadrilateral', 'hexahedron']:
            self.Nf = 2*self.d

        # Quadrature/flux points

        self.xq = xq
        self.Nq = xq.shape[0]
        self.xqf = xqf
        self.xqfe = xqfe

        if self.d == 1:
            self.Nqf = 1
        else:
            self.Nqf = xqf.shape[0]

        # Generalized Operators

        self.Psq = np.zeros([self.Nq, self.Ns])  # set_solution_to_xq
        self.D = np.zeros([self.d, self.Nq, self.Nq])  # set_derivative_operator
        self.Pqs = np.zeros([self.Ns, self.Nq]) # set_xq_to_solution
        self.L = np.zeros([self.Nf, self.Ns, self.Nqf])  # set_lifting_operator

        # Vandermonde matrices associated with polynomial basis and quadrature/flux nodes

        self.Vq = self.space.vandermonde(self.xq)
        self.Vqx = self.space.grad_vandermonde(self.xq)
        self.Vqf = np.zeros([self.Nf, self.Nqf, self.space.Np])
        for gamma in range(0, self.Nf):
            self.Vqf[gamma, :, :] = space.vandermonde(self.xqfe[gamma, :, :])

    def create_operators(self):
        """Construct the local operators -- call within the child class"""

        self.set_solution_to_xq()
        self.set_xq_to_solution()
        self.set_derivative_operator()
        self.set_lifting_operator()

    def print_operators(self):
        print('xq:\n', self.xq)
        print('xqfe:\n', self.xqfe)
        print('Vq:\n', self.Vq)
        print('Vqf:\n', self.Vqf)
        print('Vqx:\n', self.Vqx)
        print('Psq:\n', self.Psq)
        print('Pqs:\n', self.Pqs)
        print('D:\n', self.D)
        print('L:\n', self.L)

    def set_solution_to_xq(self):
        raise NotImplementedError

    def set_xq_to_solution(self):
        raise NotImplementedError

    def set_derivative_operator(self):
        raise NotImplementedError

    def set_lifting_operator(self):
        raise NotImplementedError

