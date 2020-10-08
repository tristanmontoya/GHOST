# GHOST - Spatial Discretization

from Operator import DenseLinearOperator, DiagonalOperator, Identity
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import quadpy as qp


class SpatialDiscretization:
    
    def __init__(self, mesh, f, f_star, element_to_discretization, p,
                 xi_omega, xi_gamma):
        
        # mesh
        self.mesh = mesh
        
        # spatial dimension
        self.d = mesh.d  
        
        # volume flux function
        self.f = f 
        
        # numerical flux function
        self.f_star = f_star 
        
        # map from element index to discretization type index
        self.element_to_discretization = element_to_discretization  
        
        # polynomial degree
        self.p = p
        
        # number of discretization types
        self.Nd = len(self.p) 
        
        # dimension of polynomial space (assume total-degree for now)
        self.Np = [special.comb(self.p[i] + self.d, self.d, 
                                exact=True) for i in range(0,self.Nd)]
        # flux nodes
        self.xi_omega = xi_omega
        self.N_omega = [xi_omega[i].shape[0] for i in range(0,self.Nd)]
        
        # facet nodes
        self.xi_gamma = xi_gamma
        self.Nf = [len(xi_gamma[i]) for i in range(0,self.Nd)]
        self.N_gamma = [[xi_gamma[i][gamma].shape[0] for gamma in range(0,self.Nf[i])] for i in range(0,self.Nd)]
        

    def plot(self):
        
        if self.d == 1:
            
            x_L = np.amin(self.mesh.v[:,0])
            x_R = np.amax(self.mesh.v[:,0])
            L = x_R - x_L
            meshplt = plt.figure()
            ax = plt.axes()
            plt.xlim([x_L - 0.1 * L, x_R + 0.1 * L])
            plt.ylim([-0.1 * L, 0.1 * L])
            ax.get_xaxis().set_visible(False)  
            ax.get_yaxis().set_visible(False)  
            ax.set_aspect('equal')
            plt.axis('off')
        
            color = iter(plt.cm.rainbow(np.linspace(0, 1, self.mesh.K)))
            
            # loop through all elemeents
            for k in range(0, self.mesh.K):
                
                # plot flux nodes
                x_omega = self.mesh.X[k](self.xi_omega[self.element_to_discretization[k]])
                ax.plot(x_omega, np.zeros(self.N_omega[self.element_to_discretization[k]]), "o", color = next(color))
                
                # plot vertices
                for gamma in range(0, self.mesh.Nv_local[k]):
                    ax.plot(self.mesh.v[self.mesh.local_to_vertex[k][gamma][0]],0.0,
                            's', color="black")
    
            plt.show()
            meshplt.savefig("../plots/" + self.mesh.name + "_nodes.pdf", bbox_inches=0, pad_inches=0)
                
        else:
            raise NotImplementedError
                

def make_interpolation_nodes(elem_type, p):
    if elem_type == 'triangle2d':
        return qp.triangle.WitherdenVincent(p)
    else:
        raise NotImplementedError


def project_to_solution(disc, mesh, u_v):
    return [disc.P_v(u_v[k]) for k in range(0, mesh.K)]


def evaluate_at_volume_nodes(disc, mesh, u_s):
    return [disc.R_v(u_s[k]) for k in range(0, mesh.K)]


def evaluate_at_facet_nodes(disc, mesh, u_s):
    return [[disc.R[gamma](u_s[k]) for
             gamma in range(0, mesh.Nf[k])] for k in range(0, mesh.K)]


def cardinality(d: int, p: int):
    return special.comb(p + d, d, exact=True)


def vandermonde(d: int, p: int, basis: str, xv: np.ndarray) -> DenseLinearOperator:

    phi_leg = legendre_basis(d,p)
    V_leg = np.empty([xv.shape[0], p + 1])

    for i in range(0, xv.shape[0]):
        for j in range(0, p + 1):
            V_leg[i, j] = phi_leg[j](xv[i, 0])

    T = change_polynomial_basis(d, p, 'legendre', basis)
    return DenseLinearOperator(V_leg) * T


def legendre_basis(d: int, p:int) -> list:
    if d == 1:
        return list(map(lambda k: special.legendre(k), range(0, p+1)))
    else:
        raise NotImplementedError


def change_polynomial_basis(d: int, p: int, basis1: str,
                            basis2: str) -> DenseLinearOperator:
    Np = cardinality(d,p)

    if basis1 == basis2:
        return Identity(Np)
    if d == 1:

        # Nodal
        if basis1 == 'lagrange-lg':
            xs = qp.line_segment.gauss_legendre(Np).points.reshape([Np,1])
            return vandermonde(1, p, basis2, xs)
        if basis1 == 'lagrange-lgl':
            xs = qp.line_segment.gauss_lobatto(Np).points.reshape([Np,1])
            return vandermonde(1, p, basis2, xs)
        if basis1 == 'lagrange-uniform':
            xs = np.linspace(-1,1,Np).reshape([Np,1])
            return vandermonde(1, p, basis2, xs)
        if basis2 == 'lagrange-lg' or basis2 == 'lagrange-lgl' or basis2 == 'lagrange_uniform':
            return change_polynomial_basis(d,p,basis2, basis1).inv

        # Modal
        if basis1 == 'orthonormal':
            if basis2 == 'legendre':
                return DenseLinearOperator(np.diag(np.sqrt(2.0/(2.0*np.arange(Np) + 1))))
        if basis1 == 'legendre':
            if basis2 == 'orthonormal':
                return change_polynomial_basis(d, p, 'orthonormal', 'legendre').inv
    else:
        raise NotImplementedError


def reference_mass_matrix_exact(d: int, p: int, basis: str) -> DenseLinearOperator:
    Np = cardinality(d,p)

    if d == 1:
        M_orth = Identity(Np)
        T = change_polynomial_basis(d, p, 'orthonormal', basis)
        return T.T * M_orth * T
    else:
        raise NotImplementedError


def poly_deriv(d: int, p: int, der, basis: str) -> DenseLinearOperator:
    Np = cardinality(d,p)

    if d == 1:
        D_leg = DenseLinearOperator(
            np.triu(np.diag(2*np.arange(Np) + 1)
                    @ np.array([[(i + j) % 2 for i in range(Np)]
                                for j in range(Np)])))
        T = change_polynomial_basis(d, p, 'legendre', basis)
        return T.inv * D_leg**der * T
    else:  # if d > 1, der should be array with how many times to diff. each variable
        raise NotImplementedError


def fr_filter(d: int, p: int, scheme, basis: str,
              mass_matrix=False) -> DenseLinearOperator:

    Np = cardinality(d,p)

    if d == 1:
        M = reference_mass_matrix_exact(d, p, 'legendre')
        Dp = poly_deriv(d, p, p, 'legendre')
        a_p = special.legendre(p)[p]
        if scheme == 'huynh':
            c = 2.0 * (p + 1.0) / ((2.0 * p + 1.0) * p *(np.math.factorial(p) * a_p) ** 2.0)
        elif scheme == 'dg':
            c = 0.0
        elif scheme == 'sd':
            c = 2.0 * p / ((2.0 * p + 1.0) * (p+1.0) * (np.math.factorial(p) * a_p) ** 2.0)
        else:
            c = scheme
        Finv = (Identity(Np) + c*M.inv*Dp.T*Dp)
        T = change_polynomial_basis(d, p, 'legendre', basis)
        if mass_matrix:
            return T.T * M * Finv * T
        return T.inv * Finv.inv * T
    else:
        raise NotImplementedError


def fr_c(d: int, p: int, scheme, basis: str):

    if d == 1:
        a_p = special.legendre(p)[p]
        if scheme == 'huynh':
            return 2.0 * (p + 1.0) / ((2.0 * p + 1.0) * p * (np.math.factorial(p) * a_p) ** 2.0)
        elif scheme == 'dg':
            return 0.0
        elif scheme == 'sd':
            return 2.0 * p / ((2.0 * p + 1.0) * (p + 1.0) * (np.math.factorial(p) * a_p) ** 2.0)
        else:
            return scheme


def fr_K(d: int, p: int, scheme, basis: str, use_metric=False, metric_data=None, k=0) -> DenseLinearOperator:

    Np = cardinality(d,p)

    if d == 1:
        M = reference_mass_matrix_exact(d, p, 'legendre')
        Dp = poly_deriv(d, p, p, 'legendre')
        a_p = special.legendre(p)[p]
        if scheme == 'huynh':
            c = 2.0 * (p + 1.0) / ((2.0 * p + 1.0) * p *(np.math.factorial(p) * a_p) ** 2.0)
        elif scheme == 'dg':
            c = 0.0
        elif scheme == 'sd':
            c = 2.0 * p / ((2.0 * p + 1.0) * (p+1.0) * (np.math.factorial(p) * a_p) ** 2.0)
        else:
            c = scheme

        if use_metric:
            Jinv = metric_data.inv_proj_jac[k]
        else:
            Jinv = Identity(Np)

        T = change_polynomial_basis(d, p, 'legendre', basis)
        return T.T * (c*(Jinv*Dp).T*M*(Jinv*Dp)) * T

    else:
        raise NotImplementedError


def volume_project(d: int, p: int, Nv: int, basis: str,
                        quadrature='lg') -> DenseLinearOperator:
    if d == 1:
        if quadrature == 'lg':
            xv = qp.line_segment.gauss_legendre(Nv).points.reshape([Nv, 1])
            W = DiagonalOperator(qp.line_segment.GaussLegendre(Nv).weights)
        elif quadrature == 'lgl':
            xv = qp.line_segment.gauss_lobatto(Nv).points.reshape([Nv, 1])
            W = DiagonalOperator(qp.line_segment.GaussLobatto(Nv).weights)
        else:
            raise NotImplementedError
        V = vandermonde(d, p, basis, xv)

    else:
        raise NotImplementedError

    return (V.T * W * V).inv * V.T * W


def lift(d: int, p: int, basis: str, elem_type='simplex',
         cubature='endpoints', N_gamma=1.0, scheme=0.0):

    if elem_type == 'simplex':
        Nf = d + 1
    else:
        raise NotImplementedError

    if d == 1:
        Vf = [vandermonde(d,p,basis,np.array([[-1.0]])), vandermonde(d,p,basis,np.array([[1.0]]))]
        M = fr_filter(d,p,scheme,basis,mass_matrix=True)
        Wf = DenseLinearOperator(np.array([[1.0]]))
    else:
        raise NotImplementedError

    return [M.inv * Vf[i].T * Wf for i in range(0, Nf)]
