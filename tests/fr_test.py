import quadpy as qp
import numpy as np


def huynh_modal_filter(p,c):
    # one-parameter family
    F = np.eye(p+1)
    F[p, p] = p/(2.0*p+1.0)
    return F


def c(p):
    # legendre coefficient
    return 1.0/(2.0**p)*1.0*np.math.factorial(2*p)/(np.math.factorial(p)**2)


def c_hu(p):
    # Huynh's G2 scheme

    return 2.0*(p+1.0)/((2.0*p + 1.0)*(np.math.factorial(p)*c(p))**2.0)


p = 4
F = huynh_modal_filter(p,c)
x = qp.line_segment.GaussLegendre(p+1).points
w = qp.line_segment.GaussLegendre(p+1).weights
print("x: ", x)
V = np.polynomial.legendre.legvander(x,p)
V_normalized = np.polynomial.legendre.legvander(x,p)
for j in range(0,p+1):
    V_normalized[:,j] = V_normalized[:,j]/np.sqrt(2. / (2 * j + 1))
M_dg = np.diag(w)
M_modal = V.T @ M_dg @ V
M_fr = M_dg @ np.linalg.inv(V @ F @ np.linalg.inv(V))
w, v = np.linalg.eigh(M_fr)
W_fr = np.diag(w)

print("DG Mass Matrix on LG Nodal Basis: \n", M_dg)
print("Modal DG Mass Matrix: \n", M_modal)
print("Huynh FR Mass Matrix on LG Nodal Basis: \n", M_fr)
print("Diagonalized Huynh FR Mass Matrix: \n:", W_fr)
print("LGL Quadrature Weights :\n", qp.line_segment.GaussLobatto(p+1).weights)
