import Discretization
import Operator
import Mesh
import numpy as np
import quadpy as qp

d = 1
p = 3
q = 3
N_Omega =q+1
x = qp.line_segment.gauss_lobatto(N_Omega).points
V = Discretization.vandermonde(d,p, "orthonormal", x.reshape([N_Omega, 1]))
M = Discretization.reference_mass_matrix_exact(d, q, "lagrange-lgl")
W = Operator.DiagonalOperator(qp.line_segment.gauss_lobatto(N_Omega).weights)
J = Operator.DiagonalOperator(Mesh.nonsingular_map_1d(3*q, x, 1)+1.0)

P_M = (V.T * M * V).inv * V.T * M

P_W = (V.T * W * V).inv * V.T * W

PJV_M = P_M * J * V
PJV_W = P_M * J * V
