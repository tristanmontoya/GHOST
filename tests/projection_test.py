import Discretization
import Operator
import numpy as np
import quadpy as qp


# interpolation-based discretization
d = 1
p = 3
q = 4
basis = 'legendre'
Nv = q+1

M = Discretization.reference_mass_matrix_exact(d, p, basis)
V = Discretization.vandermonde(d,p, basis, qp.line_segment.newton_cotes_closed(q).points.reshape([Nv, 1]))
W = Operator.DiagonalOperator(qp.line_segment.newton_cotes_closed(q).weights)


V_omega = Discretization.vandermonde(d,q, 'orthonormal', qp.line_segment.newton_cotes_closed(q).points.reshape([Nv, 1]))

M_omega = (V_omega*V_omega.T).inv

M_lumped = V.T*W*V # degree q=5, but needs to be degree 2p = 6 to be exact
M_test = V.T*M_omega*V # should be diagonal

P_lumped = M_lumped.inv*V.T*W
P_interp = M_test.inv*V.T*M_omega