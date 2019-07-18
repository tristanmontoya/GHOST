from Simplex1D import Simplex1D

def test_local_discretization_1D():

    element = Simplex1D(p=2, Nq=3, basis='orthonormal', quadrature_type='LGL')
    print('xq:\n', element.xq)
    print('xqfe:\n', element.xqfe)
    print('M:\n', element.M)
    print('W:\n', element.W)
    print('V:\n', element.V)
    print('Vf:\n', element.Vf)
    print('Vx:\n', element.Vx)

test_local_discretization_1D()