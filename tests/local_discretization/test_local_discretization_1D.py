from local_discretization.DGQuadratureSimplex import DGQuadratureSimplex


def test_local_discretization_1d():
    disc = DGQuadratureSimplex(1, 'legendre-normalized', 3, 'LGL', 'pointwise', 6, 1)
    disc.print_operators()


test_local_discretization_1d()
