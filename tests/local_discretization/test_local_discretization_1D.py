from local_discretization.DGLegendreSimplex import DGLegendreSimplex


def test_local_discretization_1d():
    disc = DGLegendreSimplex(3,1, 'LGL', 'pointwise', 6, 1)
    disc.print_operators()


test_local_discretization_1d()
