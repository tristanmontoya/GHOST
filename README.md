# Generalized High-Order Solver Toolbox (GHOST)
GHOST is a Python implementation of discontinuous Galerkin and flux reconstruction methods for first-order systems of conservation laws within the framework described in:

Tristan Montoya and David W. Zingg, "A unifying algebraic framework for discontinuous Galerkin and flux reconstruction methods based on the summation-by-parts property." Preprint, [arXiv:2101.10478v1](https://arxiv.org/abs/2101.10478) (2021).

As GHOST was developed for use as a test bed for numerical schemes and to run numerical experiments for simple model problems in support of the theoretical studies in the above manuscript, competitive performance for practical problems is not expected.

## Features

### Supported element types

- Line segments in 1D
- Triangles in 2D

### Supported PDEs

- Linear advection equation (constant advection coefficient)
- Euler equations (compressible flow, ideal gas with constant specific heat)

### Supported discretization options

- Discontinuous Galerkin methods and energy-stable flux reconstruction schemes using Vincent-Castonguay-Jameson-Huynh (VCJH) correction fields, implemented in strong and weak form
- Nodal (Lagrange) or modal (orthonormal) bases
- Quadrature-based or collocation-based treatment of nonlinear fluxes
- Fourth-order Runge-Kutta time marching

### Supported numerical flux functions

- Standard upwind/central/blended numerical flux for linear advection
- Roe's approximate Riemann solver for Euler equations 

## Usage

The basic usage of GHOST is demonstrated in the included Jupyter notebook `notebooks/example_usage.ipynb`, in which the linear advection equation is solved on a periodic square domain with a sinusoidal initial condition. The Jupyter notebooks used for the computations in the referenced manuscript are also provided (see `notebooks/advection_driver.ipynb` and `notebooks/euler_driver.ipynb`).

## Dependencies

[NumPy](https://numpy.org/), [scipy](https://scipy.org/), [matplotlib](https://matplotlib.org/), [quadpy](https://github.com/nschloe/quadpy)
[modepy](https://github.com/inducer/modepy), [meshio](https://github.com/nschloe/meshio),
[meshzoo](https://github.com/nschloe/meshzoo)

## License

This software is released under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).
