# Generalized High-Order Solver Toolbox (GHOST)
GHOST is a simple yet flexible Python implementation of discontinuous Galerkin and flux reconstruction methods for first-order systems of conservation laws within the framework described in:

Tristan Montoya and David W. Zingg, "A unifying algebraic framework for discontinuous Galerkin and flux reconstruction methods based on the summation-by-parts property." Preprint, [arXiv:2101.10478v1](https://arxiv.org/abs/2101.10478) (2021).

As GHOST was developed for the prototyping and demonstration of numerical schemes and to run numerical experiments for simple model problems in support of the theoretical studies in the above manuscript, competitive performance for practical problems is not expected.

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
- Fourth-order Runge-Kutta or explicit Euler time marching

### Supported numerical flux functions

- Standard upwind/central/blended numerical flux for linear advection
- Roe's approximate Riemann solver for Euler equations 

## Usage

The basic usage of GHOST is demonstrated in the Jupyter notebooks provided in the `examples` directory, in which the periodic linear advection equation is solved in one and two dimensions. The Jupyter notebooks used for the computations in the referenced manuscript are also provided (see `notebooks/advection_driver.ipynb` and `notebooks/euler_driver.ipynb`).

## Data availability and naming conventions

The tables in the manuscript (consistent with the version provided in the  `manuscript` directory of this repository) were generated using the Jupyter notebooks `notebooks/make_tables_advection.ipynb` and `notebooks/make_tables_euler.ipynb`, which retrieve data from the `results` directory for each set of discretization parameters. The subdirectory names are formatted as `advection_pAbBcCtD_E` or `euler_m04pAcCtD_E`,  where `A` is the polynomial degree of the discretization, `B` is the upwinding parameter for the numerical flux (i.e. `0` for an central flux and `1` for an upwind flux), `C` determines the VCJH parameter (i.e. `0` for $c=c_{\mathrm{DG}}$ and `p` for $c=c_+$), `D` determines the discretization type (i.e. `1` for Quadrature I, `2` for Collocation, and `3` for Quadrature II), and `E` determines the form of the semi-discrete residual (i.e. `strong` or `weak`).
## Dependencies

[NumPy](https://numpy.org/), [scipy](https://scipy.org/), [matplotlib](https://matplotlib.org/), [quadpy](https://github.com/nschloe/quadpy)
[modepy](https://github.com/inducer/modepy), [meshio](https://github.com/nschloe/meshio),
[meshzoo](https://github.com/nschloe/meshzoo)

## License

This software is released under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).
