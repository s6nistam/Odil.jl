# Odil.jl

Odil.jl is a Julia implementation of the "Optimizing a Discrete Loss" (ODIL) framework described in the [original paper](https://link.springer.com/article/10.1140/epje/s10189-023-00313-7) by Petr Karnakov, Sergey Litvinov, and Petros Koumoutsakos. The original Python version is available at [cselab/odil](https://github.com/cselab/odil).

The package was developed with two main goals in mind:

1. To gain a better understanding of how ODIL works and how it is implemented.
2. To make ODIL compatible with [Trixi.jl](https://zenodo.org/records/21708122) and its DGSEM solvers as a discretization backend.

## What this package provides

Odil.jl currently includes:

- A Julia implementation of the ODIL optimization workflow.
- Support for two nonlinear solvers at this time: Gauss-Newton and L-BFGS.
- Time integration helpers such as explicit Euler and Carpenter-Kennedy methods.
- A callback system for logging, visualization, and custom extensions.
- Support for VTK/visualization workflows and plotting utilities.
- Example setups for both finite-difference discretizations and DGSEM-based problems.

## Software architecture

Odil.jl is organized into six main components:

- [src/io/](src/io/) for saving and loading ODIL states and results.
- [src/callbacks/](src/callbacks/) for callbacks that can monitor or visualize the optimization process.
- [src/solvers/](src/solvers/) for the main ODIL problem definition and the supported solvers.
- [src/sparsity/](src/sparsity/) for Jacobian sparsity handling used by Gauss-Newton.
- [src/time_integration/](src/time_integration/) for turning ODE right-hand sides into ODIL timestep functions.
- [src/visualization/](src/visualization/) for plotting and comparison of solutions.

## Loss formulation

In Odil.jl, the loss is built from three contributions:

- a reference-data term, which penalizes deviation from known values,
- an inner consistency term, which enforces that the time-stepping relation is satisfied,
- and an optional extra term, which can be used for regularization or additional constraints.

In compact form, one can think of the loss as

$$L(u, p) = L_{\mathrm{ref}}(u, p) + L_{\mathrm{inner}}(u, p) + L_{\mathrm{extra}}(u, p),$$

with the precise form depending on whether the solver uses a residual-based formulation (Gauss-Newton) or a gradient-based formulation (L-BFGS).

## Compatibility with [Trixi.jl](https://zenodo.org/records/21708122)

A central design goal of Odil.jl is compatibility with Trixi.jl. The package is intended to work naturally with semi-discretizations of PDEs, where the right-hand side $f(u)$ and the initial state $u_0$ are available. From these, a timestep function can be constructed and plugged into the ODIL workflow, which keeps the interface close to the workflow used in DGSEM-based PDE solvers.

## Repository layout

- [Examples/](Examples/) contains runnable example scripts.
  - [Examples/dgsem/](Examples/dgsem/) contains DGSEM/Trixi-related examples.
  - [Examples/finite_differences/](Examples/finite_differences/) contains finite-difference examples.
- [src/](src/) contains the package implementation split into solver, IO, callback, sparsity, time integration, and visualization modules.
- [Experiments/](Experiments/) contains additional experimental scripts and setups.

## Installation

This package is currently developed as a local Julia project. From the repository root, run:

```julia
using Pkg
Pkg.instantiate()
```

If you want to use the package from another project, you can also add the local checkout as a development dependency:

```julia
using Pkg
Pkg.develop(path="/path/to/Odil.jl")
```

## Basic usage

A typical ODIL workflow looks like this:

```julia
using Odil

# Define or obtain a timestep function
timestep! = get_timestep(Odil.CarpenterKennedy2N54())

# Construct an OdilProblem with your discretization, reference data, and time grid
problem = OdilProblem(timestep!, p_timestep, Nx, u0, 1:length(u0), t, x)

# Run the optimizer
res = odil_gauss_newton(problem; max_iterations = 200)
```

The concrete examples in [Examples/dgsem/1d/advection/](Examples/dgsem/1d/advection/) show how this is done in practice for advection problems.

## Examples

The repository already includes runnable examples for several test cases:

- DGSEM-based advection and Euler examples in 1D, 2D, and 3D.
- Finite-difference examples for Burgers-type and wave-equation setups.

The finite-difference examples were included because they also appear in the original Python implementation and provide a useful reference point for comparing the Julia port with the earlier work.

A good starting point is the 1D advection example:

- [Examples/dgsem/1d/advection/1d_advection_gauss_newton.jl](Examples/dgsem/1d/advection/1d_advection_gauss_newton.jl)

## Notes

This project is still under active development. The API may evolve as the implementation is extended and refined, especially as the DGSEM/Trixi integration matures.

## Acknowledgements

This work builds on the ODIL idea introduced by the authors of the original paper and is intended as a Julia-based exploration and extension of that framework.