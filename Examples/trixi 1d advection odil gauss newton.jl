# The same setup as tree_1d_dgsem/elixir_advection_basic.jl
# to verify the StructuredMesh implementation against TreeMesh
using Odil
using OrdinaryDiffEqLowStorageRK
using Trixi
import Pkg
Pkg.add("SciMLBase")
Pkg.add("OptimizationBase")
Pkg.add("OptimizationOptimJL")
Pkg.add("ADTypes")
Pkg.add("Enzyme")
Pkg.add("ColorSchemes")
Pkg.add("Plots")
include("../src/solvers/odil_gauss_newton.jl")

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
# solver = DGSEM(polydeg = 1, surface_flux = flux_central)
polydeg = 1
solver = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0,) # minimum coordinate
coordinates_max = (1.0,) # maximum coordinate
cells_per_dimension = (16,)

# Create curved mesh with 16 cells
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max,
                      periodicity = true)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 1.0))

function lhs!(du, u, p, it)
    dt, x_array, t_array = p 
    fill!(du, 0.0)
    Nx = length(x_array)
    Nt = length(t_array)
    
    for ix in 1:Nx
        du[ix] = (u[ix, it + 1] - u[ix, it]) / dt
    end
    
    return nothing
end
coords = semi.cache.elements.node_coordinates
x = vec(coords[1, :, :])
Nx = length(x)
Nt = Nx 
t = range(0.0, 1.0, length=Nt)
dt = t[2] - t[1]
p_lhs = (dt, x, t)


sol = odil_gauss_newton(lhs!, ode.f, p_lhs, ode.p, size(ode.u0), ode.u0, 1:length(ode.u0), [1 for i in 1:length(ode.u0)], Nt, 50)

plot_2d(x, t, sol)