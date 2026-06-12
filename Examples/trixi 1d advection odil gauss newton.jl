# The same setup as tree_1d_dgsem/elixir_advection_basic.jl
# to verify the StructuredMesh implementation against TreeMesh
using Odil
using OrdinaryDiffEqLowStorageRK
using Trixi
import Pkg
# Pkg.add("SciMLBase")
# Pkg.add("OptimizationBase")
# Pkg.add("OptimizationOptimJL")
# Pkg.add("ADTypes")
# Pkg.add("Enzyme")
# Pkg.add("ColorSchemes")
# Pkg.add("Plots")
# Pkg.add("FiniteDifferences")
include("../src/aux/plot.jl")
include("../src/semidiscretization/bdf.jl")
include("../src/solvers/odil_gauss_newton.jl")

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 0.3
equations = LinearScalarAdvectionEquation1D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
# solver = DGSEM(polydeg = 1, surface_flux = flux_central)
polydeg = 3
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

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
# stepsize_callback = StepsizeCallback(cfl = 1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        # stepsize_callback
                        )

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks

lhs! = get_lhs(polydeg)

coords = semi.cache.elements.node_coordinates
x = vec(coords[1, :, :])
Nx = length(x)
Nt = 16 * (2 * polydeg + 1)
t = range(0.0, 1.0, length=Nt)
dt = t[2] - t[1]
p_lhs = (dt, x, t)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = dt, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()...,
            save_everystep = true, callback = callbacks);

res = odil_gauss_newton(lhs!, ode.f, p_lhs, ode.p, size(ode.u0), ode.u0, eachindex(ode.u0), [1 for _ in eachindex(ode.u0)], Nt; max_iterations = 100)

# plot_2d(x, t, sol)
prsol = reshape(sol, Nx, Nt)
plot_comparison(x, t, prsol, res)
