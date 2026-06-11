using Odil
using OrdinaryDiffEqLowStorageRK
using Trixi
include("../src/aux/plot.jl")

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (20.0, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0, -1.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = (1.0, 1.0, 1.0) # maximum coordinates (max(x), max(y), max(z))

# Create a uniformly refined mesh with periodic boundaries
refinement_level = 3
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = refinement_level,
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
#
# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 1.2)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()...,
            save_everystep = true,callback = callbacks);

# Grab the full multi-dimensional array of coordinates
coords = semi.cache.elements.node_coordinates

# Extract and flatten the X, Y, and Z values into 1D vectors
x = coords[1, :, :, :, :]
y = coords[2, :, :, :, :]
z = coords[3, :, :, :, :]
e = 1:(2^refinement_level)^3

t = sol.t

# Format 'u' for the plot function:
# 1. vec.(sol.u) flattens the 3D Trixi grid (and the 1 variable) into a 1D array for each time step.
# 2. reduce(hcat, ...) stacks these time steps side-by-side into a single 2D Matrix.
u_matrix = reduce(hcat, vec.(sol.u))
u = reshape(u_matrix, 4, 4, 4, (2^refinement_level)^3, length(t))
plot_fe_3d_time(x, y, z, e, u)