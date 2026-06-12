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
include("../src/solvers/odil_gauss_newton.jl")
include("../src/aux/plot.jl")
include("../src/semidiscretization/bdf.jl")

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.5, 0.0, 0.0)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
polydeg = 1
solver = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs)

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
                        # stepsize_callback
                        )

###############################################################################
# run the simulation

# Grab the full multi-dimensional array of coordinates
coords = semi.cache.elements.node_coordinates

# Extract and flatten the X, Y, and Z values into 1D vectors
x = coords[1, :, :, :, :]
y = coords[2, :, :, :, :]
z = coords[3, :, :, :, :]
e = 1:(2^refinement_level)^3

lhs! = get_lhs(polydeg)

coords = semi.cache.elements.node_coordinates
# x_o = [[x[i],y[i],z[i]] for i in 1:length(x)]
x_o = eachindex(ode.u0)
Nx = length(x_o)
Nt = 2 * (polydeg + 1)
t = range(0.0, 1.0, length=Nt)
dt = t[2] - t[1]
p_lhs = (dt, x_o, t)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = dt, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()...,
            save_everystep = true,callback = callbacks);

u_matrix = reduce(hcat, vec.(sol.u))
u_exact = reshape(u_matrix, polydeg + 1, polydeg + 1, polydeg + 1, (2^refinement_level)^3, length(t))
# plot_fe_3d_time(x, y, z, e, u_exact)
# plot_fe_3d_time_compare(x, y, z, e, u_exact, u_exact)
res = odil_gauss_newton(lhs!, ode.f, p_lhs, ode.p, size(ode.u0), ode.u0, eachindex(ode.u0), [1 for _ in eachindex(ode.u0)], Nt; max_iterations = 10)

u_approx = reshape(res, polydeg + 1, polydeg + 1, polydeg + 1, (2^refinement_level)^3, length(t))
plot_fe_3d_time_compare(x, y, z, e, u_exact, u_approx, c_min = 0.0, c_max = 1.5)