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
Pkg.add("FiniteDifferences")
include("../src/aux/plot.jl")
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


# function lhs!(du, u, p, it)
#     dt, x_array, t_array = p 
#     fill!(du, 0.0)
#     Nx = length(x_array)
#     Nt = length(t_array)
    
#     for ix in 1:Nx
#         du[ix] = (u[ix, it + 1] - u[ix, it]) / dt
#     end
    
#     return nothing
# end

# function lhs!(du, u, p, it)
#     dt, x_array, t_array, rhs!, p_rhs = p
#     fill!(du, 0.0)
#     Nx = length(x_array)
#     Nt = length(t_array)

#     # Use a higher-order time discretization, e.g., RK4
#     k1 = zeros(Nx)
#     k2 = zeros(Nx)
#     k3 = zeros(Nx)
#     k4 = zeros(Nx)

#     rhs!(k1, u[:, it], p_rhs, it)
#     rhs!(k2, u[:, it] + 0.5 * dt * k1, p_rhs, it)
#     rhs!(k3, u[:, it] + 0.5 * dt * k2, p_rhs, it)
#     rhs!(k4, u[:, it] + dt * k3, p_rhs, it)

#     for ix in 1:Nx
#         du[ix] = (k1[ix] + 2 * k2[ix] + 2 * k3[ix] + k4[ix]) / 6
#     end

#     return nothing
# end

using FiniteDifferences

function precompute_bdf_weights(max_order::Int)
    # Ein Array, das Arrays enthält: weights[k] speichert die Gewichte für BDF-Ordnung k
    weights = Vector{Vector{Float64}}(undef, max_order)
    
    for order in 1:max_order
        # Generiere das Finite-Differenzen-Schema: (Genauigkeitsordnung, 1. Ableitung)
        fdm = backward_fdm(order + 1, 1)
        
        # fdm.grid liefert die relativen Zeitpunkte, z.B. [-2.0, -1.0, 0.0]
        # Wir wollen die Gewichte absteigend sortiert (0.0 zuerst, dann -1.0, etc.),
        # damit sie exakt zur Reihenfolge [it+1, it, it-1, ...] passen.
        grid_and_coefs = sort(collect(zip(fdm.grid, fdm.coefs)), by = x -> x[1], rev = true)
        
        # Isoliere nur die Gewichte
        weights[order] = [c for (g, c) in grid_and_coefs]
    end
    
    return weights
end

function lhs!(du, u, p, it)
    # Entpacke p (jetzt mit den vorberechneten BDF-Gewichten)
    dt, x_array, t_array, bdf_weights = p 
    fill!(du, 0.0)
    Nx = length(x_array)
    max_order = length(bdf_weights)
    
    # --- KALTSTART-LOGIK ---
    # Die maximal nutzbare Ordnung wird durch die verfügbare Historie limitiert.
    # Bei it=1 gibt es 1 vergangenen Schritt, also BDF1. Bei it=2 gibt es BDF2.
    current_order = min(it, max_order)
    
    # Hole die korrekten Gewichte für die aktuelle Ordnung
    weights = bdf_weights[current_order]
    
    # Die Gewichte von FiniteDifferences gehen von dt=1 aus, also skalieren wir:
    inv_dt = 1.0 / dt
    
    @inbounds for ix in 1:Nx
        val = 0.0
        # Multipliziere die Gewichte mit den entsprechenden u-Werten der Historie
        # w_idx = 1 entspricht it+1
        # w_idx = 2 entspricht it
        # w_idx = 3 entspricht it-1, usw.
        for w_idx in 1:length(weights)
            time_idx = (it + 1) - (w_idx - 1)
            val += weights[w_idx] * u[ix, time_idx]
        end
        du[ix] = val * inv_dt
    end
    
    return nothing
end

coords = semi.cache.elements.node_coordinates
x = vec(coords[1, :, :])
Nx = length(x)
Nt = 16 * (polydeg + 1)
t = range(0.0, 1.0, length=Nt)
dt = t[2] - t[1]
bdf_weights = precompute_bdf_weights(polydeg)  # Beispiel für BDF-Ordnung 4
p_lhs = (dt, x, t, bdf_weights)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = dt, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()...,
            save_everystep = true, callback = callbacks);

res = odil_gauss_newton(lhs!, ode.f, p_lhs, ode.p, size(ode.u0), ode.u0, eachindex(ode.u0), [1 for _ in eachindex(ode.u0)], Nt; max_iterations = 1000)

# plot_2d(x, t, sol)
prsol = reshape(sol, Nx, Nt)
plot_comparison(x, t, prsol, res)
