using Odil
include("./dgsem_advection.jl")

Trixi.TrixiBase.disable_debug_timings()

polydeg = 3
refinement_level = 4
ndims = 1
variables = Int64(length(ode.u0)/((polydeg + 1)^ndims * (2^refinement_level)^ndims))

coords = semi.cache.elements.node_coordinates
x = coords[1, :, :]

Nx = variables * length(x)
t = sol.t
Nt = length(t)
dt = [t[i + 1] - t[i] for i in 1:Nt-1]

u_exact = reduce(hcat, vec.(sol.u))

timestep! = get_timestep(Odil.CarpenterKennedy2N54())
p_timestep = (ode.f, ode.p)

callback_set = OdilCallbackSet(PlotCallback(100))

problem = OdilProblem(timestep!, p_timestep, Nx, ode.u0, 1:length(ode.u0), t, x; timestep_alloc_size = 2 * Nx)
res = odil_timestepping(problem, odil_lbfgs, "odil_1d_advection_lbfgs_timestepping"; t_chunk_size = 10, max_iterations_per_chunk = 1000, callback_set = callback_set)
write_vtk(problem, res, "odil_1d_advection_lbfgs_timestepping")

function exact_solution(x, t)
    return 1 + 0.5 * sin(π * (x - t))
end

exact = [exact_solution(x[ix], t[it]) for ix in 1:Nx, it in 1:Nt]

for i in 1:length(res)
    if res[i] != u_exact[i]
        res[i] = log10(abs(u_exact[i] - res[i]))
    end
end

min = minimum(res)

for i in 1:length(res)
    if res[i] == u_exact[i]
        res[i] = min - 1
    end
end

# for i in 1:length(res)
#     if res[i] != exact[i]
#         res[i] = log10(abs(exact[i] - res[i]))
#     end
# end

# min = minimum(res)

# for i in 1:length(res)
#     if res[i] == exact[i]
#         res[i] = min - 1
#     end
# end

for i in 1:length(res)
    if u_exact[i] != exact[i]
        u_exact[i] = log10(abs(exact[i] - u_exact[i]))
    end
end

min = minimum(u_exact)

for i in 1:length(u_exact)
    if u_exact[i] == exact[i]
        u_exact[i] = min - 1
    end
end



# plot(problem, exact, u_exact)
# plot(problem, res)
plot(problem, u_exact, res)


write_vtk(problem, res, "odil_1d_advection_lbfgs_timestepping_errors")
