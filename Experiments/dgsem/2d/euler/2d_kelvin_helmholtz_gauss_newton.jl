using Odil
include("./dgsem_euler_kelvin_helmholtz.jl")

Trixi.TrixiBase.disable_debug_timings()

polydeg = 3
refinement_level = 5
ndims = 2
variables = Int64(length(ode.u0)/((polydeg + 1)^ndims * (2^refinement_level)^ndims))

coords = semi.cache.elements.node_coordinates
x = coords[1, :, :, :]
y = coords[2, :, :, :]

Nx = variables * length(x)
t = sol.t
Nt = length(t)
dt = [t[i + 1] - t[i] for i in 1:Nt-1]

u_exact = reduce(hcat, vec.(sol.u))

# function extra(du, u, p, iter)
#     Nx, Nt, dt = p
#     idx = LinearIndices((Nx, Nt))
    
#     k = 2.0^(-iter/2.0)

#     idx = LinearIndices((Nx, Nt))

#     for i in 1:Nt - 1
#         du[idx[:, i]] .= k * (u[idx[:, i + 1]] - u[idx[:, i]]) / dt[i]
#     end

#     return nothing
# end

# p_extra = (Nx, Nt, dt)
# len_extra = Nx * (Nt - 1)

timestep! = get_timestep(Odil.CarpenterKennedy2N54())
p_timestep = (ode.f, ode.p)

callback_set = OdilCallbackSet(PlotCallback(1), InfoPrintCallback(1))

problem = OdilProblem(timestep!, p_timestep, Nx, ode.u0, 1:length(ode.u0), t, x, y; timestep_alloc_size = 2 * Nx, u_iter0 = repeat(ode.u0, Nt))
# problem = OdilProblem(timestep!, p_timestep, Nx, ode.u0, 1:length(ode.u0), t, x, y; timestep_alloc_size = 2 * Nx, u_iter0 = repeat(ode.u0, Nt), extra = extra, p_extra = p_extra, len_extra = len_extra)
res = odil_gauss_newton(problem; max_iterations = 200, callback_set = callback_set)
# res = reconstruct_solution_from_chunks(problem, "odil_2d_kelvin_helmholtz_gauss_newton"; t_chunk_size = 8)
write_vtk(problem, res, "odil_2d_kelvin_helmholtz_gauss_newton")

# plot(problem, u_exact, res)
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

plot(problem, res)