using Odil
include("./dgsem_euler_blast_wave.jl")

Trixi.TrixiBase.disable_debug_timings()

polydeg = 3
refinement_level = 4
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

timestep! = get_timestep(Odil.CarpenterKennedy2N54())
p_timestep = (ode.f, ode.p)

callback_set = OdilCallbackSet(
    PlotCallback(100)
)

problem = OdilProblem(timestep!, p_timestep, Nx, ode.u0, 1:length(ode.u0), t, x, y; timestep_alloc_size = 2 * Nx, u_iter0 = repeat(ode.u0, Nt))

res = odil_timestepping(problem, odil_gauss_newton, "odil_2d_blast_wave"; t_chunk_size = 4, max_iterations_per_chunk = 2000, callback_set = callback_set, sub_solver_info_prints = true)
# res = odil_gauss_newton(problem; max_iterations = 100, callback_set = callback_set)
write_vtk(problem, res, "odil_2d_blast_wave")
write_csv(problem, res, "odil_2d_blast_wave")

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

write_vtk(problem, res, "odil_2d_blast_wave_errors")
write_csv(problem, res, "odil_2d_blast_wave_errors")