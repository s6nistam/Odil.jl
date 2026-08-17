using Odil
include("./dgsem_euler_sedov_blast_wave.jl")

Trixi.TrixiBase.disable_debug_timings()

polydeg = 3
refinement_level = 5
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
# timestep! = get_timestep(Odil.ExplicitEuler())
p_timestep = (ode.f, ode.p)

callback_set = OdilCallbackSet(PlotCallback(100))

problem = OdilProblem(timestep!, p_timestep, Nx, ode.u0, 1:length(ode.u0), t, x; timestep_alloc_size = 2 * Nx)
# problem = OdilProblem(timestep!, p_timestep, Nx, ode.u0, 1:length(ode.u0), t, x; timestep_alloc_size = Nx)
# res = odil_gauss_newton(problem; max_iterations = 200)
res = odil_timestepping(problem, odil_gauss_newton, "odil_1d_euler_sedov_blast_wave"; t_chunk_size = 4, max_iterations_per_chunk = 200, callback_set = callback_set)

plot(problem, u_exact, res)

write_vtk(problem, res, "odil_1d_euler_sedov_blast_wave")