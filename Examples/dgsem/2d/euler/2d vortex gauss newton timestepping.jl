using Odil
include("./dgsem euler vortex.jl")

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

e = 1:(2^refinement_level)^ndims
sol_shape = (variables, (polydeg + 1 for _ in 1:ndims)..., (2^refinement_level)^ndims, Nt)

u_exact = reduce(hcat, vec.(sol.u))

timestep! = get_timestep(Odil.CarpenterKennedy2N54())
p_timestep = (ode.f, ode.p)

problem = OdilProblem(timestep!, p_timestep, Nx, ode.u0, 1:length(ode.u0), t, x, y; timestep_alloc_size = 2 * Nx)
# res = odil_gauss_newton(problem; max_iterations = 100, u_iter0 = repeat(ode.u0, Nt))
res = odil_timestepping(problem, odil_gauss_newton, "odil_2d_vortex_gauss_newton"; t_chunk_size = 8, max_iterations = 20)
# res = reconstruct_solution_from_chunks(problem, "odil_2d_vortex_gauss_newton"; t_chunk_size = 8)
write_vtk(problem, res, "odil_2d_vortex_gauss_newton")

plot(problem, u_exact, res)