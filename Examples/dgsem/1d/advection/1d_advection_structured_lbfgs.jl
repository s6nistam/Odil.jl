using Odil
include("./structured_dgsem_advection.jl")

polydeg = 2
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

problem = OdilProblem(timestep!, p_timestep, Nx, ode.u0, 1:length(ode.u0), t, x; timestep_alloc_size = 2 * Nx)
res = odil_lbfgs(problem; max_iterations = 200)

plot(problem, u_exact, res)