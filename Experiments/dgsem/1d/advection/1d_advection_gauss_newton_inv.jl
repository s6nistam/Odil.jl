using Odil
include("./dgsem_advection.jl")

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
# index = Int((Nt/2) * Nx + 1) : Int((Nt/2 + 1) * Nx)
# index = [i^2 for i in 1:min(Nt, Nx)]
index = rand(1:Nt*Nx, Int(round(Nt*Nx*(1/100))))

callback_set = OdilCallbackSet(PlotCallback(100))

problem = OdilProblem(timestep!, p_timestep, Nx, u_exact[index], index, t, x; timestep_alloc_size = 2 * Nx)
res = odil_gauss_newton(problem; max_iterations = 200, callback_set = callback_set)

plot(problem, u_exact, res)

write_vtk(problem, res, "odil_1d_advection_gauss_newton")