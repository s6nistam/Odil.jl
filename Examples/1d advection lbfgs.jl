using Odil
include("./trixi/structured_1d_dgsem/elixir_advection_basic.jl")
include("../src/semidiscretization/bdf.jl")
include("../src/solvers/odil_lbfgs.jl")

polydeg = 3

lhs = get_lhs(polydeg)

coords = semi.cache.elements.node_coordinates
x = vec(coords[1, :, :])
Nx = length(x)
t = sol.t
Nt = length(t)
p_lhs = (x, t)

u_approx = odil_lbfgs(lhs, ode.f, p_lhs, ode.p, size(ode.u0), ode.u0, eachindex(ode.u0), [1 for _ in eachindex(ode.u0)], Nt; max_iterations = 100000)

# plot_2d(x, t, u_approx)
u_exact = reshape(sol, Nx, Nt)
plot_comparison(x, t, u_exact, u_approx)