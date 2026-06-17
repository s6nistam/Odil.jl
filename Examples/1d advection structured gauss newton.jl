using Odil
include("./trixi/structured_1d_dgsem/elixir_advection_basic.jl")
include("../src/semidiscretization/bdf.jl")
include("../src/solvers/odil_gauss_newton.jl")

polydeg = 3

lhs = get_lhs(polydeg)

coords = semi.cache.elements.node_coordinates
x = vec(coords[1, :, :])
Nx = length(x)
t = sol.t
p_lhs = (x, t)

u_approx = odil_gauss_newton(lhs, ode.f, p_lhs, ode.p, size(ode.u0), ode.u0, eachindex(ode.u0), [1 for _ in eachindex(ode.u0)], t; max_iterations = 1000)

# plot_1d_time(x, t, u_approx)
u_exact = reshape(sol, Nx, Nt)
plot_1d_time_comparison(x, t, u_exact, u_approx)
