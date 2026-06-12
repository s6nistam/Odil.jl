using Odil
include("./trixi/tree_3d_dgsem/elixir_advection_basic.jl")
include("../src/aux/plot.jl")
include("../src/semidiscretization/bdf.jl")
include("../src/solvers/odil_gauss_newton.jl")

polydeg = 1
refinement_level = 3
coords = semi.cache.elements.node_coordinates
x = coords[1, :, :, :, :]
y = coords[2, :, :, :, :]
z = coords[3, :, :, :, :]
e = 1:(2^refinement_level)^3

lhs! = get_lhs(polydeg)

coords = semi.cache.elements.node_coordinates
x_o = eachindex(ode.u0)
Nx = length(x_o)
t = sol.t
Nt = length(t)
p_lhs = (x_o, t)

u_matrix = reduce(hcat, vec.(sol.u))
u_exact = reshape(u_matrix, polydeg + 1, polydeg + 1, polydeg + 1, (2^refinement_level)^3, length(t))
# plot_fe_3d_time(x, y, z, e, u_exact)
# plot_fe_3d_time_compare(x, y, z, e, u_exact, u_exact)
res = odil_gauss_newton(lhs!, ode.f, p_lhs, ode.p, size(ode.u0), ode.u0, eachindex(ode.u0), [1 for _ in eachindex(ode.u0)], Nt; max_iterations = 10)

u_approx = reshape(res, polydeg + 1, polydeg + 1, polydeg + 1, (2^refinement_level)^3, length(t))
plot_fe_3d_time_compare(x, y, z, e, u_exact, u_approx, c_min = 0.0, c_max = 1.5)