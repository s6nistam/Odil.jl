using Odil
include("./trixi/tree_2d_dgsem/elixir_euler_blast_wave.jl")
include("../src/aux/plot.jl")
include("../src/semidiscretization/bdf.jl")
include("../src/solvers/odil_gauss_newton.jl")

println(size(sol))

polydeg = 2
refinement_level = 3
coords = semi.cache.elements.node_coordinates
x = coords[1, :, :, :]
y = coords[2, :, :, :]
e = 1:(2^refinement_level)^2

lhs! = get_lhs(polydeg)

coords = semi.cache.elements.node_coordinates
x_o = eachindex(ode.u0)
Nx = length(x_o)
t = sol.t
Nt = length(t)
p_lhs = (x_o, t)
variables = Int64(Nx/((polydeg + 1)^2 * (2^refinement_level)^2))

u_matrix = reduce(hcat, vec.(sol.u))
u_exact = reshape(u_matrix, variables, polydeg + 1, polydeg + 1, (2^refinement_level)^2, length(t))
# plot_fe_3d_time(x, y, z, e, u_exact)
# plot_fe_3d_time_compare(x, y, z, e, u_exact, u_exact)
res = odil_gauss_newton(lhs!, ode.f, p_lhs, ode.p, size(ode.u0), ode.u0, eachindex(ode.u0), [1 for _ in eachindex(ode.u0)], t; max_iterations = 100, u_iter0 = repeat(ode.u0, Nt), autodiff = AutoFiniteDiff())

u_approx = reshape(res, variables, polydeg + 1, polydeg + 1, (2^refinement_level)^2, length(t))
for var in 1:variables
    plot_fe_2d_time_compare(x, y, e, u_exact[var, :, :, :, :], u_approx[var, :, :, :, :])
end