using Odil
include("./trixi/tree_2d_dgsem/elixir_advection_basic.jl")
include("../src/aux/plot.jl")
include("../src/semidiscretization/bdf.jl")
# include("../src/semidiscretization/implicit euler.jl")

polydeg = 2
refinement_level = 3
ndims = 2
variables = Int64(length(ode.u0)/((polydeg + 1)^ndims * (2^refinement_level)^ndims))

coords = semi.cache.elements.node_coordinates
x = coords[1, :, :, :]
y = coords[2, :, :, :]

lhs! = get_lhs(polydeg)

coords = semi.cache.elements.node_coordinates
x_o = vec(repeat(x, variables))
Nx = length(x_o)
dx = [x_o[i + 1] - x_o[i] for i in 1:Nx-1]
t = sol.t
Nt = length(t)
dt = [t[i + 1] - t[i] for i in 1:Nt-1]
p_lhs = (x_o, Nx, dx, t, Nt, dt)

e = 1:(2^refinement_level)^ndims
sol_shape = (variables, (polydeg + 1 for _ in 1:ndims)..., (2^refinement_level)^ndims, Nt)

u_matrix = reduce(hcat, vec.(sol.u))
u_exact = reshape(u_matrix, sol_shape...)
# plot_fe_3d_time(x, y, z, e, u_exact)
# plot_fe_3d_time_compare(x, y, z, e, u_exact, u_exact)
problem = Odil2D(lhs!, ode.f, p_lhs, ode.p, Nx, ode.u0, 1:length(ode.u0), t, x, y)
res = odil_gauss_newton(problem; max_iterations = 20, u_iter0 = vec(repeat(ode.u0, Nt)))

u_approx = reshape(res, sol_shape...)
# plot_fe_2d_time_compare(x, y, e, u_exact, u_approx, c_min = 0.0, c_max = 1.5)
for var in 1:variables
    plot_fe_2d_time_compare(x, y, e, u_exact[var, :, :, :, :], u_approx[var, :, :, :, :], c_min = 0.0, c_max = 1.5)
end