using Odil
include("./trixi/tree_2d_dgsem/elixir_euler_blast_wave.jl")
include("../src/aux/plot.jl")
include("../src/semidiscretization/bdf.jl")

println(size(sol))

polydeg = 2
refinement_level = 3
ndims = 2
variables = Int64(length(ode.u0)/((polydeg + 1)^ndims * (2^refinement_level)^ndims))

coords = semi.cache.elements.node_coordinates
x = coords[1, :, :, :]
y = coords[2, :, :, :]

lhs! = get_lhs(polydeg)

Nx = variables * length(x)
t = sol.t
Nt = length(t)
dt = [t[i + 1] - t[i] for i in 1:Nt-1]
p_lhs = (Nx, t, Nt, dt)

e = 1:(2^refinement_level)^ndims
sol_shape = (variables, (polydeg + 1 for _ in 1:ndims)..., (2^refinement_level)^ndims, Nt)

u_matrix = reduce(hcat, vec.(sol.u))
u_exact = reshape(u_matrix, sol_shape...)
# plot_fe_3d_time(x, y, z, e, u_exact)
# plot_fe_3d_time_compare(x, y, z, e, u_exact, u_exact)
problem = OdilProblem(lhs!, ode.f, p_lhs, ode.p, Nx, ode.u0, 1:length(ode.u0), t, x, y)
res = odil_gauss_newton(problem; max_iterations = 100, u_iter0 = repeat(ode.u0, Nt), autodiff = AutoFiniteDiff())

u_approx = reshape(res, sol_shape...)
for var in 1:variables
    plot_fe_2d_time_compare(x, y, e, u_exact[var, :, :, :, :], u_approx[var, :, :, :, :])
end