using Odil
include("./trixi/structured_1d_dgsem/elixir_advection_basic.jl")
include("../src/semidiscretization/bdf.jl")
include("../src/solvers/odil_lbfgs.jl")

polydeg = 3
refinement_level = 4
ndims = 1

coords = semi.cache.elements.node_coordinates
x = coords[1, :, :]

lhs = get_lhs(polydeg)

x_o = eachindex(ode.u0)
Nx = length(x_o)
t = sol.t
Nt = length(t)
p_lhs = (x_o, t)

e = 1:(2^refinement_level)^ndims
variables = Int64(Nx/((polydeg + 1)^ndims * (2^refinement_level)^ndims))
sol_shape = (variables, (polydeg + 1 for _ in 1:ndims)..., (2^refinement_level)^ndims, Nt)

u_matrix = reduce(hcat, vec.(sol.u))
u_exact = reshape(u_matrix, sol_shape...)
res = odil_lbfgs(lhs, ode.f, p_lhs, ode.p, Nx, ode.u0, 1:length(ode.u0), t; max_iterations = 10000)

u_approx = reshape(res, sol_shape...)
for var in 1:variables
    plot_fe_1d_time_compare(x, e, u_exact[var, :, :, :], u_approx[var, :, :, :], c_min = 0.0, c_max = 1.5)
end