using Odil
include("./trixi/tree_3d_dgsem/elixir_advection_basic.jl")
include("../src/aux/plot.jl")

polydeg = 1
refinement_level = 3
ndims = 3

coords = semi.cache.elements.node_coordinates
x = coords[1, :, :, :, :]
y = coords[2, :, :, :, :]
z = coords[3, :, :, :, :]
t = sol.t
Nt = length(t)

e = 1:(2^refinement_level)^ndims
variables = Int64(Nx/((polydeg + 1)^ndims * (2^refinement_level)^ndims))
sol_shape = (variables, (polydeg + 1 for _ in 1:ndims)..., (2^refinement_level)^ndims, Nt)

u_matrix = reduce(hcat, vec.(sol.u))
u = reshape(u_matrix, sol_shape...)
for var in 1:variables
    plot_fe_3d_time(x, y, z, e, u[var, :, :, :, :, :], c_min = 0.0, c_max = 1.5)
end