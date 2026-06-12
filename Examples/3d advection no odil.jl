using Odil
include("./trixi/tree_3d_dgsem/elixir_advection_basic.jl")
include("../src/aux/plot.jl")

polydeg = 1
refinement_level = 3
coords = semi.cache.elements.node_coordinates
x = coords[1, :, :, :, :]
y = coords[2, :, :, :, :]
z = coords[3, :, :, :, :]
e = 1:(2^refinement_level)^3
t = sol.t

u_matrix = reduce(hcat, vec.(sol.u))
u = reshape(res, polydeg + 1, polydeg + 1, polydeg + 1, (2^refinement_level)^3, length(t))
plot_fe_3d_time(x, y, z, e, u, c_min = 0.0, c_max = 1.5)