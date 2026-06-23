using Odil
import Pkg
# Pkg.add("SciMLBase")
# Pkg.add("OptimizationBase")
# Pkg.add("OptimizationOptimJL")
# Pkg.add("ADTypes")
# Pkg.add("Enzyme")
# Pkg.add("ColorSchemes")
# Pkg.add("Plots")
include("../src/semidiscretization/burgers.jl")
include("../src/references/burgers.jl")

Nx = 32
Nt = 32

x = range(0, 1, length=Nx)
t = range(0, 1, length=Nt)

dx = [x[i + 1] - x[i] for i in 1:Nx-1]
dt = [t[i + 1] - t[i] for i in 1:Nt-1]

p_lhs = (x, Nx, dx, t, Nt, dt)
p_rhs = (x, Nx, dx, t, Nt, dt)

u_t0  = [get_initial_burgers(x[ix]) for ix in 1:Nx]
u_bounds_left = [0 for it in 2:Nt]
u_bounds_right = [0 for it in 2:Nt]
u_reference_vals = [u_t0; u_bounds_left; u_bounds_right]

idx = LinearIndices((Nx, Nt))

idx_t0  = [idx[ix, 1] for ix in 1:Nx]
idx_bounds_left = [idx[1, it] for it in 2:Nt]
idx_bounds_right = [idx[Nx, it] for it in 2:Nt]
reference_val_indices = [idx_t0; idx_bounds_left; idx_bounds_right]

max_iterations = 10000

problem = Odil1D(lhs!, rhs!, p_lhs, p_rhs, Nx, u_reference_vals, reference_val_indices, t, x)
u = odil_lbfgs(problem; max_iterations = max_iterations)

u = reshape(u, Nx, Nt)

plot_1d_time(x, t, u)