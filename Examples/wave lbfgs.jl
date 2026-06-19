using Odil
import Pkg
include("../src/semidiscretization/wave.jl")
include("../src/solvers/odil_lbfgs.jl")

Nx = 32
Nt = 32

x = range(0, 1, length=Nx)
t = range(-.5, .5, length=Nt)

dx = [x[i + 1] - x[i] for i in 1:Nx-1]
dt = [t[i + 1] - t[i] for i in 1:Nt-1]

u_exact = [get_exact_wave(x[ix], t[it]) for ix in 1:Nx, it in 1:Nt]

p_lhs = (x, Nx, dx, t, Nt, dt)
p_rhs = (x, Nx, dx, t, Nt, dt)

u_t0  = [get_exact_wave(x[ix], t[1]) for ix in 1:Nx]
u_bounds_left = [get_exact_wave(x[1], t[it]) for it in 2:Nt]
u_bounds_right = [get_exact_wave(x[Nx], t[it]) for it in 2:Nt]
u_reference_vals = [u_t0; u_bounds_left; u_bounds_right]

idx = LinearIndices((Nx, Nt))

idx_t0  = [idx[ix, 1] for ix in 1:Nx]
idx_bounds_left = [idx[1, it] for it in 2:Nt]
idx_bounds_right = [idx[Nx, it] for it in 2:Nt]
reference_val_indices = [idx_t0; idx_bounds_left; idx_bounds_right]

u = odil_lbfgs(lhs!, rhs!, p_lhs, p_rhs, Nx, u_reference_vals, reference_val_indices, t)

plot_1d_time_comparison(x, t, u_exact, u)

