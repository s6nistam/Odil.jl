using Odil
import Pkg
include("../src/semidiscretization/wave.jl")

Nx = 32
Nt = 32

x = range(-1, 1, length=Nx)
t = range(0, 1, length=Nt)

dx = [x[i + 1] - x[i] for i in 1:Nx-1]
dt = [t[i + 1] - t[i] for i in 1:Nt-1]

u_exact = [get_exact_wave(x[ix], t[it]) for ix in 1:Nx, it in 1:Nt]
u_t_exact = [get_exact_wave_velocity(x[ix], t[it]) for ix in 1:Nx, it in 1:Nt]

p_lhs = (x, Nx, dx, t, Nt, dt)
p_rhs = (x, Nx, dx, t, Nt, dt)

u_t0  = [get_exact_wave(x[ix], t[1]) for ix in 1:Nx]
u_bounds_left = [get_exact_wave(x[1], t[it]) for it in 2:Nt]
u_bounds_right = [get_exact_wave(x[Nx], t[it]) for it in 2:Nt]

u_t_t0  = [get_exact_wave_velocity(x[ix], t[1]) for ix in 1:Nx]

u_reference_vals = [u_t0; u_bounds_left; u_bounds_right; u_t_t0]

idx = LinearIndices((Nx, 2, Nt))

idx_t0  = [idx[ix, 1, 1] for ix in 1:Nx]
idx_bounds_left = [idx[1, 1, it] for it in 2:Nt]
idx_bounds_right = [idx[Nx, 1, it] for it in 2:Nt]

idx_t_t0  = [idx[ix, 2, 1] for ix in 1:Nx]

reference_val_indices = [idx_t0; idx_bounds_left; idx_bounds_right; idx_t_t0]

problem = Odil1D(lhs!, rhs!, p_lhs, p_rhs, 2 * Nx, u_reference_vals, reference_val_indices, t, x)
u = odil_lbfgs(problem; max_iterations = 100000)

u = reshape(u, Nx, 2, Nt)

plot_1d_time_comparison(x, t, u_exact, u[:, 1, :])
# plot_1d_time_comparison(x, t, u_t_exact, u[:, 2, :])

