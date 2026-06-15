using Odil
include("../src/semidiscretization/wave.jl")
include("../src/solvers/odil_gauss_newton.jl")

Nx = 32
Nt = 32

x = range(0, 1, length=Nx)
t = range(-.5, .5, length=Nt)

p_lhs = (x, t)
p_rhs = (x, t)

u_t0  = [get_exact_wave(x[ix], t[1]) for ix in 1:Nx]
u_bounds_left = [get_exact_wave(x[1], t[it]) for it in 2:Nt]
u_bounds_right = [get_exact_wave(x[Nx], t[it]) for it in 2:Nt]
u_fixed_vals = [u_t0; u_bounds_left; u_bounds_right]

x_t0  = [ix for ix in 1:Nx]
x_bounds_left = [1 for it in 2:Nt]
x_bounds_right = [Nx for it in 2:Nt]
x_fixed_indicies = [x_t0; x_bounds_left; x_bounds_right]

t_t0  = [1 for ix in 1:Nx]
t_bounds_left = [it for it in 2:Nt]
t_bounds_right = [it for it in 2:Nt]
t_fixed_indicies = [t_t0; t_bounds_left; t_bounds_right]

u_size_x = Nx

u = odil_gauss_newton(lhs!, rhs!, p_lhs, p_rhs, u_size_x, u_fixed_vals, x_fixed_indicies, t_fixed_indicies, Nt)

plot_1d_time(x, t, u)
# plot_1d_time_comparison(x, t, u_exact, u)

