using Odil
import Pkg
Pkg.add("SciMLBase")
Pkg.add("NonlinearSolveFirstOrder")
Pkg.add("ADTypes")
Pkg.add("Enzyme")
include("../src/semidiscretization/wave.jl")
include("../src/solvers/odil_gauss_newton.jl")

Nx = 32
Nt = 32

x = range(0, 1, length=Nx)
t = range(-.5, .5, length=Nt)

dx = x[2] - x[1]
dt = t[2] - t[1]

@info "dx ", dx
@info "dt ", dt

p_lhs = (dt, x, t)
p_rhs = (dx, x, t)

u_t0  = [(get_exact_wave(x[ix], t[1]), ix, 1) for ix in 1:Nx]
u_bounds_left = [(get_exact_wave(x[1], t[it]), 1, it) for it in 2:Nt]
u_bounds_right = [(get_exact_wave(x[Nx], t[it]), Nx, it) for it in 2:Nt]
u_exact_vals = [u_t0; u_bounds_left; u_bounds_right]

u_size_x = Nx

u = odil_gauss_newton(lhs!, rhs!, p_lhs, p_rhs, u_size_x, u_exact_vals, Nt)

plot_comparison(x, t, u_exact, u)

