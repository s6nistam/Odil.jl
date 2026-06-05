using Odil
import Pkg
Pkg.add("SciMLBase")
Pkg.add("OptimizationBase")
Pkg.add("OptimizationOptimJL")
Pkg.add("ADTypes")
Pkg.add("Enzyme")
include("../src/FD/wave.jl")
include("../src/solvers/odil_lbfgs.jl")

Nx = 32
Nt = 32

x = range(0, 1, length=Nx)
t = range(-.5, .5, length=Nt)

dx = x[2] - x[1]
dt = t[2] - t[1]

@info "dx ", dx
@info "dt ", dt

u_exact  = [get_exact_wave(xi, ti) for xi in x, ti in t]

u = odil_lbfgs(lhs!, rhs!, (dt, x, t), (dx, x, t), u_exact[:, 1], Nt)

plot_comparison(x, t, u_exact, u)

