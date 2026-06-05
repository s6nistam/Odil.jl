using Odil
import Pkg
Pkg.add("SciMLBase")
Pkg.add("NonlinearSolveFirstOrder")
Pkg.add("ADTypes")
Pkg.add("Enzyme")
include("../src/FD/wave.jl")
include("../src/solvers/odil_gauss_newton.jl")

Odil.greet()

Nx = 32
Nt = 32

x = range(0, 1, length=Nx)
t = range(-.5, .5, length=Nt)
tspan = (t[1], t[end])

dx = x[2] - x[1]
dt = t[2] - t[1]

@info "dx ", dx
@info "dt ", dt

u_exact  = [get_exact_wave(xi, ti) for xi in x, ti in t]

# u_fd = zero(u_exact)

# # initial condition
# u_fd[:, 1] .= u_exact[:, 1]
# u_fd[:, 2] .= u_exact[:, 2]

# # boundary conditions
# u_fd[1, :] .= u_exact[1, :]
# u_fd[end, :] .= u_exact[end, :]

# solve_d2dt2_central!(u_fd, rhs_wave, dx, dt)
u = odil_gauss_newton(lhs!, rhs!, (dt, x, t), (dx, x, t), u_exact[:, 1], Nt)
# u_opt_matrix = reshape(res.u, Nx, Nt)

plot_comparison(x, t, u_exact, u)

