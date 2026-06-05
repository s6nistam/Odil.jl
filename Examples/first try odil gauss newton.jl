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
f = ODEFunction{true}(rhs!)
ode = ODEProblem{true}(f, u_exact[:, 1], tspan, (dx, x, t))

function lhs!(du, u, p, it)
    dt, x_array, t_array = p 
    fill!(du, 0.0)
    Nx = length(x_array)
    Nt = length(t_array)
    
    if it == 1
        t0 = t_array[1]
        penalty_vel = Nt 
        
        for j in 2:Nx-1
            v0 = get_exact_wave_velocity(x_array[j], t0)
            du[j] = penalty_vel * (u[j, 2] - u[j, 1] - dt * v0) / dt^2
        end
    elseif it < size(u, ndims(u))
        for j in 2:Nx-1
            du[j] = (u[j, it+1] - 2 * u[j, it] + u[j, it-1]) / dt^2
        end
    end

    penalty_bc = Nx/2
    du[1]  = penalty_bc * (u[1, it + 1] - get_exact_wave(x_array[1], t_array[it + 1]))
    du[Nx] = penalty_bc * (u[Nx, it + 1] - get_exact_wave(x_array[Nx], t_array[it + 1]))
    
    return nothing
end

u = odil_gauss_newton(ode, lhs!, (dt, x, t), Nt)
# u_opt_matrix = reshape(res.u, Nx, Nt)

plot_comparison(x, t, u_exact, u)

