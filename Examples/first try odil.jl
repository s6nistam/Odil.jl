using Odil
import Pkg
Pkg.add("SciMLBase")
Pkg.add("OptimizationBase")
Pkg.add("OptimizationLBFGSB")
Pkg.add("ADTypes")
Pkg.add("Enzyme")
include("../src/FD/wave.jl")
include("../src/solvers/odil_lbfgsb.jl")

Odil.greet()

Nx = 16
Nt = 16

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
f = ODEFunction{true}((du, u, p, t) -> rhs!(du, u, t, p))
ode = ODEProblem{true}(f, u_exact[:, 1], tspan, dx)

function lhs!(du, u, p, it)
    # Entpacke die benötigten Konstanten und Gitter-Vektoren aus p
    dt = p
    
    fill!(du, 0.0)
    Nx = size(u, 1)
    
    if it == 1
        t0 = t[1]
        
        for j in 2:Nx-1
            u0 = get_exact_wave(t0, x[j])
            v0 = get_exact_wave_velocity(t0, x[j])
            
            du[j] = 2 * (u[j, 2] - u0 - dt * v0) / dt^2
            # du[j] = 0
        end
        
    elseif it < size(u, ndims(u))
        for j in 2:Nx-1
            du[j] = (u[j, it+1] - 2 * u[j, it] + u[j, it-1]) / dt^2
        end
        
    else
        du .= 0
    end
    
    return nothing
end

u = odil_lbfgsb(ode, lhs!, dx, Nt)
# u_opt_matrix = reshape(res.u, Nx, Nt)

plot_comparison(x, t, u_exact, u)

