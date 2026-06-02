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

Nx = 64
Nt = 64

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

function lhs!(du, u, dt, it)
    fill!(du, 0.0)
    
    # Indizes für den Raum (wir lassen die Ränder j=1 und j=end bei 0)
    Nx = size(u, 1)
    
    if it == 1
        # t=0: Wir benötigen hier eine konsistente Beschleunigung.
        # Wir extrapolieren aus den ersten drei Zeitschichten (t=1, t=2, t=3),
        # um den Wert der zweiten Ableitung bei t=1 zu schätzen.
        # Approximation: u_tt(t_1) ≈ (u_1 - 2*u_2 + u_3) / dt^2
        for j in 2:Nx-1
            du[j] = (u[j, 1] - 2*u[j, 2] + u[j, 3]) / dt^2
        end
        
    elseif it < size(u, ndims(u))
        # Standard: Zentrale Differenz zweiter Ordnung (t > 0)
        for j in 2:Nx-1
            du[j] = (u[j, it+1] - 2 * u[j, it] + u[j, it-1]) / dt^2
        end
        
    else
        # Letzter Zeitschritt: Hier ist der Standard-Zentraldifferenz-Operator 
        # nicht definiert, da it+1 außerhalb liegt.
        du .= 0
    end
    
    return nothing
end

u = odil_lbfgsb(ode, lhs!, dx, Nt)
# u_opt_matrix = reshape(res.u, Nx, Nt)

plot_comparison(x, t, u_exact, u)

