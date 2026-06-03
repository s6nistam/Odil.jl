function rhs!(du, u, it, p)
    # Entpacke die Parameter
    dx = p 
    
    fill!(du, 0.0)
    Nx = size(u, 1)
    t_val = t[it]
    
    # 1. Linker Rand-Knoten (j = 2)
    # Hier ersetzen wir u[1, it] durch die exakte Dirichlet-Bedingung
    bc_left = get_exact_wave(t_val, x[1])
    du[2] = (bc_left - 2 * u[2, it] + u[3, it]) / (dx * dx)
    
    # 2. Das strikte Innere (j = 3 bis Nx-2)
    for j in 3:Nx-2
        du[j] = (u[j-1, it] - 2 * u[j, it] + u[j+1, it]) / (dx * dx)
    end
    
    # 3. Rechter Rand-Knoten (j = Nx-1)
    # Hier ersetzen wir u[Nx, it] durch die exakte Dirichlet-Bedingung
    bc_right = get_exact_wave(t_val, x[end])
    du[Nx-1] = (u[Nx-2, it] - 2 * u[Nx-1, it] + bc_right) / (dx * dx)
    
    # Die tatsächlichen Ränder du[1] und du[Nx] bleiben 0.0, 
    # da ihre Position durch die Dirichlet-Bedingung feststeht 
    # und sie nicht durch die Wellengleichung beschleunigt werden.
    return nothing
end