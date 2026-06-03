function rhs!(du, u, p, it)
    dx, x_array, t_array = p 
    fill!(du, 0.0)
    Nx = size(u, 1)
    t_val = t_array[it]
    
    bc_left = get_exact_wave(x_array[1], t_val)
    bc_right = get_exact_wave(x_array[Nx], t_val)
    
    # --- 1. DAS INNERE (Semidiskretisierung) ---
    if Nx >= 3
        # NEU: Wenn it=1, multiplizieren wir auch die rechte Seite mit penalty_vel
        mult = (it == 1) ? 10 : 1.0
        
        du[2] = mult * (bc_left - 2 * u[2, it] + u[3, it]) / (dx^2)
        
        for j in 3:Nx-2
            du[j] = mult * (u[j-1, it] - 2 * u[j, it] + u[j+1, it]) / (dx^2)
        end
        
        du[Nx-1] = mult * (u[Nx-2, it] - 2 * u[Nx-1, it] + bc_right) / (dx^2)
    end
    
    # --- 2. DIE RÄNDER ---
    penalty_bc = 10
    du[1]  = penalty_bc * bc_left
    du[Nx] = penalty_bc * bc_right
    
    return nothing
end