function rhs!(du, u, p, it)
    dx, x_array, t_array = p 
    fill!(du, 0.0)
    Nx = length(x_array)
    Nt = length(t_array)
    t_val = t_array[it]
    
    bc_left = get_exact_wave(x_array[1], t_val)
    bc_right = get_exact_wave(x_array[Nx], t_val)
    
    if it == 1
        du[2] = Nt * (bc_left - 2 * u[2, it] + u[3, it]) / (dx^2)
    
        for j in 3:Nx-2
            du[j] = Nt * (u[j-1, it] - 2 * u[j, it] + u[j+1, it]) / (dx^2)
        end
        
        du[Nx-1] = Nt * (u[Nx-2, it] - 2 * u[Nx-1, it] + bc_right) / (dx^2)
    else
        du[2] = (bc_left - 2 * u[2, it] + u[3, it]) / (dx^2)
        
        for j in 3:Nx-2
            du[j] = (u[j-1, it] - 2 * u[j, it] + u[j+1, it]) / (dx^2)
        end
        
        du[Nx-1] = (u[Nx-2, it] - 2 * u[Nx-1, it] + bc_right) / (dx^2)
    end
    
    return nothing
end