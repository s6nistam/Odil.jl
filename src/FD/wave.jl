function rhs!(du, u, it, p)
    dx = p 
    
    fill!(du, 0.0)
    
    for j in 2:size(u, 1)-1
        du[j] = (u[j-1, it] - 2 * u[j, it] + u[j+1, it]) / (dx * dx)
    end

    t_val = t[it] 
    
    u[1, it] = get_exact_wave(0.0, t_val)
    u[end, it] = get_exact_wave(1.0, t_val)
    
    return nothing
end