function rhs!(du, u, p, it)
    x, t = p 
    fill!(du, 0.0)
    Nx = length(x)
    Nt = length(t)
    t_val = t[it]
    
    bc_left = get_exact_wave(x[1], t_val)
    bc_right = get_exact_wave(x[Nx], t_val)
    
    if it == 1
        dx = x[2] - x[1]
        du[2] = Nt * (bc_left - 2 * u[2] + u[3]) / (dx^2)
    
        for ix in 3:Nx-2
            dx = x[ix + 1] - x[ix]
            du[ix] = Nt * (u[ix-1] - 2 * u[ix] + u[ix+1]) / (dx^2)
        end
        
        dx = x[Nx] - x[Nx - 1]
        du[Nx-1] = Nt * (u[Nx-2] - 2 * u[Nx-1] + bc_right) / (dx^2)
    else
        dx = x[2] - x[1]
        du[2] = (bc_left - 2 * u[2] + u[3]) / (dx^2)
        
        for ix in 3:Nx-2
            dx = x[ix + 1] - x[ix]
            du[ix] = (u[ix-1] - 2 * u[ix] + u[ix+1]) / (dx^2)
        end
        
        dx = x[Nx] - x[Nx - 1]
        du[Nx-1] = (u[Nx-2] - 2 * u[Nx-1] + bc_right) / (dx^2)
    end
    
    return nothing
end

function lhs!(du, u, p, it)
    x, t = p 
    fill!(du, 0.0)
    Nx = length(x)
    Nt = length(t)
    
    if it == 1
        dt = t[2] - t[1]
        t0 = t[1]
        penalty_vel = Nt 
        
        for j in 2:Nx-1
            v0 = get_exact_wave_velocity(x[j], t0)
            du[j] = penalty_vel * (u[j, 2] - u[j, 1] - dt * v0) / dt^2
        end
    elseif it < size(u, ndims(u))
        dt = t[it + 1] - t[it]
        for j in 2:Nx-1
            du[j] = (u[j, it+1] - 2 * u[j, it] + u[j, it-1]) / dt^2
        end
    end

    # penalty_bc = Nx/2
    # du[1]  = penalty_bc * (u[1, it + 1] - get_exact_wave(x[1], t[it + 1]))
    # du[Nx] = penalty_bc * (u[Nx, it + 1] - get_exact_wave(x[Nx], t[it + 1]))
    
    return nothing
end