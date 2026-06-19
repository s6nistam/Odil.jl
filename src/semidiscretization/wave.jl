function rhs!(du, u, p, t_val)
    x, Nx, dx, t, Nt, dt = p 
    fill!(du, 0.0)
    
    bc_left = get_exact_wave(x[1], t_val)
    bc_right = get_exact_wave(x[Nx], t_val)
    
    if t_val == t[1]
        du[2] = Nt * (bc_left - 2 * u[2] + u[3]) / (dx[1]^2)
    
        for ix in 3:Nx-2
            du[ix] = Nt * (u[ix-1] - 2 * u[ix] + u[ix+1]) / (dx[ix]^2)
        end
        
        du[Nx-1] = Nt * (u[Nx-2] - 2 * u[Nx-1] + bc_right) / (dx[Nx-1]^2)
    else
        du[2] = (bc_left - 2 * u[2] + u[3]) / (dx[1]^2)
        
        for ix in 3:Nx-2
            du[ix] = (u[ix-1] - 2 * u[ix] + u[ix+1]) / (dx[ix]^2)
        end
    
        du[Nx-1] = (u[Nx-2] - 2 * u[Nx-1] + bc_right) / (dx[Nx-1]^2)
    end
    
    return nothing
end

function lhs!(du, u, p, it)
    x, Nx, dx, t, Nt, dt = p 
    fill!(du, 0.0)
    idx = LinearIndices((Nx, Nt))
    
    if it == 1
        t0 = t[1]
        penalty_vel = Nt 
        
        for ix in 2:Nx-1
            v0 = get_exact_wave_velocity(x[ix], t0)
            du[ix] = penalty_vel * (u[idx[ix, 2]] - u[idx[ix, 1]] - dt[1] * v0) / dt[1]^2
        end
    elseif it < size(u, ndims(u))
        for ix in 2:Nx-1
            du[ix] = (u[idx[ix, it+1]] - 2 * u[idx[ix, it]] + u[idx[ix, it-1]]) / dt[it]^2
        end
    end

    # penalty_bc = Nx/2
    # du[1]  = penalty_bc * (u[idx[1, it + 1]] - get_exact_wave(x[1], t[it + 1]))
    # du[idx[Nx, it + 1]] = penalty_bc * (u[idx[Nx, it + 1]] - get_exact_wave(x[Nx], t[it + 1]))
    
    return nothing
end