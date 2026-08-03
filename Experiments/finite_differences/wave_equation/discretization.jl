function f!(du, u, p, t_val)
    x, Nx, dx = p 
    fill!(du, 0.0)
    idx = LinearIndices((Nx, 2))
    
    du[idx[:, 1]] .= u[idx[:, 2]]
    
    for ix in 2:(Nx - 1)
        du[idx[ix, 2]] = (u[idx[ix - 1, 1]] - 2 * u[idx[ix, 1]] + u[idx[ix + 1, 1]]) / (dx^2)
    end

    u_l = get_exact_wave(x[1] - dx, t_val)
    u_r = get_exact_wave(x[Nx] + dx, t_val)

    du[idx[1, 2]] = (u_l - 2 * u[idx[1, 1]] + u[idx[2, 1]]) / (dx^2)
    du[idx[Nx, 2]] = (u[idx[Nx - 1, 1]] - 2 * u[idx[Nx, 1]] + u_r) / (dx^2)

    return nothing
end

function timestep!(timestep_mem, u_timestep, u, t, dt, p)
    x, Nx, dx = p
    u_timestep .= u
    du = @view(timestep_mem[1:length(u_timestep)])
    du .= zero(eltype(u_timestep))
    
    f!(du, u_timestep, p, t)
    
    idx = LinearIndices((Nx, 2))
    
    u_timestep[idx[:, 2]] .+= dt .* du[idx[:, 2]]
    u_timestep[idx[:, 1]] .+= dt .* u_timestep[idx[:, 2]]
    
    return nothing
end