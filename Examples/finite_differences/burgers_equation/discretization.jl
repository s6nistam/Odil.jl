function f!(du, u, p)
    Nx, dx = p 
    fill!(du, zero(eltype(u)))
    
    for ix in 2:(Nx-1)
        du[ix] = - u[ix] * (u[ix] - u[ix - 1]) / dx
    end
    
    return nothing
end

function timestep!(timestep_mem, u_timestep, u, t, dt, p)
    u_timestep .= u
    du = @view(timestep_mem[1:length(u_timestep)])
    du .= zero(eltype(u_timestep))
    f!(du, u_timestep, p)
    u_timestep .+= dt .* du
    return nothing
end


function extra(du, u, p, iter)
    Nx, dx, Nt, dt = p
    idx = LinearIndices((Nx, Nt))
    
    k = 0.01 * 2.0^(-iter/6.0)

    idx = LinearIndices((Nx, Nt))
    du_idx = LinearIndices((Nx - 2, Nt - 1, 2))

    for ix in 2:(Nx-1)
        for it in 1:(Nt-1)
            ux = (u[idx[ix, it + 1]] - u[idx[ix - 1, it + 1]]) / dx
            ut = (u[idx[ix, it + 1]] - u[idx[ix, it]]) / dt
            du[du_idx[ix - 1, it, 1]] = k * ux
            du[du_idx[ix - 1, it, 2]] = k * ut
        end
    end

    return nothing
end