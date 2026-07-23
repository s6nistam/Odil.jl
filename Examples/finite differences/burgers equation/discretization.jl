function rhs!(du, u, p, t_val)
    x, Nx, dx, t, Nt, dt = p 
    fill!(du, zero(eltype(u)))
    
    for ix in 2:(Nx-1)
        du[ix] = - u[ix] * (u[ix] - u[ix - 1]) / dx[ix - 1]
    end
    
    return nothing
end

function lhs!(du, u, p, it)
    Nx, t, Nt, dt = p 
    fill!(du, zero(eltype(u)))
    idx = LinearIndices((Nx, Nt))

    for ix in 2:(Nx-1)
        du[ix] = (u[idx[ix, it]] - u[idx[ix, it - 1]]) / dt[it - 1]
    end
    
    return nothing
end


function extra(du, u, p, iter)
    x, Nx, dx, t, Nt, dt = p
    idx = LinearIndices((Nx, Nt))
    
    k = 0.01 * 2.0^(-iter/6.0)

    idx = LinearIndices((Nx, Nt))
    du_idx = LinearIndices((Nx - 2, Nt - 1, 2))

    for ix in 2:(Nx-1)
        for it in 1:(Nt-1)
            ux = (u[idx[ix, it + 1]] - u[idx[ix - 1, it + 1]]) / dx[ix - 1]
            ut = (u[idx[ix, it + 1]] - u[idx[ix, it]]) / dt[it]
            du[du_idx[ix - 1, it, 1]] = k * ux
            du[du_idx[ix - 1, it, 2]] = k * ut
        end
    end

    return nothing
end