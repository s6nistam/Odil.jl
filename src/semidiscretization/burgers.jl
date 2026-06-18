function rhs!(du, u, p, t_val)
    x, t = p 
    fill!(du, 0.0)
    Nx = length(x)
    Nt = length(t)
    
    for ix in 2:(Nx-1)
        dx = x[ix + 1] - x[ix]
        du[ix] = - u[ix] * (u[ix] - u[ix - 1]) / dx
    end
    
    return nothing
end

function lhs!(du, u, p, it)
    x, t = p 
    fill!(du, 0.0)
    Nx = length(x)
    Nt = length(t)
    dt = it == length(t) ? t[it] - t[it - 1] : t[it + 1] - t[it]
    idx = LinearIndices((Nx, Nt))

    for ix in 2:(Nx-1)
        du[ix] = (u[idx[ix, it + 1]] - u[idx[ix, it]]) / dt
    end
    
    return nothing
end


function extra(du, u, p, iter)
    x, t = p
    Nx = length(x)
    Nt = length(t)
    idx = LinearIndices((Nx, Nt))
    
    k = 0.01 * 2.0^(-iter/6.0)

    idx = LinearIndices((Nx - 2, Nt - 1, 2))

    for ix in 2:(Nx-1)
        dx = x[ix + 1] - x[ix]
        for it in 1:(Nt-1)
            dt = t[it + 1] - t[it]
            ux = (u[idx[ix, it + 1]] - u[idx[ix - 1, it + 1]]) / dx
            ut = (u[idx[ix, it + 1]] - u[idx[ix, it]]) / dt
            du[idx[ix - 1, it, 1]] = k * ux
            du[idx[ix - 1, it, 2]] = k * ut
        end
    end

    return nothing
end