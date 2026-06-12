function rhs!(du, u, p, it)
    x, t = p 
    fill!(du, 0.0)
    Nx = length(x)
    Nt = length(t)
    t_val = t[it]

    
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

    for ix in 2:(Nx-1)
        du[ix] = (u[ix, it + 1] - u[ix, it]) / dt
    end
    
    return nothing
end


function extra(u, p, iter)
    x, t = p
    Nx = length(x)
    Nt = length(t)
    
    k = 0.01 * 2.0^(-iter/6.0)

    res = zeros(Nx - 2, Nt - 1, 2)

    for ix in 2:(Nx-1)
        dx = x[ix + 1] - x[ix]
        for it in 1:(Nt-1)
            dt = t[it + 1] - t[it]
            ux = (u[ix, it + 1] - u[ix - 1, it + 1]) / dx
            ut = (u[ix, it + 1] - u[ix, it]) / dt
            res[ix - 1, it, 1] = k * ux
            res[ix - 1, it, 2] = k * ut
        end
    end

    return vec(res)
end