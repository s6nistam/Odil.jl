function lhs!(du, u, p, it)
    x, t = p 
    fill!(du, 0.0)
    Nx = length(x)
    Nt = length(t)
    idx = LinearIndices((Nx, Nt))
    dt = t[it] - t[it - 1]
    if it == Nt
        for ix in 1:Nx
            du[ix...] = (u[idx[ix, it]] - u[idx[ix, it - 1]]) / dt
        end
    else
        for ix in 1:Nx
            du[ix...] = (u[idx[ix, it + 1]] - u[idx[ix, it - 1]]) / (2 * dt)
        end
    end
    
    return nothing
end