function lhs!(du, u, p, it)
    Nx, t, Nt, dt = p 
    fill!(du, 0.0)
    idx = LinearIndices((Nx, Nt))
    if it == Nt
        for ix in 1:Nx
            du[ix] = (u[idx[ix, it]] - u[idx[ix, it - 1]]) / dt[it - 1]
        end
    else
        for ix in 1:Nx
            du[ix] = (u[idx[ix, it + 1]] - u[idx[ix, it - 1]]) / (2 * dt[it - 1])
        end
    end
    
    return nothing
end