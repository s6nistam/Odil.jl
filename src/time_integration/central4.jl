function lhs!(du, u, p, it)
    Nx, t, Nt, dt = p 
    fill!(du, 0.0)
    idx = LinearIndices((Nx, Nt))
    if it == 2
        for ix in 1:Nx
            du[ix] = (u[idx[ix, it]] - u[idx[ix, it - 1]]) / dt[it - 1]
        end
    elseif it == Nt
        for ix in 1:Nx
            du[ix] = (u[idx[ix, it]] - u[idx[ix, it - 1]]) / dt[it - 1]
        end
    elseif it == Nt - 1 || it == Nt - 2
        for ix in 1:Nx
            du[ix] = (u[idx[ix, it + 1]] - u[idx[ix, it - 1]]) / (2 * dt[it - 1])
        end
    else 
        for ix in 1:Nx
            du[ix] = (- u[idx[ix, it + 2]] + 8 * u[idx[ix, it + 1]] - 8 * u[idx[ix, it - 1]] + u[idx[ix, it - 2]]) / (12 * dt[it - 1])
        end
    end
    
    return nothing
end