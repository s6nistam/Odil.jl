function lhs!(du, u, p, it)
    x, t = p 
    fill!(du, 0.0)
    Nx = length(x)
    Nt = length(t)
    idx = LinearIndices((Nx, Nt))
    dt = t[it] - t[it - 1]

    for ix in 1:Nx
        du[ix] = (u[idx[ix, it]] - u[idx[ix, it - 1]]) / dt
    end
    
    return nothing
end