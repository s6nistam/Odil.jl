function lhs!(du, u, p, it)
    x, t = p 
    fill!(du, 0.0)
    Nx = length(x)
    Nt = length(t)
    idx = LinearIndices((Nx, Nt))
    dt = it == length(t) ? t[it] - t[it - 1] : t[it + 1] - t[it]

    for ix in 1:Nx
        du[ix...] = (u[idx[ix, it + 1]] - u[idx[ix, it]]) / dt
    end
    
    return nothing
end