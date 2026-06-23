function lhs!(du, u, p, it)
    Nx, t, Nt, dt = p 
    fill!(du, 0.0)
    idx = LinearIndices((Nx, Nt))

    for ix in 1:Nx
        du[ix] = (u[idx[ix, it + 1]] - u[idx[ix, it]]) / dt[it]
    end
    
    return nothing
end