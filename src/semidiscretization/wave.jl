function rhs!(du, u, p, t_val)
    x, Nx, dx, t, Nt, dt = p 
    fill!(du, 0.0)
    idx = LinearIndices((Nx, 2))
    
    du[idx[:, 1]] .= u[idx[:, 2]]
    for ix in 2:(Nx - 1)
        du[idx[ix, 2]] = (u[idx[ix - 1, 1]] - 2 * u[idx[ix, 1]] + u[idx[ix + 1, 1]]) / (dx[ix]^2)
    end

    u_l = (u[idx[2, 1]] - 6*u[idx[1, 1]] + 8*get_exact_wave(x[1] - dx[1], t_val))/3
    u_r = (u[idx[Nx - 1, 1]] - 6 * u[idx[Nx, 1]] + 8 * get_exact_wave(x[Nx] + dx[Nx - 1], t_val)) / 3

    du[idx[1, 2]] = (u_l - 2 * u[idx[1, 1]] + u[idx[2, 1]]) / (dx[1]^2)
    du[idx[Nx, 2]] = (u[idx[Nx - 1, 1]] - 2 * u[idx[Nx, 1]] + u_r) / (dx[Nx - 1]^2)

    return nothing
end

function lhs!(du, u, p, it)
    x, Nx, dx, t, Nt, dt = p 
    fill!(du, zero(eltype(u)))
    idx = LinearIndices((Nx, 2, Nt))
    idx_x = LinearIndices((Nx, 2))
    for ix in 1:Nx
        du[idx_x[ix, 1]] = (u[idx[ix, 1, it]] - u[idx[ix, 1, it - 1]]) / dt[it - 1]
    end

    for ix in 1:Nx
        du[idx_x[ix, 2]] = (u[idx[ix, 2, it]] - u[idx[ix, 2, it - 1]]) / dt[it - 1]
    end
    return nothing
end