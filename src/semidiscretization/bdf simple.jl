using FiniteDifferences

function get_lhs(max_order::Int)
    fdm_pre = [backward_fdm(order + 1, 1) for order in 1:max_order]
    lhs = (du, u, p, it) -> begin
        x, Nx, dx, t, Nt, dt = p 
        fill!(du, zero(eltype(u)))
        idx = LinearIndices((Nx, Nt))

        current_order = min(it - 1, max_order)
        fdm = fdm_pre[current_order]
        indices = fdm.grid
        coefs = fdm.coefs
        
        @inbounds for ix in 1:Nx
            for i in eachindex(indices)
                du[ix] += coefs[i] * u[idx[ix, it + indices[i]]]
            end
            du[ix] /= dt[it]
        end
        return nothing
    end

    return lhs
end