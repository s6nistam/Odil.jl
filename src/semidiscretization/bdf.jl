using FiniteDifferences

function precompute_bdf_weights(max_order::Int)
    weights = zeros(Float64, max_order, max_order + 1)
    
    for order in 1:max_order
        fdm = backward_fdm(order + 1, 1)
        grid_and_coefs = sort(collect(zip(fdm.grid, fdm.coefs)), by = x -> x[1], rev = true)
        for (i, (g, c)) in enumerate(grid_and_coefs)
            weights[order, i] = c
        end
    end
    
    return weights
end

function get_lhs(max_order::Int)
    bdf_weights = precompute_bdf_weights(max_order)

    lhs = (du, u, p, it) -> begin
        x, Nx, dx, t, Nt, dt = p 
        fill!(du, zero(eltype(u)))
        
        current_order = min(it, max_order)
        
        weights = bdf_weights[current_order , 1:(current_order + 1)]
        
        @inbounds for ix in 1:Nx
            val = 0.0
            for w_idx in 1:length(weights)
                time_idx = (it + 1) - (w_idx - 1)
                val += weights[w_idx] * u[LinearIndices((Nx, Nt))[ix, time_idx]]
            end
            du[ix] = val / dt[it]
        end
        return nothing
    end

    return lhs
end