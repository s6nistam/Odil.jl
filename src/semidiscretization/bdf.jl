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
        x_array, t_array = p 
        dt = it == length(t_array) ? t_array[it] - t_array[it - 1] : t_array[it + 1] - t_array[it]
        fill!(du, 0.0)
        Nx = length(x_array)
        
        current_order = min(it, max_order)
        
        weights = bdf_weights[current_order , 1:(current_order + 1)]
        
        inv_dt = 1.0 / dt
        
        @inbounds for ix in 1:Nx
            val = 0.0
            for w_idx in 1:length(weights)
                time_idx = (it + 1) - (w_idx - 1)
                val += weights[w_idx] * u[ix, time_idx]
            end
            du[ix] = val * inv_dt
        end
        return nothing
    end

    return lhs
end