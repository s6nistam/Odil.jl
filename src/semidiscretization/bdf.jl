using FiniteDifferences

function get_lhs(max_order::Int)
    coef_matrix = zeros(max_order, max_order + 1)
    for order in 1:max_order
        c = backward_fdm(order + 1, 1).coefs
        for i in 1:length(c)
            coef_matrix[order, i] = c[i]
        end
    end
    lhs = (du, u, p, it) -> begin
        x, Nx, dx, t, Nt, dt = p 
        dt = dt[1]
        current_order = min(it - 1, max_order)
        
        @inbounds for ix in 1:Nx
            val = zero(eltype(u))
            for i in -current_order:0
                val += coef_matrix[current_order, i + current_order + 1] * u[(it + i - 1) * Nx + ix]
            end
            du[ix] = val / dt
        end
        return nothing
    end

    return lhs
end