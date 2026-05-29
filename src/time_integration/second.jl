function solve_d2dt2_central!(u, rhs, dx, dt)
    for it in 3:size(u, 2)
        @info "Step ", it
        _rhs = rhs(u, it-1, dx)
        for j in 2:size(u, 1)-1
            u[j ,it] = 2 * u[j, it-1] .- u[j, it-2] +
                       (dt * dt) .* _rhs[j]
        end
    end
end
