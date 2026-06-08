using SciMLBase, OptimizationBase, OptimizationOptimJL, ADTypes, Enzyme, LinearAlgebra

function odil_lbfgs(lhs, rhs, p_lhs, p_rhs, u_size_x, u_fixed_vals, x_fixed_indicies, t_fixed_indicies, Nt = 1000, extra = nothing,  p_extra = nothing)
    space_dims = u_size_x
    num_cells = prod(space_dims)
    num_unknowns = num_cells * Nt
    iter = Ref(0)
    u_iter0 = zeros(num_unknowns)

    p_all = (lhs, rhs, extra, p_lhs, p_rhs, p_extra, u_fixed_vals, x_fixed_indicies, t_fixed_indicies, space_dims, num_unknowns, Nt, iter)

    function loss(u_vec, p)
        lhs_inner, rhs_inner, extra_inner, p_lhs_inner, p_rhs_inner, p_extra_inner, u_fixed_vals_inner, x_fixed_indicies_inner, t_fixed_indicies_inner, space_dims_inner, num_unknowns_inner, Nt_inner, iter_inner = p
        iter_inner[] += 1

        u_local = reshape(u_vec, space_dims_inner..., Nt_inner)
        
        l_exact = zero(eltype(u_vec))

        for (u_val, ix, it) in zip(u_fixed_vals_inner, x_fixed_indicies_inner, t_fixed_indicies_inner)
            l_exact += (num_unknowns_inner/length(u_fixed_vals_inner))^2 *(u_local[ix..., it] - u_val)^2
        end

        l_pde = zero(eltype(u_vec))
        
        du_rhs = similar(u_local)
        du_lhs = similar(u_local)
        
        for it in 1:(Nt_inner - 1)
            fill!(du_rhs, 0.0)
            fill!(du_lhs, 0.0)
            
            rhs_inner(du_rhs, u_local, p_rhs_inner, it)
            lhs_inner(du_lhs, u_local, p_lhs_inner, it)
            
            for i in eachindex(du_rhs)
                l_pde += (du_rhs[i] - du_lhs[i])^2
            end
        end

        l_extra = zero(eltype(u_vec))

        if extra_inner !== nothing && p_extra_inner !== nothing
            for it in 1:(Nt_inner - 1)
                du_extra = similar(u_local)
                fill!(du_extra, 0.0)
                extra_inner(du_extra, u_local, p_extra_inner, it, iter_inner[])
                for i in eachindex(du_extra)
                    l_extra += abs(du_extra[i])
                end
            end
        end

        l = l_exact + l_pde + l_extra

        return l
    end
    
    optf = OptimizationFunction(loss, ADTypes.AutoEnzyme())
    prob = OptimizationProblem(optf, u_iter0, p_all) 
    # opt = LBFGS(m = 50)
    opt = LBFGS()

    callback = function (state, l)
        if state.iter % 1000 == 0 || state.iter == 1
            println("Iteration ", state.iter, ": Loss = ", l)

            u_current = reshape(state.u, space_dims..., Nt)
            # plot_comparison(x, t, u_exact, u_current)
            plot_2d(x, t, u_current)
        end
        return false
    end
    
    println("Starte Lösung...")
    res = solve(prob, opt, maxiters = 10000000, callback = callback)
    
    u_final = reshape(res.u, space_dims..., Nt)
    println("Optimierung beendet!")
    println("Return Code: ", res.retcode)
    return u_final
end