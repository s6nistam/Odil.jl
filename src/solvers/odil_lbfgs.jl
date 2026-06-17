using SciMLBase, OptimizationBase, OptimizationOptimJL, ADTypes, Enzyme, LinearAlgebra

function odil_lbfgs(lhs, rhs, p_lhs, p_rhs, u_iter0_size, u_fixed_vals, x_fixed_indicies, t_fixed_indicies, t; extra = nothing,  p_extra = nothing, u_iter0 = nothing, max_iterations = 10000000, autodiff = AutoEnzyme())
    space_dims = u_iter0_size
    num_cells = prod(space_dims)
    Nt = length(t)
    num_unknowns = num_cells * Nt
    iter = Ref(0)
    if u_iter0 === nothing
        u_iter0 = zeros(num_unknowns)
    end

    p_all = (lhs, rhs, extra, p_lhs, p_rhs, p_extra, u_fixed_vals, x_fixed_indicies, t_fixed_indicies, space_dims, num_cells, num_unknowns, t, Nt, iter)

    function loss(u_vec, p)
        lhs_inner, rhs_inner, extra_inner, p_lhs_inner, p_rhs_inner, p_extra_inner, u_fixed_vals_inner, x_fixed_indicies_inner, t_fixed_indicies_inner, space_dims_inner, num_cells_inner, num_unknowns_inner, t_inner, Nt_inner, iter_inner = p
        iter_inner[] += 1

        u_local = reshape(u_vec, space_dims_inner..., Nt_inner)
        
        l_exact = zero(eltype(u_vec))

        for (u_val, ix, it) in zip(u_fixed_vals_inner, x_fixed_indicies_inner, t_fixed_indicies_inner)
            l_exact += (num_unknowns_inner/length(u_fixed_vals_inner))^2 *(u_local[ix..., it] - u_val)^2
        end

        l_pde = zero(eltype(u_vec))
        
        du_rhs = zeros(eltype(u_vec), num_cells_inner)
        du_lhs = zeros(eltype(u_vec), num_cells_inner)
        
        for it in 1:(Nt_inner - 1)
            u_rhs = vec(selectdim(u_local, length(space_dims_inner) + 1, it))
            fill!(du_rhs, 0.0)
            fill!(du_lhs, 0.0)
            
            t_val = t_inner[it]
            rhs_inner(du_rhs, u_rhs, p_rhs_inner, t_val)
            lhs_inner(du_lhs, u_local, p_lhs_inner, it)
            
            for i in eachindex(du_rhs)
                l_pde += (du_rhs[i] - du_lhs[i])^2
            end
        end

        l_extra = zero(eltype(u_vec))

        if extra_inner !== nothing && p_extra_inner !== nothing
            l_extra = extra_inner(u_local, p_extra_inner, iter_inner[])
            l_extra = sum(l_extra.^2)
        end

        l = l_exact + l_pde + l_extra

        return l
    end
    
    optf = OptimizationFunction(loss, autodiff)
    prob = OptimizationProblem(optf, u_iter0, p_all) 
    # opt = LBFGS(m = 50)
    opt = LBFGS()

    callback = function (state, l)
        if state.iter % 1000 == 0 || state.iter == 1
            println("Iteration ", state.iter, ": Loss = ", l)

            # u_current = reshape(state.u, space_dims..., Nt)
            # plot_1d_time_comparison(x, t, u_exact, u_current)
        end
        return false
    end
    
    println("Starte Lösung...")
    res = solve(prob, opt, maxiters = max_iterations, callback = callback)
    
    u_final = reshape(res.u, space_dims..., Nt)
    println("Optimierung beendet!")
    println("Return Code: ", res.retcode)
    return u_final
end