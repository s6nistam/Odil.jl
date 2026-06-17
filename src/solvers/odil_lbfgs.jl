using SciMLBase, OptimizationBase, OptimizationOptimJL, ADTypes, Enzyme, LinearAlgebra

function odil_lbfgs(lhs, rhs, p_lhs, p_rhs, Nx, u_reference_vals, reference_val_indices, t; extra = nothing,  p_extra = nothing, u_iter0 = nothing, max_iterations = 10000000, autodiff = AutoEnzyme())
    Nt = length(t)
    num_unknowns = Nx * Nt
    iter = Ref(0)
    if u_iter0 === nothing
        u_iter0 = zeros(num_unknowns)
    end

    p_all = (lhs, rhs, extra, p_lhs, p_rhs, p_extra, u_reference_vals, reference_val_indices, Nx, t, Nt, iter)

    function loss(u_vec, p)
        lhs_inner, rhs_inner, extra_inner, p_lhs_inner, p_rhs_inner, p_extra_inner, u_reference_vals_inner, reference_val_indices_inner, Nx_inner, t_inner, Nt_inner, iter_inner = p
        iter_inner[] += 1
        
        l_exact = zero(eltype(u_vec))

        for (u_val, idx) in zip(u_reference_vals_inner, reference_val_indices_inner)
            l_exact += ((u_vec[idx] - u_val)/length(u_reference_vals_inner))^2
        end

        l_pde = zero(eltype(u_vec))
        
        du_rhs = zeros(eltype(u_vec), Nx_inner)
        du_lhs = zeros(eltype(u_vec), Nx_inner)
        
        for it in 1:(Nt_inner - 1)
            u_rhs = u_vec[((it - 1) * Nx_inner + 1):(it * Nx_inner)]
            fill!(du_rhs, 0.0)
            fill!(du_lhs, 0.0)
            
            t_val = t_inner[it]
            t_val_next = t_inner[it + 1]
            rhs_inner(du_rhs, u_rhs, p_rhs_inner, t_val_next)
            lhs_inner(du_lhs, u_vec, p_lhs_inner, it)
            
            for i in eachindex(du_rhs)
                l_pde += ((du_rhs[i] - du_lhs[i])/Nx_inner)^2
            end
        end

        l_extra = zero(eltype(u_vec))

        if extra_inner !== nothing && p_extra_inner !== nothing
            l_extra = extra_inner(u_vec, p_extra_inner, iter_inner[])
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

            u_current = reshape(state.u, Nx, Nt)
            plot_1d_time(x, t, u_current)
        end
        return false
    end
    
    println("Starte Lösung...")
    res = solve(prob, opt, maxiters = max_iterations, callback = callback)
    
    u_final = reshape(res.u, Nx, Nt)
    println("Optimierung beendet!")
    println("Return Code: ", res.retcode)
    return u_final
end