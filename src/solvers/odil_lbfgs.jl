using SciMLBase, OptimizationBase, OptimizationOptimJL, ADTypes, Enzyme, LinearAlgebra

function odil_lbfgs(problem::OdilProblem; max_iterations = 100000, extra = problem.extra, p_extra = problem.p_extra, len_extra = problem.len_extra, u_iter0 = problem.u_iter0, autodiff = AutoEnzyme(), info_prints = true)
    return odil_lbfgs(problem.lhs, problem.rhs, problem.p_lhs, problem.p_rhs, problem.N_coords, problem.u_reference_vals, problem.reference_val_indices, problem.t; max_iterations = max_iterations, extra = extra, p_extra = p_extra, len_extra = len_extra, u_iter0 = u_iter0, autodiff = autodiff, problem = problem, info_prints = info_prints)
end

function odil_lbfgs(lhs, rhs, p_lhs, p_rhs, Nx, u_reference_vals, reference_val_indices, t; extra = nothing,  p_extra = nothing, len_extra = 0, u_iter0 = nothing, max_iterations = 100000, autodiff = AutoEnzyme(), problem = nothing, info_prints = true)
    Nref = length(u_reference_vals)
    Nt = length(t)
    num_unknowns = Nx * Nt
    iter = Ref(0)
    if u_iter0 === nothing
        u_iter0 = zeros(num_unknowns)
    end

    p_all = (lhs, rhs, extra, p_lhs, p_rhs, p_extra, u_reference_vals, reference_val_indices, Nref, Nx, t, Nt, iter)

    function loss(u_vec, p)
        lhs_inner, rhs_inner, extra_inner, p_lhs_inner, p_rhs_inner, p_extra_inner, u_reference_vals_inner, reference_val_indices_inner, Nref_inner, Nx_inner, t_inner, Nt_inner, iter_inner = p
        l = zero(eltype(u_vec))
        l_exact = zero(eltype(u_vec))
        l_pde = zero(eltype(u_vec))

        for i in eachindex(u_reference_vals_inner)
            u_val = u_reference_vals_inner[i]
            idx = reference_val_indices_inner[i]
            l_exact += ((u_vec[idx] - u_val)^2)/Nref_inner
        end
        
        du_rhs = zeros(eltype(u_vec), Nx_inner)
        du_lhs = zeros(eltype(u_vec), Nx_inner)
        
        for it in 2:Nt_inner
            fill!(du_rhs, 0.0)
            fill!(du_lhs, 0.0)

            u_rhs = @view(u_vec[((it - 2) * Nx_inner + 1):((it - 1) * Nx_inner)])
            
            t_val = t_inner[it - 1]
            # t_val_next = t_inner[it + 1]
            rhs_inner(du_rhs, u_rhs, p_rhs_inner, t_val)
            lhs_inner(du_lhs, u_vec, p_lhs_inner, it)
            
            for i in eachindex(du_rhs)
                l_pde += ((du_rhs[i] - du_lhs[i])^2)/(Nx_inner * (Nt_inner - 1))
            end
        end

        l = l_exact + l_pde

        if extra_inner !== nothing && p_extra_inner !== nothing
            du_extra = zeros(eltype(u_vec), len_extra)
            extra_inner(du_extra, u_vec, p_extra_inner, iter_inner[])
            l_extra = sum(du_extra.^2)/(len_extra)
            l += l_extra
        end

        return l
    end
    
    optf = OptimizationFunction(loss, autodiff)
    prob = OptimizationProblem(optf, u_iter0, p_all) 
    opt = LBFGS(m = 50)
    # opt = LBFGS()

    callback = function (state, l)
        iter = state.iter
        # if iter == 1
        #     println("Gradient= ", state.grad)
        # end
        if state.iter % 10 == 0 || state.iter == 1
            println("Iteration ", state.iter, ": Loss = ", l, " Loss gradient = ", norm(state.grad))

            plot(problem, state.u)
            # u_current = reshape(state.u, Nx, Nt)
            # plot_1d_time(x, t, u_current)
        end
        return false
    end
    
    if info_prints
        println("Starte Lösung...")
    end
    res = solve(prob, opt, maxiters = max_iterations, callback = callback)
    
    if info_prints
        println("Optimierung beendet!")
        println("Return Code: ", res.retcode)
    end
    return res.u
end