using SciMLBase, NonlinearSolveFirstOrder, ADTypes, Enzyme, LinearAlgebra

function odil_gauss_newton(lhs, rhs, p_lhs, p_rhs, Nx, u_reference_vals, reference_val_indices, t; max_iterations = 2, extra = nothing,  p_extra = nothing, u_iter0 = nothing, autodiff = AutoEnzyme())
    Nt = length(t)
    p_iter = Ref(0)
    if u_iter0 === nothing
        u_iter0 = zeros(Nx * Nt)
    end

    p_all = (lhs, rhs, extra, p_lhs, p_rhs, p_extra, u_reference_vals, reference_val_indices, Nx, t, Nt, p_iter)

    function operator_loss(u_vec, p)
        lhs_inner, rhs_inner, extra_inner, p_lhs_inner, p_rhs_inner, p_extra_inner, u_reference_vals_inner, reference_val_indices_inner, Nx_inner, t_inner, Nt_inner, iter_inner = p
        
        l_exact = zeros(eltype(u_vec), length(u_reference_vals_inner))

        for (i, (u_val, idx)) in enumerate(zip(u_reference_vals_inner, reference_val_indices_inner))
            l_exact[i] = (u_vec[idx] - u_val)/length(u_reference_vals_inner)
        end

        l_pde = zeros(eltype(u_vec), Nx_inner * (Nt_inner - 1))
        
        du_rhs = zeros(eltype(u_vec), Nx_inner)
        du_lhs = zeros(eltype(u_vec), Nx_inner)
        
        for it in 1:(Nt_inner - 1)
            t_val = t_inner[it]
            t_val_next = t_inner[it + 1]
            u_rhs = u_vec[((it - 1) * Nx_inner + 1):(it * Nx_inner)]
            fill!(du_rhs, 0.0)
            fill!(du_lhs, 0.0)
            
            rhs_inner(du_rhs, u_rhs, p_rhs_inner, t_val_next)
            lhs_inner(du_lhs, u_vec, p_lhs_inner, it)
            
            for i in 1:Nx_inner
                l_pde[(it - 1) * Nx_inner + i] = (du_rhs[i] - du_lhs[i])/Nx_inner
            end
        end

        l = vcat(l_exact, l_pde)

        l_extra = zeros(eltype(u_vec), Nx_inner, Nt_inner)

        if extra_inner !== nothing && p_extra_inner !== nothing
            l_extra = extra_inner(u_vec, p_extra_inner, iter_inner[])
            l = vcat(l, vec(l_extra))
        end

        return l
    end
    
    prob = NonlinearLeastSquaresProblem(operator_loss, u_iter0, p_all)
    opt = GaussNewton(autodiff = autodiff)
    # opt = LevenbergMarquardt(autodiff = autodiff, damping_initial = 0.01)

    callback = function (cache, iter)
        println("Iteration ", iter, ": Loss = ", norm(cache.fu))
        u_current = reshape(cache.u, Nx, Nt)
        # plot_1d_time(x, t, u_current)
        return false # false = Optimierung weiterlaufen lassen
    end
    
    println("Starte Lösung...")
    
    cache = init(prob, opt, maxiters = max_iterations)
    
    for iter in 1:max_iterations
        p_iter[] = iter
        step!(cache)
        
        callback(cache, iter)
        
        if cache.retcode != SciMLBase.ReturnCode.Default
            break
        end
    end
    
    # 3. Final reconstruction
    u_final = reshape(cache.u, Nx, Nt)
    
    println("Optimierung beendet!")
    println("Return Code: ", cache.retcode)
    return u_final
end