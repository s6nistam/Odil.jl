using SciMLBase, NonlinearSolveFirstOrder, ADTypes, Enzyme, LinearAlgebra

function odil_gauss_newton(problem::OdilProblem; max_iterations = 2, extra = problem.extra, p_extra = problem.p_extra, len_extra = problem.len_extra, u_iter0 = problem.u_iter0, autodiff = AutoEnzyme(), info_prints = true)
    return odil_gauss_newton(problem.lhs, problem.rhs, problem.p_lhs, problem.p_rhs, problem.N_coords, problem.u_reference_vals, problem.reference_val_indices, problem.t; max_iterations = max_iterations, extra = extra, p_extra = p_extra, len_extra = len_extra, u_iter0 = u_iter0, autodiff = autodiff, problem = problem, info_prints = info_prints)
end
function odil_gauss_newton(lhs, rhs, p_lhs, p_rhs, Nx, u_reference_vals, reference_val_indices, t; max_iterations = 2, extra = nothing,  p_extra = nothing, len_extra = 0, u_iter0 = nothing, autodiff = AutoEnzyme(), problem = nothing, info_prints = true)
    Nref = length(u_reference_vals)
    Nt = length(t)
    p_iter = Ref(0)
    if u_iter0 === nothing
        u_iter0 = zeros(Nx * Nt)
    end

    p_all = (lhs, rhs, extra, p_lhs, p_rhs, p_extra, u_reference_vals, reference_val_indices, Nref, Nx, t, Nt, p_iter)
    resid_prototype = zeros(eltype(u_iter0), Nref + Nx * (Nt - 1) + len_extra)

    function operator_loss!(du, u_vec, p)
        lhs_inner, rhs_inner, extra_inner, p_lhs_inner, p_rhs_inner, p_extra_inner, u_reference_vals_inner, reference_val_indices_inner, Nref_inner, Nx_inner, t_inner, Nt_inner, iter_inner = p
        
        du[1:Nref_inner] .= zero(eltype(u_vec))
        l_exact = @view(du[1:Nref_inner])

        for i in eachindex(u_reference_vals_inner)
            u_val = u_reference_vals_inner[i]
            idx = reference_val_indices_inner[i]
            l_exact[i] = (u_vec[idx] - u_val)/sqrt(Nref_inner)
        end

        du[Nref_inner + 1:Nref_inner + Nx_inner * (Nt_inner - 1)] .= zero(eltype(u_vec))
        l_pde = @view(du[Nref_inner + 1:Nref_inner + Nx_inner * (Nt_inner - 1)])

        du_rhs = zeros(eltype(u_vec), Nx_inner)
        du_lhs = zeros(eltype(u_vec), Nx_inner)

        for it in 2:Nt_inner
            fill!(du_rhs, 0.0)
            fill!(du_lhs, 0.0)

            u_rhs = @view(u_vec[((it - 2) * Nx_inner + 1):((it - 1) * Nx_inner)])
            # u_rhs = u_vec[(it * Nx_inner + 1):((it + 1) * Nx_inner)]

            t_val = t_inner[it - 1]
            # t_val_next = t_inner[it + 1]
            rhs_inner(du_rhs, u_rhs, p_rhs_inner, t_val)
            # rhs_inner(du_rhs, u_rhs, p_rhs_inner, t_val_next)
            lhs_inner(du_lhs, u_vec, p_lhs_inner, it)
            
            for i in 1:Nx_inner
                l_pde[(it - 2) * Nx_inner + i] = (du_rhs[i] - du_lhs[i])/sqrt(Nx_inner * (Nt_inner - 1))
            end
        end

        du[Nref_inner + Nx_inner * (Nt_inner - 1) + 1:end] .= zero(eltype(u_vec))
        l_extra = @view(du[Nref_inner + Nx_inner * (Nt_inner - 1) + 1:end])

        if extra_inner !== nothing && p_extra_inner !== nothing
            extra_inner(l_extra, u_vec, p_extra_inner, iter_inner[])
            l_extra ./= sqrt(len_extra)
        end

        return nothing
    end
    
    optf = NonlinearFunction(operator_loss!, resid_prototype = resid_prototype)
    prob = NonlinearLeastSquaresProblem(optf, u_iter0, p_all)
    opt = GaussNewton(autodiff = autodiff)
    # opt = LevenbergMarquardt(autodiff = autodiff, damping_initial = 0.01)

    callback = function (cache, iter)
        println("Iteration ", iter, ": Loss = ", norm(cache.fu), " descent direction = ", norm(get_du(cache.descent_cache)))
        
        plot(problem, cache.u)
        # plot_1d_time(x, t, u_current)
        return false # false = Optimierung weiterlaufen lassen
    end
    
    if info_prints
        println("Starte Lösung...")
    end
    
    cache = init(prob, opt, maxiters = max_iterations)
    
    for iter in 1:max_iterations
        p_iter[] = iter
        step!(cache)
        
        if iter % 50 == 0 || iter == 1
            callback(cache, iter)
        end
        
        if cache.retcode != SciMLBase.ReturnCode.Default
            break
        end
    end
    
    if info_prints
        println("Optimierung beendet!")
        println("Return Code: ", cache.retcode)
    end
    return cache.u
end