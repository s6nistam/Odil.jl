using SciMLBase, NonlinearSolveFirstOrder, ADTypes, Enzyme, LinearAlgebra, Symbolics, SparseDiffTools, SparseArrays, SparseMatrixColorings

function odil_gauss_newton(problem::OdilProblem; max_iterations = 2, timestep_alloc_size = problem.timestep_alloc_size, extra = problem.extra, p_extra = problem.p_extra, len_extra = problem.len_extra, u_iter0 = problem.u_iter0, autodiff = AutoEnzyme(), info_prints = true, jac_sparse = nothing, colors = nothing)
    return odil_gauss_newton(problem.timestep, problem.p_timestep, problem.N_coords, problem.u_reference_vals, problem.reference_val_indices, problem.t; max_iterations = max_iterations, timestep_alloc_size = timestep_alloc_size, extra = extra, p_extra = p_extra, len_extra = len_extra, u_iter0 = u_iter0, autodiff = autodiff, problem = problem, info_prints = info_prints, jac_sparse = jac_sparse, colors = colors)
end
function odil_gauss_newton(timestep, p_timestep, Nx, u_reference_vals, reference_val_indices, t; max_iterations = 2, timestep_alloc_size = 0, extra = nothing,  p_extra = nothing, len_extra = 0, u_iter0 = nothing, autodiff = AutoEnzyme(), problem = nothing, info_prints = true, jac_sparse = nothing, colors = nothing)
    Nref = length(u_reference_vals)
    Nt = length(t)
    p_iter = Ref(0)
    if u_iter0 === nothing
        u_iter0 = zeros(Nx * Nt)
    end

    p_all = (timestep, p_timestep, timestep_alloc_size, u_reference_vals, reference_val_indices, Nref, Nx, Nt, t, extra, p_extra, len_extra, p_iter)
    resid_prototype = zeros(eltype(u_iter0), Nref + Nx * (Nt - 1) + len_extra)

    function operator_loss!(du, u_vec, p)
        timestep_inner, p_timestep_inner, timestep_alloc_size_inner, u_reference_vals_inner, reference_val_indices_inner, Nref_inner, Nx_inner, Nt_inner, t_inner, extra_inner, p_extra_inner, len_extra_inner, iter_inner = p
        
        du[1:Nref_inner] .= zero(eltype(u_vec))
        l_exact = @view(du[1:Nref_inner])

        for i in eachindex(u_reference_vals_inner)
            u_val = u_reference_vals_inner[i]
            idx = reference_val_indices_inner[i]
            l_exact[i] = (u_vec[idx] - u_val)/sqrt(Nref_inner)
        end

        du[Nref_inner + 1:Nref_inner + Nx_inner * (Nt_inner - 1)] .= zero(eltype(u_vec))
        l_pde = @view(du[Nref_inner + 1:Nref_inner + Nx_inner * (Nt_inner - 1)])

        u_timestep = zeros(eltype(u_vec), Nx_inner)
        timestep_mem = zeros(eltype(u_vec), timestep_alloc_size_inner)
        for it in 2:Nt_inner
            fill!(u_timestep, zero(eltype(u_vec)))
            fill!(timestep_mem, zero(eltype(u_vec)))
            u_it_last = @view(u_vec[(it - 2) * Nx_inner + 1:(it - 1) * Nx_inner])
            u_it = @view(u_vec[(it - 1) * Nx_inner + 1:it * Nx_inner])
            timestep_inner(timestep_mem, u_timestep, u_it_last, t_inner[it - 1], t_inner[it] - t_inner[it - 1], p_timestep_inner)
            l_pde[(it - 2) * Nx_inner + 1: (it - 1) * Nx_inner] .= (u_timestep - u_it)/sqrt(Nx_inner * Nt_inner)
        end

        du[Nref_inner + Nx_inner * (Nt_inner - 1) + 1:end] .= zero(eltype(u_vec))
        l_extra = @view(du[Nref_inner + Nx_inner * (Nt_inner - 1) + 1:end])

        if extra_inner !== nothing && p_extra_inner !== nothing
            extra_inner(l_extra, u_vec, p_extra_inner, iter_inner[])
            l_extra ./= sqrt(len_extra_inner)
        end

        return nothing
    end

    if jac_sparse === nothing
        if info_prints
            println("Computing Jacobian sparsity pattern...")
        end
        jac_sparse = get_jac_sparse(timestep, p_timestep, timestep_alloc_size, t, Nref, Nx, Nt, reference_val_indices, extra, p_extra, len_extra, u_iter0)
    end

    if colors === nothing
        colors = matrix_colors(jac_sparse)
    end
    
    optf = NonlinearFunction(operator_loss!, resid_prototype = resid_prototype, jac_prototype = jac_sparse, colorvec = colors)
    prob = NonlinearLeastSquaresProblem(optf, u_iter0, p_all)
    opt = GaussNewton(autodiff = autodiff)
    # opt = LevenbergMarquardt(autodiff = autodiff, damping_initial = 0.01)

    callback = function (cache, iter)
        println("Iteration ", iter, ": Loss = ", norm(cache.fu), " descent direction = ", norm(get_du(cache.descent_cache)))
        
        plot(problem, cache.u)
        return false 
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