using SciMLBase, NonlinearSolveFirstOrder, ADTypes, Enzyme, LinearAlgebra

function odil_gauss_newton(lhs, rhs, p_lhs, p_rhs, u_iter0_size, u_fixed_vals, x_fixed_indicies, t_fixed_indicies, t; max_iterations = 2, extra = nothing,  p_extra = nothing, u_iter0 = nothing, autodiff = AutoEnzyme())
    space_dims = u_iter0_size
    num_cells = prod(space_dims)
    Nt = length(t)
    num_unknowns = num_cells * Nt
    p_iter = Ref(0)
    if u_iter0 === nothing
        u_iter0 = zeros(num_unknowns)
    end

    p_all = (lhs, rhs, extra, p_lhs, p_rhs, p_extra, u_fixed_vals, x_fixed_indicies, t_fixed_indicies, space_dims, num_cells, num_unknowns, t, Nt, p_iter)

    function operator_loss(u_vec, p)
        lhs_inner, rhs_inner, extra_inner, p_lhs_inner, p_rhs_inner, p_extra_inner, u_fixed_vals_inner, x_fixed_indicies_inner, t_fixed_indicies_inner, space_dims_inner, num_cells_inner, num_unknowns_inner, t_inner, Nt_inner, iter_inner = p

        u_local = reshape(u_vec, space_dims_inner..., Nt_inner)
        
        l_exact = zeros(eltype(u_vec), length(u_fixed_vals_inner))

        for (i, (u_val, ix, it)) in enumerate(zip(u_fixed_vals_inner, x_fixed_indicies_inner, t_fixed_indicies_inner))
            l_exact[i] = (num_unknowns_inner/length(u_fixed_vals_inner)) * (u_local[ix..., it] - u_val)
        end

        l_pde = zeros(eltype(u_vec), num_cells_inner * (Nt_inner - 1))
        
        du_rhs = zeros(eltype(u_vec), num_cells_inner)
        du_lhs = zeros(eltype(u_vec), num_cells_inner)
        
        for it in 1:(Nt_inner - 1)
            t_val = t_inner[it]
            u_rhs = vec(selectdim(u_local, length(space_dims_inner) + 1, it))
            fill!(du_rhs, 0.0)
            fill!(du_lhs, 0.0)
            
            rhs_inner(du_rhs, u_rhs, p_rhs_inner, t_val)
            lhs_inner(du_lhs, u_local, p_lhs_inner, it)
            
            for i in 1:num_cells_inner
                l_pde[(it - 1) * num_cells_inner + i] = (du_rhs[i] - du_lhs[i])
            end
        end

        l = vcat(l_exact, l_pde)

        l_extra = zeros(eltype(u_vec), space_dims_inner..., Nt_inner)

        if extra_inner !== nothing && p_extra_inner !== nothing
            l_extra = extra_inner(u_local, p_extra_inner, iter_inner[])
            l = vcat(l, vec(l_extra))
        end

        return l
    end
    
    prob = NonlinearLeastSquaresProblem(operator_loss, u_iter0, p_all)
    opt = GaussNewton(autodiff = autodiff)
    # opt = LevenbergMarquardt(autodiff = autodiff, damping_initial = 0.01)

    callback = function (cache, iter)
        println("Iteration ", iter, ": Loss = ", norm(cache.fu))
        u_current = reshape(cache.u, space_dims..., Nt)
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
    u_final = reshape(cache.u, space_dims..., Nt)
    
    println("Optimierung beendet!")
    println("Return Code: ", cache.retcode)
    return u_final
end