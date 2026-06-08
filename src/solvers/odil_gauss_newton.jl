using SciMLBase, NonlinearSolveFirstOrder, ADTypes, Enzyme, LinearAlgebra

function odil_gauss_newton(lhs, rhs, p_lhs, p_rhs, u_size_x, u_fixed_vals, x_fixed_indicies, t_fixed_indicies, Nt = 1000)
    space_dims = (u_size_x,)
    num_cells = prod(space_dims)
    num_unknowns = num_cells * Nt
    
    u_iter0 = zeros(num_unknowns)

    p_all = (p_rhs, p_lhs, u_fixed_vals, x_fixed_indicies, t_fixed_indicies, space_dims, num_cells, num_unknowns, Nt)

    function operator_loss(u_vec, p)
        p_rhs_inner, p_lhs_inner, u_fixed_vals_inner, x_fixed_indicies_inner, t_fixed_indicies_inner, space_dims_inner, num_cells_inner, num_unknowns_inner, Nt_inner = p
        
        u_local = reshape(u_vec, space_dims_inner..., Nt_inner)
        
        l_exact = zeros(eltype(u_vec), space_dims_inner..., Nt_inner)

        for (u_val, ix, it) in zip(u_fixed_vals_inner, x_fixed_indicies_inner, t_fixed_indicies_inner)
            l_exact[ix..., it] += (num_unknowns_inner/length(u_fixed_vals_inner)) *(u_local[ix..., it] - u_val)
        end

        l_pde = zeros(eltype(u_vec), space_dims_inner..., Nt_inner)
        
        du_rhs = similar(u_local)
        du_lhs = similar(u_local)
        
        for it in 1:(Nt_inner - 1)
            fill!(du_rhs, 0.0)
            fill!(du_lhs, 0.0)
            
            rhs(du_rhs, u_local, p_rhs_inner, it)
            lhs(du_lhs, u_local, p_lhs_inner, it)
            
            for i in 1:num_cells_inner
                l_pde[it * num_cells_inner + i] += (du_rhs[i] - du_lhs[i])
            end
        end

        l = l_exact + l_pde

        return l
    end
    
    prob = NonlinearLeastSquaresProblem(operator_loss, u_iter0, p_all)
    opt = GaussNewton(autodiff = AutoEnzyme())

    callback = function (cache, iter)
        println("Iteration ", iter, ": Loss = ", norm(cache.fu))
        u_current = reshape(cache.u, space_dims..., Nt)
        plot_2d(x, t, u_current)
        return false # false = Optimierung weiterlaufen lassen
    end
    
    println("Starte Lösung...")
    
    max_iterations = 15
    cache = init(prob, opt, maxiters = max_iterations)
    
    for iter in 1:max_iterations
        step!(cache)
        
        current_loss = norm(cache.fu)
        println("Iteration ", iter, ": Loss = ", current_loss)

        u_current = reshape(cache.u, space_dims..., Nt)
        
        plot_2d(x, t, u_current)
        
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