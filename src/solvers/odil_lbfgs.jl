using SciMLBase, OptimizationBase, OptimizationOptimJL, ADTypes, Enzyme, LinearAlgebra

function odil_lbfgs(lhs, rhs, p_lhs, p_rhs, u_size_x, u_exact_vals, Nt = 1000)
    space_dims = u_size_x
    num_cells = prod(space_dims)
    num_unknowns = num_cells * Nt
    
    u_iter0 = zeros(num_unknowns)

    p_all = (p_rhs, p_lhs, u_exact_vals, space_dims, num_cells, num_unknowns, Nt)

    function loss(u_vec, p)
        p_rhs_inner, p_lhs_inner, u_exact_vals_inner, space_dims_inner, num_cells_inner, num_unknowns_inner, Nt_inner = p
        
        u_local = zeros(eltype(u_vec), space_dims_inner..., Nt)
        
        for i in 1:num_unknowns_inner
            u_local[i] = u_vec[i]
        end
        
        l_exact = zero(eltype(u_vec))

        for (u_val, ix, it) in u_exact_vals_inner
            l_exact += (num_unknowns_inner/length(u_exact_vals))^2 *(u_local[ix..., it] - u_val)^2
        end

        l_pde = zero(eltype(u_vec))
        
        du_rhs = similar(u_local)
        du_lhs = similar(u_local)
        
        for it in 1:(Nt_inner - 1)
            fill!(du_rhs, 0.0)
            fill!(du_lhs, 0.0)
            
            rhs(du_rhs, u_local, p_rhs_inner, it)
            lhs(du_lhs, u_local, p_lhs_inner, it)
            
            for i in eachindex(du_rhs)
                l_pde += (du_rhs[i] - du_lhs[i])^2
            end
        end

        l = l_exact + l_pde

        return l
    end
    
    optf = OptimizationFunction(loss, ADTypes.AutoEnzyme())
    prob = OptimizationProblem(optf, u_iter0, p_all) 
    # opt = LBFGS(m = 50)
    opt = LBFGS()

    counter = 0
    callback = function (state, l)
        counter += 1
        if counter % 1000 == 0 || counter == 1
            println("Iteration ", counter, ": Loss = ", l)

            u_current = zeros(space_dims..., Nt)
            for i in 1:length(state.u)
                u_current[i] = state.u[i]
            end
            # plot_comparison(x, t, u_exact, u_current)
            plot_2d(x, t, u_current)
        end
        return false # false = Optimierung weiterlaufen lassen
    end
    
    println("Starte Lösung...")
    res = solve(prob, opt, maxiters = 10000000, callback = callback)
    
    # Ergebnis für den Output wieder rekonstruieren
    u_final = zeros(space_dims..., Nt)
    for i in 1:length(res.u)
        u_final[i] = res.u[i]
    end
    println("Optimierung beendet!")
    println("Return Code: ", res.retcode)
    return u_final
end