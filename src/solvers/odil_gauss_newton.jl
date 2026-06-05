using SciMLBase, NonlinearSolveFirstOrder, ADTypes, Enzyme, LinearAlgebra

function odil_gauss_newton(lhs, rhs, p_lhs, p_rhs, u_t0, Nt = 1000)
    space_dims = size(u_t0)
    num_cells = prod(space_dims)
    num_unknowns = num_cells * Nt
    
    u_iter0 = zeros(eltype(u_t0), num_unknowns)

    p_all = (p_rhs, p_lhs, u_t0, space_dims, num_cells, num_unknowns, Nt)

    function operator_loss(u_vec, p)
        p_rhs_inner, p_lhs_inner, u_t0_inner, space_dims_inner, num_cells_inner, num_unknowns_inner, Nt_inner = p
        
        u_local = zeros(eltype(u_vec), space_dims_inner..., Nt_inner)
        
        for i in 1:num_unknowns_inner
            u_local[i] = u_vec[i]
        end
        
        l_init = zeros(eltype(u_vec), space_dims_inner..., Nt_inner)

        for i in 1:num_cells_inner
            l_init[i] += Nt_inner *(u_local[i] - u_t0_inner[i])
        end

        l_inner = zeros(eltype(u_vec), space_dims_inner..., Nt_inner)
        
        du_rhs = similar(u_local)
        du_lhs = similar(u_local)
        
        for it in 1:(Nt_inner - 1)
            fill!(du_rhs, 0.0)
            fill!(du_lhs, 0.0)
            
            rhs(du_rhs, u_local, p_rhs_inner, it)
            lhs(du_lhs, u_local, p_lhs_inner, it)
            
            for i in 1:num_cells_inner
                l_inner[it * num_cells_inner + i] += (du_rhs[i] - du_lhs[i])
            end
        end

        l = l_init + l_inner

        return l
    end
    
    prob = NonlinearLeastSquaresProblem(operator_loss, u_iter0, p_all)
    opt = GaussNewton(autodiff = AutoEnzyme())

    counter = 0
    callback = function (state, l)
        counter += 1
        if counter % 1000 == 0 || counter == 1
            println("Iteration ", counter, ": Loss = ", norm(l))

            u_current = zeros(eltype(u_t0), space_dims..., Nt)
            for i in 1:length(state.u)
                u_current[i] = state.u[i]
            end
            plot_comparison(x, t, u_exact, u_current)
        end
        return false # false = Optimierung weiterlaufen lassen
    end
    
    println("Starte Lösung...")
    res = solve(prob, opt, maxiters = 2, callback = callback)
    
    # Ergebnis für den Output wieder rekonstruieren
    u_final = zeros(eltype(u_t0), space_dims..., Nt)
    for i in 1:length(res.u)
        u_final[i] = res.u[i]
    end
    println("Optimierung beendet!")
    println("Return Code: ", res.retcode)
    return u_final
end