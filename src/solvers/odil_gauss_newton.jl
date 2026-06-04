using SciMLBase, NonlinearSolveFirstOrder, ADTypes, Enzyme, LinearAlgebra

function odil_gauss_newton(ode::ODEProblem, lhs_func, p_lhs, Nt = 1000)
    u_t0 = ode.u0
    
    space_dims = size(u_t0)
    num_cells = prod(space_dims)
    # num_unknowns = num_cells * (Nt - 1)
    num_unknowns = num_cells * Nt
    
    u_iter0 = zeros(eltype(u_t0), num_unknowns)
    
    rhs_func = ode.f.f

    p_all = (ode.p, p_lhs, u_t0)

    function operator_loss(u_vec, p)
        p_rhs, p_lhs_inner, u_t0_inner = p
        
        u_local = zeros(eltype(u_vec), space_dims..., Nt)
        
        for i in 1:length(u_vec)
            # u_local[num_cells + i] = u_vec[i]
            u_local[i] = u_vec[i]
        end
        
        l_init = zeros(eltype(u_vec), space_dims..., Nt)

        for i in 1:length(u_t0_inner)
            l_init[i] += Nt *(u_local[i] - u_t0_inner[i])
        end

        l_inner = zeros(eltype(u_vec), space_dims..., Nt)
        
        du_rhs = similar(u_local)
        du_lhs = similar(u_local)
        
        for it in 1:Nt
            fill!(du_rhs, 0.0)
            fill!(du_lhs, 0.0)
            
            rhs_func(du_rhs, u_local, p_rhs, it)
            lhs_func(du_lhs, u_local, p_lhs_inner, it)
            
            for i in 1:num_cells
                l_inner[(it - 1) * num_cells + i] += (du_rhs[i] - du_lhs[i])
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
    res = solve(prob, opt, maxiters = 10000000, callback = callback)
    
    # Ergebnis für den Output wieder rekonstruieren
    u_final = zeros(eltype(u_t0), space_dims..., Nt)
    for i in 1:length(res.u)
        u_final[i] = res.u[i]
    end
    println("Optimierung beendet!")
    println("Return Code: ", res.retcode)
    return u_final
end