using SciMLBase, OptimizationBase, OptimizationLBFGSB, ADTypes, Enzyme, LinearAlgebra

function odil_lbfgsb(ode::ODEProblem, lhs_func, p_lhs, Nt = 1000)
    u_t0 = ode.u0
    
    space_dims = size(u_t0)
    num_cells = prod(space_dims)
    num_unknowns = num_cells * (Nt - 1)
    
    u_iter0 = zeros(eltype(u_t0), num_unknowns)
    
    rhs_func = ode.f.f

    p_all = (ode.p, p_lhs, u_t0)
    c = 0

    function loss(u_vec, p)
        p_rhs, p_lhs_inner, u_t0_inner = p
        
        u_local = zeros(eltype(u_vec), space_dims..., Nt)

        for i in 1:num_cells
            u_local[i] = u_t0_inner[i]
        end
        
        for i in 1:length(u_vec)
            u_local[num_cells + i] = u_vec[i]
        end
        
        l = zero(eltype(u_vec))
        
        du_rhs = similar(u_local)
        du_lhs = similar(u_local)
        
        for it in 1:Nt
            fill!(du_rhs, 0.0)
            fill!(du_lhs, 0.0)
            
            rhs_func(du_rhs, u_local, p_rhs, it)
            lhs_func(du_lhs, u_local, p_lhs_inner, it)
            
            for i in eachindex(du_rhs)
                l += (du_rhs[i] - du_lhs[i])^2
            end
        end
        c += 1
        if c % 100 == 0
            println("Iteration ", c, ": Loss = ", l)
        end

        return l
    end
    
    optf = OptimizationFunction(loss, ADTypes.AutoEnzyme())
    prob = OptimizationProblem(optf, u_iter0, p_all) 
    opt = LBFGSB()
    
    println("Starte Lösung...")
    res = solve(prob, opt, pgtol = 1e-16, tol = 1e-16, maxiters = 100000)
    
    # Ergebnis für den Output wieder rekonstruieren
    u_final = zeros(eltype(u_t0), space_dims..., Nt)
    for i in 1:num_cells
        u_final[i] = u_t0[i]
    end
    for i in 1:length(res.u)
        u_final[num_cells + i] = res.u[i]
    end
    
    return u_final
end