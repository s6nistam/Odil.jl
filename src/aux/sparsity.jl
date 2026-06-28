function get_jac_sparse(lhs, p_lhs, Nref, Nx, Nt, reference_val_indices, extra, p_extra, len_extra, u_iter0)
    I = Int[]
    J = Int[]
    for i in 1:Nref
        push!(I, i)
        push!(J, reference_val_indices[i])
    end

    for it in 2:Nt
        for ix in 1:Nx
            row = Nref + (it - 2) * Nx + ix
            for jx in 1:Nx
                col = (it - 2) * Nx + jx
                push!(I, row)
                push!(J, col)
            end
        end

        lhs_wrapper! = (du, u) -> lhs(du, u, p_lhs, it)
        
        jac_lhs_bool = jacobian_sparsity(lhs_wrapper!, zeros(eltype(u_iter0), Nx), u_iter0, SymbolicsSparsityDetector())
        
        I_lhs, J_lhs, _ = findnz(jac_lhs_bool)
        
        for k in eachindex(I_lhs)
            row = Nref + (it - 2) * Nx + I_lhs[k]
            col = J_lhs[k]
            push!(I, row)
            push!(J, col)
        end
    end

    if extra !== nothing && len_extra > 0
        extra_wrapper! = (du, u) -> extra(du, u, p_extra, 1)
        
        jac_extra_bool = jacobian_sparsity(extra_wrapper!, zeros(eltype(u_iter0), len_extra), u_iter0, SymbolicsSparsityDetector())
        
        I_extra, J_extra, _ = findnz(jac_extra_bool)
        
        for k in eachindex(I_extra)
            row = Nref + Nx * (Nt - 1) + I_extra[k]
            col = J_extra[k]
            push!(I, row)
            push!(J, col)
        end
    end

    sparse_vals = ones(eltype(u_iter0), length(I))
    return sparse(I, J, sparse_vals, Nref + Nx * (Nt - 1) + len_extra, Nx * Nt)
end