function get_jac_sparse(step, p_step, step_alloc_size, t, Nref, Nx, Nt, reference_val_indices, extra, p_extra, len_extra, u_iter0)
    I = Int[]
    J = Int[]
    for i in 1:Nref
        push!(I, i)
        push!(J, reference_val_indices[i])
    end
    u0 = u_iter0[1:Nx]
    noise = rand(eltype(u_iter0), Nx) .* 1e-8
    u_sparsity = u0 .+ noise
    u_step_out = zeros(eltype(u0), Nx)
    step_mem = zeros(eltype(u0), step_alloc_size)
    d_step_mem = zeros(eltype(u0), step_alloc_size)
    jac_step = zeros(eltype(u0), Nx, Nx)
    
    for i in 1:Nx
        d_u = zeros(eltype(u0), Nx)
        d_u[i] = 1.0 
        d_u_step = zeros(eltype(u0), Nx)
        
        Enzyme.autodiff(
            Enzyme.Forward, 
            Enzyme.Const(step), 
            Enzyme.Duplicated(step_mem, d_step_mem), 
            Enzyme.Duplicated(u_step_out, d_u_step), 
            Enzyme.Duplicated(u_sparsity, d_u), 
            Enzyme.Const(t[1]), 
            Enzyme.Const(t[2] - t[1]),
            Enzyme.Const(p_step), 
        )
        jac_step[:, i] = d_u_step
    end
    jac_step_bool = sparse(jac_step .!= 0)
    I_step, J_step, _ = findnz(jac_step_bool)
    for it in 2:Nt
        for k in eachindex(I_step)
            row = Nref + (it - 2) * Nx + I_step[k]
            col = (it - 2) * Nx + J_step[k]
            push!(I, row)
            push!(J, col)
        end
        
        for i in 1:Nx
            row = Nref + (it-2)*Nx + i
            col = (it-1)*Nx + i
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