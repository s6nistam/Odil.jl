"""
Compute the Jacobian sparsity pattern for the ODIL least-squares residual.
"""
function get_jac_sparse(problem::OdilProblem)
    return get_jac_sparse(problem.timestep, problem.p_timestep, problem.timestep_alloc_size, problem.N_coords, length(problem.u_reference_vals), problem.reference_val_indices, length(problem.t), problem.t, problem.extra, problem.p_extra, problem.len_extra, problem.u_iter0)
end

"""
Compute the Jacobian sparsity pattern directly from the low-level ODIL components.
"""
function get_jac_sparse(timestep, p_timestep, timestep_alloc_size, N_coords, Nref, reference_val_indices, Nt, t, extra, p_extra, len_extra, u_iter0)
    I = Int[]
    J = Int[]
    for i in 1:Nref
        push!(I, i)
        push!(J, reference_val_indices[i])
    end
    u0 = u_iter0[1:N_coords]
    noise = rand(eltype(u_iter0), N_coords) .* 1e-8
    u_sparsity = u0 .+ noise
    u_timestep_out = zeros(eltype(u0), N_coords)
    timestep_mem = zeros(eltype(u0), timestep_alloc_size)
    d_timestep_mem = zeros(eltype(u0), timestep_alloc_size)

    I_timestep = Int[]
    J_timestep = Int[]
    d_u = zeros(eltype(u0), N_coords)
    d_u_timestep = zeros(eltype(u0), N_coords)

    for i in 1:N_coords
        d_u .= zero(eltype(u0))
        d_u_timestep .= zero(eltype(u0))
        d_u[i] = one(eltype(u0)) 
        
        Enzyme.autodiff(
            Enzyme.Forward, 
            Enzyme.Const(timestep), 
            Enzyme.Duplicated(timestep_mem, d_timestep_mem), 
            Enzyme.Duplicated(u_timestep_out, d_u_timestep), 
            Enzyme.Duplicated(u_sparsity, d_u), 
            Enzyme.Const(t[1]), 
            Enzyme.Const(t[2] - t[1]),
            Enzyme.Const(p_timestep), 
        )

        for row in findall(!iszero, d_u_timestep)
            push!(I_timestep, row)
            push!(J_timestep, i)
        end
    end


    for it in 2:Nt
        for k in eachindex(I_timestep)
            row = Nref + (it - 2) * N_coords + I_timestep[k]
            col = (it - 2) * N_coords + J_timestep[k]
            push!(I, row)
            push!(J, col)
        end
        
        for i in 1:N_coords
            row = Nref + (it - 2) * N_coords + i
            col = (it - 1) * N_coords + i
            push!(I, row)
            push!(J, col)
        end
    end

    if extra !== nothing && len_extra > 0
        extra_wrapper! = (du, u) -> extra(du, u, p_extra, 1)
        
        jac_extra_bool = jacobian_sparsity(extra_wrapper!, zeros(eltype(u_iter0), len_extra), u_iter0, SymbolicsSparsityDetector())
        
        I_extra, J_extra, _ = findnz(jac_extra_bool)
        
        for k in eachindex(I_extra)
            row = Nref + N_coords * (Nt - 1) + I_extra[k]
            col = J_extra[k]
            push!(I, row)
            push!(J, col)
        end
    end

    sparse_vals = ones(eltype(u_iter0), length(I))
    return sparse(I, J, sparse_vals, Nref + N_coords * (Nt - 1) + len_extra, N_coords * Nt)
end