function odil_timestepping(problem::OdilProblem, odil_func, filename_prefix; t_chunk_size::Int = 2, start_state = nothing, kwargs...)
    Nt = length(problem.t)
    N_coords = problem.N_coords
    Nref = length(problem.u_reference_vals)
    res = zeros(eltype(problem.u_iter0), N_coords, Nt)
    idx = LinearIndices((N_coords, Nt))
    idx_chunk = LinearIndices((N_coords, t_chunk_size))
    u_iter0 = repeat(problem.u_reference_vals, t_chunk_size)

    if start_state === nothing
        state = OdilState(u_iter0, 1)
    else
        state = start_state
    end

    if odil_func == odil_gauss_newton
        println("Computing Jacobian sparsity pattern...")
        jac_sparse = get_jac_sparse(problem.step, problem.p_step, problem.step_alloc_size, problem.t[1 : t_chunk_size], Nref, N_coords, t_chunk_size, problem.reference_val_indices, problem.extra, problem.p_extra, problem.len_extra, u_iter0)
        println("Computing coloring for Jacobian sparsity pattern...")
        colors = matrix_colors(jac_sparse)
    end

    for iter in 1:Int(ceil((Nt - 1)/(t_chunk_size - 1)))
        it_start = state.it_last
        it_end = min(it_start + t_chunk_size - 1, Nt)

        u_reference_vals = state.u[idx_chunk[:, t_chunk_size]]
        reference_val_indices = idx_chunk[:, 1]
        u_iter0 = repeat(u_reference_vals, it_end - it_start + 1)


        if iter == Int(ceil((Nt - 1)/(t_chunk_size - 1)))
            if odil_func == odil_gauss_newton
                
                n_cols_final = length(u_iter0) 
                row_ratio = size(jac_sparse, 1) / size(jac_sparse, 2)
                n_rows_final = Int(n_cols_final * row_ratio)
                
                jac_sparse = jac_sparse[1:n_rows_final, 1:n_cols_final] 
                
                colors = colors[1:n_cols_final]
            end
        end

        problem_chunk = OdilProblem(problem.step, problem.p_step, problem.N_coords, u_reference_vals, reference_val_indices, problem.t[it_start:it_end], problem.xyz...; problem.step_alloc_size, problem.extra, problem.p_extra, problem.len_extra)
        println("Solving chunk $iter: time steps $it_start to $it_end")
        if odil_func == odil_gauss_newton
            res_chunk = odil_func(problem_chunk; u_iter0 = u_iter0, info_prints = false, jac_sparse = jac_sparse, colors = colors, kwargs...)
        else
            res_chunk = odil_func(problem_chunk; u_iter0 = u_iter0, info_prints = false, kwargs...)
        end
        res[:, it_start:it_end] = res_chunk

        state = OdilState(res_chunk, it_end)
        write_h5(state, filename_prefix * "_iter$iter.h5")
    end
    return res
end

function reconstruct_solution_from_chunks(problem::OdilProblem, filename_prefix; t_chunk_size = 2)
    N_coords = problem.N_coords
    Nt = length(problem.t)
    res = zeros(Float64, N_coords * Nt)
    idx = LinearIndices((N_coords, Nt))
    idx_chunk = LinearIndices((N_coords, t_chunk_size))

    for iter in 1:Int(ceil((Nt - 1)/(t_chunk_size - 1)))
        it_start = (iter - 1) * (t_chunk_size - 1) + 1
        it_end = min(it_start + t_chunk_size - 1, Nt)

        if isfile(filename_prefix * "_iter$iter.h5")
            state = read_h5(filename_prefix * "_iter$iter.h5")
            res[idx[:, it_start:it_end]] = state.u[idx_chunk[:, 1:(it_end - it_start + 1)]]
        else
            return res
        end
    end
    return res
end