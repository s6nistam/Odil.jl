function odil_timestepping(problem::OdilProblem, odil_func, filename_prefix; t_chunk_size::Int = 2, start_state = nothing, kwargs...)
    Nt = length(problem.t)
    N_coords = problem.N_coords
    res = zeros(eltype(problem.u_iter0), N_coords, Nt)
    idx = LinearIndices((N_coords, Nt))
    idx_chunk = LinearIndices((N_coords, t_chunk_size))

    if start_state === nothing
        u_iter0 = problem.u_iter0
        u_iter0[idx_chunk[:, t_chunk_size]] = problem.u_reference_vals
        state = OdilState(u_iter0, 1)
    else
        state = start_state
    end

    for iter in 1:Int(ceil((Nt - 1)/(t_chunk_size - 1)))
        it_start = state.it_last
        it_end = min(it_start + t_chunk_size - 1, Nt)

        u_reference_vals = state.u[idx_chunk[:, t_chunk_size]]
        reference_val_indices = idx_chunk[:, 1]

        problem_chunk = OdilProblem(problem.lhs, problem.rhs, problem.p_lhs, problem.p_rhs, problem.N_coords, u_reference_vals, reference_val_indices, problem.t[it_start:it_end], problem.xyz...; problem.extra, problem.p_extra, problem.len_extra)
        u_iter0 = repeat(u_reference_vals, it_end - it_start + 1)
        res_chunk = odil_func(problem_chunk; u_iter0 = u_iter0, info_prints = false, kwargs...)
        res[:, it_start:it_end] = res_chunk

        state = OdilState(res_chunk, it_end)
        write_h5(state, filename_prefix * "_iter$iter.h5")
    end
    return res
end