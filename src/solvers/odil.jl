using ADTypes

struct OdilProblem{N, XYZ<:Tuple, FS, PS, T, TT, II, FE, PE}
    step::FS
    p_step::PS
    step_alloc_size::Int
    N_coords::Int
    u_reference_vals::Vector{T}
    reference_val_indices::II
    t::TT
    extra::FE
    p_extra::PE
    len_extra::Int
    u_iter0::Vector{T}
    xyz::XYZ

    function OdilProblem(step::FS, p_step::PS, step_alloc_size::Int, N_coords::Int,
        u_reference_vals::Vector{T}, reference_val_indices::II,
        t::TT, extra::FE, p_extra::PE, len_extra::Int, u_iter0::Vector{T}, xyz::XYZ) where 
        {FS, PS, T, II, TT, FE, PE, XYZ<:Tuple}
        N = length(xyz)
        new{N, XYZ, FS, PS, T, TT, II, FE, PE}(
            step, p_step, step_alloc_size, N_coords, 
            u_reference_vals, reference_val_indices, 
            t, extra, p_extra, len_extra, u_iter0, xyz
        )
    end
end

function OdilProblem(step, p_step, N_coords, u_reference_vals,
    reference_val_indices, t, xyz...; extra = nothing,  p_extra = nothing,
    len_extra = 0, u_iter0 = nothing, step_alloc_size = 0)
    return OdilProblem(step, p_step, step_alloc_size, N_coords, u_reference_vals, 
        reference_val_indices, t, extra, p_extra, len_extra, 
        u_iter0 === nothing ? zeros(N_coords * length(t)) : u_iter0, xyz
    )
end

struct OdilState{T}
    u::Vector{T}
    it_last::Int
end