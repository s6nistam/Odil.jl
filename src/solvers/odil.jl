using ADTypes

struct OdilProblem{N, XYZ<:Tuple, T, TT, FL, FR, PL, PR, II, FE, PE}
    lhs::FL
    rhs::FR
    p_lhs::PL
    p_rhs::PR
    N_coords::Int
    u_reference_vals::Vector{T}
    reference_val_indices::II
    t::TT
    extra::FE
    p_extra::PE
    len_extra::Int
    u_iter0::Vector{T}
    xyz::XYZ

    function OdilProblem(lhs::FL, rhs::FR, p_lhs::PL, p_rhs::PR, N_coords::Int,
        u_reference_vals::Vector{T}, reference_val_indices::II,
        t::TT, extra::FE, p_extra::PE, len_extra::Int, u_iter0::Vector{T}, xyz::XYZ) where 
        {FL, FR, PL, PR, T, II, TT, FE, PE, XYZ<:Tuple}
        N = length(xyz)
        new{N, XYZ, T, TT, FL, FR, PL, PR, II, FE, PE}(
            lhs, rhs, p_lhs, p_rhs, N_coords, 
            u_reference_vals, reference_val_indices, 
            t, extra, p_extra, len_extra, u_iter0, xyz
        )
    end
end

function OdilProblem(lhs, rhs, p_lhs, p_rhs, N_coords, u_reference_vals,
    reference_val_indices, t, xyz...; extra = nothing,  p_extra = nothing,
    len_extra = 0, u_iter0 = nothing, max_iterations = 100000, autodiff = AutoEnzyme())
    return OdilProblem(lhs, rhs, p_lhs, p_rhs, N_coords, u_reference_vals, 
        reference_val_indices, t, extra, p_extra, len_extra, 
        u_iter0 === nothing ? zeros(N_coords * length(t)) : u_iter0, xyz
    )
end

struct OdilState{T}
    u::Vector{T}
    it_last::Int
end