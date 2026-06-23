using ADTypes

struct Odil{T, TT, FL, FR, PL, PR, II, FE, PE}
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
end

function Odil(lhs, rhs, p_lhs, p_rhs, N_coords, u_reference_vals, reference_val_indices, t; extra = nothing,  p_extra = nothing, len_extra = 0, u_iter0 = nothing, max_iterations = 100000, autodiff = AutoEnzyme())
    return Odil(lhs,
    rhs, 
    p_lhs, 
    p_rhs, 
    N_coords, 
    u_reference_vals, 
    reference_val_indices, 
    t, 
    extra, 
    p_extra, 
    len_extra, 
    u_iter0 === nothing ? zeros(N_coords * length(t)) : u_iter0)
end

struct Odil1D{P<:Odil, XX}
    problem::P
    x::XX
end

function Odil1D(lhs, rhs, p_lhs, p_rhs, N_coords, u_reference_vals, reference_val_indices, t, x; extra = nothing,  p_extra = nothing, len_extra = 0, u_iter0 = nothing, max_iterations = 100000, autodiff = AutoEnzyme())
    problem = Odil(lhs,
        rhs,
        p_lhs,
        p_rhs,
        N_coords,
        u_reference_vals,
        reference_val_indices,
        t,
        extra,
        p_extra,
        len_extra,
        u_iter0 === nothing ? zeros(N_coords * length(t)) : u_iter0)
    return Odil1D(problem, x)
end

struct Odil2D{P<:Odil, XX, YY}
    problem::P
    x::XX
    y::YY
end

function Odil2D(lhs, rhs, p_lhs, p_rhs, N_coords, u_reference_vals, reference_val_indices, t, x, y; extra = nothing,  p_extra = nothing, len_extra = 0, u_iter0 = nothing, max_iterations = 100000, autodiff = AutoEnzyme())
    problem = Odil(lhs,
        rhs,
        p_lhs,
        p_rhs,
        N_coords,
        u_reference_vals,
        reference_val_indices,
        t,
        extra,
        p_extra,
        len_extra,
        u_iter0 === nothing ? zeros(N_coords * length(t)) : u_iter0)
    return Odil2D(problem, x, y)
end

struct Odil3D{P<:Odil, XX, YY, ZZ}
    problem::P
    x::XX
    y::YY
    z::ZZ
end

function Odil3D(lhs, rhs, p_lhs, p_rhs, N_coords, u_reference_vals, reference_val_indices, t, x, y, z; extra = nothing,  p_extra = nothing, len_extra = 0, u_iter0 = nothing, max_iterations = 100000, autodiff = AutoEnzyme())
    problem = Odil(lhs,
        rhs,
        p_lhs,
        p_rhs,
        N_coords,
        u_reference_vals,
        reference_val_indices,
        t,
        extra,
        p_extra,
        len_extra,
        u_iter0 === nothing ? zeros(N_coords * length(t)) : u_iter0)
    return Odil3D(problem, x, y, z)
end

base_problem(problem::Odil) = problem
base_problem(problem::Odil1D) = problem.problem
base_problem(problem::Odil2D) = problem.problem
base_problem(problem::Odil3D) = problem.problem