module Odil

using GLMakie

include("solvers/odil.jl")
include("solvers/odil_gauss_newton.jl")
include("solvers/odil_lbfgs.jl")
include("solvers/odil_timestepping.jl")
include("aux/h5.jl")
include("aux/plot.jl")
include("aux/sparsity.jl")
include("aux/vtk.jl")

export OdilProblem, OdilState
export odil_gauss_newton, odil_lbfgs
export odil_timestepping
export reconstruct_solution_from_chunks
export AutoEnzyme, AutoFiniteDiff
export plot
export plot_1d_time_comparison
export plot_1d_time
export plot_fe_1d_time_compare
export plot_fe_1d_time
export plot_fe_2d_time_compare
export plot_fe_2d_time
export plot_fe_3d_time_compare
export plot_fe_3d_time
export get_jac_sparse
export write_vtk
export write_h5
export read_h5

end # module Odil
