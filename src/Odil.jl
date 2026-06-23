module Odil

using GLMakie

include("solvers/odil.jl")
include("solvers/odil_gauss_newton.jl")
include("solvers/odil_lbfgs.jl")
include("aux/plot.jl")
include("semidiscretization/wave.jl")
include("references/wave.jl")

export Odil, Odil1D, Odil2D, Odil3D
export odil_gauss_newton, odil_lbfgs
export AutoEnzyme, AutoFiniteDiff
export plot_1d_time_comparison
export plot_1d_time
export plot_fe_1d_time_compare
export plot_fe_1d_time
export plot_fe_2d_time_compare
export plot_fe_2d_time
export plot_fe_3d_time_compare
export plot_fe_3d_time
export get_exact_wave
export get_exact_wave_velocity

greet() = print("Hello ODIL!")

end # module Odil
