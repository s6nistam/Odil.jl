module Odil

using GLMakie

include("aux/plot.jl")
include("semidiscretization/wave.jl")
include("references/wave.jl")

export plot_1d_time_comparison
export plot_1d_time
export get_exact_wave
export get_exact_wave_velocity
export rhs!

greet() = print("Hello ODIL!")

end # module Odil
