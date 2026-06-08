module Odil

using Plots

include("aux/plot.jl")
include("semidiscretization/wave.jl")
include("references/wave.jl")

export plot_comparison
export plot_2d
export get_exact_wave
export get_exact_wave_velocity
export rhs!

greet() = print("Hello ODIL!")

end # module Odil
