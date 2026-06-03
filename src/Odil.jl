module Odil

using Plots

include("aux/plot.jl")
include("FD/wave.jl")
include("references/wave.jl")
include("time_integration/second.jl")

export plot_comparison
export get_exact_wave
export get_exact_wave_velocity
export rhs!
export rhs_wave
export solve_d2dt2_central!

greet() = print("Hello ODIL!")

end # module Odil
