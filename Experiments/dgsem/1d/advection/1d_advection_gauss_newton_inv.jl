using Odil
include("./dgsem_advection.jl")

Trixi.TrixiBase.disable_debug_timings()

polydeg = 3
refinement_level = 4
ndims = 1
variables = Int64(length(ode.u0)/((polydeg + 1)^ndims * (2^refinement_level)^ndims))

coords = semi.cache.elements.node_coordinates
x = coords[1, :, :]

Nx = variables * length(x)
t = sol.t
Nt = length(t)
dt = [t[i + 1] - t[i] for i in 1:Nt-1]

u_exact = reduce(hcat, vec.(sol.u))

timestep! = get_timestep(Odil.CarpenterKennedy2N54())
p_timestep = (ode.f, ode.p)
standard_index_t0 = 1:Nx
# index = Int((Nt/2) * Nx + 1) : Int((Nt/2 + 1) * Nx)
# index = [i^2 for i in 1:min(Nt, Nx)]
using Random
# shuffled_index = shuffle(1:Nx)
# index = shuffled_index[1:round(Int, Nx * 0.95)]
shuffled_index = shuffle(1:Nx*Nt)
# index = shuffled_index[1:round(Int, Nx * Nt * 0.085)]
index = shuffled_index[1:round(Int, Nx * Nt * 0.2)]

div_count = 0
import LinearAlgebra: norm
function DivergenceCountCallback(div_count, interval::Int = 200)
    return (problem, u, loss, d_loss, iter) -> begin
        if iter % interval == 0
            if norm(loss) > 1e-3
                div_count += 1
                println("Divergence count: $div_count")
            end
        end
        return nothing
    end
end

callback_set = OdilCallbackSet(PlotCallback(100), DivergenceCountCallback(div_count, 200))


for i in 1:100
    shuffled_index = shuffle(1:Nx*Nt)
    index = shuffled_index[1:round(Int, Nx * Nt * 0.10)]
    problem = OdilProblem(timestep!, p_timestep, Nx, u_exact[index], index, t, x; timestep_alloc_size = 2 * Nx)
    res = odil_gauss_newton(problem; max_iterations = 200, callback_set = callback_set)
    write_vtk(problem, res, "odil_1d_advection_gauss_newton_inv_$i")
    write_csv(problem, res, "odil_1d_advection_gauss_newton_inv_$i")
end
println("Divergence count: $div_count")

# for i in 1:length(res)
#     if res[i] != u_exact[i]
#         res[i] = log10(abs(u_exact[i] - res[i]))
#     end
# end

# min = minimum(res)

# for i in 1:length(res)
#     if res[i] == u_exact[i]
#         res[i] = min - 1
#     end
# end

# plot(problem, u_exact, res)
# plot(problem, res)

# write_vtk(problem, res, "odil_1d_advection_gauss_newton_inv")