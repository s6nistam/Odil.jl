struct CarpenterKennedy2N54
end

const ABc = [
    0,
    - (567301805773.0 / 1357537059087.0),
    - (2404267990393.0 / 2016746695238.0),
    - (3550918686646.0 / 2091501179385.0),
    - (1275806237668.0 / 842570457699.0),
    1432997174477.0 / 9575080441755.0,
    5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0,
    3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0,
    0,
    1432997174477.0 / 9575080441755.0,
    2526269341429.0 / 6820363962896.0,
    2006345519317.0 / 3224310063776.0,
    2802321613138.0 / 2924317926251.0
]

# const ABc_1 = [
#     0,
#     -0.4812317431372,
#     -1.049562606709,
#     -1.602529574275,
#     -1.778267193916,
#     0.097618354692056,
#     0.4122532929155,
#     0.4402169639311,
#     1.426311463224,
#     0.1978760536318,
#     0,
#     0.097618354692056,
#     0.3114822768438,
#     0.5120100121666,
#     0.8971360011895
# ]

# function time_integration(u0, f, dt, tspan)
#     t0, tf = tspan
#     t = t0
#     u = u0
#     du = zeros(size(f(u0, t0)))
#     while t < tf
#         for i in 1:5
#             du .= ABc[i] * du .+ dt * f(u, t + ABc[10 + i] * dt)
#             u .+= ABc[5 + i] * du
#         end
#         t += dt
#     end
#     return u
# end

function timestep!(timestep_mem, u_timestep, u, t, dt, p)
    f!, p_f = p
    u_timestep .= u
    du = @view(timestep_mem[1:length(u_timestep)])
    du .= zero(eltype(u_timestep))
    du_old = @view(timestep_mem[length(u_timestep) + 1 : 2 * length(u_timestep)])
    for i in 1:5
        du_old .= du
        f!(du, u_timestep, p_f, t + ABc[10 + i] * dt)
        du .= ABc[i] .* du_old .+ dt .* du
        u_timestep .+= ABc[5 + i] .* du
    end
    return nothing
end

function get_timestep(method::CarpenterKennedy2N54)
    return timestep!
end