const ABc = [
    0,
    -0.4812317431372,
    -1.049562606709,
    -1.602529574275,
    -1.778267193916,
    0.097618354692056,
    0.4122532929155,
    0.4402169639311,
    1.426311463224,
    0.1978760536318,
    0,
    0.097618354692056,
    0.3114822768438,
    0.5120100121666,
    0.8971360011895
]

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

function step!(step_mem, u_step, u, t, dt, p)
    f!, p_f = p
    u_step .= u
    du = @view(step_mem[1:length(u_step)])
    du .= zero(eltype(u_step))
    du_old = @view(step_mem[length(u_step) + 1 : 2 * length(u_step)])
    for i in 1:5
        du_old .= du
        f!(du, u_step, p_f, t + ABc[10 + i] * dt)
        du .= ABc[i] .* du_old .+ dt .* du
        u_step .+= ABc[5 + i] .* du
    end
    return nothing
end