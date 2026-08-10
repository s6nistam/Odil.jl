"""
Marker type selecting the explicit Euler time-integration scheme.
"""
struct ExplicitEuler
end

"""
Advance one explicit-Euler time step for the ODIL residual evaluation.
"""
function timestep_explicit_euler!(timestep_mem, u_timestep, u, t, dt, p)
    f!, p_f = p
    u_timestep .= u
    du = @view(timestep_mem[1:length(u_timestep)])
    du .= zero(eltype(u_timestep))
    f!(du, u_timestep, p_f, t)
    u_timestep .+= dt .* du
    return nothing
end

"""
Return the timestep function associated with the selected integration method.
"""
function get_timestep(method::ExplicitEuler)
    return timestep_explicit_euler!
end