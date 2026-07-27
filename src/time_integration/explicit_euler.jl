struct ExplicitEuler
end

function timestep_explicit_euler!(timestep_mem, u_timestep, u, t, dt, p)
    f!, p_f = p
    u_timestep .= u
    du = @view(timestep_mem[1:length(u_timestep)])
    du .= zero(eltype(u_timestep))
    f!(du, u_timestep, p_f, t)
    u_timestep .+= dt .* du
    return nothing
end

function get_timestep(method::ExplicitEuler)
    return timestep_explicit_euler!
end