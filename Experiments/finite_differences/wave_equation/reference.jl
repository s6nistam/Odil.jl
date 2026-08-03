function get_exact_wave(x, t)
    ii = 1:5
    u = sum(i -> begin
        k = i * π
        cos.((x .- t .+ 0.5) .* k) .+
        cos.((x .+ t .- 0.5) .* k)
    end, ii) ./ (2 * length(ii))
    
    return u
end

function get_exact_wave_velocity(x, t)
    ii = 1:5
    u_t = sum(i -> begin
        k = i * π
        (k .* sin.((x .- t .+ 0.5) .* k)) .- 
        (k .* sin.((x .+ t .- 0.5) .* k))
    end, ii) ./ (2 * length(ii))
    
    return u_t
end