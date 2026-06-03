function get_exact_wave(t, x)
    ii = 1:5

    u = sum(i -> begin
        k = i * π
        cos.((x .- t .+ 0.5) .* k) .+
        cos.((x .+ t .- 0.5) .* k)
    end, ii) ./ (2 * length(ii))
    
    return u
end

function get_exact_wave_velocity(t, x)
    ii = 1:5

    u_t = sum(i -> begin
        k = i * π
        # Das k wird durch die innere Ableitung nach t herausgezogen
        (k .* sin.((x .- t .+ 0.5) .* k)) .- 
        (k .* sin.((x .+ t .- 0.5) .* k))
    end, ii) ./ (2 * length(ii))
    
    return u_t
end