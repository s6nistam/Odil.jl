function rhs_wave(u, it, dx)
   
    rhs = zero(u[:,it])
    
    # Differenzenquotient im Inneren
    for j in 2:size(u, 1)-1
        rhs[j] = (u[j-1, it] - 2 * u[j, it] + u[j+1, it]) / (dx*dx)
    end

    return rhs
end