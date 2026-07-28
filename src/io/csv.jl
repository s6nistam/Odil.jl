using CSV, DataFrames, Morton

function write_csv(problem::OdilProblem{1}, u, filename::String)
    (x,) = problem.xyz
    t = problem.t
    N_coords = problem.N_coords
    Nt = length(t)
    Ne = length(x[1, :])
    Nx = length(x[:, 1])
    variables = Int(N_coords/(Nx * Ne))
    it_last = Int(length(u) / (variables * Nx * Ne))
    t = t[1:it_last]
    Nt = length(t)

    u_reshaped = reshape(u, (variables, Nx * Ne, Nt))

    x_col = repeat(vec(x), outer = Nt)
    t_col = repeat(t, inner = Nx * Ne)
    
    df = DataFrame(
        x = x_col, 
        t = t_col
    )

    for var in 1:variables
        df[!, Symbol("u_$var")] = vec(u_reshaped[var, :, :])
    end

    CSV.write("$(filename).csv", df)
end

function write_csv(problem::OdilProblem{2}, u, filename::String)
    x, y = problem.xyz
    t = problem.t
    N_coords = problem.N_coords
    Nt = length(t)
    Ne = length(x[1, 1, :])
    Ne_per_dim = Int(round(sqrt(Ne)))
    Nx = length(x[:, 1, 1])
    Ny = length(y[1, :, 1])
    variables = Int(N_coords/(Nx * Ny * Ne))
    it_last = Int(length(u) / (variables * Nx * Ny * Ne))
    t = t[1:it_last]
    Nt = length(t)
    
    u_linear = reshape(u, variables, Nx, Ny, Ne, Nt)

    u_cartesian = zeros(eltype(u), variables, Nx, Ny, Ne_per_dim, Ne_per_dim, Nt)
    x_cartesian = zeros(eltype(x), Nx, Ny, Ne_per_dim, Ne_per_dim)
    y_cartesian = zeros(eltype(y), Nx, Ny, Ne_per_dim, Ne_per_dim)

    for e in 1:Ne
        ix, iy = morton2cartesian(e)
        
        u_cartesian[:, :, :, ix, iy, :] = u_linear[:, :, :, e, :]
        x_cartesian[:, :, ix, iy] = x[:, :, e]
        y_cartesian[:, :, ix, iy] = y[:, :, e]
    end
    x_re = reshape(x_cartesian, Nx, Ny, Ne_per_dim, Ne_per_dim)
    y_re = reshape(y_cartesian, Nx, Ny, Ne_per_dim, Ne_per_dim)
    u_perm = permutedims(u_cartesian, (1, 2, 4, 3, 5, 6))
    u_reshaped = reshape(u_perm, variables, Nx * Ne_per_dim, Ny * Ne_per_dim, Nt)
    
    x_col = repeat(vec(x_re[:, 1, :, 1]), outer = Ny * Ne_per_dim * Nt)
    y_col = repeat(vec(y_re[1, :, 1, :]), inner = Nx * Ne_per_dim * Nt)
    t_col = repeat(t, inner = Nx * Ne_per_dim * Ny * Ne_per_dim)

    df = DataFrame(
        x = x_col,
        y = y_col,
        t = t_col
    )

    for var in 1:variables
        df[!, Symbol("u_$var")] = vec(u_reshaped[var, :, :, :])
    end

    CSV.write("$(filename).csv", df)
end

function write_csv(problem::OdilProblem{3}, u, filename::String)
    x, y, z = problem.xyz
    t = problem.t
    N_coords = problem.N_coords
    Ne = length(x[1, 1, 1, :])
    Ne_per_dim = Int(round(Ne^(1/3)))
    Nx = length(x[:, 1, 1, 1])
    Ny = length(y[1, :, 1, 1])
    Nz = length(z[1, 1, :, 1])
    variables = Int(N_coords / (Nx * Ny * Nz * Ne))
    it_last = Int(length(u) / (variables * Nx * Ny * Nz * Ne))
    t = t[1:it_last]
    Nt = length(t)

    u_linear = reshape(u, variables, Nx, Ny, Nz, Ne, Nt)
    
    u_cartesian = zeros(eltype(u), variables, Nx, Ny, Nz, Ne_per_dim, Ne_per_dim, Ne_per_dim, Nt)
    x_cartesian = zeros(eltype(x), Nx, Ny, Nz, Ne_per_dim, Ne_per_dim, Ne_per_dim)
    y_cartesian = zeros(eltype(y), Nx, Ny, Nz, Ne_per_dim, Ne_per_dim, Ne_per_dim)
    z_cartesian = zeros(eltype(z), Nx, Ny, Nz, Ne_per_dim, Ne_per_dim, Ne_per_dim)

    for e in 1:Ne
        ix, iy, iz = morton3cartesian(e) 
        
        u_cartesian[:, :, :, :, ix, iy, iz, :] = u_linear[:, :, :, :, e, :]
        x_cartesian[:, :, :, ix, iy, iz] = x[:, :, :, e]
        y_cartesian[:, :, :, ix, iy, iz] = y[:, :, :, e]
        z_cartesian[:, :, :, ix, iy, iz] = z[:, :, :, e]
    end

    u_perm = permutedims(u_cartesian, (1, 2, 5, 3, 6, 4, 7, 8))
    u_reshaped = reshape(u_perm, variables, Nx * Ne_per_dim, Ny * Ne_per_dim, Nz * Ne_per_dim, Nt)

    x_perm = permutedims(x_cartesian, (1, 4, 2, 5, 3, 6))
    y_perm = permutedims(y_cartesian, (1, 4, 2, 5, 3, 6))
    z_perm = permutedims(z_cartesian, (1, 4, 2, 5, 3, 6))
    
    x_global = reshape(x_perm, Nx * Ne_per_dim, Ny * Ne_per_dim, Nz * Ne_per_dim)
    y_global = reshape(y_perm, Nx * Ne_per_dim, Ny * Ne_per_dim, Nz * Ne_per_dim)
    z_global = reshape(z_perm, Nx * Ne_per_dim, Ny * Ne_per_dim, Nz * Ne_per_dim)


    x_col = repeat(x_global[:, 1, 1], outer = Ny * Ne_per_dim * Nz * Ne_per_dim * Nt)
    y_col = repeat(repeat(y_global[1, :, 1], inner = Nx * Ne_per_dim), outer = Nz * Ne_per_dim * Nt)
    z_col = repeat(z_global[1, 1, :], inner = Nx * Ne_per_dim * Ny * Ne_per_dim * Nt)
    t_col = repeat(t, inner = Nx * Ne_per_dim * Ny * Ne_per_dim)

    df = DataFrame(
        x = x_col,
        y = y_col,
        z = z_col,
        t = t_col,
        u = vec(u_reshaped[var, :, :, :, it])
    )

    for var in 1:variables
        df[!, Symbol("u_$var")] = vec(u_reshaped[var, :, :, :, :])
    end

    CSV.write("$(filename).csv", df)
end