using WriteVTK, ReadVTK, Morton

function write_vtk(problem::Odil1D, u, filename::String)
    x = problem.x
    t = problem.problem.t
    N_coords = problem.problem.N_coords
    Nt = length(t)
    Ne = length(x[1, :])
    Nx = length(x[:, 1])
    variables = Int(N_coords/(Nx * Ne))
    u_reshaped = reshape(u, (variables, Nx * Ne, Nt))

    vtk_grid(filename, vec(x), t) do vtk
        for var in 1:variables
            vtk["u_$var"] = u_reshaped[var, :, :]
        end
    end
end

function write_vtk(problem::Odil2D, u, filename::String)
    x = problem.x
    y = problem.y
    t = problem.problem.t
    N_coords = problem.problem.N_coords
    Nt = length(t)
    Ne = length(x[1, 1, :])
    Ne_per_dim = Int(round(sqrt(Ne)))
    Nx = length(x[:, 1, 1])
    Ny = length(y[1, :, 1])
    variables = Int(N_coords/(Nx * Ny * Ne))
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

    paraview_collection(filename) do pvd
        for (i, it) in enumerate(eachindex(t))
            vtk_grid("timestep_$i", vec(x_re[:, 1, :, 1]), vec(y_re[1, :, 1, :])) do vtk
                for var in 1:variables
                    vtk["u_$var"] = u_reshaped[var, :, :, it]
                end
                pvd[it] = vtk
            end
        end
    end
end

function write_vtk(problem::Odil3D, u, filename::String)
    x = problem.x
    y = problem.y
    z = problem.z
    t = problem.problem.t
    N_coords = problem.problem.N_coords
    Nt = length(t)
    Ne = length(x[1, 1, 1, :])
    Ne_per_dim = Int(round(Ne^(1/3)))
    Nx = length(x[:, 1, 1, 1])
    Ny = length(y[1, :, 1, 1])
    Nz = length(z[1, 1, :, 1])
    variables = Int(N_coords / (Nx * Ny * Nz * Ne))

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

    paraview_collection(filename) do pvd
        for (i, it) in enumerate(eachindex(t))
            vtk_grid("timestep_$i", x_global[:, 1, 1], y_global[1, :, 1], z_global[1, 1, :]) do vtk
                for var in 1:variables
                    vtk["u_$var"] = u_reshaped[var, :, :, :, it]
                end
                pvd[it] = vtk
            end
        end
    end
end