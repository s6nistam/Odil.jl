using GLMakie
using ColorSchemes

function plot_1d_time_comparison(x, t, u_exact, u_approx;
    save_file = false, filename = "comparison.png")

    # We determine the min and max values to ensure both plots share the same color scale
    z_min = min(minimum(u_exact), minimum(u_approx))
    z_max = max(maximum(u_exact), maximum(u_approx))

    # Create a new figure
    fig = Figure(size = (800, 400))

    # Add heatmaps to the figure
    ax1 = Axis(fig[1, 1], title = "Exact Solution", xlabel = "x", ylabel = "t")
    hm1 = heatmap!(ax1, x, t, u_exact, colorrange = (z_min, z_max), colormap = :viridis)

    ax2 = Axis(fig[1, 2], title = "Approximation", xlabel = "x", ylabel = "t")
    hm2 = heatmap!(ax2, x, t, u_approx, colorrange = (z_min, z_max), colormap = :viridis)

    # Add a colorbar
    Colorbar(fig[1, 3], hm1, label = "Value")

    # Display the figure
    display(fig)

    # Optional: Save to file
    if save_file
        save(filename, fig)
    end

    return fig
end

function plot_1d_time(x, t, u;
    save_file = false, filename = "solution.png")

    # We determine the min and max values
    z_min = minimum(u)
    z_max = maximum(u)

    # Create a new figure
    fig = Figure(size = (400, 400))

    # Add heatmap to the figure
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "t")
    hm = heatmap!(ax, x, t, u, colorrange = (z_min, z_max), colormap = :viridis)

    # Add a colorbar
    Colorbar(fig[1, 2], hm, label = "Value")

    # Display the figure
    display(fig)

    # Optional: Save to file
    if save_file
        save(filename, fig)
    end

    return fig
end

function plot_fe_1d_time(x, e, u;
    save_file = false, filename = "solution.png", c_min = nothing, c_max = nothing)

    fig = Figure(size = (400, 400))
    it = Observable(1)
    ax = Axis(fig[1, 1], xlabel = "x")

    if c_min === nothing
        c_min = minimum(u)
    end
    if c_max === nothing
        c_max = maximum(u)
    end

    for element in e
        ue = u[:, element, :]
        xe = x[:, element]
        u_t = lift(it) do current_t
            return ue[:, current_t]
        end

        hm_e = lines!(ax, 
            xe, u_t,
            color = u_t,
            colorrange = (c_min, c_max), 
            colormap = :viridis,
            transparency = true, 
        )
    end
        Colorbar(fig[1, 2], limits=(c_min, c_max), label = "Value")


    sg = SliderGrid(fig[2, 1],
        (label = "Time Step", range = 1:size(u, 3), startvalue = 1))
    
    on(sg.sliders[1].value) do val
        it[] = val
        ax.title = "Time index: $val"
    end

    display(fig)

    if save_file
        save(filename, fig)
    end

    return fig
end

function plot_fe_1d_time_compare(x, e, u_exact, u_approx;
    save_file = false, filename = "solution.png", c_min = nothing, c_max = nothing)

    # Create a new figure
    fig = Figure(size = (800, 400))
    it = Observable(1)
    ax_exact = Axis(fig[1, 1], xlabel = "x", title = "Exact Solution")
    ax_approx = Axis(fig[1, 2], xlabel = "x", title = "Approximate Solution")

    # We determine the min and max values
    if c_min === nothing
        c_min = minimum(u_exact)
    end
    if c_max === nothing
        c_max = maximum(u_exact)
    end

    for element in e
        ue_exact = u_exact[:, element, :]
        ue_approx = u_approx[:, element, :]
        xe = x[:, element]
        u_t_exact = lift(it) do current_t
            return ue_exact[:, current_t]
        end
        u_t_approx = lift(it) do current_t
            return ue_approx[:, current_t]
        end

        hm_e_exact = lines!(ax_exact, 
            xe, u_t_exact,
            color = u_t_exact,
            colorrange = (c_min, c_max), 
            colormap = :viridis, 
            transparency = true, 
        )

        hm_e_approx = lines!(ax_approx, 
            xe, u_t_approx,
            color = u_t_approx,
            colorrange = (c_min, c_max), 
            colormap = :viridis,
            transparency = true, 
        )
    end
        Colorbar(fig[1, 3], limits=(c_min, c_max), label = "Value")


    sg = SliderGrid(fig[2, :],
        (label = "Time Step", range = 1:size(u_exact, 3), startvalue = 1))
    
    on(sg.sliders[1].value) do val
        it[] = val
    end

    display(fig)

    if save_file
        save(filename, fig)
    end

    return fig
end

function plot_fe_2d_time(x, y, e, u;
    save_file = false, filename = "solution.png", c_min = nothing, c_max = nothing)

    # Create a new figure
    fig = Figure(size = (400, 400))
    it = Observable(1)
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y")

    # We determine the min and max values
    if c_min === nothing
        c_min = minimum(u)
    end
    if c_max === nothing
        c_max = maximum(u)
    end

    for element in e
        ue = u[:, :, element, :]
        xe = x[:, :, element]
        ye = y[:, :, element]
        u_t = lift(it) do current_t
            return ue[:, :, current_t]
        end

        hm_e = surface!(ax, 
            xe, ye, u_t,
            colorrange = (c_min, c_max), 
            colormap = :viridis,
            transparency = true,
            shading = NoShading
        )
    end
        Colorbar(fig[1, 2], limits=(c_min, c_max), label = "Value")


    sg = SliderGrid(fig[2, 1],
        (label = "Time Step", range = 1:size(u, 4), startvalue = 1))
    
    on(sg.sliders[1].value) do val
        it[] = val
        ax.title = "Time index: $val"
    end

    display(fig)

    if save_file
        save(filename, fig)
    end

    return fig
end

function plot_fe_2d_time_compare(x, y, e, u_exact, u_approx;
    save_file = false, filename = "solution.png", c_min = nothing, c_max = nothing)

    # Create a new figure
    fig = Figure(size = (800, 400))
    it = Observable(1)
    ax_exact = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "Exact Solution")
    ax_approx = Axis(fig[1, 2], xlabel = "x", ylabel = "y", title = "Approximate Solution")

    # We determine the min and max values
    if c_min === nothing
        c_min = minimum(u_exact)
    end
    if c_max === nothing
        c_max = maximum(u_exact)
    end

    for element in e
        ue_exact = u_exact[:, :, element, :]
        ue_approx = u_approx[:, :, element, :]
        xe = x[:, :, element]
        ye = y[:, :, element]
        u_t_exact = lift(it) do current_t
            return ue_exact[:, :, current_t]
        end
        u_t_approx = lift(it) do current_t
            return ue_approx[:, :, current_t]
        end

        hm_e_exact = surface!(ax_exact, 
            xe, ye, u_t_exact,
            colorrange = (c_min, c_max), 
            colormap = :viridis,
            transparency = true,
            shading = NoShading
        )
        
        hm_e_approx = surface!(ax_approx, 
            xe, ye, u_t_approx,
            colorrange = (c_min, c_max), 
            colormap = :viridis,
            transparency = true,
            shading = NoShading
        )
    end
        Colorbar(fig[1, 3], limits=(c_min, c_max), label = "Value")


    sg = SliderGrid(fig[2, :],
        (label = "Time Step", range = 1:size(u_exact, 4), startvalue = 1))
    
    on(sg.sliders[1].value) do val
        it[] = val
    end

    display(fig)

    if save_file
        save(filename, fig)
    end

    return fig
end

function interpolate_to_equidistant(x, y, z, u)
    # Define the new equidistant grid
    nx, ny, nz = size(u, 1), size(u, 2), size(u, 3)
    x_new = range(minimum(x), maximum(x), length=nx)
    y_new = range(minimum(y), maximum(y), length=ny)
    z_new = range(minimum(z), maximum(z), length=nz)

    u_out = zeros(nx, ny, nz)
    
    for i in 1:nx, j in 1:ny, k in 1:nz
        xi, yi, zi = x_new[i], y_new[j], z_new[k]

        ref_i = searchsortedlast(x[:, 1, 1], xi)
        ref_j = searchsortedlast(y[1, :, 1], yi)
        ref_k = searchsortedlast(z[1, 1, :], zi)
        ref_ip = min(searchsortedlast(x[:, 1, 1], xi)+ 1, nx)
        ref_jp = min(searchsortedlast(y[1, :, 1], yi)+ 1, ny)
        ref_kp = min(searchsortedlast(z[1, 1, :], zi)+ 1, nz)
        xn = x[ref_i, ref_j, ref_k]
        yn = y[ref_i, ref_j, ref_k]
        zn = z[ref_i, ref_j, ref_k]

        # 2. Get relative distances (normalized 0 to 1)
        xd = (xi - xn) / (x[ref_ip, ref_j, ref_k] - xn)
        yd = (yi - yn) / (y[ref_i, ref_jp, ref_k] - yn)
        zd = (zi - zn) / (z[ref_i, ref_j, ref_kp] - zn)
        
        # 3. Perform trilinear interpolation
        # Extract the 8 surrounding corner values
        c000 = u[ref_i,   ref_j,   ref_k]
        c100 = u[ref_ip, ref_j,   ref_k]
        c010 = u[ref_i,   ref_jp,  ref_k]
        c110 = u[ref_ip, ref_jp,  ref_k]
        c001 = u[ref_i,   ref_j,   ref_kp]
        c101 = u[ref_ip, ref_j,   ref_kp]
        c011 = u[ref_i,   ref_jp,  ref_kp]
        c111 = u[ref_ip, ref_jp,  ref_kp]
        
        # Interpolate along x
        c00 = ref_i != ref_ip ? c000 * (1 - xd) + c100 * xd : c000
        c01 = ref_i != ref_ip ? c001 * (1 - xd) + c101 * xd : c001
        c10 = ref_i != ref_ip ? c010 * (1 - xd) + c110 * xd : c010
        c11 = ref_i != ref_ip ? c011 * (1 - xd) + c111 * xd : c011
        
        # Interpolate along y
        c0 = ref_j != ref_jp ? c00 * (1 - yd) + c10 * yd : c00
        c1 = ref_j != ref_jp ? c01 * (1 - yd) + c11 * yd : c01
        
        # Interpolate along z
        u_out[i, j, k] = ref_k != ref_kp ? c0 * (1 - zd) + c1 * zd : c0
    end
    
    return x_new, y_new, z_new, u_out
end

function plot_fe_3d_time(x, y, z, e, u;
    save_file = false, filename = "solution.png", c_min = nothing, c_max = nothing)

    # Create a new figure
    fig = Figure(size = (400, 400))
    it = Observable(1)
    ax = Axis3(fig[1, 1], xlabel = "x", ylabel = "y", zlabel = "z")

    # We determine the min and max values
    if c_min === nothing
        c_min = minimum(u)
    end
    if c_max === nothing
        c_max = maximum(u)
    end

    for element in e
        ue = u[:, :, :, element, :]
        xe = x[:, :, :, element]
        ye = y[:, :, :, element]
        ze = z[:, :, :, element]

        # 1. Calculate the spatial grid bounds ONCE (using t=1 as a dummy)
        # We only need x_eq, y_eq, z_eq; they don't change with time.
        x_eq, y_eq, z_eq, _ = interpolate_to_equidistant(xe, ye, ze, ue[:, :, :, 1])

        # 2. Lift the interpolation logic. 
        # Whenever `it` changes, this block re-runs and generates a new u_out array.
        u_eq_obs = lift(it) do current_t
            # Extract current slice
            u_slice = ue[:, :, :, current_t]
            # Interpolate and return ONLY the u_out part
            _, _, _, u_out = interpolate_to_equidistant(xe, ye, ze, u_slice)
            return u_out
        end

        # 3. Pass the observable (u_eq_obs) directly to volume!
        hm_e = volume!(ax, 
            (x_eq[1], x_eq[end]), 
            (y_eq[1], y_eq[end]), 
            (z_eq[1], z_eq[end]), 
            u_eq_obs, # <- Observable passed here!
            colorrange = (c_min, c_max), 
            colormap = :viridis, 
            algorithm = :mip, 
            transparency = true, 
            interpolate = true
        )
    end
        # Add a colorbar
        Colorbar(fig[1, 2], limits=(c_min, c_max), label = "Value")


    sg = SliderGrid(fig[2, 1],
        (label = "Time Step", range = 1:size(u, 5), startvalue = 1))
    
    # Link slider to the Observable
    on(sg.sliders[1].value) do val
        it[] = val
        ax.title = "Time index: $val" # Update title dynamically
    end

    # Display the figure
    display(fig)

    # Optional: Save to file
    if save_file
        save(filename, fig)
    end

    return fig
end

function plot_fe_3d_time_compare(x, y, z, e, u_exact, u_approx;
    save_file = false, filename = "solution.png", c_min = nothing, c_max = nothing)

    # Create a new figure
    fig = Figure(size = (800, 400))
    it = Observable(1)
    ax_exact = Axis3(fig[1, 1], xlabel = "x", ylabel = "y", zlabel = "z", title = "Exact Solution")
    ax_approx = Axis3(fig[1, 2], xlabel = "x", ylabel = "y", zlabel = "z", title = "Approximate Solution")

    # We determine the min and max values
    if c_min === nothing
        c_min = minimum(u_exact)
    end
    if c_max === nothing
        c_max = maximum(u_exact)
    end

    for element in e
        ue_exact = u_exact[:, :, :, element, :]
        ue_approx = u_approx[:, :, :, element, :]
        xe = x[:, :, :, element]
        ye = y[:, :, :, element]
        ze = z[:, :, :, element]

        # 1. Calculate the spatial grid bounds ONCE (using t=1 as a dummy)
        # We only need x_eq, y_eq, z_eq; they don't change with time.
        x_eq, y_eq, z_eq, _ = interpolate_to_equidistant(xe, ye, ze, ue_exact[:, :, :, 1])

        # 2. Lift the interpolation logic. 
        # Whenever `it` changes, this block re-runs and generates a new u_out array.
        u_eq_exact_obs = lift(it) do current_t
            # Extract current slice
            u_slice_exact = ue_exact[:, :, :, current_t]
            # Interpolate and return ONLY the u_out part
            _, _, _, u_out_exact = interpolate_to_equidistant(xe, ye, ze, u_slice_exact)
            return u_out_exact
        end

        u_eq_approx_obs = lift(it) do current_t
            # Extract current slice
            u_slice_approx = ue_approx[:, :, :, current_t]
            # Interpolate and return ONLY the u_out part
            _, _, _, u_out_approx = interpolate_to_equidistant(xe, ye, ze, u_slice_approx)
            return u_out_approx
        end

        # 3. Pass the observable (u_eq_obs) directly to volume!
        hm_e_exact = volume!(ax_exact, 
            (x_eq[1], x_eq[end]), 
            (y_eq[1], y_eq[end]), 
            (z_eq[1], z_eq[end]), 
            u_eq_exact_obs, # <- Observable passed here!
            colorrange = (c_min, c_max), 
            colormap = :viridis, 
            algorithm = :mip, 
            transparency = true, 
            interpolate = true
        )

        hm_e_approx = volume!(ax_approx, 
            (x_eq[1], x_eq[end]), 
            (y_eq[1], y_eq[end]), 
            (z_eq[1], z_eq[end]), 
            u_eq_approx_obs, # <- Observable passed here!
            colorrange = (c_min, c_max), 
            colormap = :viridis, 
            algorithm = :mip, 
            transparency = true, 
            interpolate = true
        )
    end
        # Add a colorbar
        Colorbar(fig[1, 3], limits=(c_min, c_max), label = "Value")


    sg = SliderGrid(fig[2, :],
        (label = "Time Step", range = 1:size(u_exact, 5), startvalue = 1))
    
    # Link slider to the Observable
    on(sg.sliders[1].value) do val
        it[] = val
    end

    # Display the figure
    display(fig)

    # Optional: Save to file
    if save_file
        save(filename, fig)
    end

    return fig
end