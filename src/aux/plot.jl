using GLMakie
using ColorSchemes

function plot_comparison(x, t, u_exact, u_approx;
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

function plot_2d(x, t, u;
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