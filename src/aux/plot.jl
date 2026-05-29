function plot_comparison(x, t, u_exact, u_approx;
    save_file = false)

# We determine the min and max values to ensure both plots share the same color scale
z_min = min(minimum(u_exact), minimum(u_approx))
z_max = max(maximum(u_exact), maximum(u_approx))

# Create the plot
# Note: If your data is u[time, space], add a transpose like: u_exact'
p = plot(
    heatmap(x, t, u_exact',    title="Exact Solution", xlabel="x", ylabel="t", clims=(z_min, z_max)),
    heatmap(x, t, u_approx',   title="Approximation",  xlabel="x", ylabel="t", clims=(z_min, z_max)),
    layout = (1, 2),           # 1 row, 2 columns
    size = (800, 400),         # Resolution
    c = :viridis               # Color map (optional)
)

# Display the plot
display(p)

# Optional: Save to file
# png(p, "simulation_result.png")

end