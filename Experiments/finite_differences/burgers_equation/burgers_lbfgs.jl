using Odil
include("discretization.jl")
include("reference.jl")

Nx = 32
Nt = 32

x = range(0, 1, length=Nx)
t = range(0, 1, length=Nt)

dx = x[2] - x[1]

p = (Nx, dx)

u_t0  = [get_initial_burgers(x[ix]) for ix in 1:Nx]
u_bounds_left = [0 for it in 2:Nt]
u_bounds_right = [0 for it in 2:Nt]
u_reference_vals = [u_t0; u_bounds_left; u_bounds_right]

idx = LinearIndices((Nx, Nt))

idx_t0  = [idx[ix, 1] for ix in 1:Nx]
idx_bounds_left = [idx[1, it] for it in 2:Nt]
idx_bounds_right = [idx[Nx, it] for it in 2:Nt]
reference_val_indices = [idx_t0; idx_bounds_left; idx_bounds_right]

max_iterations = 10000

callback_set = OdilCallbackSet(PlotCallback(100))

problem = OdilProblem(timestep!, p, Nx, u_reference_vals, reference_val_indices, t, x; timestep_alloc_size = Nx)
u = odil_lbfgs(problem; max_iterations = max_iterations, callback_set = callback_set)

u = reshape(u, Nx, Nt)

plot_1d_time(x, t, u)