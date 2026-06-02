using SciMLBase, OptimizationBase, OptimizationLBFGSB, ADTypes, Zygote

function odil_gauss_newton(ode:ODEProblem, lhs, p_lhs)
    u_t0 = ode.u0
    u_iter0 = zeros(size(u_t0), length(ode.tspan))
    du_rhs = zeros(size(u_t0), length(ode.tspan))
    du_lhs = zeros(size(u_t0), length(ode.tspan))
    function loss(ut, p)
        p_rhs, p_lhs = p
        u, t = ut
        l_init = u[:, :, end] - u_t0
        l_inner_boundary = ode.f(du_rhs, u, t, p_rhs) - lhs(du_lhs, u, t, p_lhs)
        l = norm(l_init)^2 + norm(l_inner_boundary)^2
        return l
    end
    prob = OptimizationProblem(loss, u_iter0, (ode.p, p_lhs))
    opt = LBFGSB()
    res = optimize(prob, opt)
    return res
end