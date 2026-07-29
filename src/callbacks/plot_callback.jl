function PlotCallback(interval::Int = 100)
    return (problem, u, loss, d_loss, iter) -> begin
        if iter % interval == 0 || iter == 1
            plot(problem, u)
        end
        return nothing
    end
end