"""
Create a callback that prints the current iteration and loss norm every `interval` steps.
"""
function InfoPrintCallback(interval::Int = 10)
    return (problem, u, loss, d_loss, iter) -> begin
        if iter % interval == 0 || iter == 1
            println("Iteration ", iter, ": Loss = ", norm(loss))
        end
        return nothing
    end
end