"""
Collects callbacks that are invoked during an ODIL optimization run.
"""
mutable struct OdilCallbackSet
    callbacks::Vector{Function}
end

OdilCallbackSet(cb::Function) = OdilCallbackSet(Function[cb])

OdilCallbackSet(cbs...) = OdilCallbackSet(Function[cbs...])

Base.iterate(cs::OdilCallbackSet, state...) = iterate(cs.callbacks, state...)
Base.length(cs::OdilCallbackSet) = length(cs.callbacks)

"""
Add a callback to the set unless it is already present.
"""
function add!(cs::OdilCallbackSet, cb::Function)
    if !(cb in cs.callbacks)
        push!(cs.callbacks, cb)
    end
end

"""
Call every registered callback for the current optimization state and iteration.
"""
function (cs::OdilCallbackSet)(problem, u, loss, d_loss, iter)
    for cb in cs.callbacks
        cb(problem, u, loss, d_loss, iter)
    end
end