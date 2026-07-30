mutable struct OdilCallbackSet
    callbacks::Vector{Function}
end

OdilCallbackSet(cb::Function) = OdilCallbackSet(Function[cb])

OdilCallbackSet(cbs...) = OdilCallbackSet(Function[cbs...])

Base.iterate(cs::OdilCallbackSet, state...) = iterate(cs.callbacks, state...)
Base.length(cs::OdilCallbackSet) = length(cs.callbacks)

function add!(cs::OdilCallbackSet, cb::Function)
    if !(cb in cs.callbacks)
        push!(cs.callbacks, cb)
    end
end

function (cs::OdilCallbackSet)(problem, u, loss, d_loss, iter)
    for cb in cs.callbacks
        cb(problem, u, loss, d_loss, iter)
    end
end