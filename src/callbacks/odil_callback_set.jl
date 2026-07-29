mutable struct OdilCallbackSet{F}
    callbacks::Vector{F}
end

OdilCallbackSet(cbs...) = OdilCallbackSet(collect(cbs))

Base.iterate(cs::OdilCallbackSet, state...) = iterate(cs.callbacks, state...)
Base.length(cs::OdilCallbackSet) = length(cs.callbacks)

function add!(cs::OdilCallbackSet, cb::F) where {F}
    if !(cb in cs.callbacks)
        push!(cs.callbacks, cb)
    end
end

function (cs::OdilCallbackSet)(problem, u, loss, d_loss, iter)
    for cb in cs.callbacks
        cb(problem, u, loss, d_loss, iter)
    end
end