export TEvoCallback,
    NoTEvoCallback, SpecCallback, measurement_ts, measurements_norm, callback_dt, expvalues

"""
A TEvoCallback can implement the following methods:

- apply!(cb::TEvoCallback, psi ; t, kwargs...): apply the callback with the
current state `psi` (e.g. perform some measurement)

- checkdone!(cb::TEvoCallback, psi; t, kwargs...): check whether some criterion
to stop the time evolution is satisfied (e.g. convergence of cbervable, error
too large) and return `true` if so.

- callback_dt(cb::TEvoCallback): time-steps at which the callback needs access
for the wave-function (e.g. for measurements). This is used for TEBD evolution
where several time-steps are bunched together to reduce the cost.
"""
abstract type TEvoCallback end

"""
    NoTEvoCallback is a trivial implementation of an evolution callback (<:TEvoCallback)
    object.
"""
struct NoTEvoCallback <: TEvoCallback end

apply!(cb::NoTEvoCallback, args...; kwargs...) = nothing
checkdone!(cb::NoTEvoCallback, args...; kwargs...) = false
callback_dt(cb::NoTEvoCallback) = 0

"""
    expvalues(cb::ExpValueCallback)
    expvalues(cb::ExpValueCallback, lop::LocalOperator)
    expvalues(cb::ExpValueCallback, name::AbstractString)

    expvalues(cb::SuperfermionCallback)
    expvalues(cb::SuperfermionCallback, lop::LocalOperator)
    expvalues(cb::SuperfermionCallback, name::AbstractString)

Retrieve the expectation values of the operators stored in the callback `cb`.  The time
series of an individual operator `lop` within `cb` can be directly accessed by
`expvalues(cb, lop)`, where `lop` is either a `LocalOperator` or a string that defines a
single `LocalOperator`.
"""
function expvalues end

expvalues(cb::NoTEvoCallback) = nothing

"""
    A Measurement object is an alias for `Vector{Float64}`, in other words an
    array of real numbers.

    Given a Measurement `M`, the result for the measurement at step `n` is `M[n]`.
"""
const Measurement = Vector{Float64}

struct SpecCallback <: TEvoCallback
    truncerrs::Vector{Float64}
    current_truncerr::Base.RefValue{Float64}
    entropies::Measurement
    bonddims::Vector{Vector{Int64}}
    bonds::Vector{Int64}
    ts::Vector{Float64}
    dt_measure::Float64
end

function SpecCallback(dt, psi::MPS, bonds::Vector{Int64}=collect(1:(length(psi) - 1)))
    bonds = sort(unique(bonds))
    if maximum(bonds) > length(psi) - 1 || minimum(bonds) < 1
        throw("bonds must be between 1 and $(length(psi)-1)")
    end
    return SpecCallback(
        Vector{Float64}(),
        Ref(0.0),
        Measurement(),
        Vector{Vector{Int64}}(),
        bonds,
        Vector{Float64}(),
        dt,
    )
end

measurement_ts(cb::SpecCallback) = cb.ts

function measurements(cb::SpecCallback)
    return Dict(
        "entropy" => cb.entropies, "bonddim" => cb.bonddims, "truncerrs" => cb.truncerrs
    )
end

callback_dt(cb::SpecCallback) = cb.dt_measure
bonds(cb::SpecCallback) = cb.bonds

function Base.show(io::IO, cb::SpecCallback)
    println(io, "SpecCallback")
    if length(measurement_ts(cb)) > 0
        println(
            io, "Measured times: ", callback_dt(cb):callback_dt(cb):measurement_ts(cb)[end]
        )
    else
        println(io, "No measurements performed")
    end
end

function apply!(cb::SpecCallback, psi; t, sweepend, bond, spec, sweepdir, kwargs...)
    cb.current_truncerr[] += truncerror(spec)
    prev_t = length(measurement_ts(cb)) > 0 ? measurement_ts(cb)[end] : 0
    if (t - prev_t ≈ callback_dt(cb) || t == prev_t) && sweepend
        if t != prev_t
            push!(measurement_ts(cb), t)
            push!(cb.bonddims, zeros(Int64, length(cb.bonds)))
            push!(cb.entropies, zeros(length(cb.bonds)))
        end

        if bond in bonds(cb)
            i = findfirst(x -> x == bond, bonds(cb))
            cb.bonddims[end][i] = length(eigs(spec))
            cb.entropies[end][i] = entropy(spec)
        end
        if sweepdir == "right" && bond == length(psi) - 1
            push!(cb.truncerrs, cb.current_truncerr[])
        elseif sweepdir == "left" && bond == 1
            push!(cb.truncerrs, cb.current_truncerr[])
        end
    end
end

checkdone!(cb::SpecCallback, args...) = false
