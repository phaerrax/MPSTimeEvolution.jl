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

function previous_recorded_time(cb::TEvoCallback)
    return if isempty(measurement_ts(cb))  # = is this the very first measurement?
        zero(callback_dt(cb))
    else
        last(measurement_ts(cb))
    end
    # The first case is obvious, it is triggered if we called `apply!` for the first time at
    # a time step where measurements should be performed. Since `apply!` can be invoked
    # multiple times at each time step, e.g. once for each site/bond within a sweep, we need
    # also to consider the second case.
end

function is_measurement_time(cb, current_time)
    prev_t = previous_recorded_time(cb)
    return current_time - prev_t ≈ callback_dt(cb) || current_time ≈ prev_t
end

"""
    register_time!(cb::TEvoCallback, current_time)

Given the current simulation time `current_time`, determine whether `current_time` lies on a
measurement instant for `cb`. Returns two `Bool` results (`on_schedule`, `is_new_step`):

- `on_schedule`: `true` if `current_time` is either exactly `callback_dt(cb)` past the last
  recorded time, or if this is a repeated call at the same `current_time`
- `is_new_step`: `true` only on the first call at a new `current_time`

If `is_new_step` is `true`, `current_time` is pushed onto `measurement_ts(cb)` as a side
effect.
"""
function register_time!(cb::TEvoCallback, current_time)
    prev_t = previous_recorded_time(cb)
    on_schedule = is_measurement_time(cb, current_time)

    is_new_step = if on_schedule && !isempty(measurement_ts(cb))
        current_time != prev_t
    else
        on_schedule
    end
    if is_new_step
        push!(measurement_ts(cb), current_time)
        @debug "Adding t = $current_time to the time instants recorded by the callback."
    end
    # We need to discriminate whether this is the first time that `current_time` is hit, in
    # which case the callback will need to allocate a fresh storage slot for each quantity
    # it tracks, before measuring into it.

    return on_schedule, is_new_step
end

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

function apply!(
    cb::SpecCallback, psi; current_time, sweepend, bond, spec, sweepdir, kwargs...
)
    cb.current_truncerr[] += truncerror(spec)

    if sweepend
        on_schedule, is_new_step = register_time!(cb, current_time)
        if on_schedule
            is_new_step && foreach(v -> push!(v, zero(eltype(v))), values(measurements(cb)))
            measure_localops!(cb, state, alg)

            if is_new_step
                push!(measurement_ts(cb), current_time)
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
end

checkdone!(cb::SpecCallback, args...) = false
