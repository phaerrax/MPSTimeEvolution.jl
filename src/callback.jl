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

### SpecCallback

# The following callback type is not actively used in this package (either in tests or in
# the documentation) but it's an interesting example of what we could do with callback: not
# just recording expectation values, but also more complicated operations on the MPS such as
# computing the bipartition entropy or storing the bond dimensions.

mutable struct SpecCallback <: TEvoCallback
    truncerrs::Vector{Float64}
    entropies::Vector{Float64}
    bonddims::Vector{Vector{Int}}
    bonds::Vector{Int}
    ts::Vector{Float64}
    dt_measure::Float64

    # This callback accumulates values (truncation errors, bond dimensions on swept bonds,
    # etc.) across multiple `apply!` calls (one per bond) within a single sweep. It needs
    # somewhere to hold the in-progress values between calls, since the function has no
    # state or storage of its own. For this reason we use a "scratch space" for the
    # `truncerrs`, `entropies` and `bonddims` fields above.
    # (This implies that we need to make the struct mutable --- another approach would be to
    # declare `current_truncerr` as `Base.RefValue{Float64}`, but that's discouraged now.)
    current_truncerr::Float64
    current_entropies::Vector{Float64}
    current_bonddims::Vector{Int}
end

function SpecCallback(dt, psi::MPS, bonds=1:(length(psi) - 1))
    bonds = sort(unique(bonds))
    if maximum(bonds) > length(psi) - 1 || minimum(bonds) < 1
        throw("bonds must be between 1 and $(length(psi)-1)")
    end
    return SpecCallback(
        Float64[],
        Float64[],
        Vector{Int}[],
        bonds,
        Float64[],
        dt,
        0.0,
        zeros(Float64, length(bonds)),
        zeros(Int, length(bonds)),
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
    cb::SpecCallback, state, alg; current_time, sweepend, bond, spec, sweepdir, kwargs...
)
    cb.current_truncerr += truncerror(spec)

    if sweepend
        on_schedule, _ = register_time!(cb, current_time)
        if on_schedule
            # measure_localops!(cb, state, alg)
            # This is disabled, for now, since there's no `measure_localops!` method anyway
            # that's targeted towards the SpecCallback type.

            if bond in bonds(cb)
                i = findfirst(==(bond), bonds(cb))
                cb.current_bonddims[i] = length(eigs(spec))
                cb.current_entropies[i] = entropy(spec)
            end

            if sweepdir == "left" && bond == 1
                # We're at the end of the time step: flush the temporary storage to the
                # actual result arrays. 
                push!(cb.truncerrs, cb.current_truncerr)
                push!(cb.bonddims, cb.current_bonddims)
                push!(cb.entropies, cb.current_entropies)
                # Reset the scratch arrays to zero right after each flush, so that the next
                # sweep starts writing from a clean slate. Not the truncation error though:
                # that one is a cumulative counter.
                cb.current_bonddims = zeros(Int, length(bonds(cb)))
                cb.current_entropies = zeros(Float64, length(bonds(cb)))
            end
        end
    end
end

checkdone!(cb::SpecCallback, args...) = false
