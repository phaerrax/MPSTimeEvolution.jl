export SuperfermionCallback

struct SuperfermionCallback <: TEvoCallback
    operators::Vector{LocalOperator}
    sites::Vector{<:Index}
    measurements::Dict{LocalOperator,ExpValueSeries}
    times::Vector{Float64}
    measure_timestep::Float64
end

"""
    SuperfermionCallback(
        operators::Vector{LocalOperator}, sites::Vector{<:Index}, measure_timestep::Float64
    )

Construct a `SuperfermionCallback` providing an array `operators` of `LocalOperator` objects
representing operators associated to specific sites. Each of them will be measured
on the given site during every step of the time evolution, and the results recorded inside
the `SuperfermionCallback` object as an `ExpValueSeries` for later analysis. The array
`sites` is the same basis of sites used to define the MPS and MPO for the calculations.

This struct is actually defined the same say as [`ExpValueCallback`](@ref), but it allows
using Julia's multiple dispatch features to choose the correct method to measure expectation
values.
"""
function SuperfermionCallback(
    operators::Vector{LocalOperator}, sites::Vector{<:Index}, measure_timestep::Float64
)
    return SuperfermionCallback(
        operators,
        sites,
        Dict(op => ExpValueSeries() for op in operators),
        # A single ExpValueSeries for each operator in the list.
        Vector{Float64}(),
        measure_timestep,
    )
end

measurement_ts(cb::SuperfermionCallback) = cb.times
measurements(cb::SuperfermionCallback) = cb.measurements
callback_dt(cb::SuperfermionCallback) = cb.measure_timestep
ops(cb::SuperfermionCallback) = cb.operators
sites(cb::SuperfermionCallback) = cb.sites

function Base.show(io::IO, cb::SuperfermionCallback)
    println(io, "SuperfermionCallback")
    # Print the list of operators
    println(io, "Operators: ", join(name.(ops(cb)), ", ", " and "))
    if Base.length(measurement_ts(cb)) > 0
        println(
            io, "Measured times: ", callback_dt(cb):callback_dt(cb):measurement_ts(cb)[end]
        )
    else
        println(io, "No measurements performed")
    end
end

checkdone!(cb::SuperfermionCallback, args...) = false

@memoize function identity_sf(sites)
    N = length(sites)
    @assert iseven(N)
    pairs = [
        add(MPS(sites[n:(n + 1)], "Emp"), MPS(sites[n:(n + 1)], "Occ"); alg="directsum") for
        n in 1:2:N if n + 1 ≤ N
    ]
    # Use the "direct sum" method for summing the MPSs so that ITensors doesn't waste time
    # re-orthogonalizing the result, which might also reverse the direction of some QN
    # arrows, and we don't want that.

    id = MPS(
        collect(Iterators.flatten((pairs[n][1], pairs[n][2]) for n in eachindex(pairs)))
    )

    vacuum = MPS(sites, "Emp")  # copy link indices from here
    # Here "Up" or "Dn" makes no difference, because we are interested only in the link
    # indices between sites 2j and 2j+1, and an even number of "Up" or "Dn" states doesn't
    # change the parity.

    for n in 2:2:(N - 1)
        id[n] *= state(linkind(vacuum, n), 1)
        id[n + 1] *= state(dag(linkind(vacuum, n)), 1)
    end
    return id
end

"""
    adj(x)

Returns the adjoint (conjugate transpose) of x. It can be an ITensor, an MPS or an MPO.
Note that it is not the same as ITensors.adjoint.
"""
adj(x) = swapprime(dag(x), 0 => 1)

@memoize function _sf_observable_mps(op, sites)
    id = identity_sf(sites)
    x = mpo(sites, op)
    return apply(adj(x), id)
end

"""
    measure_localops!(cb::SuperfermionCallback, state::MPS, alg::TDVP1vec)

Measure each operator defined inside the callback object `cb` on the state `state`.
"""
function measure_localops!(cb::SuperfermionCallback, state::MPS, alg::TDVP1vec)
    # With TDVP1vec algorithms the situation is much simpler than with simple TDVP1: since
    # we need to contract any site which is not "occupied" (by the operator which is to be
    # measured) anyway with vec(I), we don't need to care about the orthocenter, we just
    # measure everything at the end of the sweep.

    for localop in ops(cb)
        # Strategy:
        # 1. create the identity MPS (the so-called “left vacuum”)
        # 2. apply the adjoint of the local operator to it (this way we can memoize it)
        # 3. contract the result with the state
        measurements(cb)[localop][end] = dot(_sf_observable_mps(localop, sites(cb)), state)
        # measurements(cb)[localop][end] is the last element in the measurements of localop,
        # which we (must) have created in apply! before calling this function.
    end
end

function apply!(cb::SuperfermionCallback, state::MPS, alg::TDVP1vec; t, sweepend, kwargs...)
    if isempty(measurement_ts(cb))
        prev_t = 0
    else
        prev_t = measurement_ts(cb)[end]
    end

    if (t - prev_t ≈ callback_dt(cb) || t == prev_t) && sweepend
        if (t != prev_t || t == 0)
            # Add the current time to the list of time instants at which we measured
            # something.
            push!(measurement_ts(cb), t)
            # Create a new slot in which we will put the measurement result.
            foreach(x -> push!(x, zero(eltype(x))), values(measurements(cb)))
        end
        measure_localops!(cb, state, alg)
    end

    return nothing
end
