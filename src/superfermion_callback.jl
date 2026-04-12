export SuperfermionCallback

struct SuperfermionCallback <: TEvoCallback
    operators::Vector{LocalOperator}
    sites::Vector{<:Index}
    measurements::Dict{LocalOperator,ExpValueSeries}
    norm::ExpValueSeries
    times::Vector{Float64}
    measure_timestep::Float64
end

_sf_translate_sites(n::Int) = 2n-1
_sf_translate_sites_inv(n::Int) = div(n+1, 2)
function _sf_translate_sites(op::LocalOperator)
    LocalOperator(Dict(_sf_translate_sites(k) => v for (k, v) in op.terms))
end
function _sf_translate_sites_inv(op::LocalOperator)
    LocalOperator(Dict(_sf_translate_sites_inv(k) => v for (k, v) in op.terms))
end

"""
    SuperfermionCallback(
        operators, sites::Vector{<:Index}, measure_timestep::Float64
    )

Construct a `SuperfermionCallback` providing some `operators` and a list of ITensor `sites`.
The `operator` variable can be either a vector of `LocalOperator` objects, or a string (see
`parseoperators` for instructions on the allowed syntax).  Each operator will be measured
on the given site during every step of the time evolution, and the results recorded inside
the `SuperfermionCallback` object as an `ExpValueSeries` for later analysis. The array
`sites` is the same basis of sites used to define the MPS and MPO for the calculations.

This struct is defined the same say as [`ExpValueCallback`](@ref), but it allows using
Julia's multiple dispatch features to choose the correct method to measure expectation
values.
"""
function SuperfermionCallback end

function SuperfermionCallback(
    operators::Vector{LocalOperator}, sites::Vector{<:Index}, measure_timestep::Float64
)
    return SuperfermionCallback(
        operators,
        sites,
        Dict(x => ExpValueSeries() for x in operators),
        # A single ExpValueSeries for each operator in the list.
        ExpValueSeries(),  # for the norm, or the trace
        Vector{Float64}(),  # time instants of measurement steps
        measure_timestep,
    )
end

function SuperfermionCallback(
    operators::AbstractString, sites::Vector{<:Index}, measure_timestep::Float64
)
    localops = parseoperators(operators)
    return SuperfermionCallback(
        localops,
        sites,
        Dict(x => ExpValueSeries() for x in localops),
        # A single ExpValueSeries for each operator in the list.
        ExpValueSeries(),  # for the norm, or the trace
        Vector{Float64}(),  # time instants of measurement steps
        measure_timestep,
    )
end

measurement_ts(cb::SuperfermionCallback) = cb.times
measurements(cb::SuperfermionCallback) = cb.measurements
measurements_norm(cb::SuperfermionCallback) = cb.norm
callback_dt(cb::SuperfermionCallback) = cb.measure_timestep
ops(cb::SuperfermionCallback) = cb.operators
sites(cb::SuperfermionCallback) = cb.sites

expvalues(cb::SuperfermionCallback) = sort(cb.measurements)
expvalues(cb::SuperfermionCallback, lop::LocalOperator) = cb.measurements[lop]
function expvalues(cb::SuperfermionCallback, name::AbstractString)
    expvalues(cb, LocalOperator(name))
end

function Base.show(io::IO, cb::SuperfermionCallback)
    println(io, "SuperfermionCallback")
    # Print the list of operators
    println(io, "Operators: ", join(name.(ops(cb)), ", ", " and "))
    if !isempty(measurement_ts(cb))
        println(
            io,
            "Measured times:",
            "\n  from ",
            first(measurement_ts(cb)),
            "\n  to ",
            last(measurement_ts(cb)),
            "\n  each ",
            callback_dt(cb),
        )
    else
        println(io, "No measurements performed")
    end
end

checkdone!(cb::SuperfermionCallback, args...) = false

"""
    adj(x)

Returns the adjoint (conjugate transpose) of x. It can be an ITensor, an MPS or an MPO.
Note that it is not the same as ITensors.adjoint.
"""
adj(x) = swapprime(dag(x), 0 => 1)

@memoize function _sf_id_pairs(sites)
    return [
        state(sites[n], "Emp") * state(sites[n + 1], "Occ") +
        state(sites[n], "Occ") * state(sites[n + 1], "Emp") for
        n in eachindex(sites)[1:2:end]
    ]
end

"""
    measure_localops!(cb::SuperfermionCallback, ψ::MPS, alg::TDVP1vec)

Measure each operator defined inside the callback object `cb` on the state `ψ`.
"""
function measure_localops!(cb::SuperfermionCallback, ψ::MPS, alg::TDVP1vec)
    # We follow the same logic as in
    # `measure_localops!(cb::ExpValueCallback, ψ::MPS, alg::TDVP1vec)`, but we work with
    # 2-site blocks at a time.

    # Contract each tensor from `ψ` with the identity, separately.
    sf_id_blocks = _sf_id_pairs(siteinds(ψ))
    ids = [
        dag(sf_id_blocks[_sf_translate_sites_inv(n)]) * ψ[n] * ψ[n + 1] for
        n in eachindex(ψ)[1:2:end]
    ]
    # Where the identity is not needed, we put a OneITensor as a placeholder, so that the
    # site enumeration is preserved.

    for l in ops(cb)
        # Compute the expectation values by multiplying the tensor of the LocalOperator and
        # the precomputed identities on the other sites.
        x = OneITensor()
        for n in eachindex(ψ)[1:2:end]
            if n in domain(l)
                lop = if n + 1 in domain(l)
                    # We loop over odd sites only, so we check manually that the next site
                    # is in the domain of the operator.
                    op(l[n], siteind(ψ, n)) * op(l[n + 1], siteind(ψ, n + 1))
                else
                    op(l[n], siteind(ψ, n))
                end
                x *=
                    dag(apply(adj(lop), sf_id_blocks[_sf_translate_sites_inv(n)])) *
                    ψ[n] *
                    ψ[n + 1]
            else
                x *= ids[_sf_translate_sites_inv(n)]
            end
        end
        measurements(cb)[l][end] = scalar(x)
        # `measurements(cb)[l][end]` is the last element in the measurements of `l`,
        # which we (must) have created in `apply!` before calling this function.
    end

    # Since computing `ids` might require a little time, we return it so that other methods
    # can reuse the results.
    return ids
end

function compute_trace!(
    cb::SuperfermionCallback, ids::Vector{ITensor}, alg::TDVP1vec; current_time
)
    if isempty(measurement_ts(cb))
        prev_t = 0
    else
        prev_t = measurement_ts(cb)[end]
    end
    # From precomputed `ids`: we just multiply the elements together.
    if current_time - prev_t ≈ callback_dt(cb) || current_time ≈ prev_t
        push!(measurements_norm(cb), scalar(prod(ids)))
    end

    return nothing
end

function compute_trace!(cb::SuperfermionCallback, ψ::MPS, alg::TDVP1vec; current_time)
    if isempty(measurement_ts(cb))
        prev_t = 0
    else
        prev_t = measurement_ts(cb)[end]
    end
    # From scratch: we contract each tensor from `ψ` with the identity, separately.
    if current_time - prev_t ≈ callback_dt(cb) || current_time ≈ prev_t
        sf_id_blocks = _sf_id_pairs(siteinds(ψ))
        ids = [
            dag(sf_id_blocks[_sf_translate_sites_inv(n)]) * ψ[n] * ψ[n + 1] for
            n in eachindex(ψ)[1:2:end]
        ]
        push!(measurements_norm(cb), scalar(prod(ids)))
    end

    return nothing
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
            #push!(measurements_norm(cb), zero(eltype(measurements_norm(cb))))
        end
        measure_localops!(cb, state, alg)
    end

    return nothing
end
