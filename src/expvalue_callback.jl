using ITensors: OneITensor

export ExpValueCallback

const ExpValueSeries = Vector{ComplexF64}

struct ExpValueCallback <: TEvoCallback
    operators::Vector{LocalOperator}
    sites::Vector{<:Index}
    measurements::Dict{LocalOperator,ExpValueSeries}
    norm::ExpValueSeries
    times::Vector{Float64}
    measure_timestep::Float64
end

"""
    ExpValueCallback(
        operators::Vector{LocalOperator}, sites::Vector{<:Index}, measure_timestep::Float64
    )

Construct an `ExpValueCallback`, providing an array `operators` of `LocalOperator` objects
representing operators associated to specific sites. Each of them will be measured
on the given site during every step of the time evolution, and the results recorded inside
the `ExpValueCallback` object as an `ExpValueSeries` for later analysis. The norm of the
state, or the an equivalent quantity (trace, overlap of two sites...) where applicable, will
also be computed after each step.
The array `sites` is the same list of sites indices used to define MPSs and MPOs for the
calculations.
"""
function ExpValueCallback(
    operators::Vector{LocalOperator}, sites::Vector{<:Index}, measure_timestep::Float64
)
    return ExpValueCallback(
        operators,
        sites,
        Dict(x => ExpValueSeries() for x in operators),
        # A single ExpValueSeries for each operator in the list.
        ExpValueSeries(),  # for the norm, or the trace
        Vector{Float64}(),  # time instants of measurement steps
        measure_timestep,
    )
end

measurement_ts(cb::ExpValueCallback) = cb.times
measurements(cb::ExpValueCallback) = cb.measurements
callback_dt(cb::ExpValueCallback) = cb.measure_timestep
measurements_norm(cb::ExpValueCallback) = cb.norm
ops(cb::ExpValueCallback) = cb.operators
sites(cb::ExpValueCallback) = cb.sites

function Base.show(io::IO, cb::ExpValueCallback)
    println(io, "ExpValueCallback")
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

checkdone!(cb::ExpValueCallback, args...) = false

function _expval_while_sweeping(state::MPS, l::LocalOperator)
    # Find the relevant site range as the smallest interval of (consecutive) sites that
    # includes the orthocentre of the state and the support of the operator. We contract
    # the MPS only on those sites, and rely on the canonical simplification rules for the
    # other tensors.
    # This function is (ideally) called with `l` such that the lower bound of its domain
    # is also the orthocentre of the state MPS, but let's calculate the site range in a more
    # generic way anyway.
    site_range =
        minimum([orthocenter(state); domain(l)]):maximum([orthocenter(state); domain(l)])

    x = OneITensor()
    for n in site_range
        if n in domain(l)
            x *=
                prime(dag(state[n]); tags="Link") *
                apply(op(l[n], siteind(state, n)), state[n])
        else
            x *= prime(dag(state[n]); tags="Link") * state[n]
        end
    end
    # Now `x` is a tensor with indices
    #   (dim=##|id=##|"Link,l=L")'
    #   (dim=##|id=##|"Link,l=L")
    #   (dim=##|id=##|"Link,l=R")'
    #   (dim=##|id=##|"Link,l=R")
    # where L is the minimum of `site_range` and R its maximum. We contract these dangling
    # indices and obtain the expectation value. Anyway there will be two pairs of
    # primed/unprimed indices to contract. We'll find them and pair them in a more automatic
    # way, without looking at their site number (let's not rely on the presence of this tag,
    # in the future it may not be there anymore).
    i0 = inds(x; plev=0)
    i1 = inds(x; plev=1)
    for j in i0
        k = i1[findfirst(isequal(j'), i1)]
        x *= delta(dag(j), dag(k))
    end
    return scalar(x)
end

"""
    measure_localops!(cb::ExpValueCallback, state::MPS, site::Int, alg::TDVP1)

Measure on the MPS `state` each operator whose support starts on `site` defined inside the
callback object `cb`.
"""
function measure_localops!(cb::ExpValueCallback, state::MPS, site::Int, alg::TDVP1)
    # The `state[site]` block has just been updated and we should be in the middle of an
    # evolution step, before the 0-site evolution happens and the orthocentre of the state
    # is shifted left. We will measure all operators whose support starts on `site`.
    # Operators whose support is contained in `site+1:end` have already been measured in
    # previous calls of this function.
    for localop in filter(l -> first(domain(l)) == site, ops(cb))
        measurements(cb)[localop][end] = _expval_while_sweeping(state, localop)
        # `measurements(cb)[localop][end]` is the last line in the measurements of `localop`
        # which we (must) have created in `apply!` before calling this function.
    end

    if site == 1
        measurements_norm(cb)[end] = norm(state)
        # No optimisation needed here---ITensors uses the orthocentre only already.
    end

    return nothing
end

"""
    measure_localops!(cb::ExpValueCallback, ψ₁::MPS, ψ₂::MPS, alg::TDVP1)

Compute the inner product ``⟨ψ₁, Aψ₂⟩`` for each operator ``A``, defined
in the callback object ``cb``.
"""
function measure_localops!(cb::ExpValueCallback, psiL::MPS, psiR::MPS, alg::TDVP1)
    # Here we can't use `_expval_while_sweeping` because the two MPS which sandwich the
    # operator are different, so the "free" tensors do not cancel when contracting.
    # This function is meant to be called at the end of the sweep.
    for l in ops(cb)
        # TODO Consider whether it makes sense to memoize the product of operators here.
        # We could at least save the partial contractions, since we're measuring all the
        # observables at the same time (if anything, we get the overlap as a bonus).
        lop = prod(
            op(opname, siteind(psiR, opsite)) for
            (opname, opsite) in zip(factors(l), domain(l))
        )
        measurements(cb)[l][end] = dot(psiL, apply(lop, psiR))
        # measurements(cb)[localop][end] is the last line in the measurements of localop,
        # which we (must) have created in apply! before calling this function.
    end

    measurements_norm(cb)[end] = dot(psiL, psiR)

    return nothing
end

"""
    measure_localops!(cb::ExpValueCallback, ψ::MPS, alg::TDVP1vec)

Measure each operator defined inside the callback object `cb` on the state `ψ`.
"""
function measure_localops!(cb::ExpValueCallback, ψ::MPS, alg::TDVP1vec)
    # With TDVP1vec algorithms the situation is much simpler than with simple TDVP1: since
    # we need to contract any site which is not "occupied" (by the operator which is to be
    # measured) anyway with vec(I), we don't need to care about the orthocenter, we just
    # measure everything at the end of the sweep.

    # We contract each tensor from `ψ` with the identity, separately.
    ids = [state("vId", siteind(ψ, n)) * ψ[n] for n in eachindex(ψ)]

    for l in ops(cb)
        # Compute the expectation values by multiplying the tensor of the LocalOperator and
        # the precomputed identities on the otherx sites.
        x = OneITensor()
        for n in eachindex(ψ)
            if n in domain(l)
                x *= state("v" * l[n], siteind(ψ, n)) * ψ[n]
                # Note that contrary to `inner` or `dot`, this simple product of tensors
                # does not imply any complex conjugation, i.e.
                #
                #   state("vA", s) = apply(op("A⋅", s), state("vId", s))
                #
                # behaves as follows (`t` is an ITensor with index `s`):
                #
                #   state("vA", s) * t == state("vId", s) * apply(op("A⋅", s), t)
                #
                # so we should not use `dag` on the measured operator here.
            else
                x *= ids[n]
            end
        end
        measurements(cb)[l][end] = scalar(x)
        # `measurements(cb)[l][end]` is the last element in the measurements of `l`,
        # which we (must) have created in `apply!` before calling this function.
    end

    # Now measure the trace, too.
    measurements_norm(cb)[end] = scalar(prod(ids))

    return nothing
end

function apply!(
    cb::ExpValueCallback, state::MPS, alg::TDVP1; t, sweepend, sweepdir, site, kwargs...
)
    if isempty(measurement_ts(cb))
        prev_t = 0.0
        # Initialize `cb` here.
        push!(measurement_ts(cb), t)
        foreach(x -> push!(x, zero(eltype(x))), values(measurements(cb)))
        push!(measurements_norm(cb), zero(eltype(measurements_norm(cb))))
    else
        prev_t = measurement_ts(cb)[end]
    end

    # We perform measurements only at the end of a sweep and at measurement steps.
    # For TDVP we can perform measurements to the right of each site when sweeping back left.
    if (t - prev_t ≈ callback_dt(cb) || t == prev_t) && sweepend && sweepdir == "left"
        @debug "Computing expectation values on site $site at t = $t (prev_t = $prev_t)"
        if t != prev_t
            # Add the current time to the list of time instants at which we measured
            # something.
            push!(measurement_ts(cb), t)
            # Create a new slot in which we will put the measurement result.
            foreach(x -> push!(x, zero(eltype(x))), values(measurements(cb)))
            if site == 1
                push!(measurements_norm(cb), zero(eltype(measurements_norm(cb))))
            end
        end
        measure_localops!(cb, state, site, alg)
    end

    return nothing
end

function apply!(
    cb::ExpValueCallback,
    state1::MPS,
    state2::MPS,
    alg::TDVP1;
    t,
    sweepend,
    sweepdir,
    kwargs...,
)
    if isempty(measurement_ts(cb))
        prev_t = 0
    else
        prev_t = measurement_ts(cb)[end]
    end

    # We perform measurements only at the end of a sweep and at measurement steps.
    # For TDVP we can perform measurements to the right of each site when sweeping back left.
    if (t - prev_t ≈ callback_dt(cb) || t == prev_t) && sweepend && sweepdir == "left"
        @debug "Computing expectation values at t = $t (prev_t = $prev_t)"
        if (t != prev_t || t == 0)
            # Add the current time to the list of time instants at which we measured
            # something.
            push!(measurement_ts(cb), t)
            # Create a new slot in which we will put the measurement result.
            foreach(x -> push!(x, zero(eltype(x))), values(measurements(cb)))
            push!(measurements_norm(cb), zero(eltype(measurements_norm(cb))))
        end
        measure_localops!(cb, state1, state2, alg)
    end

    return nothing
end

function apply!(cb::ExpValueCallback, state::MPS, alg::TDVP1vec; t, sweepend, kwargs...)
    if isempty(measurement_ts(cb))
        prev_t = 0
    else
        prev_t = measurement_ts(cb)[end]
    end

    # We perform measurements only at the end of a sweep and at measurement steps.
    if (t - prev_t ≈ callback_dt(cb) || t == prev_t) && sweepend
        @debug "Computing expectation values at t = $t (prev_t = $prev_t)"
        if (t != prev_t || t == 0)
            # Add the current time to the list of time instants at which we measured
            # something.
            push!(measurement_ts(cb), t)
            # Create a new slot in which we will put the measurement result.
            foreach(x -> push!(x, zero(eltype(x))), values(measurements(cb)))
            push!(measurements_norm(cb), zero(eltype(measurements_norm(cb))))
        end
        measure_localops!(cb, state, alg)
    end

    return nothing
end
