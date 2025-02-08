export ExpValueCallback

const ExpValueSeries = Vector{ComplexF64}

struct ExpValueCallback <: TEvoCallback
    operators::Vector{LocalOperator}
    sites::Vector{<:Index}
    measurements::Dict{LocalOperator,ExpValueSeries}
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
the `ExpValueCallback` object as an `ExpValueSeries` for later analysis. The array
`sites` is the same basis of sites used to define the MPS and MPO for the calculations.
"""
function ExpValueCallback(
    operators::Vector{LocalOperator}, sites::Vector{<:Index}, measure_timestep::Float64
)
    return ExpValueCallback(
        operators,
        sites,
        Dict(op => ExpValueSeries() for op in operators),
        # A single ExpValueSeries for each operator in the list.
        Vector{Float64}(),
        measure_timestep,
    )
end

measurement_ts(cb::ExpValueCallback) = cb.times
measurements(cb::ExpValueCallback) = cb.measurements
callback_dt(cb::ExpValueCallback) = cb.measure_timestep
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

    x = ITensors.OneITensor()
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
        # This works, but calculating the MPO from scratch every time might take too much
        # time, especially when it has to be repeated thousands of times. For example,
        # executing TimeEvoVecMPS.mpo(s, o) with
        #   s = siteinds("Osc", 400; dim=16)
        #   o = LocalOperator(Dict(20 => "A", 19 => "Adag"))
        # takes 177.951 ms (2313338 allocations: 329.80 MiB).
        # Memoizing this function allows us to cut the time (after the first call, which is
        # expensive anyway since Julia needs to compile the function) to 45.368 ns
        # (1 allocation: 32 bytes) for each call.
        lop = prod(
            op(opname, siteind(psiR, opsite)) for
            (opname, opsite) in zip(factors(l), domain(l))
        )
        measurements(cb)[l][end] = dot(psiL, apply(lop, psiR))
        # measurements(cb)[localop][end] is the last line in the measurements of localop,
        # which we (must) have created in apply! before calling this function.
    end
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

    for localop in ops(cb)
        # Transform each `localop` into an MPS, filling with `vId` states.
        measurements(cb)[localop][end] = dot(mps(sites(cb), localop), ψ)
        # measurements(cb)[localop][end] is the last element in the measurements of localop,
        # which we (must) have created in apply! before calling this function.
    end
end

function apply!(
    cb::ExpValueCallback, state::MPS, alg::TDVP1; t, sweepend, sweepdir, site, kwargs...
)
    if isempty(measurement_ts(cb))
        prev_t = 0.0
        # Initialize `cb` here.
        push!(measurement_ts(cb), t)
        foreach(x -> push!(x, zero(eltype(x))), values(measurements(cb)))
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
        end
        measure_localops!(cb, state, alg)
    end

    return nothing
end
