using ITensors: OneITensor

export ExpValueCallback, SuperfermionCallback

const ExpValueSeries = Vector{ComplexF64}

### Struct definitions
# (They are identical, but are meant to represent two different ways of extracting physical
# information from the MPS. Method defined on them will be different.) 

struct ExpValueCallback <: TEvoCallback
    operators::Vector{LocalOperator}
    sites::Vector{<:Index}
    measurements::Dict{LocalOperator,ExpValueSeries}
    norm::ExpValueSeries
    times::Vector{Float64}
    measure_timestep::Float64
end

struct SuperfermionCallback <: TEvoCallback
    operators::Vector{LocalOperator}
    sites::Vector{<:Index}
    measurements::Dict{LocalOperator,ExpValueSeries}
    norm::ExpValueSeries
    times::Vector{Float64}
    measure_timestep::Float64
end

const MeasurementCallback = Union{ExpValueCallback,SuperfermionCallback}

# Helper functions for SuperfermionCallbacks (superfermion site mappings) 
_sf_translate_sites(n::Int) = 2n-1
_sf_translate_sites_inv(n::Int) = div(n+1, 2)
function _sf_translate_sites(op::LocalOperator)
    return LocalOperator(Dict(_sf_translate_sites(k) => v for (k, v) in op.terms))
end
function _sf_translate_sites_inv(op::LocalOperator)
    return LocalOperator(Dict(_sf_translate_sites_inv(k) => v for (k, v) in op.terms))
end

"""
    adj(x)

Returns the adjoint (conjugate transpose) of x. It can be an ITensor, an MPS or an MPO.
Note that it is not the same as ITensors.adjoint.
"""
adj(x) = swapprime(dag(x), 0 => 1)

### Constructors

"""
    ExpValueCallback(operators, sites::Vector{<:Index}, measure_timestep)

Construct an `ExpValueCallback`, providing some `operators` and a list of ITensor `sites`. 
The `operator` variable can be either a vector of `LocalOperator` objects, or a string (see
`parseoperators` for instructions on the allowed syntax).  Each operator will be
measured on the associated sites during every step of the time evolution, and the results
recorded inside the `ExpValueCallback` object as an `ExpValueSeries` for later analysis. The
norm of the state, or the an equivalent quantity (trace, overlap of two sites...) where
applicable, will also be computed after each step.
The array `sites` is the same list of sites indices used to define MPSs and MPOs for the
calculations.
"""
function ExpValueCallback end

"""
    SuperfermionCallback(
        operators::Vector{LocalOperator}, sites::Vector{<:Index}, measure_timestep
    )

Construct a `SuperfermionCallback` providing some `operators` and a list of ITensor `sites`.
The `operator` variable must be either a vector of `LocalOperator` objects, or a string (see
`parseoperators` for instructions on the allowed syntax). When adding multi-site operators,
remember to add fermionic strings, e.g. Jordan-Wigner factors, where appropriate. Each
operator will be measured on the given site during every step of the time evolution, and the
results recorded inside the `SuperfermionCallback` object as an `ExpValueSeries` for later
analysis. The trace of the state will also be computed after each step.
The array `sites` is the same basis of sites used to define the MPS and MPO for the
calculations.
"""
function SuperfermionCallback end

# Common ExpValueCallback-SuperfermionCallback constructor (the default one).
function (::Type{MCT})(
    operators::Vector{LocalOperator}, sites::Vector{<:Index}, measure_timestep
) where {MCT<:MeasurementCallback}
    return MCT(
        operators,
        sites,
        Dict(x => ExpValueSeries() for x in operators),
        # A single ExpValueSeries for each operator in the list.
        ExpValueSeries(),  # for the norm, or the trace
        Vector{Float64}(),  # time instants of measurement steps
        measure_timestep,
    )
end

# We can also allow the user to provide the operator with the "more natural" syntax of
# `parseoperators`.
function (::Type{MCT})(
    operators::AbstractString, sites::Vector{<:Index}, measure_timestep
) where {MCT<:MeasurementCallback}
    return MCT(parseoperators(operators), sites, measure_timestep)
end

### Common getter methods

measurement_ts(cb::MeasurementCallback) = cb.times
measurements(cb::MeasurementCallback) = cb.measurements
callback_dt(cb::MeasurementCallback) = cb.measure_timestep
measurements_norm(cb::MeasurementCallback) = cb.norm
ops(cb::MeasurementCallback) = cb.operators
sites(cb::MeasurementCallback) = cb.sites

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

expvalues(cb::MeasurementCallback) = sort(cb.measurements)  # sort to make output consistent
expvalues(cb::MeasurementCallback, lop::LocalOperator) = cb.measurements[lop]
function expvalues(cb::MeasurementCallback, name::AbstractString)
    return expvalues(cb, LocalOperator(name))
end

function Base.show(io::IO, cb::MCT) where {MCT<:MeasurementCallback}
    println(io, MCT)
    # Print the list of operators
    println(io, "Operators: ", join(name.(ops(cb)), ", ", " and "))
    if !isempty(measurement_ts(cb))
        print(
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
        print(io, "No measurements performed")
    end
end

### apply! methods

# Standard MPS + TDVP method: meant to be applied at every site along the final sweep of a
# time step. It measures expectation values only for observables specific to that site (thus
# it needs the `site` keyword argument).
function apply!(
    cb::ExpValueCallback,
    state::MPS,
    alg::Union{TDVP1,TDVP2};
    current_time,
    sweepend,
    sweepdir,
    site,
    kwargs...,
)
    # We perform measurements only at the end of a sweep and at measurement steps.
    # During the final leftwards sweep, all sites to the right of the currently evolved site
    # are updated to the current time, so they are safe to be considered for expectation
    # values.
    if sweepend && sweepdir == "left"
        on_schedule = register_time!(cb, current_time)
        if on_schedule
            @debug "Computing expectation values on site $site at t = $current_time"
            measure_localops!(cb, state, site, alg)
        end
    end

    return nothing
end

# TDVP1 & TDVP2 variant for inverse-canonical MPSs.
function apply!(
    cb::ExpValueCallback,
    state::InverseCanonicalMPS,
    alg::Union{TDVP1,TDVP2};
    current_time,
    sweepend,
    kwargs...,
)
    # Since with inverse-canonical MPSs we can compute expectation values in parallel, we
    # can do everything at the end of the time step.
    if sweepend
        on_schedule = register_time!(cb, current_time)
        if on_schedule
            @debug "Computing expectation values at t = $current_time"
            measure_localops!(cb, state, alg)
        end
    end

    return nothing
end

# Slightly different version with two MPSs, used to compute running overlaps and other inner
# products of operators between two different states.
function apply!(
    cb::ExpValueCallback,
    state1::MPS,
    state2::MPS,
    alg::TDVP1;
    current_time,
    sweepend,
    sweepdir,
    kwargs...,
)
    if sweepend && sweepdir == "left"
        on_schedule = register_time!(cb, current_time)
        if on_schedule
            @debug "Computing expectation values at t = $current_time"
            measure_localops!(cb, state1, state2, alg)
        end
    end

    return nothing
end

# Vectorised-TDVP1 version for both ExpValueCallback and SuperfermionCallback. Here we have
# no specific `site` because it makes no sense to optimise the function that way, so
# measurements are performed at the end of each time step.
function apply!(
    cb::MeasurementCallback, state::MPS, alg::TDVP1vec; current_time, sweepend, kwargs...
)
    if sweepend
        on_schedule = register_time!(cb, current_time)
        if on_schedule
            @debug "Computing expectation values at t = $current_time"
            measure_localops!(cb, state, alg)
        end
    end

    return nothing
end

# VidalMPS-specific version. There's no `measure_localops!` inside, because computing
# expectation values is done efficiently simply by calling `expect`, which is already
# optimised and only considers the sites affected by the operator we are going to measure.
function apply!(cb::ExpValueCallback, ψ::VidalMPS, alg::TEBD; current_time, kwargs...)
    on_schedule = register_time!(cb, current_time)
    if on_schedule
        @debug "Computing expectation values on site $site at t = $current_time"
        for localop in ops(cb)
            push!(measurements(cb)[localop], expect(ψ, localop))
        end
    end

    return nothing
end

### measure_localops! methods

# Helper function for `measure_localops!(::ExpValueCallback, ::MPS, ::Int, ::TDVP1)`.
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

# Compute the inner product ⟨ψ, Aψ⟩ for each operator A defined inside the callback object
# whose support starts on `site`.
function measure_localops!(cb::ExpValueCallback, ψ::MPS, site::Int, alg::TDVP1)
    # The `ψ[site]` block has just been updated and we should be in the middle of an
    # evolution step, before the 0-site evolution happens and ψ's orthocentre is shifted
    # left. We will measure all operators whose support starts on `site`.  Operators whose
    # support is contained in `site+1:end` have already been measured in previous calls of
    # this function.
    for localop in filter(l -> first(domain(l)) == site, ops(cb))
        push!(measurements(cb)[localop], _expval_while_sweeping(ψ, localop))
    end

    return nothing
end

function measure_localops!(cb::ExpValueCallback, state::MPS, b::Int, alg::TDVP2)
    # When we are sweeping right-to-left, once the block at sites (b, b+1) has been evolved,
    # the tensor at ψ[b + 1] has completed its evolution within the time step dt.
    # The MPS is
    # • left-orthogonal from ψ[1] to ψ[b - 1]
    # • right-orthogonal from ψ[b] to ψ[end]
    # so this is a good time to measure observables that are local to site b+1: when
    # contracting in inner(ψ', A(n), ψ), all the sites left of ψ[b] (excluded) give the
    # identity, and so do all those right of ψ[b + 1].  The measurement can then be
    # performed using the tensor composed by only ψ[b] and ψ[b + 1].

    # Operators whose support is contained in `site+1:end` have already been measured in
    # previous calls of this function.
    for localop in filter(l -> first(domain(l)) == b+1, ops(cb))
        push!(measurements(cb)[localop], _expval_while_sweeping(state, localop))
    end
    # If b == 1, meaning that the right-to-left sweep has ended, we also measure on the
    # first site.
    if b == 1
        for localop in filter(l -> first(domain(l)) == 1, ops(cb))
            push!(measurements(cb)[localop], _expval_while_sweeping(state, localop))
        end
    end
    return nothing
end

# Compute the inner product ⟨ψ, Aψ⟩ for each operator A defined inside the callback object.
function measure_localops!(
    cb::ExpValueCallback, state::InverseCanonicalMPS, alg::Union{TDVP1,TDVP2}
)
    # In the inverse-canonical gauge, measurements can be performed in parallel; we don't
    # need to reorthogonalise the MPS each time.
    Threads.@threads :greedy for localop in ops(cb)
        push!(measurements(cb)[localop], expect(state, localop))
    end

    return nothing
end

# Compute the inner product ⟨ψₗ, Aψᵣ⟩ for each operator A defined in the callback object.
# This function is meant to be called at the end of the time step.
function measure_localops!(cb::ExpValueCallback, psiL::MPS, psiR::MPS, alg::TDVP1)
    # Here we can't use `_expval_while_sweeping` because the two MPS which sandwich the
    # operator are different, so the "free" tensors do not cancel when contracting.
    for l in ops(cb)
        # TODO Consider whether it makes sense to memoize the product of operators here.
        # We could at least save the partial contractions, since we're measuring all the
        # observables at the same time (if anything, we get the overlap as a bonus).
        lop = prod(
            op(opname, siteind(psiR, opsite)) for
            (opname, opsite) in zip(factors(l), domain(l))
        )
        push!(measurements(cb)[l], dot(psiL, apply(lop, psiR)))
    end

    return nothing
end

# Compute the inner product tr(ρA) for each operator A defined inside the callback object.
# Here the MPS represents a vectorised density matrix, which we denote by `ρ`.
function measure_localops!(cb::ExpValueCallback, ρ::MPS, alg::TDVP1vec)
    # With TDVP1vec algorithms the situation is much simpler than with simple TDVP1: since
    # we need to contract any site which is not "occupied" (by the operator which is to be
    # measured) anyway with vec(I), we don't need to care about the orthocentre, we just
    # measure everything at the end of the sweep.

    # We contract each tensor from `ρ` with the identity, separately.
    ids = [state("Id", siteind(ρ, n)) * ρ[n] for n in eachindex(ρ)]

    for l in ops(cb)
        # Compute the expectation values by multiplying the tensor of the LocalOperator and
        # the identity on the remaining sites.
        x = OneITensor()
        for n in eachindex(ρ)
            if n in domain(l)
                x *= state(l[n], siteind(ρ, n)) * ρ[n]
                # Note that contrary to `inner` or `dot`, this simple product of tensors
                # does not imply any complex conjugation, i.e.
                #
                #   state("A", s) = apply(op("A⋅", s), state("Id", s))
                #
                # behaves as follows (`t` is an ITensor with index `s`):
                #
                #   state("A", s) * t == state("Id", s) * apply(op("A⋅", s), t)
                #
                # so we should not take the adjoint of the measured operator.
            else
                x *= ids[n]
            end
        end
        push!(measurements(cb)[l], scalar(x))
    end

    # Since computing `ids` might require a little time, we return it so that other methods
    # can reuse the results.
    return ids
end

# Helper function for `measure_localops!(::SuperfermionCallback, ::MPS, ::TDVP1vec)`.
# Since the same _sf_id_pairs get reused during a time step, it makes sense to cache them to
# avoid recomputing them as much as possible.
@memoize function _sf_id_pairs(sites)
    return [
        state(sites[n], "Emp") * state(sites[n + 1], "Occ") +
        state(sites[n], "Occ") * state(sites[n + 1], "Emp") for
        n in eachindex(sites)[1:2:end]
    ]
end

# Compute the inner product tr(ρA) for each operator A defined inside the callback object.
# Same as `measure_localops!(::ExpValueCallback, ::MPS, ::TDVP1vec)`, but here the density
# matrix is supposed to be vectorised using the superfermion formalism.
function measure_localops!(cb::SuperfermionCallback, ρ::MPS, alg::TDVP1vec)
    # We follow the same logic as in the ExpValueCallback method, but working with 2-site
    # blocks at a time.

    # Contract each tensor from `ρ` with the identity, separately.
    sf_id_blocks = _sf_id_pairs(siteinds(ρ))
    ids = [
        dag(sf_id_blocks[_sf_translate_sites_inv(n)]) * ρ[n] * ρ[n + 1] for
        n in eachindex(ρ)[1:2:end]
    ]

    for l in ops(cb)
        # Compute the expectation values by multiplying the tensor of the LocalOperator and
        # the identity on the remaining sites.
        x = OneITensor()
        for n in eachindex(ρ)[1:2:end]
            if n in domain(l)
                lop = if n + 1 in domain(l)
                    # We loop over odd sites only, so we check manually that the next site
                    # is in the domain of the operator (the auxiliary sites may contain a
                    # factor of a Jordan-Wigner string).
                    op(l[n], siteind(ρ, n)) * op(l[n + 1], siteind(ρ, n + 1))
                else
                    op(l[n], siteind(ρ, n))
                end
                x *=
                    dag(apply(adj(lop), sf_id_blocks[_sf_translate_sites_inv(n)])) *
                    ρ[n] *
                    ρ[n + 1]
            else
                x *= ids[_sf_translate_sites_inv(n)]
            end
        end
        push!(measurements(cb)[l], scalar(x))
    end

    # Since computing `ids` might require a little time, we return it so that other methods
    # can reuse the results.
    return ids
end

### compute_normalization! functions

# Each callback+algorithm combination has a specific "norm"-like quantity, typically used to
# normalise the expectation values/inner products.  We use the same name for all functions,
# the specific meaning will be clear from context.

# For standard TDVP/TEBD, we use the norm of the state.
function compute_normalization!(
    cb::ExpValueCallback,
    ψ::Union{MPS,VidalMPS,InverseCanonicalMPS},
    alg::Union{TDVP1,TDVP2,TEBD};
    current_time,
)
    # No optimisation needed here. We already have efficient algorithms for both standard
    # MPSs and canonical MPSs.
    is_measurement_time(cb, current_time) && push!(measurements_norm(cb), norm(ψ))

    return nothing
end

# For the "twin TDVP" method we use the inner product of the two states.
function compute_normalization!(
    cb::ExpValueCallback, psiL::MPS, psiR::MPS, alg::TDVP1; current_time
)
    is_measurement_time(cb, current_time) && push!(measurements_norm(cb), dot(psiL, psiR))

    return nothing
end

# For vectorised TDVP, the appropriate normalisation is the trace of the density matrix.
# These two functions compute it from scratch: we contract each tensor from `ρ` with the
# identity, separately.
function compute_normalization!(cb::ExpValueCallback, ρ::MPS, alg::TDVP1vec; current_time)
    return if is_measurement_time(cb, current_time)
        ids = [state("Id", siteind(ρ, n)) * ρ[n] for n in eachindex(ρ)]
        compute_normalization!(cb, ids, alg; current_time)
    else
        nothing
    end
end

function compute_normalization!(
    cb::SuperfermionCallback, ρ::MPS, alg::TDVP1vec; current_time
)
    return if is_measurement_time(cb, current_time)
        sf_id_blocks = _sf_id_pairs(siteinds(ρ))
        ids = [
            dag(sf_id_blocks[_sf_translate_sites_inv(n)]) * ρ[n] * ρ[n + 1] for
            n in eachindex(ρ)[1:2:end]
        ]
        compute_normalization!(cb, ids, alg; current_time)
    else
        nothing
    end
end

# We can also reuse the contractions with the single-site identities, if they have already
# been computed e.g. for the expectation values. They will be a vector of ITensors.
function compute_normalization!(
    cb::MeasurementCallback, ids::Vector{ITensor}, alg::TDVP1vec; current_time
)
    # From precomputed `ids`: we just multiply the elements together.
    is_measurement_time(cb, current_time) && push!(measurements_norm(cb), scalar(prod(ids)))

    return nothing
end
