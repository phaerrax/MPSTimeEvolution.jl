export VidalMPS, InverseCanonicalMPS, canonicalize

# Most of the functions work the same with either VidalMPSs or InverseCanonicalMPSs, so
# we define a common interface. This way we will sometimes be able to write a single
# function for both types.
abstract type ExplicitBondMPS end

mutable struct VidalMPS <: ExplicitBondMPS
    #        │       │       │         │       │       │
    #    ◇···○───◇───○───◇───○─╶╶  ╶╶╶─○───◇───○───◇───○···◇
    #       Γ[1]    Γ[2]    Γ[3]      Γ[N-2]  Γ[N-1]  Γ[N]
    #   Λ[0]    Λ[1]    Λ[2]             Λ[N-2]  Λ[N-1]   Λ[N]
    #
    site_tensors::Vector{ITensor}
    bond_tensors::OffsetVector{ITensor}
    # In order to make the logic in most functions simpler, we add two additional dummy bond
    # tensors at the edges of the MPS: one at the left of the "real" first one, and one to
    # the right of the "real" last one.  For the trivial bond tensors we use ITensor(1.0),
    # and not e.g. OneITensor(), because we need to call `inv.` on it, and there's no such
    # method for OneITensors.  Moreover, there are no actual indices linking these trivial
    # bond tensors to the adjacent site tensors, so any index operation (contraction,
    # computing common indices, etc... I think) involving these tensors does not actually do
    # anything.
    # This allows us for example to iterate with something like `k in
    # eachindex(site_tensors)` and still be able to call `bond_tensors[k-1]` and
    # `bond_tensors[k+1]` for all k's without triggering errors.  In order to preserve the
    # "natural" indexing of the actual bond tensors from 1 to N-1, we use an OffsetVector to
    # store the bond tensors, storing the dummy ones in  `bond_tensors[0]` and
    # `bond_tensors[N]`.
    norm::Float64
    # For a canonical Vidal MPS there is (or seems to be) no way of absorbing an arbitrary
    # (complex) number into the state while keeping all bond tensors normalised to one. In
    # order to be able to represent vectors with arbitrary norm, we have two choices.
    # 1. "Park" the norm into a specific bond tensor, relaxing the convention that every
    #    bond tensor is normalised. This is very practical but it's fragile, because every
    #    function assuming the canonical gauge condition must then be aware of this
    #    convention, and it's easy to overlook something.
    # 2. Store the norm in a separate field and keep all bond tensors normalised. This is a
    #    clearer (and probably more robust) choice, as it's clear where the norm resides.
    # How should the norm be initialised?
    # *  If the MPS is completely empty, that is, it has no tensors, then its norm is 1.
    #    Basically we are saying that an empty tensor network contracts to 1, just as the
    #    result of an empty product is 1. Moreover, this way an empty MPS is (correctly) a
    #    multiplicative identity for the tensor product.
    # *  If the MPS contains a non-empty list of empty tensors, we set its norm to zero,
    #    consistently with the fact that `norm(t)` is zero if `t` is an empty ITensor.
    #    Moreover, such an MPS could be added to another MPS of similar shape, and in this
    #    case it should act as an identity for the addition.
    # *  Genuine, non-empty product states are always a specific basis element, therefore
    #    their norm is 1.

    function VidalMPS(
        site_tensors::Vector{ITensor}, bond_tensors::OffsetVector{ITensor}, norm
    )
        # This constructor seems pretty useless, but we need to define it otherwise Julia
        # would automatically define a default constructor
        #   VidalMPS(site_tensors, bond_tensors, norm)
        # with no type annotations, which might conflict with some of the methods below.
        return new(site_tensors, bond_tensors, norm)
    end
end

mutable struct InverseCanonicalMPS <: ExplicitBondMPS
    #        │       │       │         │       │       │
    #    ◇···▧───◆───▧───◆───▧─╶╶  ╶╶╶─▧───◆───▧───◆───▧···◇
    #       C[1]    C[2]    C[3]      C[N-2]  C[N-1]  C[N]
    #   V[0]    V[1]    V[2]             V[N-2]  V[N-1]   V[N]
    #
    # The tensors in an inverse-canonical MPS are usually denoted by Ψₙ and Vₙ. I guess V
    # is intended to be a vertically reflected Λ, which is nice. However Ψ feels too “heavy”
    # to be used so much throughout the code, so we'll use C to denote site tensors in
    # drawings.
    # This struct is designed analogously to the VidalMPS type above.
    site_tensors::Vector{ITensor}
    bond_tensors::OffsetVector{ITensor}
    norm::Float64

    function InverseCanonicalMPS(
        site_tensors::Vector{ITensor}, bond_tensors::OffsetVector{ITensor}, norm
    )
        return new(site_tensors, bond_tensors, norm)
    end
end

site_tensors(ψ::ExplicitBondMPS) = ψ.site_tensors
bond_tensors(ψ::ExplicitBondMPS) = ψ.bond_tensors
LinearAlgebra.norm(ψ::ExplicitBondMPS) = ψ.norm

Base.length(ψ::ExplicitBondMPS) = length(site_tensors(ψ)) + length(bond_tensors(ψ))
Base.keys(ψ::ExplicitBondMPS) = 1:length(ψ)
# `length` and `keys` here should be considered as internal functions, that shouldn't really
# be called by end users; they were added because they were required by some iterators.
# Consider using `nsites` instead.

nsites(ψ::ExplicitBondMPS) = length(site_tensors(ψ))

# bond tensors --> (0, n), n from 0 to N
# site tensors --> (1, n), n from 1 to N
# iteration sequence: (0, 0) -> (1, 0) -> (1, 1) -> ... -> (0, N-1) -> (1, N) -> (0, N)
Base.iterate(ψ::ExplicitBondMPS) = (first(bond_tensors(ψ)), (0, 0))
function Base.iterate(ψ::ExplicitBondMPS, state)
    t_type, n = state
    return if state == (0, nsites(ψ)) || n > nsites(ψ)
        nothing
    else
        if t_type == 0  # then `state` is a bond tensor. Return the next site tensor.
            site_tensors(ψ)[n + 1], (rem(t_type + 1, 2), n+1)
        else  # then `state` is a site tensor. Return the next bond tensor.
            bond_tensors(ψ)[n], (rem(t_type + 1, 2), n)
        end
    end
end

### Constructors
# (carried over from the MPS type from ITensorMPS)

# Empty MPSs with no sites.
VidalMPS() = VidalMPS(ITensor[], OffsetVector(ITensor[]), 1.0)
InverseCanonicalMPS() = InverseCanonicalMPS(ITensor[], OffsetVector(ITensor[]), 1.0)

### Type-parametric functions
# The following syntax is equivalent to defining two functions, 
#   function VidalMPS(N::Int)
#   function InverseCanonicalMPS(N::Int)
# at once, with `MPST` inside of them replaced by either `VidalMPS` or `InverseCanonicalMPS`
# depending on which version is called. There are many functions like this in this package:
# this syntax is necessary because by declaring the function as
#   function f(x::MPST) where {MPST<:ExplicitBondMPS}
#       ...
#   end
# we can then call `MPST` within the function body to get the actual type the function is
# compiled with, either VidalMPS or InverseCanonicalMPSs. This would not be possible if we
# just wrote
#   function f(x::ExplicitBondMPS)
#       ...
#   end
# (The following is a very particular instance of this syntax where the MPST type parameter
# is in the function name itself.)
function (::Type{MPST})(N::Int) where {MPST<:ExplicitBondMPS}
    # Construct a VidalMPS or an InverseCanonicalMPS with N sites, with default-constructed
    # ITensors.
    #
    # Beware that N is the number of the site tensors.
    # (This is a default constructor that is not meant to be called directly.)
    return MPST(
        Vector{ITensor}(undef, N), OffsetVector(Vector{ITensor}(undef, N+1), 0:N), 0.0
    )
end

"""
    VidalMPS([::Type{ElT} = Float64, ]sites)
    InverseCanonicalMPS([::Type{ElT} = Float64, ]sites)

Construct an MPS filled with empty ITensors of type `ElT` from a collection of indices.
"""
function (::Type{MPST})(
    ::Type{T}, sites::Vector{<:Index}
) where {T<:Number,MPST<:ExplicitBondMPS}
    N = length(sites)
    site_tensors = Vector{ITensor}(undef, N)
    bond_tensors = OffsetVector(Vector{ITensor}(undef, N+1), 0:N)

    link_indices_left = [Index(1, "Link,l=$i") for i in 2:N]
    link_indices_right = [Index(1, "Link,r=$i") for i in 1:(N - 1)]
    for i in 1:N
        s = sites[i]
        if i == 1
            site_tensors[i] = ITensor(T, link_indices_right[i], s)
        elseif i == N
            site_tensors[i] = ITensor(T, dag(link_indices_left[i - 1]), s)
        else
            site_tensors[i] = ITensor(
                T, dag(link_indices_left[i - 1]), s, link_indices_right[i]
            )
        end
    end

    bond_tensors[0] = ITensor(1.0)
    for i in 1:(N - 1)
        bond_tensors[i] = diag_itensor(T, dag(link_indices_right[i]), link_indices_left[i])
        bond_tensors[i][dag(link_indices_right[i]) => 1, link_indices_left[i] => 1] = 1.0
    end
    bond_tensors[N] = ITensor(1.0)

    return MPST(site_tensors, bond_tensors, 0.0)
end

function (::Type{MPST})(
    sites::Vector{<:Index}, args...; kwargs...
) where {MPST<:ExplicitBondMPS}
    return MPST(Float64, sites, args...; kwargs...)
end

"""
    VidalMPS(
        ::Type{T},
        sites::Vector{<:Index},
        states::Union{Vector{String}, Vector{Int}, String, Int}
    )
    InverseCanonicalMPS(
        ::Type{T},
        sites::Vector{<:Index},
        states::Union{Vector{String}, Vector{Int}, String, Int}
    )

Construct a product state MPS of element type `T`, having site indices `sites`, and which
corresponds to the initial state given by the array `states`. The input `states` may be an
array of strings or an array of ints recognized by the `state` function defined for the tag
types in `sites`. In addition, a single string or int can be input to create a uniform
state.

# Examples

```julia
N = 10
sites = siteinds("Boson", N; dim=4)
states = [isodd(n) ? "1" : "2" for n in 1:N]
psi = VidalMPS(ComplexF64, sites, states)
phi = InverseCanonicalMPS(sites, "1")
```
"""
function (::Type{MPST})(
    eltype::Type{<:Number}, sites::Vector{<:Index}, states_
) where {MPST<:ExplicitBondMPS}
    N = length(states_)
    if N != length(sites)
        error("sites and states do not have the same number of elements")
    end

    site_ts = Vector{ITensor}(undef, N)
    bond_ts = Vector{ITensor}(undef, N-1)
    # We'll add the trivial edge bonds and convert it to an OffsetVector later.

    link_indices_left = [Index(1; tags="Link,l=$n") for n in 2:N]
    link_indices_right = [Index(1; tags="Link,r=$n") for n in 1:(N - 1)]

    if N == 1
        site_ts[1] = state(only(sites), only(states_))
    else
        site_ts[1] = state(sites[1], states_[1]) * state(link_indices_right[1], 1)
        for n in 2:(N - 1)
            site_ts[n] =
                state(dag(link_indices_left[n - 1]), 1) *
                state(sites[n], states_[n]) *
                state(link_indices_right[n], 1)
        end
        site_ts[N] = state(dag(link_indices_left[N - 1]), 1) * state(sites[N], states_[N])
        for n in 1:(N - 1)
            bond_ts[n] = diag_itensor(dag(link_indices_right[n]), link_indices_left[n])
            bond_ts[n][dag(link_indices_right[n]) => 1, link_indices_left[n] => 1] = 1.0
        end
    end

    # convert_leaf_eltype is not defined on OffsetArrays so we apply first to a standard
    # vector of ITensors.
    return MPST(
        convert_leaf_eltype(eltype, site_ts),
        OffsetVector(
            convert_leaf_eltype(eltype, [ITensor(1.0); bond_ts; ITensor(1.0)]), 0:N
        ),
        1.0,
    )
end

function (::Type{MPST})(
    ::Type{T}, sites::Vector{<:Index}, state::Union{String,Integer}
) where {T<:Number,MPST<:ExplicitBondMPS}
    return MPST(T, sites, fill(state, length(sites)))
end

function (::Type{MPST})(
    ::Type{T}, sites::Vector{<:Index}, states::Function
) where {T<:Number,MPST<:ExplicitBondMPS}
    states_vec = [states(n) for n in 1:length(sites)]
    return MPST(T, sites, states_vec)
end

"""
    VidalMPS(sites::Vector{<:Index}, states)
    InverseCanonicalMPS(sites::Vector{<:Index}, states)

Construct a product-state MPS having site indices `sites`, and which corresponds to the
initial state given by the array `states`. The `states` array may consist of either an array
of integers or strings, as recognized by the `state` function defined for the relevant Index
tag type.

# Examples

```julia
N = 10
sites = siteinds("Boson", N; dim=4)
states = [isodd(n) ? "1" : "0" for n in 1:N]
psi = VidalMPS(sites, states)
```
"""
function (::Type{MPST})(sites::Vector{<:Index}, states) where {MPST<:ExplicitBondMPS}
    return MPST(Float64, sites, states)
end

"""
    copy(::VidalMPS)
    copy(::InverseCanonicalMPS)

Make a shallow copy of a `VidalMPS` or an `InverseCanonicalMPS`. By shallow copy, it means
that a new MPS is returned, but the data of the tensors are still shared between the
returned MPS and the original MPS.

Therefore, replacing an entire tensor of the returned MPS will not modify the input MPS, but
modifying the data of the returned MPS will modify the input MPS.

Use [`deepcopy`](@ref) for an alternative that copies the ITensors as well.
"""
function Base.copy(ψ::MPST) where {MPST<:ExplicitBondMPS}
    return MPST(copy(site_tensors(ψ)), copy(bond_tensors(ψ)), norm(ψ))
end

function Base.similar(ψ::MPST) where {MPST<:ExplicitBondMPS}
    return MPST(similar(site_tensors(ψ)), similar(bond_tensors(ψ)), NaN)
end

"""
    deepcopy(::VidalMPS)
    deepcopy(::InverseCanonicalMPS)

Make a deep copy of a `VidalMPS` or an `InverseCanonicalMPS`. By deep copy, it means that a
new MPS is returned that doesn't share any data with the input MPS.

Therefore, modifying the resulting MPS will note modify the original MPS.

Use [`copy`](@ref) for an alternative that performs a shallow copy that avoids copying the
ITensor data.
"""
function ITensorMPS.deepcopy(ψ::MPST) where {MPST<:ExplicitBondMPS}
    return MPST(copy.(site_tensors(ψ)), copy.(bond_tensors(ψ)), norm(ψ))
end

function LinearAlgebra.promote_leaf_eltypes(ψ::ExplicitBondMPS)
    # `LinearAlgebra.promote_leaf_eltypes` requires one-based indexing, so we use `parent`
    # to retrieve the standard array underlying the OffsetArray.
    return LinearAlgebra.promote_leaf_eltypes([site_tensors(ψ); parent(bond_tensors(ψ))])
end
NDTensors.scalartype(ψ::ExplicitBondMPS) = LinearAlgebra.promote_leaf_eltypes(ψ)

# Compact printing of MPS contents on the REPL (otherwise it dumps the contents of _all_ the
# tensors, creating a huge output)
function Base.show(io::IO, ψ::ExplicitBondMPS)
    print(io, "$(typeof(ψ)) ($(nsites(ψ)) sites)")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", ψ::ExplicitBondMPS)
    N = nsites(ψ)
    println(io, "$(N)-site $(typeof(ψ)):")
    st = site_tensors(ψ)
    bt = bond_tensors(ψ)[1:(N - 1)]
    # We don't show the trivial bond tensors at the edges of the MPS.

    siteinds_vec = map(eachindex(st)) do j
        !isassigned(st, j) && return ITensorMPS.Undef()
        return inds(st[j])
    end
    bondinds_vec = map(eachindex(bt)) do j
        !isassigned(bt, j) && return ITensorMPS.Undef()
        return inds(bt[j])
    end
    Base.print_array(
        io, [collect(Iterators.flatten(zip(siteinds_vec, bondinds_vec))); siteinds_vec[end]]
    )
    return nothing
end

function check_vidal_form(ψ::VidalMPS; verbose=true, kwargs...)
    N = nsites(ψ)

    ψdag = sim(linkinds, dag(ψ))
    st = site_tensors(ψ)
    bt = bond_tensors(ψ)
    stdag = site_tensors(ψdag)
    btdag = bond_tensors(ψdag)

    errors = String[]

    # Check whether the singular values are real, not negative (excluding the trivial
    # bond tensors at the edges), and the sum of their squares is 1.
    for (j, Λ) in enumerate(bt[1:(N - 1)])
        if any(!isreal, diag(Λ))
            push!(errors, "non-real singular values on bond $j")
        end
        if any(<(0), diag(Λ))
            push!(errors, "negative singular values on bond $j")
        end
        if !isapprox(scalar(Λ * Λ), 1; kwargs...)
            push!(errors, "bond tensor $j not normalised")
        end
    end

    # Check whether the cancellation rules hold.

    #   ───Γ[j]───Λ[j]───╮                     ───╮
    #       │            │                        │
    #       │            │   =  tr(Λ[j]²)  ×      │
    #       │            │                        │
    #   ───Γ[j]───Λ[j]───╯                     ───╯

    #   ───Γ[N]       ───╮
    #       │            │
    #       │     =      │
    #       │            │
    #   ───Γ[N]       ───╯

    for j in 2:N
        Mⱼ = st[j] * bt[j] * stdag[j] * btdag[j]
        Mⱼ = Mⱼ * delta(commoninds(Mⱼ, bt[j] * btdag[j]))
        # Everything works even if j = N: then bt[N] * btdag[N] == ITensor(1.0), and
        # delta(commoninds(Mⱼ, bt[j] * btdag[j])) becomes a unit tensor, which does nothing
        # to Mⱼ.
        if !isapprox(ITensors.matrix(Mⱼ), I; kwargs...)
            push!(errors, "right-orthonormality condition not satisfied on site $j")
        end
    end

    #  ╭───Λ[j-1]───Γ[j]───                       ╭───       ╭───
    #  │             │                            │          │
    #  │             │        =  tr(Λ[j-1]²)  ×   │      =   │
    #  │             │                            │          │
    #  ╰───Λ[j-1]───Γ[j]───                       ╰───       ╰───
    #
    #   Γ[1]───       ╭───
    #    │            │
    #    │        =   │
    #    │            │
    #   Γ[1]───       ╰───

    for j in 1:(N - 1)
        Mⱼ = bt[j - 1] * st[j] * btdag[j - 1] * stdag[j]
        Mⱼ = Mⱼ * delta(commoninds(Mⱼ, bt[j - 1] * btdag[j - 1]))
        if !isapprox(ITensors.matrix(Mⱼ), I; kwargs...)
            push!(errors, "left-orthonormality condition not satisfied on site $j")
        end
    end

    if isempty(errors)
        return true
    else
        verbose && foreach(println, errors)
        return false
    end
end

function check_inverse_canonical_form(ψ::InverseCanonicalMPS; verbose=true, kwargs...)
    N = nsites(ψ)

    ψdag = sim(linkinds, dag(ψ))
    st = site_tensors(ψ)
    bt = bond_tensors(ψ)
    stdag = site_tensors(ψdag)
    btdag = bond_tensors(ψdag)

    errors = String[]

    # Check whether the singular values are real and not negative (excluding the trivial
    # bond tensors at the edges), and that there are no NaNs or Infs.
    for (j, V) in enumerate(bt[1:(N - 1)])
        if any(!isfinite, diag(V))
            push!(errors, "Inf or NaN values on bond $j")
        end
        if any(!isreal, diag(V))
            push!(errors, "non-real singular values on bond $j")
        end
        if any(<(0), diag(V))
            push!(errors, "negative singular values on bond $j")
        end
        if !isapprox(scalar(inv.(V) * inv.(V)), 1; kwargs...)
            push!(errors, "bond tensor $j not normalised")
        end
    end

    # Check whether the cancellation rules hold.

    #  ╭───C[j]───V[j]───     ╭───Λ[j-1]───Γ[j]───                    ╭───       ╭───
    #  │    │                 │             │                         │          │
    #  │    │              =  │             │       =  tr(Λ[j-1]²) ×  │      =   │
    #  │    │                 │             │                         │          │
    #  ╰───C[j]───V[j]───     ╰───Λ[j-1]───Γ[j]───                    ╰───       ╰───

    #   C[1]───V[1]───       Γ[j]───       ╭───
    #    │                    │            │
    #    │               =    │        =   │
    #    │                    │            │
    #   C[1]───V[1]───       Γ[j]───       ╰───

    for j in 1:(N - 1)
        Mⱼ = st[j] * bt[j] * stdag[j] * btdag[j]
        Mⱼ = Mⱼ * delta(commoninds(Mⱼ, st[j] * stdag[j]))
        if !isapprox(ITensors.matrix(Mⱼ), I; kwargs...)
            push!(errors, "left-orthonormality condition not satisfied on site $j")
        end
    end

    #   ───V[j-1]───C[j]───╮       ───Γ[j]───Λ[j]───╮                     ───╮       ───╮
    #                │     │           │            │                        │          │
    #                │     │   =       │            │   =  tr(Λ[j]²)  ×      │   =      │
    #                │     │           │            │                        │          │
    #   ───V[j-1]───C[j]───╯       ───Γ[j]───Λ[j]───╯                     ───╯       ───╯

    #   ───V[N-1]───C[N]        ───Γ[N]       ───╮
    #                │              │            │
    #                │     =        │     =      │
    #                │              │            │
    #   ───V[N-1]───C[N]        ───Γ[N]       ───╯

    for j in 2:N
        Mⱼ = bt[j - 1] * st[j] * btdag[j - 1] * stdag[j]
        Mⱼ = Mⱼ * delta(commoninds(Mⱼ, st[j] * stdag[j]))
        if !isapprox(ITensors.matrix(Mⱼ), I; kwargs...)
            push!(errors, "right-orthonormality condition not satisfied on site $j")
        end
    end

    if isempty(errors)
        return true
    else
        verbose && foreach(println, errors)
        return false
    end
end

"""
    canonicalize(ψ::VidalMPS; kwargs...)
    canonicalize(ψ::InverseCanonicalMPS; kwargs...)

Return a new MPS of the same type as `ψ` which is equivalent to it and satisfies the
(inverse-) canonical gauge conditions, whether `ψ` satisfies them or not.
The returner MPS will be normalised, regardless of whether the original `ψ` should have unit
norm.

The process involves a sequence of singular-value decompositions, to which a `cutoff`
keyword argument can be forwarded.
"""
function canonicalize(ψ::MPST; kwargs...) where {MPST<:ExplicitBondMPS}
    # See the comments in the `+(::Algorithm"directsum", ψs::VidalMPS...)` method to
    # understand these choices.
    #
    # Steps:
    # 1. The original MPS ψ is converted into a standard MPS. Bond tensors are absorbed into
    #    site tensors, it's just matrix multiplications and no truncation is performed.
    #    We also need to normalise the state, otherwise if ‖ψ‖ != 1 the IC gauge will be
    #    broken and the ‖ψ‖ - 1 discrepancy will come back to bite us in the IC gauge check
    #    function.
    ψ_mps = convert(MPS, ψ)
    normalize!(ψ_mps) # TODO
    # 2. The resulting MPS might not be a valid state, if the original ψ did not respect its
    #    gauge. However, we can fix the MPS easily by re-orthogonalising it (we
    #    orthogonalise it on the last site, but I think we can choose whatever site we
    #    like). This gives us an equivalent MPS but in a correct mixed-canonical form.
    orthogonalize!(ψ_mps, nsites(ψ))
    # 3. We convert it back to the original canonical form.
    #    a. If the target is a VidalMPS, then we decompose the MPS in the Vidal form,
    #       possibly with truncation (set the `cutoff` keyword argument accordingly).
    #    b. If the target is an InverseCanonicalMPS, we still compute the Vidal form, then
    #       we convert it into the inverse-canonical form via some matrix inversions.
    return convert(MPST, ψ_mps; kwargs...)
    # TODO is there a way to do this without passing through an ordinary MPS?
end
