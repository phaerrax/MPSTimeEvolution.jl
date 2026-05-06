export VidalMPS

mutable struct VidalMPS
    #        │       │       │         │       │       │
    #    ◇···○───◇───○───◇───○─╶╶  ╶╶╶─○───◇───○───◇───○···◇
    #       Γ[1]    Γ[2]    Γ[3]      Γ[N-2]  Γ[N-1]  Γ[N]
    #   Λ[0]    Λ[1]    Λ[2]             Λ[N-2]  Λ[N-1]   Λ[N]
    #
    site_tensors::Vector{ITensor}
    bond_tensors::OffsetVector{ITensor}
    # The `bond_tensors` member is an OffsetVector because we want to add trivial bond
    # tensors at the edges of the MPS (this simplifies the logic in some functions a lot),
    # while at the same time preserving the "natural" indexing of the actual bond tensors
    # from 1 to N-1.
    # For the trivial bond tensors we will use ITensor(1.0), and not e.g. OneITensor(),
    # because we need to call `inv.` on it, and there's no such method for OneITensors.
    # There are no actual indices linking these trivial bond tensors to the adjacent site
    # tensors.
end

site_tensors(v::VidalMPS) = v.site_tensors
bond_tensors(v::VidalMPS) = v.bond_tensors

Base.length(ψ::VidalMPS) = length(site_tensors(ψ)) + length(bond_tensors(ψ))
# `length` here should be considered as an internal function that shouldn't really be called
# by end users; it has beed added because it is required by some iterators. Consider using
# `nsites` instead.

nsites(ψ::VidalMPS) = length(site_tensors(ψ))

# bond tensors --> (0, n), n from 0 to N
# site tensors --> (1, n), n from 1 to N
# iteration sequence: (0, 0) -> (1, 0) -> (1, 1) -> ... -> (0, N-1) -> (1, N) -> (0, N)
Base.iterate(ψ::VidalMPS) = (first(bond_tensors(ψ)), (0, 0))
function Base.iterate(ψ::VidalMPS, state)
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

Base.keys(ψ::VidalMPS) = 1:length(ψ)

### Constructors
# (carried over from the MPS type from ITensorMPS)

VidalMPS() = VidalMPS(ITensor[], OffsetVector(ITensor[]))  # Empty VidalMPS with no sites.

function VidalMPS(N::Int)
    # Construct a VidalMPS with N sites, with default-constructed ITensors.
    #
    # Beware that N is the number of the site tensors.
    # (This is a default constructor that is not meant to be called directly.)
    return VidalMPS(
        Vector{ITensor}(undef, N), OffsetVector(Vector{ITensor}(undef, N+1), 0:N)
    )
end

"""
    VidalMPS([::Type{ElT} = Float64, ]sites)

Construct a `VidalMPS` filled with empty ITensors of type `ElT` from a collection of
indices.
"""
function VidalMPS(::Type{T}, sites::Vector{<:Index}) where {T<:Number}
    N = length(sites)
    site_tensors = Vector{ITensor}(undef, N)
    bond_tensors = OffsetVector(Vector{ITensor}(undef, N+1), 0:N)

    link_indices_left = [Index(1, "Link,l=$i") for i in 1:(N - 1)]
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

    return VidalMPS(site_tensors, bond_tensors)
end

function VidalMPS(sites::Vector{<:Index}, args...; kwargs...)
    VidalMPS(Float64, sites, args...; kwargs...)
end

"""
    VidalMPS(
        ::Type{T},
        sites::Vector{<:Index},
        states::Union{Vector{String}, Vector{Int}, String, Int}
    )

Construct a product state `VidalMPS` of element type `T`, having site indices `sites`,
and which corresponds to the initial state given by the array `states`. The input `states`
may be an array of strings or an array of ints recognized by the `state` function defined
for the tag types in `sites`. In addition, a single string or int can be input to create a
uniform state.

# Examples

```julia
N = 10
sites = siteinds("Boson", N; dim=4)
states = [isodd(n) ? "1" : "2" for n in 1:N]
psi = VidalMPS(ComplexF64, sites, states)
phi = VidalMPS(sites, "1")
```
"""
function VidalMPS(eltype::Type{<:Number}, sites::Vector{<:Index}, states_)
    N = length(states_)
    if N != length(sites)
        error("sites and states do not have the same number of elements")
    end

    site_ts = Vector{ITensor}(undef, N)
    bond_ts = Vector{ITensor}(undef, N-1)
    # We'll add the trivial edge bonds and convert it to an OffsetVector later.

    link_indices_left = [Index(1; tags="Link,l=$n") for n in 1:(N - 1)]
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
    return VidalMPS(
        convert_leaf_eltype(eltype, site_ts),
        OffsetVector(
            convert_leaf_eltype(eltype, [ITensor(1.0); bond_ts; ITensor(1.0)]), 0:N
        ),
    )
end

function VidalMPS(
    ::Type{T}, sites::Vector{<:Index}, state::Union{String,Integer}
) where {T<:Number}
    return VidalMPS(T, sites, fill(state, length(sites)))
end

function VidalMPS(::Type{T}, sites::Vector{<:Index}, states::Function) where {T<:Number}
    states_vec = [states(n) for n in 1:length(sites)]
    return VidalMPS(T, sites, states_vec)
end

"""
    VidalMPS(sites::Vector{<:Index}, states)

Construct a product-state `VidalMPS` having site indices `sites`, and which corresponds
to the initial state given by the array `states`. The `states` array may consist of either
an array of integers or strings, as recognized by the `state` function defined for the
relevant Index tag type.

# Examples

```julia
N = 10
sites = siteinds("Boson", N; dim=4)
states = [isodd(n) ? "1" : "0" for n in 1:N]
psi = VidalMPS(sites, states)
```
"""
VidalMPS(sites::Vector{<:Index}, states) = VidalMPS(Float64, sites, states)

"""
    copy(::VidalMPS)

Make a shallow copy of a VidalMPS. By shallow copy, it means that a new VidalMPS is
returned, but the data of the tensors are still shared between the returned VidalMPS and the
original VidalMPS.

Therefore, replacing an entire tensor of the returned VidalMPS will not modify the input
VidalMPS, but modifying the data of the returned VidalMPS will modify the input VidalMPS.

Use [`deepcopy`](@ref) for an alternative that copies the ITensors as well.
"""
Base.copy(ψ::VidalMPS) = VidalMPS(copy(site_tensors(ψ)), copy(bond_tensors(ψ)))

Base.similar(ψ::VidalMPS) = VidalMPS(similar(site_tensors(ψ)), similar(bond_tensors(ψ)))

"""
    deepcopy(::VidalMPS)

Make a deep copy of a VidalMPS. By deep copy, it means that a new VidalMPS is returned that
doesn't share any data with the input VidalMPS.

Therefore, modifying the resulting VidalMPS will note modify the original VidalMPS.

Use [`copy`](@ref) for an alternative that performs a shallow copy that avoids
copying the ITensor data.
"""
ITensorMPS.deepcopy(ψ::VidalMPS) = VidalMPS(copy.(site_tensors(ψ)), copy.(bond_tensors(ψ)))

function LinearAlgebra.promote_leaf_eltypes(ψ::VidalMPS)
    # `LinearAlgebra.promote_leaf_eltypes` requires one-based indexing, so we use `parent`
    # to retrieve the standard array underlying the OffsetArray.
    return LinearAlgebra.promote_leaf_eltypes([site_tensors(ψ); parent(bond_tensors(ψ))])
end
NDTensors.scalartype(ψ::VidalMPS) = LinearAlgebra.promote_leaf_eltypes(ψ)

# Compact printing of MPS contents on the REPL (otherwise it dumps the contents of _all_ the
# tensors, creating a huge output)
function Base.show(io::IO, ψ::VidalMPS)
    print(io, "VidalMPS ($(nsites(ψ)) sites)")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", ψ::VidalMPS)
    N = nsites(ψ)
    println(io, "$(N)-site VidalMPS:")
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

function check_vidal_form(ψ::VidalMPS)
    N = nsites(ψ)

    ψdag = sim(linkinds, dag(ψ))
    st = site_tensors(ψ)
    bt = bond_tensors(ψ)
    stdag = site_tensors(ψdag)
    btdag = bond_tensors(ψdag)

    # Check whether the singular values are real and not negative (excluding the trivial
    # bond tensors at the edges).
    for (j, Λ) in enumerate(bt[1:(N - 1)])
        if any(!isreal, diag(Λ))
            error("non-real singular values on bond $j.")
        end
        if any(<(0), diag(Λ))
            error("negative singular values on bond $j.")
        end
    end

    # Check whether the cancellation rules hold.
    #
    #   Γ[1]───       ╭───
    #    │            │
    #    │        =   │
    #    │            │
    #   Γ[1]───       ╰───

    M₁ = st[1] * stdag[1]
    if !isapprox(ITensors.matrix(M₁), I)
        error("orthogonality condition not satisfied on site 1.")
    end

    #   ───Γ[N]       ───╮
    #       │            │
    #       │     =      │
    #       │            │
    #   ───Γ[N]       ───╯

    Mₙ = st[N] * stdag[N]
    if !isapprox(ITensors.matrix(Mₙ), I)
        error("orthogonality condition not satisfied on site $N.")
    end

    #   ───Γ[j]───Λ[j]───╮                     ───╮
    #       │            │                        │
    #       │            │   =  tr(Λ[j]²)  ×      │
    #       │            │                        │
    #   ───Γ[j]───Λ[j]───╯                     ───╯

    for j in 2:(N - 1)
        Mⱼ = st[j] * bt[j] * stdag[j] * btdag[j]
        Mⱼ = Mⱼ * delta(commoninds(Mⱼ, bt[j] * btdag[j]))
        if !isapprox(ITensors.matrix(Mⱼ), scalar(bt[j] * bt[j]) * I)
            error("orthogonality condition not satisfied on site $j.")
        end
    end

    #  ╭───Λ[j-1]───Γ[j]───                       ╭───
    #  │             │                            │
    #  │             │        =  tr(Λ[j-1]²)  ×   │
    #  │             │                            │
    #  ╰───Λ[j-1]───Γ[j]───                       ╰───

    for j in 2:(N - 1)
        Mⱼ = bt[j - 1] * st[j] * btdag[j - 1] * stdag[j]
        Mⱼ = Mⱼ * delta(commoninds(Mⱼ, bt[j - 1] * btdag[j - 1]))
        if !isapprox(ITensors.matrix(Mⱼ), scalar(bt[j - 1] * bt[j - 1]) * I)
            error("orthogonality condition not satisfied on site $j.")
        end
    end

    return true
end
