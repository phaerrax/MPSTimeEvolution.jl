export VidalMPS

mutable struct VidalMPS
    site_tensors::Vector{ITensor}
    bond_tensors::Vector{ITensor}
end

site_tensors(v::VidalMPS) = v.site_tensors
bond_tensors(v::VidalMPS) = v.bond_tensors

Base.length(ψ::VidalMPS) = length(site_tensors(ψ)) + length(bond_tensors(ψ))
nsites(ψ::VidalMPS) = length(site_tensors(ψ))

# site tensors --> (0, n), n from 1 to N
# bond tensors --> (1, n), n from 1 to N-1
# iteration sequence: (0, 1) -> (1, 1) -> (0, 2) -> ... -> (1, ns-1) -> (0, ns)
Base.iterate(ψ::VidalMPS) = (site_tensors(ψ)[1], (0, 1))
function Base.iterate(ψ::VidalMPS, state)
    t_type, n = state
    return if n ≥ nsites(ψ)
        nothing
    else
        if t_type == 0
            bond_tensors(ψ)[n], (rem(t_type + 1, 2), n)
        else
            site_tensors(ψ)[n + 1], (rem(t_type + 1, 2), n+1)
        end
    end
end

Base.keys(ψ::VidalMPS) = 1:length(ψ)

### Constructors
# (carried over from the MPS type from ITensorMPS)

VidalMPS() = VidalMPS(ITensor[], ITensor[])  # Empty VidalMPS with no sites.

function VidalMPS(N::Int)
    # Construct a VidalMPS with N sites, with default-constructed ITensors.
    #
    # Beware that N is the number of the site tensors.
    # (This is a default constructor that is not meant to be called directly.)
    return VidalMPS(Vector{ITensor}(undef, N), Vector{ITensor}(undef, N-1))
end

"""
    VidalMPS([::Type{ElT} = Float64, ]sites)

Construct a `VidalMPS` filled with empty ITensors of type `ElT` from a collection of
indices.
"""
function VidalMPS(::Type{T}, sites::Vector{<:Index}) where {T<:Number}
    N = length(sites)
    site_tensors = Vector{ITensor}(undef, N)
    bond_tensors = Vector{ITensor}(undef, N-1)

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
    for i in 1:(N - 1)
        bond_tensors[i] = diag_itensor(T, dag(link_indices_right[i]), link_indices_left[i])
        bond_tensors[i][dag(link_indices_right[i]) => 1, link_indices_left[i] => 1] = 1.0
    end

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
    ψ = VidalMPS(N)

    link_indices_left = [Index(1; tags="Link,l=$n") for n in 1:(N - 1)]
    link_indices_right = [Index(1; tags="Link,r=$n") for n in 1:(N - 1)]

    if N == 1
        site_tensors(ψ)[1] = state(only(sites), only(states_))
    else
        site_tensors(ψ)[1] = state(sites[1], states_[1]) * state(link_indices_right[1], 1)
        for n in 2:(N - 1)
            site_tensors(ψ)[n] =
                state(dag(link_indices_left[n - 1]), 1) *
                state(sites[n], states_[n]) *
                state(link_indices_right[n], 1)
        end
        site_tensors(ψ)[N] =
            state(dag(link_indices_left[N - 1]), 1) * state(sites[N], states_[N])
        for n in 1:(N - 1)
            bond_tensors(ψ)[n] = diag_itensor(
                dag(link_indices_right[n]), link_indices_left[n]
            )
            bond_tensors(ψ)[n][dag(link_indices_right[n]) => 1, link_indices_left[n] => 1] =
                1.0
        end
    end

    return VidalMPS(
        convert_leaf_eltype(eltype, site_tensors(ψ)),
        convert_leaf_eltype(eltype, bond_tensors(ψ)),
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
    return LinearAlgebra.promote_leaf_eltypes([site_tensors(ψ); bond_tensors(ψ)])
end
NDTensors.scalartype(ψ::VidalMPS) = LinearAlgebra.promote_leaf_eltypes(ψ)
