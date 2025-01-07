# some extensions for ITensors functionality
# the goal is to eventually contribute these upstream if found appropriate

export growbond!, recompute!, maxlinkdims, stretchBondDim, growMPS, growMPS!, enlargelinks

using ITensorMPS: setleftlim!, setrightlim!, AbstractProjMPO, linkdims

function findprimeinds(is::IndexSet, plevel::Int=-1)
    if plevel >= 0
        return filter(x -> plev(x) == plevel, is)
    else
        return filter(x -> plev(x) > 0, is)
    end
end

findprimeinds(A::ITensor, args...) = findprimeinds(A.inds, args...)

function isleftortho(M, i)
    i == length(M) && return true
    L = M[i] * prime(dag(M[i]), i == 1 ? "Link" : commonindex(M[i], M[i + 1]))
    l = linkindex(M, i)
    return norm(L - delta(l, l')) < 1E-12
end

function isrightortho(M, i)
    i == 1 && return true
    R = M[i] * prime(dag(M[i]), i == length(M) ? "Link" : commonindex(M[i], M[i - 1]))
    r = linkindex(M, i - 1)
    return norm(R - delta(r, r')) < 1E-12
end

function reorthogonalize!(psi::MPS)
    setleftlim!(psi, -1)
    setrightlim!(psi, length(psi) + 2)
    orthogonalize!(psi, 1)
    return psi[1] /= sqrt(inner(psi, psi))
end

"""
    growbond!(v::MPS, bond::Integer; increment::Integer=1)

Grow the bond dimension of the MPS `v` between sites `bond` and `bond+1` by `increment`.
Return the new bond dimension.
"""
function growbond!(v::MPS, bond::Integer; increment::Integer=1)::Integer
    bond_index = commonind(v[bond], v[bond + 1])
    current_bonddim = ITensors.dim(bond_index)
    aux = Index(current_bonddim + increment; tags=tags(bond_index))
    v[bond] = v[bond] * delta(bond_index, aux)
    v[bond + 1] = v[bond + 1] * delta(bond_index, aux)
    return current_bonddim + increment
end

"""
    growMPS(v::MPS, dims::Vector{<:Integer})

Grow the dimension of `v`'s bond indices so that the result has a bond dimension of
`dims[n]` on bond (`n`, `n+1`).
Return the new MPS and its overlap with the original one.
"""
function growMPS(v::MPS, dims::Vector{<:Integer})
    @assert length(dims) == length(v) - 1
    currentdims = linkdims(v)
    v_ext = copy(v)
    for (n, new_d, d) in zip(1:(length(v) - 1), dims, currentdims)
        growbond!(v_ext, n; increment=new_d - d)
    end
    v_overlap = dot(v, v_ext)
    @debug "Overlap ⟨original|extended⟩: $v_overlap"
    return v_ext, v_overlap
end

"""
    growMPS(v::MPS, d::Integer)

Grow the dimension of `v`'s bond indices so that the result has a bond dimension of
`d` on each of its bonds.
Return the new MPS and its overlap with the original one.
"""
growMPS(v::MPS, d::Integer) = growMPS(v, fill(d, length(v) - 1))

const stretchBondDim = growMPS

"""
    growMPS!(v::MPS, dims::Vector{<:Integer})

Grow the dimension of `v`'s bond indices (in place) so that the result has a bond dimension
of `dims[n]` on bond (`n`, `n+1`).
Return the overlap of the new MPS with the original one.
"""
function growMPS!(v::MPS, dims::Vector{<:Integer})
    @assert length(dims) == length(v) - 1
    v_prev = copy(v)
    currentdims = linkdims(v)
    for (n, new_d, d) in zip(1:(length(v) - 1), dims, currentdims)
        growbond!(v, n; increment=new_d - d)
    end
    v_overlap = dot(v, v_prev)
    @debug "Overlap ⟨original|extended⟩: $v_overlap"
    return v_overlap
end

"""
    growMPS!(v::MPS, d::Integer)

Grow the dimension of `v`'s bond indices (in place) so that the result has a bond dimension
of `d` on each of its bonds.
Return the overlap of the new MPS with the original one.
"""
growMPS!(v::MPS, d::Integer) = growMPS!(v, fill(d, length(v) - 1))

"""
    enlargelinks(v, dims; ref_state)

Increase the bond dimensions of the MPS `v` to `dims` by adding an all-zero MPS to it,
with a direct sum. The result is an MPS orthogonalized on the first site whose maximum bond
dimensions are `dims`, except possibly at its edges, where the bond dimension is naturally
lower.

In order to obtain the all-zero MPS, a random MPS with the given bond dimensions is
computed, so that the correct link structure is maintained; it is then multiplied by zero
and added to `v`.

If the MPS has some conserved quantum numbers, then a product state must also be supplied
as the keyword argument `ref_state`, because the random MPS is obtained by randomising this
initial state (which also determines the total QN of the resulting random MPS).
This argument may take the form of one of the many available ways to build an MPS from the
`MPS(sites, ...)` function, such as a string, an array of strings, or a function.

# Examples

Without quantum numbers:

```julia-repl
julia> N = 9; s = siteinds("Fermion", N);

julia> v = MPS(s, "Emp");

julia> v_ext = enlargelinks(v, 10; ref_state="Emp");

julia> print(linkdims(v_ext))
[2, 4, 8, 10, 10, 8, 4, 2]
```

With quantum numbers:

```julia-repl
julia> N = 9; s = siteinds("Fermion", N; conserve_nfparity=true);

julia> v = MPS(s, n -> isodd(n) ? "Occ" : "Emp");

julia> v_ext = enlargelinks(v, 10; ref_state=n -> isodd(n) ? "Occ" : "Emp");

julia> print(linkdims(v_ext))
[1, 2, 4, 8, 10, 8, 4, 2]
```
"""
function enlargelinks(v, dims::Vector{<:Integer}; ref_state=nothing)
    diff_linkdims = max.(dims .- linkdims(v), 1)
    x = if hasqns(first(v))
        if isnothing(ref_state)
            error("Initial state required to use enlargelinks with QNs")
        else
            random_mps(siteinds(v), ref_state; linkdims=diff_linkdims)
        end
    else
        random_mps(siteinds(v); linkdims=diff_linkdims)
    end
    orthogonalize!(x, 1)
    v_ext = add(orthogonalize(v, 1), 0 * x; alg="directsum")
    orthogonalize!(v_ext, length(v_ext))
    return orthogonalize(v_ext, 1)
end

function enlargelinks(v, dim::Integer; kwargs...)
    return enlargelinks(v, fill(dim, length(v) - 1); kwargs...)
end

# TODO deprecate `growMPS` and its siblings? They don't really do exactly the same thing,
# but realistically `enlargelinks` is safer to use...

"""
    recompute!(P::AbstractProjMPO, psi::MPS, n::Int)

Recompute `P`'s projection operators assuming that `psi` has changed on sites
(`n`, `n+1`). The position of the projection is not changed.
"""
function recompute!(P::AbstractProjMPO, v::MPS, n::Int)
    N = length(P.H)
    @assert n ≤ N - 1
    # Since v[n] and v[n+1] have changed, we assume that all projection operators
    # currently stored in P which contain those two sites are invalid.
    # Consequently, P.LR[n] and P.LR[n+1] must be recomputed.
    # We spot three cases:
    # 1) n, n+1 ≤ P.lpos
    #    In this case, the right projections are fine, and we recompute P.LR[i] for each
    #    i = n, n+1, ..., P.lpos.
    # 2) n, n+1 ≥ P.rpos
    #    We redo P.LR[N], P.LR[N-1], ..., P.LR[n].
    # 3) n = P.lpos
    #    Here n+1 may or may not be = P.rpos, it depends on P.nsite.
    #    The projection in P.LR[n] must surely be recomputed, but if n+1 is in the "open"
    #    (unprojected) part of the ProjMPO, what happens if we recompute P.LR[n+1]?
    #    It doesn't affect the product of P with a tensor in this configuration, but
    #    it might affect later calculations when we shift the projection on another site.
    #    Now, if we move the projection to the right, P.LR[n+1] will get overwritten, so
    #    no problem there; if we move to the left, or we decrease P.nsite, then P.LR[n+1]
    #    will _not_ be recomputed by position!, which will instead reuse (assuming it has
    #    already been computed) what is already in P.LR. So anyway it is best to just
    #    recompute P.LR[n], P.LR[n+1] and be done with it.
    if n + 1 ≤ P.lpos
        L = (n - 1 < 1 ? ITensors.OneITensor() : P.LR[n - 1])

        L = L * v[n] * P.H[n] * dag(prime(v[n]))
        P.LR[n] = L
        L = L * v[n + 1] * P.H[n + 1] * dag(prime(v[n + 1]))
        P.LR[n + 1] = L
    elseif n ≥ P.rpos
        R = (n + 2 > N ? ITensors.OneITensor() : P.LR[n + 2])

        R = R * v[n + 1] * P.H[n + 1] * dag(prime(v[n + 1]))
        P.LR[n + 1] = R
        R = R * v[n] * P.H[n] * dag(prime(v[n]))
        P.LR[n] = R
    elseif n == P.lpos
        L = (n - 1 ≤ 0 ? ITensors.OneITensor() : P.LR[n - 1])
        R = (n + 2 ≥ N + 1 ? ITensors.OneITensor() : P.LR[n + 2])
        P.LR[n] = L * v[n] * P.H[n] * dag(prime(v[n]))
        P.LR[n + 1] = R * v[n + 1] * P.H[n + 1] * dag(prime(v[n + 1]))
    else
        @warn "v[n] and v[n+1] have changed but sites n and n+1 are currently associated " *
            "to the \"open part\" of the projection represented by the ProjMPO object. " *
            "This may lead to some issues. Be careful."
    end
    return nothing
end

"""
    recompute!(P::ProjMPOSum, psi::MPS, n::Int)

Recompute `P`'s projection operators assuming that `psi` has changed on sites
(`n`, `n+1`). The position of the projection is not changed.
"""
function recompute!(P::ProjMPOSum, v::MPS, n::Int)
    # See recompute! for AbstractProjMPO types. This just applies the method on
    # each of P's terms.
    for t in P.terms
        recompute!(t, v, n)
    end
    return nothing
end

"""
    maxlinkdims(v::MPS[, maxbonddim])

Return a vector containing the maximum bond dimensions that the MPS `v` can have, taking
into account both the physical dimension of its sites and (if provided) a manually set upper
bound `maxbonddim`. The function assumes that all sites share the same physical dimension.
"""
function maxlinkdims(v::MPS)
    # Naive method:
    #   dims = dim.(siteinds(only, v)) .^ (1:length(v))
    #   return min.(min.(dims, reverse(dims)), Ref(maxdim))
    # but if `v` has a lot of sites, the calculations in `dims` quickly overflow.
    # We go through the logarithms instead, so as to avoid the exponentiation.
    # Once we cap the values at `maxbonddim`, we can safely exponentiate.
    sitedims = dim.(siteinds(only, v))
    logdims = (1:length(v)) .* log2.(sitedims)
    minlogdims = min.(logdims[1:(end - 1)], reverse(logdims[1:(end - 1)]))
    return round.(Int, 2 .^ minlogdims)
end

maxlinkdims(v::MPS, maxbonddim::Int) = min.(maxlinkdims(v), Ref(maxbonddim))

"""
    maxlinkdims(st::AbstractString, N[, maxbonddim])

Return a vector containing the maximum bond dimensions that an MPS of site type `st` and
length `N` can have, taking into account both the physical dimension of its sites and (if
provided) a manually set upper bound `maxbonddim`.
"""
function maxlinkdims(st::AbstractString, N)
    logdims = (1:N) .* log2(space(SiteType(st)))
    minlogdims = min.(logdims[1:(end - 1)], reverse(logdims[1:(end - 1)]))
    return round.(Int, 2 .^ minlogdims)
end

maxlinkdims(st::AbstractString, N, maxbonddim) = min.(maxlinkdims(st, N), Ref(maxbonddim))
