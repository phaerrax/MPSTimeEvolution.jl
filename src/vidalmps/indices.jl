export leftlinkind, rightlinkind, leftlinkinds, rightlinkinds

### Index getting/setting utilities

"""
    findsites(ψ::VidalMPS, is)

Return the sites of the VidalMPS that have indices in common with the collection of site
indices `is`.

# Examples

```julia
s = siteinds("S=1/2", 5)
ψ = VidalMPS(s)
findsites(ψ, s[3]) == [3]
findsites(ψ, (s[4], s[1])) == [1, 4]
```
"""
ITensorMPS.findsites(ψ::VidalMPS, is) = findall(hascommoninds(is), siteinds(ψ))
# Bond indices are internal, so we don't even consider them.

ITensorMPS.findsites(ψ::VidalMPS, s::Index) = findsites(ψ, IndexSet(s))

"""
    siteind(ψ::VidalMPS, j::Integer; kwargs...)

Return the site Index found on the `j`th site tensor of the VidalMPS.

You can choose different filters, like prime level and tags, with the `kwargs`.
"""
function ITensors.SiteTypes.siteind(ψ::VidalMPS, j::Integer; kwargs...)
    N = nsites(ψ)

    return if N == 1
        firstind(only(site_tensors(ψ)); kwargs...)
    else
        if j == 1
            uniqueind(site_tensors(ψ)[j], bond_tensors(ψ)[j]; kwargs...)
        elseif j == N
            uniqueind(site_tensors(ψ)[j], bond_tensors(ψ)[j - 1]; kwargs...)
        else
            uniqueind(
                site_tensors(ψ)[j], bond_tensors(ψ)[j - 1], bond_tensors(ψ)[j]; kwargs...
            )
        end
    end
end

"""
    siteinds(ψ::VidalMPS, j::Integer; kwargs...)

Return the site indices found on the `j`th site of the VidalMPS.

Optionally filter prime tags and prime levels with keyword arguments like `plev` and `tags`.
"""
function ITensors.SiteTypes.siteinds(ψ::VidalMPS, j::Integer; kwargs...)
    N = nsites(ψ)

    return if N == 1
        inds(only(site_tensors(ψ)); kwargs...)
    else
        if j == 1
            uniqueinds(site_tensors(ψ)[j], bond_tensors(ψ)[j]; kwargs...)
        elseif j == N
            uniqueinds(site_tensors(ψ)[j], bond_tensors(ψ)[j - 1]; kwargs...)
        else
            uniqueinds(
                site_tensors(ψ)[j], bond_tensors(ψ)[j - 1], bond_tensors(ψ)[j]; kwargs...
            )
        end
    end
end

function ITensors.SiteTypes.siteinds(ψ::VidalMPS; kwargs...)
    return [siteind(ψ, j; kwargs...) for j in 1:nsites(ψ)]
end

"""
    leftlinkind(ψ::VidalMPS, j::Integer; kwargs...)

Return the link Index pointing to the left of the `j`th site tensor of the VidalMPS.

You can choose different filters, like prime level and tags, with the `kwargs`.
"""
function leftlinkind(ψ::VidalMPS, j::Integer; kwargs...)
    N = nsites(ψ)

    return if N == 1
        nothing
    else
        if j == 1
            nothing
        else
            commonind(site_tensors(ψ)[j], bond_tensors(ψ)[j - 1]; kwargs...)
        end
    end
end

function leftlinkinds(ψ::VidalMPS, j::Integer; kwargs...)
    N = nsites(ψ)

    return if N == 1
        nothing
    else
        if j == 1
            nothing
        else
            commoninds(site_tensors(ψ)[j], bond_tensors(ψ)[j - 1]; kwargs...)
        end
    end
end

"""
    rightlinkind(ψ::VidalMPS, j::Integer; kwargs...)

Return the link Index pointing to the right of the `j`th site tensor of the VidalMPS.

You can choose different filters, like prime level and tags, with the `kwargs`.
"""
function rightlinkind(ψ::VidalMPS, j::Integer; kwargs...)
    N = nsites(ψ)

    return if N == 1
        nothing
    else
        if j == N
            nothing
        else
            commonind(site_tensors(ψ)[j], bond_tensors(ψ)[j]; kwargs...)
        end
    end
end

function rightlinkinds(ψ::VidalMPS, j::Integer; kwargs...)
    N = nsites(ψ)

    return if N == 1
        nothing
    else
        if j == N
            nothing
        else
            commoninds(site_tensors(ψ)[j], bond_tensors(ψ)[j]; kwargs...)
        end
    end
end

"""
    linkinds(ψ::VidalMPS, j::Integer; kwargs...)

Return the link Indices found of the VidalMPS at the site `j` as an IndexSet.

Optionally filter prime tags and prime levels with keyword arguments like `plev` and `tags`.
"""
function ITensorMPS.linkinds(ψ::VidalMPS, j::Integer; kwargs...)
    N = nsites(ψ)

    return if N == 1
        nothing
    else
        if j == 1
            commoninds(site_tensors(ψ)[j], bond_tensors(ψ)[j]; kwargs...)
        elseif j == N
            commoninds(site_tensors(ψ)[j], bond_tensors(ψ)[j - 1]; kwargs...)
        else
            [
                commoninds(site_tensors(ψ)[j], bond_tensors(ψ)[j - 1]; kwargs...);
                commoninds(site_tensors(ψ)[j], bond_tensors(ψ)[j]; kwargs...)
            ]
        end
    end
end

function leftlinkinds(ψ::VidalMPS; kwargs...)
    return [leftlinkind(ψ, j; kwargs...) for j in 2:nsites(ψ)]
end

function rightlinkinds(ψ::VidalMPS; kwargs...)
    return [rightlinkind(ψ, j; kwargs...) for j in 1:(nsites(ψ) - 1)]
end

function ITensorMPS.linkinds(ψ::VidalMPS; kwargs...)
    return [linkinds(ψ, j; kwargs...) for j in 1:nsites(ψ)]
end

### Index manipulation functions

function ITensors.replaceinds!(
    ::typeof(linkinds),
    ψ::VidalMPS,
    new_ls_left::Vector{<:Index},
    new_ls_right::Vector{<:Index},
)
    # Replace the link indices of ψ with the new sets `new_ls_left` and `new_ls_right`.
    site_ts = site_tensors(ψ)
    bond_ts = bond_tensors(ψ)
    N = nsites(ψ)

    # Apparently there is no need to use `dag` to control the indices here...
    for i in 1:(N - 1)
        old_link_l = rightlinkind(ψ, i)
        bond_ts[i] = replaceinds(bond_ts[i - 1], old_link_r => new_ls_right[i])
        site_ts[i] = replaceinds(site_ts[i], old_link_r => new_ls_right[i])
    end
    for i in 2:N
        old_link_l = leftlinkind(ψ, i)
        bond_ts[i - 1] = replaceinds(bond_ts[i - 1], old_link_l => new_ls_left[i])
        site_ts[i] = replaceinds(site_ts[i], old_link_l => new_ls_left[i])
    end

    return ψ
end

function ITensors.replaceinds(
    ::typeof(linkinds),
    ψ::VidalMPS,
    new_ls_left::Vector{<:Index},
    new_ls_right::Vector{<:Index},
)
    return replaceinds!(linkinds, copy(ψ), new_ls_left, new_ls_right)
end

function Base.map!(f::Function, ψ::VidalMPS)
    site_ts = site_tensors(ψ)
    for i in eachindex(site_ts)
        site_ts[i] = f(site_ts[i])
    end

    bond_ts = bond_tensors(ψ)
    for i in eachindex(bond_ts)
        bond_ts[i] = f(bond_ts[i])
    end

    return ψ
end

Base.map(f::Function, ψ::VidalMPS) = map!(f, copy(ψ))

function Base.map!(f::Function, ::typeof(linkinds), ψ::VidalMPS)
    # Apply `f` to all link indices of the VidalMPS. In practice: replace the link indices
    # `l` of each tensor in `ψ` with `f(l)`. Be careful not to apply `f` twice (or not?)!
    site_ts = site_tensors(ψ)
    bond_ts = bond_tensors(ψ)
    N = nsites(ψ)

    for i in 1:(N - 1)
        old_l = rightlinkinds(ψ, i)
        new_l = f(old_l)
        # We can apply `f` to sets (i.e. Vectors) of indices as well.

        # `replaceinds` doesn't complain if some of the indices in `old_l` are not
        # present in the tensor...
        site_ts[i] = replaceinds(site_ts[i], old_l, new_l)
        bond_ts[i] = replaceinds(bond_ts[i], old_l, new_l)
    end
    for i in 2:N
        old_l = leftlinkinds(ψ, i)
        new_l = f(old_l)

        site_ts[i] = replaceinds(site_ts[i], old_l, new_l)
        bond_ts[i - 1] = replaceinds(bond_ts[i - 1], old_l, new_l)
    end

    return ψ
end

Base.map(f::Function, ::typeof(linkinds), ψ::VidalMPS) = map!(f, linkinds, copy(ψ))

function Base.map!(f::Function, ::typeof(siteinds), ψ::VidalMPS)
    site_ts = site_tensors(ψ)
    N = nsites(ψ)

    for i in 1:N
        s = siteinds(ψ, i)
        site_ts[i] = replaceinds(site_ts[i], s, f(s))
    end

    return ψ
end

Base.map(f::Function, ::typeof(siteinds), ψ::VidalMPS) = map!(f, siteinds, copy(ψ))

for (fname, fname!) in [
    (:(ITensors.dag), :(ITensorMPS.dag!)),
    (:(ITensors.prime), :(ITensors.prime!)),
    (:(ITensors.setprime), :(ITensors.setprime!)),
    (:(ITensors.noprime), :(ITensors.noprime!)),
    (:(ITensors.swapprime), :(ITensors.swapprime!)),
    (:(ITensors.replaceprime), :(ITensors.replaceprime!)),
    (:(ITensors.TagSets.addtags), :(ITensors.addtags!)),
    (:(ITensors.TagSets.removetags), :(ITensors.removetags!)),
    (:(ITensors.TagSets.replacetags), :(ITensors.replacetags!)),
    (:(ITensors.settags), :(ITensors.settags!)),
]
    @eval begin
        """
            $($fname)[!](ψ::VidalMPS, args...; kwargs...)

        Apply $($fname) to all ITensors of a VidalMPS, returning a new VidalMPS.

        The ITensors of the VidalMPS will be a view of the storage of the original ITensors.
        Alternatively apply the function in-place.
        """
        function $fname(ψ::VidalMPS, args...; kwargs...)
            return map(m -> $fname(m, args...; kwargs...), ψ)
        end

        function $(fname!)(ψ::VidalMPS, args...; kwargs...)
            return map!(m -> $fname(m, args...; kwargs...), ψ)
        end
    end
end

for (fname, fname!) in [
    (:(NDTensors.sim), :(ITensorMPS.sim!)),
    (:(ITensors.prime), :(ITensors.prime!)),
    (:(ITensors.setprime), :(ITensors.setprime!)),
    (:(ITensors.noprime), :(ITensors.noprime!)),
    (:(ITensors.TagSets.addtags), :(ITensors.addtags!)),
    (:(ITensors.TagSets.removetags), :(ITensors.removetags!)),
    (:(ITensors.TagSets.replacetags), :(ITensors.replacetags!)),
    (:(ITensors.settags), :(ITensors.settags!)),
]
    @eval begin
        """
            $($fname)[!](linkinds, ψ::VidalMPS, args...; kwargs...)

        Apply $($fname) to all link indices of a VidalMPS, returning a new VidalMPS.

        The ITensors of the VidalMPS will be a view of the storage of the original ITensors.
        """
        function $fname(ffilter::typeof(linkinds), ψ::VidalMPS, args...; kwargs...)
            return map(i -> $fname(i, args...; kwargs...), ffilter, ψ)
        end

        function $(fname!)(ffilter::typeof(linkinds), ψ::VidalMPS, args...; kwargs...)
            return map!(i -> $fname(i, args...; kwargs...), ffilter, ψ)
        end

        """
            $($fname)[!](siteinds, ψ::VidalMPS, args...; kwargs...)

        Apply $($fname) to all site indices of a VidalMPS, returning a new VidalMPS.

        The ITensors of the VidalMPS will be a view of the storage of the original ITensors.
        """
        function $fname(ffilter::typeof(siteinds), ψ::VidalMPS, args...; kwargs...)
            return map(i -> $fname(i, args...; kwargs...), ffilter, ψ)
        end

        function $(fname!)(ffilter::typeof(siteinds), ψ::VidalMPS, args...; kwargs...)
            return map!(i -> $fname(i, args...; kwargs...), ffilter, ψ)
        end
    end
end

function ITensorMPS.hascommoninds(::typeof(siteinds), A::VidalMPS, B::VidalMPS)
    N = nsites(A)
    for n in 1:N
        if !hascommoninds(siteinds(A, n), siteinds(B, n))
            return false
        end
    end
    return true
end

function ITensorMPS.check_hascommoninds(::typeof(siteinds), A::VidalMPS, B::VidalMPS)
    N = nsites(A)
    if nsites(B) != N
        throw(
            DimensionMismatch(
                "The two VidalMPSs have mismatched number of sites $N and $(nsites(B))."
            ),
        )
    end

    for n in 1:N
        if !hascommoninds(siteinds(A, n), siteinds(B, n))
            errmsg =
                "The two VidalMPSs must share site indices. On site $n, the former " *
                "has site indices $(siteinds(A, n)) while the latter has site indices " *
                "$(siteinds(B, n))."
            error(errmsg)
        end
    end

    return nothing
end
