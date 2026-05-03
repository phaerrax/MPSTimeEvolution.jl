using NDTensors.BackendSelection: @Algorithm_str, Algorithm

### Inner products

function LinearAlgebra.dot(ψ1::VidalMPS, ψ2::VidalMPS)::Number
    N = nsites(ψ1)
    if nsites(ψ2) != N
        throw(DimensionMismatch("inner: mismatched number of sites $N and $(nsites(ψ2))"))
    end

    ψ1dag = dag(ψ1)

    # Replace ψ1dag's link indices with a new set of indices: otherwise they might clash
    # with ψ2's indices (for example if ψ1 = ψ2).
    sim!(linkinds, ψ1dag)

    # Check whether the two MPSs are defined on the same set of site indices. Return an
    # error if false.
    check_hascommoninds(siteinds, ψ1dag, ψ2)

    # Contract the tensors lengthwise.
    x = site_tensors(ψ1dag)[N] * site_tensors(ψ2)[N]
    for j in reverse(1:(N - 1))
        x = (x * bond_tensors(ψ1dag)[j]) * bond_tensors(ψ2)[j]
        x = (x * site_tensors(ψ1dag)[j]) * site_tensors(ψ2)[j]
    end

    dot_ψ1_ψ2 = scalar(x)

    if !isfinite(dot_ψ1_ψ2)
        @warn "The inner product (or norm²) you are computing is very large " *
            "($dot_ψ1_ψ2). You should consider using `lognorm` or `loginner` instead, " *
            "which will help avoid floating point errors. For example if you are trying " *
            "to normalize your MPS/MPO `A`, the normalized MPS/MPO `B` would be given by " *
            "`B = A ./ z` where `z = exp(lognorm(A) / length(A))`."
    end

    return dot_ψ1_ψ2
end

"""
    norm(A::VidalMPS)

Compute the norm of the `VidalMPS`.
"""
function LinearAlgebra.norm(ψ::VidalMPS; neg_atol=eps(real(NDTensors.scalartype(ψ))) * 10)
    norm2_ψ = dot(ψ, ψ)
    rtol = eps(real(NDTensors.scalartype(ψ))) * 10
    atol = rtol

    if !IsApprox.isreal(norm2_ψ, IsApprox.Approx(; rtol=rtol, atol=atol))
        @warn "norm² is $norm2_ψ, which is not real up to a relative tolerance of " *
            "$rtol and an absolute tolerance of $atol. Taking the real part, which " *
            "may not be accurate."
    end
    norm2_ψ = real(norm2_ψ)

    # Sometimes it happens that ⟨ψ, ψ⟩ is slightly below zero (~1e-16, always within
    # numerical accuracy), likely because of some rounding inaccuracies.
    # UGLY HACK: check whether ⟨ψ, ψ⟩ < 0 within some small error, and if so return zero,
    # otherwise throw a genuine error.
    if norm2_ψ < 0
        if abs(norm2_ψ) < neg_atol
            norm2_ψ = zero(norm2_ψ)
        else
            error(
                "norm² is $norm2_ψ, which is negative beyond an absolute tolerance of $neg_atol.",
            )
        end
    end

    return sqrt(norm2_ψ)
end

### Truncation

"""
    truncate!(M::VidalMPS; kwargs...)

Perform a truncation of all bonds of a VidalMPS using the truncation parameters
(`cutoff`, `maxdim`, etc.) provided as keyword arguments.

Keyword arguments:

- `site_range=1:nsites(ψ)` - only truncate the bonds between these sites
- `callback=Returns(nothing)` - callback function that allows the user to save the
  per-bond truncation error. The API of `callback` expects to take two kwargs called `link`
  and `truncation_error` where `link` is of type `Pair{Int64, Int64}` and `truncation_error`
  is `Float64`. Consider the following example that illustrates one possible use case.

```julia
nbonds = 9
truncation_errors = zeros(nbonds)
function callback(; link, truncation_error)
    bond_no = last(link)
    truncation_errors[bond_no] = truncation_error
    return nothing
end
truncate!(ψ; maxdim = 5, cutoff = 1E-7, callback)
```
"""
function ITensorMPS.truncate!(
    ψ::VidalMPS; site_range=1:nsites(ψ), callback=Returns(nothing), kwargs...
)
    site_ts = site_tensors(ψ)
    bond_ts = bond_tensors(ψ)

    # Perform truncations from left to right.
    for j in first(site_range):(last(site_range) - 1)
        M = site_ts[j] * bond_ts[j] * site_ts[j + 1]

        linds = uniqueinds(site_ts[j], bond_ts[j])
        ltags = tags(commonind(site_ts[j], bond_ts[j]))
        rtags = tags(commonind(bond_ts[j], site_ts[j + 1]))

        site_ts[j], bond_ts[j], site_ts[j + 1], spec = svd(
            M, linds; lefttags=ltags, righttags=rtags, kwargs...
        )

        callback(; link=(j => j - 1), truncation_error=spec.truncerr)
    end

    return ψ
end

function ITensorMPS.truncate(ψ0::VidalMPS; kwargs...)
    ψ = copy(ψ0)
    truncate!(ψ; kwargs...)
    return ψ
end

### Sums of MPSs

"""
    +(A::VidalMPS...; kwargs...)
    add(A::VidalMPS...; kwargs...)

Add arbitrary numbers of `VidalMPS` with each other, optionally truncating the results.

A cutoff of 1e-15 is used by default, and in general users should set their own cutoff for
their particular application.

# Keywords

- `cutoff::Real`: singular value truncation cutoff
- `maxdim::Int`: maximum MPS bond dimension
- `alg = "densitymatrix"`: `"densitymatrix"` or `"directsum"`. `"densitymatrix"` adds the
  MPSs by adding up and diagonalizing local density matrices site by site in a single sweep
  through the system, truncating the density matrix with `cutoff` and `maxdim`.
  `"directsum"` performs a direct sum of each tensors on each site of the input MPS being
  summed. It doesn't perform any truncation, and therefore ignores `cutoff` and `maxdim`.
  The bond dimension of the output is the sum of the bond dimensions of the inputs. You can
  truncate the resulting MPS with the `truncate!` function.

# Examples

```julia
N = 10
s = siteinds("S=1/2", N)

state = n -> isodd(n) ? "↑" : "↓"
ψ₁ = convert(VidalMPS, random_mps(s, state; linkdims = 2))
ψ₂ = convert(VidalMPS, random_mps(s, state; linkdims = 2))
ψ₃ = convert(VidalMPS, random_mps(s, state; linkdims = 2))

ψ = +(ψ₁, ψ₂)
ψ = ψ₁ + ψ₂

println()
@show inner(ψ, ψ)
@show inner(ψ₁, ψ₂) + inner(ψ₁, ψ₂) + inner(ψ₂, ψ₁) + inner(ψ₂, ψ₂)

# Computes ψ₁ + 2ψ₂
ψ = ψ₁ + 2ψ₂

println()
@show inner(ψ, ψ)
@show inner(ψ₁, ψ₁) + 2 * inner(ψ₁, ψ₂) + 2 * inner(ψ₂, ψ₁) + 4 * inner(ψ₂, ψ₂)

# Computes ψ₁ + 2ψ₂ + ψ₃
ψ = ψ₁ + 2ψ₂ + ψ₃

println()
@show inner(ψ, ψ)
@show inner(ψ₁, ψ₁) + 2 * inner(ψ₁, ψ₂) + inner(ψ₁, ψ₃) +
      2 * inner(ψ₂, ψ₁) + 4 * inner(ψ₂, ψ₂) + 2 * inner(ψ₂, ψ₃) +
      inner(ψ₃, ψ₁) + 2 * inner(ψ₃, ψ₂) + inner(ψ₃, ψ₃)
```
"""
function Base.:(+)(ψs::VidalMPS...; alg=Algorithm"densitymatrix"(), kwargs...)
    return +(Algorithm(alg), ψs...; kwargs...)
end

function Base.:(+)(::Algorithm"directsum", ψs::VidalMPS...)
    @assert allequal(nsites, ψs)
    n = nsites(first(ψs))

    # Output tensors:
    sum_site_ts = Vector{ITensor}(undef, n)
    sum_bond_ts = Vector{ITensor}(undef, n-1)

    # First tensor of the direct sum:
    Γ₁, (r₁,) = directsum(
        (site_tensors(ψᵢ)[1] => (rightlinkind(ψᵢ, 1),) for ψᵢ in ψs)...;
        tags=[tags(rightlinkind(first(ψs), 1))],
    )
    # Γ₁ is the direct sum of the site_tensors(ψᵢ)[1]'s over the link indices: it will have
    # the site index shared by all the site_tensors(ψᵢ)[1]s, and a link index that runs over
    # all their link indices.
    # r₁ is the new collective (right) link index (the new indices are returned as a tuple
    # --- there may be more than one, as we will se later --- so we write (r₁,) to extract
    # r₁ from the tuple).

    prev_link_inds = r₁  # The link indices of the previous site
    sum_site_ts[1] = Γ₁  # Set the first tensor in the output Vidal MPS

    for j in 1:(n - 2)
        # Repeat the direct sum on the other sites. In this loop we have two sets of link
        # indices we need to group.

        # Bond tensors:
        Λⱼ, (rⱼ, lⱼ₊₁) = directsum(
            (
                bond_tensors(ψᵢ)[j] => (rightlinkind(ψᵢ, j), leftlinkind(ψᵢ, j+1)) for
                ψᵢ in ψs
            )...;
            tags=[tags(rightlinkind(first(ψs), j)), tags(leftlinkind(first(ψs), j+1))],
        )
        Λⱼ = replaceind(Λⱼ, rⱼ => dag(prev_link_inds))
        prev_link_inds = lⱼ₊₁
        sum_bond_ts[j] = Λⱼ

        # Site tensors:
        Γⱼ₊₁, (lⱼ₊₁, rⱼ₊₁) = directsum(
            (
                site_tensors(ψᵢ)[j + 1] => (leftlinkind(ψᵢ, j+1), rightlinkind(ψᵢ, j+1)) for
                ψᵢ in ψs
            )...;
            tags=[tags(leftlinkind(first(ψs), j + 1)), tags(rightlinkind(first(ψs), j+1))],
        )
        Γⱼ₊₁ = replaceind(Γⱼ₊₁, lⱼ₊₁ => dag(prev_link_inds))
        prev_link_inds = rⱼ₊₁
        sum_site_ts[j + 1] = Γⱼ₊₁
    end

    # Last bond tensor:
    Λₙ₋₁, (rₙ₋₁, lₙ) = directsum(
        (
            bond_tensors(ψᵢ)[n - 1] => (rightlinkind(ψᵢ, n-1), leftlinkind(ψᵢ, n)) for
            ψᵢ in ψs
        )...;
        tags=[tags(rightlinkind(first(ψs), n-1)), tags(leftlinkind(first(ψs), n))],
    )
    Λₙ₋₁ = replaceind(Λₙ₋₁, rₙ₋₁ => dag(prev_link_inds))
    prev_link_inds = lₙ
    sum_bond_ts[n - 1] = Λₙ₋₁

    # Last site tensor. Here once again we have just one set of link indices.
    Γₙ, (lₙ,) = directsum(
        (site_tensors(ψᵢ)[n] => (leftlinkind(ψᵢ, n),) for ψᵢ in ψs)...;
        tags=[tags(leftlinkind(first(ψs), n))],
    )
    Γₙ = replaceind(Γₙ, lₙ => dag(prev_link_inds))
    sum_site_ts[n] = Γₙ

    return VidalMPS(sum_site_ts, sum_bond_ts)
end

function Base.:(+)(::Algorithm"densitymatrix", ψs::VidalMPS...; cutoff=1e-15, kwargs...)
    return convert(
        VidalMPS,
        sum([convert(MPS, ψ) for ψ in ψs]; cutoff=cutoff, kwargs...);
        cutoff=cutoff,
        kwargs...,
    )
end

Base.:(+)(ψ::VidalMPS) = ψ

ITensorMPS.add(ψs::VidalMPS...; kwargs...) = +(ψs...; kwargs...)
