### Bond truncation methods

"""
    truncate!(ψ::VidalMPS; kwargs...)
    truncate!(ψ::InverseCanonicalMPS; kwargs...)

Perform a truncation of all bonds of the MPS using the truncation parameters (`cutoff`,
`maxdim`, etc.) provided as keyword arguments.

Keyword arguments:

- `site_range=1:nsites(ψ)` - only truncate the bonds between these sites
- `callback=Returns(nothing)` - callback function that allows the user to save the
  per-bond truncation error. The API of `callback` expects to take two kwargs called `link`
  and `truncation_error` where `link` is of type `Pair{Int64, Int64}` and `truncation_error`
  is `Float64`. Consider the following example that illustrates one possible use case.

```julia
nbonds = nsites(ψ) - 1
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
    # When we truncate the (j, j+1) bond we need to incorporate Λⱼ₋₁ and Λⱼ₊₁ in the tensor
    # we want to decompose and truncate: the trivial bond tensors help us write a simpler
    # routine, without having to discriminate the j=1 and j=N-1 case each time (where
    # normally there wouldn't be a bond tensor both on the left and on the right).

    # We perform truncations from right to left. This is how ITensor does it, and how we
    # should implement it if we want the results to match, i.e. if we want that
    #   truncate(v::MPS; ...) ≈ truncate(convert(VidalMPS, v); ...)
    # for the same `site_range`, `cutoff` and `maxdim` on both sides.
    for j in reverse((first(site_range) + 1):last(site_range))
        M = bond_ts[j - 2] * site_ts[j - 1] * bond_ts[j - 1] * site_ts[j] * bond_ts[j]
        #      │           │           │           │
        #  ╶╶╶─○─────◇─────○─────◇─────○─────◇─────○─╶╶
        #        Λ[j-2] Γ[j-1] Λ[j-1] Γ[j]  Λ[j]

        # We should have
        #   inds(M) = (rⱼ₋₂, sⱼ₋₁, sⱼ, lⱼ₊₁)
        # except if j == 1 or j == N-1, then there's one bond index less, but we don't need
        # to worry, because the trivial tensors will be picked up instead of getting an
        # error because the index is out of bounds.
        # We'll decompose M as U*S*V, then put U in the j-th site tensor, V in the
        # (j-1)-th one, then continue with the truncation on the (j-1)-th bond.

        linds = uniqueinds(M, bond_ts[j - 2] * site_ts[j - 1])
        #                           ┌╶╶╶╶╶╶╶╶╶╶╶╶┐
        #                 sⱼ₋₁      ╎  sⱼ        ╎
        #      │  rⱼ₋₂     │     M  ╎  │     lⱼ₊₁╎ │
        #  ╶╶╶─○──────▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▓▓▓▓▓▓▓▓────╎─○─╶╶
        #                           └╶╶╶╶╶╶╶╶╶╶╶╶┘
        #                                linds
        #
        # inds(bond_ts[j - 2] * site_ts[j - 1]) = (rⱼ₋₂, sⱼ₋₁)
        #  ↪ uniqueinds(M, bond_ts[j - 2] * site_ts[j - 1]) =
        #       = (rⱼ₋₂, sⱼ₋₁, sⱼ, lⱼ₊₁) \ (rⱼ₋₂, sⱼ₋₁) = (sⱼ, lⱼ₊₁)

        ltags = tags(commonind(bond_ts[j - 1], site_ts[j]))  # = "Link,l=j"
        rtags = tags(commonind(site_ts[j - 1], bond_ts[j - 1]))  # = "Link,r=j-1"

        U, bond_ts[j - 1], V, spec = svd(
            M, linds; lefttags=ltags, righttags=rtags, kwargs...
        )
        #               sⱼ₋₁                 sⱼ
        #      │   rⱼ₋₂   │ rⱼ₋₁         lⱼ  │   lⱼ₊₁  │
        #  ╶╶╶─○──────────V───────Λⱼ₋₁───────U─────────○─╶╶
        #
        # The  `lefttags` are assigned to the Λⱼ₋₁──U bond, while the `righttags` to the
        # V──Λⱼ₋₁ one.

        callback(; link=(j => j - 1), truncation_error=spec.truncerr)

        # Restore the Vidal form by "removing" the singular values we previously
        # incorporated.
        site_ts[j] = inv.(bond_ts[j]) * U
        site_ts[j - 1] = V * inv.(bond_ts[j - 2])
    end

    return ψ
end

function ITensorMPS.truncate!(
    ψ::InverseCanonicalMPS; site_range=1:nsites(ψ), callback=Returns(nothing), kwargs...
)
    site_ts = site_tensors(ψ)
    bond_ts = bond_tensors(ψ)
    # When we truncate the (j, j+1) bond we need to incorporate Λⱼ₋₁ and Λⱼ₊₁ in the tensor
    # we want to decompose and truncate: the trivial bond tensors help us write a simpler
    # routine, without having to discriminate the j=1 and j=N-1 case each time (where
    # normally there wouldn't be a bond tensor both on the left and on the right).

    # We perform truncations from right to left. This is how ITensor does it, and how we
    # should implement it if we want the results to match, i.e. if we want that
    #   truncate(v::MPS; ...) ≈ truncate(convert(VidalMPS, v); ...)
    # for the same `site_range`, `cutoff` and `maxdim` on both sides.
    for j in reverse((first(site_range) + 1):last(site_range))
        M = site_ts[j - 1] * bond_ts[j - 1] * site_ts[j]
        #      │           │           │           │
        #  ╶╶╶─○─────◇─────○─────◇─────○─────◇─────○─╶╶
        #               C[j-1] V[j-1] C[j]

        # We should have
        #   inds(M) = (lⱼ₋₁, sⱼ₋₁, sⱼ, rⱼ)
        # except if j == 1 or j == N-1, then there's one bond index less, but we don't need
        # to worry, because the trivial tensors will be picked up instead of getting an
        # error because the index is out of bounds.
        # We'll decompose M as U*S*W, then put US⁻¹ in the j-th site tensor, W in the
        # (j-1)-th one, then continue with the truncation on the (j-1)-th bond.

        linds = uniqueinds(M, site_ts[j - 1])
        #                           ┌╶╶╶╶╶╶╶╶╶╶╶╶┐
        #                 sⱼ₋₁      ╎  sⱼ        ╎
        #         lⱼ₋₁     │     M  ╎  │     rⱼ  ╎
        #  ╶╶╶─◇──────▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▓▓▓▓▓▓▓▓────╎─◇─╶╶
        #                           └╶╶╶╶╶╶╶╶╶╶╶╶┘
        #                                linds
        #
        # inds(site_ts[j - 1]) = (lⱼ₋₁, rⱼ₋₁, sⱼ₋₁)
        #  ↪ uniqueinds(M, site_ts[j - 1]) =
        #       = (lⱼ₋₁, sⱼ₋₁, sⱼ, rⱼ) \ (lⱼ₋₁, rⱼ, sⱼ₋₁) = (sⱼ, rⱼ)

        ltags = tags(commonind(bond_ts[j - 1], site_ts[j]))  # = "Link,l=j"
        rtags = tags(commonind(site_ts[j - 1], bond_ts[j - 1]))  # = "Link,r=j-1"

        U, S, W, spec = svd(M, linds; lefttags=ltags, righttags=rtags, kwargs...)
        #              sⱼ₋₁              sⱼ
        #         lⱼ₋₁   │ rⱼ₋₁      lⱼ  │  rⱼ 
        #  ╶╶╶─◇─────────W───────S───────U──────◇──╶╶
        #
        # The  `lefttags` are assigned to the S──U bond, while the `righttags` to the
        # W──S one.

        callback(; link=(j => j - 1), truncation_error=spec.truncerr)

        # Restore the inverse canonical form.
        bond_ts[j - 1] = inv.(S)
        site_ts[j] = S * U
        site_ts[j - 1] = W * S
    end

    return ψ
end

# Copying version (same for both types)
function ITensorMPS.truncate(ψ₀::ExplicitBondMPS; kwargs...)
    ψ = copy(ψ₀)
    truncate!(ψ; kwargs...)
    return ψ
end
