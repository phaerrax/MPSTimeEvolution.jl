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
function LinearAlgebra.norm(ψ::VidalMPS)
    norm2_ψ = dot(ψ, ψ)
    rtol = eps(real(NDTensors.scalartype(ψ))) * 10
    atol = rtol
    if !IsApprox.isreal(norm2_ψ, IsApprox.Approx(; rtol=rtol, atol=atol))
        @warn "norm² is $norm2_ψ, which is not real up to a relative tolerance of " *
            "$rtol and an absolute tolerance of $atol. Taking the real part, which " *
            "may not be accurate."
    end
    return sqrt(real(norm2_ψ))
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
        linds = uniqueinds(site_ts[j], bond_ts[j] * site_ts[j + 1])
        ltags = tags(commonind(site_ts[j], bond_ts[j] * site_ts[j + 1]))
        rtags = tags(commonind(site_ts[j] * bond_ts[j], site_ts[j + 1]))
        U, S, V, spec = svd(
            site_ts[j] * bond_ts[j], linds; lefttags=ltags, righttags=rtags, kwargs...
        )
        site_ts[j] = U
        bond_ts[j] = S
        site_ts[j + 1] *= V
        callback(; link=(j => j - 1), truncation_error=spec.truncerr)
    end
    return ψ
end

function ITensorMPS.truncate(ψ0::VidalMPS; kwargs...)
    ψ = copy(ψ0)
    truncate!(ψ; kwargs...)
    return ψ
end
