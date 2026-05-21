using NDTensors.BackendSelection: @Algorithm_str, Algorithm

### Inner products

"""
    inner(A::VidalMPS, B::VidalMPS)
    inner(A::InverseCanonicalMPS, B::InverseCanonicalMPS)

Compute the inner product `⟨A,B⟩`.

Same as [`dot`](@ref).
"""
ITensorMPS.inner(ψ1::MPST, ψ2::MPST) where {MPST<:ExplicitBondMPS} = dot(ψ1, ψ2)

"""
    dot(A::VidalMPS, B::VidalMPS)
    dot(A::InverseCanonicalMPS, B::InverseCanonicalMPS)

Compute the inner product `⟨A,B⟩`.

Same as [`inner`](@ref).
"""
function LinearAlgebra.dot(ψ1::MPST, ψ2::MPST)::Number where {MPST<:ExplicitBondMPS}
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

function _simplified_norm_contraction(ψ::VidalMPS)
    contr_post = prod(scalar(Λ*Λ) for Λ in bond_tensors(ψ)[2:end]; init=1.0)
    return contr_post * norm(site_tensors(ψ)[1] * bond_tensors(ψ)[1])
end

function _simplified_norm_contraction(ψ::InverseCanonicalMPS)
    contr_post = prod(scalar(inv.(V) * inv.(V)) for V in bond_tensors(ψ); init=1.0)
    return contr_post * norm(site_tensors(ψ)[1])
end

"""
    norm(ψ::VidalMPS)
    norm(ψ::InverseCanonicalMPS)

Compute the norm of the MPS, assuming that it satisfies the (inverse-) canonical gauge
conditions.
"""
function LinearAlgebra.norm(
    ψ::ExplicitBondMPS; neg_atol=eps(real(NDTensors.scalartype(ψ))) * 10
)
    rtol = eps(real(NDTensors.scalartype(ψ))) * 10
    atol = rtol

    norm2_ψ = _simplified_norm_contraction(ψ)

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
                "norm² is $norm2_ψ, which is negative beyond an absolute tolerance " *
                "of $neg_atol.",
            )
        end
    end

    return sqrt(norm2_ψ)
end

### Sums of MPSs

"""
    +(A::VidalMPS...; kwargs...)
    +(A::InverseCanonicalMPS...; kwargs...)

    add(A::VidalMPS...; kwargs...)
    add(A::InverseCanonicalMPS...; kwargs...)

Add arbitrary numbers of (inverse-) canonical MPSs with each other, optionally truncating
the results.

A cutoff of 1e-15 is used by default, and in general users should set their own cutoff for
their particular application.

See [`ITensorMPS.add`](@extref) for an explanation of the accepted arguments and some
examples.
"""
function Base.:(+)(
    ψs::MPST...; alg=Algorithm"densitymatrix"(), kwargs...
) where {MPST<:ExplicitBondMPS}
    return +(Algorithm(alg), ψs...; kwargs...)
end

function Base.:(+)(::Algorithm"directsum", ψs::MPST...) where {MPST<:ExplicitBondMPS}
    # XXX The direct-sum algorithm, in general, yields an MPS which is not in the Vidal
    # gauge (because orthonormality rules are not satisfied). The same likely happens in the
    # inverse canonical gauge as well.
    # This is something that already happens at the level of standard MPSs:
    #
    #   julia> N = 10; s = siteinds("S=1/2", N);
    #
    #   julia> v = random_mps(ComplexF64, s; linkdims=2);
    #
    #   julia> w = random_mps(ComplexF64, s; linkdims=5);
    #
    #   julia> z = +(v, w; alg="directsum");
    #
    #   julia> function right_ortho_check(z)
    #              zdag = sim(linkinds, dag(z))
    #              rchecks = [
    #                  matrix(
    #                      z[j] * zdag[j]
    #                      * delta(commoninds(z[j] * zdag[j], z[j+1] * zdag[j+1]))
    #                  ) ≈ I
    #                  for j in 2:N-1
    #              ]
    #              return [rchecks; matrix(z[end] * zdag[end]) ≈ I]
    #          end
    #
    #   julia> all(right_ortho_check(z))
    #   false
    #
    # so this is a well-known “issue”. As a matter of fact, ITensor sets the orthonormality
    # limits (i.e. the sites delimiting the position of the orthocentre) of the MPS returned
    # from the direct sum to 1 and N, that practically means that none of the tensors are
    # left- or right-orthonormal.  The other MPS functions in the library read these
    # orthonormality limits and know that they cannot simplify the contractions using the
    # cancellation rules.
    #
    #   julia> orthocenter(z)
    #   ERROR: MPS has no well-defined orthogonality center, orthogonality center is on the
    #       range 1:10.
    #
    # However, check this:
    #
    #   julia> zz = +(convert(VidalMPS, v), convert(VidalMPS, w); alg="directsum");
    #
    #   julia> MPSTimeEvolution.check_vidal_form(convert(VidalMPS, zz); verbose=false)
    #   false
    #
    # ...but
    #
    #   julia> zo = orthogonalize(z, 1);
    #
    #   julia> MPSTimeEvolution.check_vidal_form(convert(VidalMPS, zo); verbose=false)
    #   true
    #
    # This tells us that there's a way to restore the Vidal gauge, by repeating the same
    # steps as in the `orthogonalize` function. (See the `canonicalize` function.)

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
        sum_bond_ts[j] = diag_itensor(vector(diag(Λⱼ)), inds(Λⱼ)...)
        # If we just wrote
        #   sum_bond_ts[j] = Λⱼ
        # then we would get a dense tensor, which is the wrong choice since we often take
        # the inverse of the bond tensors Λ with `inv.(Λ)`. If Λ is dense (i.e. a normal
        # ITensor), then the inverse is actually taken componentwise due to the
        # broadcasting, and consequently we get a load of Infs and NaNs in our calculations.
        # If we convert the Λs to diagonal ITensors instead, the `inv.` method doesn't touch
        # the elements out of the diagonal, and gives us the actual matrix inverse of Λ.

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
    sum_bond_ts[n - 1] = diag_itensor(vector(diag(Λₙ₋₁)), inds(Λₙ₋₁)...)

    # Last site tensor. Here once again we have just one set of link indices.
    Γₙ, (lₙ,) = directsum(
        (site_tensors(ψᵢ)[n] => (leftlinkind(ψᵢ, n),) for ψᵢ in ψs)...;
        tags=[tags(leftlinkind(first(ψs), n))],
    )
    Γₙ = replaceind(Γₙ, lₙ => dag(prev_link_inds))
    sum_site_ts[n] = Γₙ

    return MPST(sum_site_ts, OffsetVector([ITensor(1.0); sum_bond_ts; ITensor(1.0)], 0:n))
end

function Base.:(+)(
    ::Algorithm"densitymatrix", ψs::MPST...; cutoff=1e-15, kwargs...
) where {MPST<:ExplicitBondMPS}
    return convert(
        MPST,
        sum([convert(MPS, ψ) for ψ in ψs]; cutoff=cutoff, kwargs...);
        cutoff=cutoff,
        kwargs...,
    )
end

ITensorMPS.add(ψs::ExplicitBondMPS...; kwargs...) = +(ψs...; kwargs...)

function scalarmult!(ψ::VidalMPS, a::Number)
    # Multiplying the MPS by a is equivalent to multiplying one of its tensors by a.
    # However, in order to preserve the Vidal form, the bond tensors must contain
    # non-negative values only, and the site tensors have some orthogonality conditions to
    # satisfy. Thus, we multiply the first of the bond tensors by |a| and the first site
    # tensor by exp(i*arg(a)), which means that we multiply the vectors associated to the
    # singular values (of the first bond tensor) by a unit complex number. This should be
    # okay.
    st = site_tensors(ψ)
    bt = bond_tensors(ψ)
    st[1] *= cis(angle(a))
    bt[2] *= abs(a)
    return ψ
end

function scalarmult!(ψ::InverseCanonicalMPS, a::Number)
    # Still unclear how to do this if a is not a unit complex number, while preserving the
    # inverse canonical form.
    if abs(a) ≈ 1
        site_tensors(ψ)[1] *= a
    else
        error("scalar factor multiplying the MPS does not have unit absolute value.")
    end
    return ψ
end

function scalarmult(ψ::ExplicitBondMPS, a::Number)
    return scalarmult!(copy(ψ), a)
end

Base.:(*)(ψ::ExplicitBondMPS, a::Number) = scalarmult(ψ, a)
Base.:(*)(a::Number, ψ::ExplicitBondMPS) = scalarmult(ψ, a)

Base.:(+)(ψ::ExplicitBondMPS) = ψ

Base.:(-)(ψ::ExplicitBondMPS) = scalarmult(ψ, -1)
Base.:(-)(ψ::MPST, ϕ::MPST) where {MPST<:ExplicitBondMPS} = +(ψ, -ϕ)

Base.:(/)(ψ::ExplicitBondMPS, a::Number) = scalarmult(ψ, inv(a))

function Base.isapprox(
    x::MPST,
    y::MPST;
    atol::Real=0,
    rtol::Real=Base.rtoldefault(
        LinearAlgebra.promote_leaf_eltypes(x), LinearAlgebra.promote_leaf_eltypes(y), atol
    ),
) where {MPST<:ExplicitBondMPS}
    d = norm(x - y)
    if isfinite(d)
        return d <= max(atol, rtol * max(norm(x), norm(y)))
    else
        error(
            "In `isapprox(x::",
            typeof(x),
            "y::",
            typeof(y),
            "`, `norm(x - y)` is not finite",
        )
    end
end
