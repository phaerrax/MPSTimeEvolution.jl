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
