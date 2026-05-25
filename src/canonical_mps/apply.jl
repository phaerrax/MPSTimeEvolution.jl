function replace_and_decompose!(ψ::VidalMPS, M::ITensor; kwargs...)
    # Replace the sites of the Vidal MPS `ψ` with the tensor `A`, splitting up `A` into MPS
    # tensors. THE TENSORS TO BE REPLACED ARE AUTOMATICALLY DETERMINED BY A'S SITE INDICES.
    # THIS FUNCTION DOESN'T TAKE INTO ACCOUNT THE BOND TENSORS ON THE LEFT AND ON THE RIGHT
    # OF A. KEEP THIS IN MIND WHEN USING THIS FUNCTION WHEN APPLYING MULTI-SITE OPERATORS.

    # The ITensor A will have some site indices. We need to factor it into an MPS-like form,
    # with a different site block for each site index.

    # 1. Get the site indices involved in the decomposition, and sort them by increasing
    #    site number.
    ns = findsites(ψ, M)
    sort!(ns)
    N = length(ns)

    # 2. Gather the indices relative to the leftmost site of A.
    site_ts = site_tensors(ψ)
    bond_ts = bond_tensors(ψ)
    linds = commoninds(M, bond_ts[ns[1] - 1] * site_ts[ns[1]])

    # 3. Recursively decompose A with an SVD, until we exhaust the site indices.
    U, S, V = svd(
        M, linds...; lefttags="Link,r=$(ns[1])", righttags="Link,l=$(ns[2])", kwargs...
    )

    # Now we assign U to the first site tensors within the segment that is being updated.
    # IF WE ARE APPLYING A MULTI-SITE OPERATOR, WE WILL ALSO NEED TO MULTIPLY site_ts[ns[1]]
    # BY THE INVERSE OF THE BOND TENSOR ON ITS LEFT. THIS MUST BE DONE AFTER THIS FUNCTION
    # HAS ENDED.
    site_ts[ns[1]] = U
    bond_ts[ns[1]] = S

    for n in 2:(N - 1)
        M = S * V
        linds = commoninds(M, bond_ts[ns[n] - 1] * site_ts[ns[n]])
        U, S, V = svd(
            M,
            linds...;
            lefttags="Link,r=$(ns[n])",
            righttags="Link,l=$(ns[n+1])",
            kwargs...,
        )

        site_ts[ns[n]] = inv.(bond_ts[ns[n - 1]]) * U
        bond_ts[ns[n]] = S
    end

    site_ts[ns[N]] = V

    return ψ
end

"""
    apply(o::ITensor, ψ::VidalMPS; kwargs...)
    apply(o::ITensor, ψ::InverseCanonicalMPS; kwargs...)
    product([...])

Get the product of the operator `o` with the MPS `ψ`.

# Keywords

- `cutoff::Real`: singular value truncation cutoff.
- `maxdim::Int`: maximum MPS dimension.
"""
function ITensors.product(o::ITensor, ψ::VidalMPS; kwargs...)
    ψ = copy(ψ)

    # Determine the sites on which `o` acts.
    ns = findsites(ψ, o)
    N = length(ns)

    # Find out if the sites are consecutive or not.
    # If they are not, we should permute the MPS sites so that they are consecutive, then
    # permute them back to their original configuration.
    # For now, let's throw an error saying it's not implemented.
    diff_ns = diff(ns)
    ns′ = ns
    if any(!=(1), diff_ns)
        error("apply not implemented for non-consecutive application sites")
        # ns′ = [ns[1] + n - 1 for n in 1:N]
        # ψ = movesites(ψ, ns .=> ns′; kwargs...)
    end

    # Multiply everything in the VidalMPS from ns′[1] to ns′[end] together, and include the
    # bond tensors to the left of ns′[1] and to the right of ns′[end].
    # (This is useless, but also harmless, if `o` is a single-site operator, so we do it
    # anyway so that the code is simpler.)
    ϕ = bond_tensors(ψ)[ns′[1] - 1] * site_tensors(ψ)[ns′[1]]
    for n in 2:N
        ϕ *= bond_tensors(ψ)[ns′[n] - 1] * site_tensors(ψ)[ns′[n]]
    end
    ϕ *= bond_tensors(ψ)[ns′[N]]

    # Apply the operator to the combined site and bond tensors.
    ϕ = ITensors.product(o, ϕ)

    if length(ns) > 1
        # Insert the result in ψ, decomposing it into site and bond tensors...
        replace_and_decompose!(ψ, ϕ; kwargs...)
    else
        # ...or just replace the affected site tensor.
        site_tensors(ψ)[only(ns)] = ϕ
    end

    # Restore the Vidal form by re-inserting the bond tensors on the left and on the right.
    # Note that the bond tensors at ns′[1] and ns′[N] from the input VidalMPS were not
    # modified, so we don't need to “reinsert” them. We just need to multiply the first and
    # last of the new site tensors.
    site_tensors(ψ)[ns′[1]] *= inv.(bond_tensors(ψ)[ns′[1] - 1])
    site_tensors(ψ)[ns′[N]] *= inv.(bond_tensors(ψ)[ns′[N]])

    # If the applied operator is not unitary, it will rescale the columns/rows of the site
    # tensors on which it acts, in such a way that the orthonormality rules are not
    # satisfied anymore.  In this case, the MPS should be reorthogonalised in order to
    # restore the Vidal form.
    return ψ
end

function replace_and_decompose!(ψ::InverseCanonicalMPS, M::ITensor; kwargs...)
    # Same as replace_and_decompose!(::VidalMPS, ::ITensor; kwargs...), but here we don't
    # need to incorporate the “lateral” bond tensors.

    # 1. Get the site indices involved in the decomposition, and sort them by increasing
    #    site number.
    ns = findsites(ψ, M)
    sort!(ns)
    N = length(ns)  # ≥ 2

    # 2. Gather the indices relative to the leftmost site of A.
    #
    #                    ┌╶╶╶╶╶╶╶╶╶╶╶╶╶┐
    #                    ╎  │       │  ╎
    #                    ╎  ▒▒▒▒▒▒▒▒▒  ╎
    #       │       │    ╎  │       │  ╎    │
    #   ╶╶╶─▧───◆───▧───◆╎──▧───◆───▧──╎◆───▧─╶╶
    #      Cⱼ₋₂    Cⱼ₋₁  ╎  Cⱼ     Cⱼ₊₁╎   Cⱼ₊₂
    #                    └╶╶╶╶╶╶╶╶╶╶╶╶╶┘
    #
    #
    #                       sⱼ     sⱼ₊₁
    #       │       │     lⱼ│       │rⱼ₊₁   │
    #   ╶╶╶─▧───◆───▧───◆───▓▓▓▓▓▓▓▓▓───◆───▧─╶╶
    #      Cⱼ₋₂    Cⱼ₋₁         M          Cⱼ₊₂

    site_ts = site_tensors(ψ)
    bond_ts = bond_tensors(ψ)

    # 3. Recursively decompose A with an SVD, until we exhaust the site indices.
    linds = commoninds(M, site_ts[ns[1]])  # = (sⱼ, lⱼ), with j = ns[1]
    U, S, V = svd(
        M, linds...; lefttags="Link,r=$(ns[1])", righttags="Link,l=$(ns[2])", kwargs...
    )

    # We assign U to the first site tensors within the segment that is being updated.
    # We will frequently multiply the tensors by delta(inds(S)): it is necessary in order to
    # correct the link indices.
    site_ts[ns[1]] = (U * S) * delta(inds(S))
    bond_ts[ns[1]] = inv.(S)

    for n in 2:(N - 1)
        M = (S * V) * delta(inds(S))
        # inds(M) = (lⱼ, sⱼ, s₊₁, ..., sₙ, r), with j = ns[n] and
        linds = commoninds(M, site_ts[ns[n]] * bond_ts[ns[n] - 1])
        U, S, V = svd(
            M,
            linds...;
            lefttags="Link,r=$(ns[n])",
            righttags="Link,l=$(ns[n+1])",
            kwargs...,
        )

        site_ts[ns[n]] = (U * S) * delta(inds(S))  # = Λₙ₋₁ U S
        # This tensor already contains the singular values on the left, because it comes
        # from the decomposition of M = S * V.
        bond_ts[ns[n]] = inv.(S)
    end

    site_ts[ns[N]] = (S * V) * delta(inds(S))

    return ψ
end

function ITensors.product(o::ITensor, ψ::InverseCanonicalMPS; kwargs...)
    # Same as product(::ITensor, ::InverseCanonicalMPS; kwargs...), but here we don't
    # need to incorporate the “lateral” bond tensors.
    ψ = copy(ψ)

    site_ts = site_tensors(ψ)
    bond_ts = bond_tensors(ψ)

    # Determine the sites on which `o` acts.
    ns = findsites(ψ, o)
    N = length(ns)

    # Find out if the sites are consecutive or not.
    # If they are not, we should permute the MPS sites so that they are consecutive, then
    # permute them back to their original configuration.
    # For now, let's throw an error saying it's not implemented.
    diff_ns = diff(ns)
    ns′ = ns
    if any(!=(1), diff_ns)
        error("apply not implemented for non-consecutive application sites")
        # ns′ = [ns[1] + n - 1 for n in 1:N]
        # ψ = movesites(ψ, ns .=> ns′; kwargs...)
    end

    # Multiply everything in the MPS from ns′[1] to ns′[end] together.
    ϕ = site_ts[ns′[1]]
    for n in 2:N
        ϕ *= bond_ts[ns′[n] - 1] * site_ts[ns′[n]]
    end

    # Apply the operator to the combined site and bond tensors.
    ϕ = ITensors.product(o, ϕ)

    if length(ns) > 1
        # Insert the result in ψ, decomposing it into site and bond tensors...
        replace_and_decompose!(ψ, ϕ; kwargs...)
    else
        # ...or just replace the affected site tensor.
        site_ts[only(ns)] = ϕ
    end

    return ψ
end

function simplified_self_contraction(ψ::VidalMPS, A::ITensor)
    # Compute ⟨ψ, Aψ⟩ by contracting only the site tensors on which A acts.  If the opposite
    # tensors are directly contracted with each other, the following cancellation rules
    # hold:
    #
    #   Γ[1]───       ╭───
    #    │            │
    #    │        =   │
    #    │            │
    #   Γ[1]───       ╰───
    #
    #
    #   ───Γ[N]       ───╮
    #       │            │
    #       │     =      │
    #       │            │
    #   ───Γ[N]       ───╯
    #
    #
    #  ╭───Λ[k-1]───Γ[k]───                       ╭───
    #  │             │                            │
    #  │             │        =  tr(Λ[k-1]²)  ×   │
    #  │             │                            │
    #  ╰───Λ[k-1]───Γ[k]───                       ╰───
    #
    #
    #   ───Γ[k]───Λ[k]───╮                     ───╮
    #       │            │                        │
    #       │            │   =  tr(Λ[k]²)  ×      │
    #       │            │                        │
    #   ───Γ[k]───Λ[k]───╯                     ───╯
    #
    #
    # This means that we don't actually need to compute the full contraction. For example,
    # if A is a single-site operator acting on site j:
    #
    #     ○──◇──○──◇──○─╶╶   ╶╶─○──◇──○──◇──○─╶╶   ╶╶─○──◇──○──◇──○
    #     │     │     │         │     │     │         │     │     │
    #     │     │     │         │     □     │         │     │     │  =
    #     │     │     │         │     │     │         │     │     │
    #     ○──◇──○──◇──○─╶╶   ╶╶─○──◇──○──◇──○─╶╶   ╶╶─○──◇──○──◇──○
    #     1     2     3        j-1    j    j+1       N-2   N-1    N
    #
    #
    #     ╭──◇──○──◇──○─╶╶   ╶╶─○──◇──○──◇──○─╶╶   ╶╶─○──◇──○──◇──╮
    #     │     │     │         │     │     │         │     │     │
    #  =  │     │     │         │     □     │         │     │     │  =
    #     │     │     │         │     │     │         │     │     │
    #     ╰──◇──○──◇──○─╶╶   ╶╶─○──◇──○──◇──○─╶╶   ╶╶─○──◇──○──◇──╯
    #           2     3        j-1    j    j+1       N-2   N-1
    #
    #
    #           ╭──◇──○─╶╶   ╶╶─○──◇──○──◇──○─╶╶   ╶╶─○──◇──╮
    #           │     │         │     │     │         │     │
    #  =        │     │         │     □     │         │     │  ×  tr(Λ[1]²) tr(Λ[N-1]²)  =
    #           │     │         │     │     │         │     │
    #           ╰──◇──○─╶╶   ╶╶─○──◇──○──◇──○─╶╶   ╶╶─○──◇──╯
    #                 3        j-1    j    j+1       N-2
    #
    #
    #                           ╭──◇──○──◇──╮
    #                           │     │     │
    #  =                        │     □     │  ×  tr(Λ[1]²) ⋯ tr(Λ[j-2]²)
    #                           │     │     │            tr(Λ[j+1]²) ⋯ tr(Λ[N-1]²).
    #                           ╰──◇──○──◇──╯
    #                                 j

    active_sites = findsites(ψ, A)
    site_range = first(active_sites):last(active_sites)
    Mⱼ =
        bond_tensors(ψ)[first(site_range) - 1] *
        prod(site_tensors(ψ)[j] * bond_tensors(ψ)[j] for j in site_range)
    contr_pre = prod(
        scalar(Λ*Λ) for Λ in bond_tensors(ψ)[1:(first(site_range) - 2)]; init=1.0
    )
    contr_post = prod(
        scalar(Λ*Λ) for Λ in bond_tensors(ψ)[(last(site_range) + 1):end]; init=1.0
    )
    return contr_pre * contr_post * inner(Mⱼ, apply(A, Mⱼ))
end

function simplified_self_contraction(ψ::InverseCanonicalMPS, A::ITensor)
    # Compute ⟨ψ, Aψ⟩ by contracting only the site tensors on which A acts.  If the opposite
    # tensors are directly contracted with each other, the following cancellation rules
    # hold:
    #
    #  ╭───C[j]───V[j]───      ╭───Λ[j-1]───Γ[j]───                       ╭───
    #  │    │                  │             │                            │
    #  │    │               =  │             │        =  tr(Λ[j-1]²)  ×   │
    #  │    │                  │             │                            │
    #  ╰───C[j]───V[j]───      ╰───Λ[j-1]───Γ[j]───                       ╰───

    #   C[1]───V[1]───       Γ[j]───       ╭───
    #    │                    │            │
    #    │               =    │        =   │
    #    │                    │            │
    #   C[1]───V[1]───       Γ[j]───       ╰───
    #
    #   ───V[j-1]───C[j]───╮       ───Γ[j]───Λ[j]───╮                     ───╮
    #                │     │           │            │                        │
    #                │     │   =       │            │   =  tr(Λ[j]²)  ×      │
    #                │     │           │            │                        │
    #   ───V[j-1]───C[j]───╯       ───Γ[j]───Λ[j]───╯                     ───╯

    #   ───V[N-1]───C[N]        ───Γ[N]       ───╮
    #                │              │            │
    #                │     =        │     =      │
    #                │              │            │
    #   ───V[N-1]───C[N]        ───Γ[N]       ───╯
    #
    # This means that we don't actually need to compute the full contraction. For example,
    # if A is a single-site operator acting on site j:
    #
    #     ▧──◆──▧──◆──▧─╶╶   ╶╶─▧──◆──▧──◆──▧─╶╶   ╶╶─▧──◆──▧──◆──▧
    #     │     │     │         │     │     │         │     │     │
    #     │     │     │         │     □     │         │     │     │  =
    #     │     │     │         │     │     │         │     │     │
    #     ▧──◆──▧──◆──▧─╶╶   ╶╶─▧──◆──▧──◆──▧─╶╶   ╶╶─▧──◆──▧──◆──▧
    #     1     2     3        j-1    j    j+1       N-2   N-1    N
    #
    #
    #        ╭──▧──◆──▧─╶╶   ╶╶─▧──◆──▧──◆──▧─╶╶   ╶╶─▧──◆──▧──╮
    #        │  │     │         │     │     │         │     │  │
    #  =     │  │     │         │     □     │         │     │  │  =
    #        │  │     │         │     │     │         │     │  │
    #        ╰──▧──◆──▧─╶╶   ╶╶─▧──◆──▧──◆──▧─╶╶   ╶╶─▧──◆──▧──╯
    #           2     3        j-1    j    j+1       N-2   N-1
    #
    #
    #              ╭──▧─╶╶   ╶╶─▧──◆──▧──◆──▧─╶╶   ╶╶─▧──╮
    #              │  │         │     │     │         │  │
    #  =           │  │         │     □     │         │  │  ×  tr(Λ[1]²) tr(Λ[N-1]²)  =
    #              │  │         │     │     │         │  │
    #              ╰──▧─╶╶   ╶╶─▧──◆──▧──◆──▧─╶╶   ╶╶─▧──╯
    #                 3        j-1    j    j+1       N-2
    #
    #
    #                              ╭──▧──╮
    #                              │  │  │
    #  =                           │  □  │  ×  tr(Λ[1]²) ⋯ tr(Λ[j-2]²)
    #                              │  │  │            tr(Λ[j+1]²) ⋯ tr(Λ[N-1]²).
    #                              ╰──▧──╯
    #                                 j

    active_sites = findsites(ψ, A)
    site_range = first(active_sites):last(active_sites)
    Mⱼ =
        prod(
            site_tensors(ψ)[j] * bond_tensors(ψ)[j] for j in site_range[1:(end - 1)];
            init=ITensors.OneITensor(),
        ) * site_tensors(ψ)[last(site_range)]
    contr_pre = prod(
        scalar(inv.(Λ)*inv.(Λ)) for Λ in bond_tensors(ψ)[1:(first(site_range) - 2)];
        init=1.0,
    )
    contr_post = prod(
        scalar(inv.(Λ)*inv.(Λ)) for Λ in bond_tensors(ψ)[(last(site_range) + 1):end];
        init=1.0,
    )
    return contr_pre * contr_post * inner(Mⱼ, apply(A, Mⱼ))
end

"""
    expect(ψ::Union{VidalMPS,InverseCanonicalMPS}, op::AbstractString...; kwargs...)
    expect(ψ::Union{VidalMPS,InverseCanonicalMPS}, op::Matrix{<:Number}...; kwargs...)
    expect(ψ::Union{VidalMPS,InverseCanonicalMPS}, op::LocalOperator...; kwargs...)
    expect(ψ::Union{VidalMPS,InverseCanonicalMPS}, ops; kwargs...)

Given a canonical or inverse-canonical MPS `ψ` and a single operator name, returns a vector
of the expected value of the operator on each site of the MPS.

See [ITensorMPS.expect](@ref) for more details.
"""
function ITensorMPS.expect(ψ::ExplicitBondMPS, ops; sites=1:nsites(ψ))
    ψ = copy(ψ)
    N = nsites(ψ)
    ElT = NDTensors.scalartype(ψ)
    s = siteinds(ψ)

    site_range = (sites isa AbstractRange) ? sites : collect(sites)
    Ns = length(site_range)
    start_site = first(site_range)

    el_types = map(o -> ishermitian(op(o, s[start_site])) ? real(ElT) : ElT, ops)

    norm2_ψ = norm(ψ)^2
    iszero(norm2_ψ) && error("MPS has zero norm in function `expect`")

    ex = map((o, el_t) -> zeros(el_t, Ns), ops, el_types)
    for (entry, j) in enumerate(site_range)
        for (n, opname) in enumerate(ops)
            oⱼ = Adapt.adapt(
                unspecify_type_parameters(NDTensors.datatype(site_tensors(ψ)[j])),
                op(opname, s[j]),
            )
            # From the docs: "The `adapt(T, x)` function acts like `convert(T, x)`, but
            # without the restriction of returning a `T`." From the code of the Adapt
            # package, if `T` is `Array` and `xs` is an `AbstractArray`, the call boils down
            # to `convert(Array, xs)`, so basically it converts `xs` to a concrete `Array`.
            # (Not really sure what is the purpose here...)
            #
            # `unspecify_type_parameters` is a function from the TypeParameterAccessors
            # package that removes the parameters of a type (a type parameter is the thing
            # inside braces after a type, for example `Int` in `Vector{Int}`). Examples:
            # `Array{ITensors}` becomes `Array`, and `Vector{Real}` becomes `Vector`.

            val = simplified_self_contraction(ψ, oⱼ) / norm2_ψ
            ex[n][entry] = (el_types[n] <: Real) ? real(val) : val
        end
    end

    if sites isa Number
        return map(arr -> only(arr), ex)
    end
    return ex
end

function ITensorMPS.expect(ψ::ExplicitBondMPS, op::AbstractString; kwargs...)
    return first(expect(ψ, (op,); kwargs...))
end

function ITensorMPS.expect(ψ::ExplicitBondMPS, op::Matrix{<:Number}; kwargs...)
    return first(expect(ψ, (op,); kwargs...))
end

function ITensorMPS.expect(
    ψ::ExplicitBondMPS, op1::AbstractString, ops::AbstractString...; kwargs...
)
    return expect(ψ, (op1, ops...); kwargs...)
end

function ITensorMPS.expect(
    ψ::ExplicitBondMPS, op1::Matrix{<:Number}, ops::Matrix{<:Number}...; kwargs...
)
    return expect(ψ, (op1, ops...); kwargs...)
end
