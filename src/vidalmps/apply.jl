function _replace_and_decompose!(ψ::VidalMPS, M::ITensor; kwargs...)
    # Replace the sites of the Vidal MPS `ψ` with the tensor `A`, splitting up `A` into MPS
    # tensors. THE TENSORS TO BE REPLACED ARE AUTOMATICALLY DETERMINED BY A'S SITE INDICES.
    # THIS MEANS THAT THIS FUNCTION CANNOT ACT ON THE BOND TENSORS ON THE LEFT AND ON THE
    # RIGHT OF A. KEEP THIS IN MIND WHEN USING THIS FUNCTION WHEN APPLYING MULTI-SITE
    # OPERATORS.

    # The ITensor A will have some site indices. We need to factor it into an MPS-like form,
    # with a different site block for each site index.

    # 1. Get the site indices involved in the decomposition, and sort them by increasing
    #    site number.
    ns = findsites(ψ, M)
    sort!(ns)
    N = length(ns)

    # 2. Gather the indices relative to the leftmost site of A.
    #    If A is not on the left edge of the MPS, we also include the link index which is
    #    dangling on the left.

    # 3. Recursively decompose A with an SVD, until we exhaust the site indices.
    site_ts = site_tensors(ψ)
    bond_ts = bond_tensors(ψ)

    linds = if ns[1] > 1
        commoninds(M, bond_ts[ns[1] - 1] * site_ts[ns[1]])
    else
        commoninds(M, site_ts[ns[1]])
    end
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
    apply(o::ITensor, ψ::VidalMPS, [ns::Vector{Int}]; kwargs...)
    product([...])

Get the product of the operator `o` with the VidalMPS `ψ`.

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
    # For now, let's throw an error saying it's not implemented
    diff_ns = diff(ns)
    ns′ = ns
    if any(!=(1), diff_ns)
        error("apply not (yet) implemented for non-consecutive application sites")
        # ns′ = [ns[1] + n - 1 for n in 1:N]
        # ψ = movesites(ψ, ns .=> ns′; kwargs...)
    end

    # Multiply everything in the VidalMPS from ns′[1] to ns′[end] together, and include the
    # bond tensors to the left of ns′[1] and to the right ns′[end].
    ϕ = site_tensors(ψ)[ns′[1]]
    if ns′[1] > 1
        Λₗ = bond_tensors(ψ)[ns′[1] - 1]
        ϕ *= Λₗ
    end
    for n in 2:N
        ϕ *= site_tensors(ψ)[ns′[n]] * bond_tensors(ψ)[ns′[n] - 1]
    end
    if ns′[N] < length(site_tensors(ψ))
        Λᵣ = bond_tensors(ψ)[ns′[N]]
        ϕ *= Λᵣ
    end

    # Apply the operator to the combined site and bond tensors.
    ϕ = ITensors.product(o, ϕ)

    # Insert the result in ψ, decomposing it into site and bond tensors.
    if length(ns) > 1
        _replace_and_decompose!(ψ, ϕ; kwargs...)
    else
        site_tensors(ψ)[only(ns)] = ϕ
    end

    # Restore the Vidal form by re-inserting the bond tensors on the left and on the right.
    # Note that the bond tensors at ns′[1] and ns′[N] from the input VidalMPS were not
    # modified, so we don't need to "reinsert" them. We just need to multiply the first and
    # last of the new site tensors.
    if ns′[1] > 1
        site_tensors(ψ)[ns′[1]] *= inv.(Λₗ)
    end
    if ns′[N] < length(site_tensors(ψ))
        site_tensors(ψ)[ns′[N]] *= inv.(Λᵣ)
    end

    return ψ
end

"""
    expect(ψ::VidalMPS, op::AbstractString...; kwargs...)
    expect(ψ::VidalMPS, op::Matrix{<:Number}...; kwargs...)
    expect(ψ::VidalMPS, ops; kwargs...)

Given a `VidalMPS` `ψ` and a single operator name, returns a vector of the expected value of
the operator on each site of the MPS.

See [ITensorMPS.expect](@ref) for more details.
"""
function ITensorMPS.expect(ψ::VidalMPS, ops; sites=1:nsites(ψ))
    ψ = copy(ψ)
    N = nsites(ψ)
    ElT = NDTensors.scalartype(ψ)
    s = siteinds(ψ)

    site_range = (sites isa AbstractRange) ? sites : collect(sites)
    Ns = length(site_range)
    start_site = first(site_range)

    el_types = map(o -> ishermitian(op(o, s[start_site])) ? real(ElT) : ElT, ops)

    norm2_ψ = norm(ψ)^2
    iszero(norm2_ψ) && error("VidalMPS has zero norm in function `expect`")

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

            val = inner(ψ, apply(oⱼ, ψ)) / norm2_ψ
            # Here ITensorMPS can optimise the calculation, because at this point of the
            # function the MPS is orthogonalised on site `j`.  They can call the `inner`
            # method on the ITensor `ψ[j]` only instead of the whole MPS, i.e.
            #   inner(ψ[j], apply(oⱼ, ψ[j])) / norm2_ψ
            # that ignores the other site tensors completely (they cancel out due to their
            # left- or right-canonicity).
            # Here we do not have (yet!) such cancellation rules, so we must contract the
            # full MPS, which is of course wasteful.
            ex[n][entry] = (el_types[n] <: Real) ? real(val) : val
        end
    end

    if sites isa Number
        return map(arr -> only(arr), ex)
    end
    return ex
end

function ITensorMPS.expect(ψ::VidalMPS, op::AbstractString; kwargs...)
    return first(expect(ψ, (op,); kwargs...))
end

function ITensorMPS.expect(ψ::VidalMPS, op::Matrix{<:Number}; kwargs...)
    return first(expect(ψ, (op,); kwargs...))
end

function ITensorMPS.expect(
    ψ::VidalMPS, op1::AbstractString, ops::AbstractString...; kwargs...
)
    return expect(ψ, (op1, ops...); kwargs...)
end

function ITensorMPS.expect(
    ψ::VidalMPS, op1::Matrix{<:Number}, ops::Matrix{<:Number}...; kwargs...
)
    return expect(ψ, (op1, ops...); kwargs...)
end
