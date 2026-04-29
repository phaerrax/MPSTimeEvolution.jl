### MPS <-> VidalMPS conversion functions

# VidalMPS to MPS: merge the bond indices into the site indices.
# For now, we always output a left-canonical MPS.
function Base.convert(::Type{MPS}, ψ::VidalMPS)
    M = Vector{ITensor}(undef, nsites(ψ))
    M[1] = site_tensors(ψ)[1]
    for n in 2:length(M)
        M[n] = bond_tensors(ψ)[n - 1] * site_tensors(ψ)[n]
    end
    return MPS(M; ortho_lims=1:1)
end

# MPS to VidalMPS: orthogonalise the MPS first, then use the SVD to separate the bond
# tensors from the site tensors until we reach the opposite edge of the MPS.
function Base.convert(::Type{VidalMPS}, ψ::MPS; cutoff=1e-8)
    # We follow Schollwöck's approach from his 2011 review article ("Conversion A, B → Γ Λ"
    # starting at page 138).

    # Start from a right-normalised MPS.
    ψ = deepcopy(ψ)
    orthogonalize!(ψ, 1)
    # (We need to deepcopy the MPS otherwise the calls to `replacetags!` below wend up
    # modifying the original ψ. I hoped that calling the non-modifying version of
    # `orthogonalize` would be enough to create a copy of the original MPS, but apparently
    # it is not enough.)

    # Replace the "l=$n" names in the link indices of the original MPS with something else,
    # in order to avoid an overlap with the "l=$n" tags we want to use for the final MPS.
    for i in eachindex(ψ)
        replacetags!(ψ[i], "l=$i" => "orig=$i")
        replacetags!(ψ[i], "l=$(i-1)" => "orig=$(i-1)")
    end

    # Create the arrays that will hold the VidalMPS tensors.
    N = length(ψ)
    site_ts = Vector{ITensor}(undef, N)
    bond_ts = Vector{ITensor}(undef, N-1)

    # Decompose the first MPS tensor.
    A, bond_ts[1], V = svd(
        ψ[1],
        uniqueinds(ψ[1], ψ[2]);
        lefttags="Link,r=1",
        righttags="Link,l=2",
        use_absolute_cutoff=true,
        cutoff=cutoff,
    )

    # From Eq. (159): A⁽ⁿ⁾ = Λ⁽ⁿ⁻¹⁾ Γ⁽ⁿ⁾, with A⁽⁰⁾ = 1.
    site_ts[1] = A

    for n in 2:(N - 1)
        # Repeat the procedure until we reach the opposite edge.
        M = bond_ts[n - 1] * V * ψ[n]

        A, bond_ts[n], V = svd(
            M,
            uniqueinds(M, ψ[n + 1]);
            lefttags="Link,r=$n",
            righttags="Link,l=$(n+1)",
            use_absolute_cutoff=true,
            cutoff=cutoff,
        )

        #    ┌────┐         ┌──────┐      ┌────┐
        # ───│A⁽ⁿ⁾│─── = ───│Λ⁽ⁿ⁻¹⁾│──────│Γ⁽ⁿ⁾│────
        #    └────┘         └──────┘      └────┘
        #      │                            │
        #      │                            │

        site_ts[n] = inv.(bond_ts[n - 1]) * A
    end

    # M = bond_ts[N-1] * V * ψ[N]
    # site_ts[N] = inv.(bond_ts[N-1]) * M = inv.(bond_ts[N-1]) * bond_ts[N-1] * V * ψ[N]
    site_ts[N] = V * ψ[N]

    return VidalMPS(site_ts, bond_ts)
end
