# VidalMPS to MPS: merge the bond indices into the site indices.
function Base.convert(::Type{MPS}, ψ::VidalMPS; ortho_center=1)
    N = nsites(ψ)
    M = Vector{ITensor}(undef, N)

    for n in 1:(ortho_center - 1)
        # To the left of the orthocenter: M[n] = Λ[n-1] Γ[n]
        M[n] = bond_tensors(ψ)[n - 1] * site_tensors(ψ)[n]
    end

    # Orthocenter site: M[n] = Λ[n-1] Γ[n] Λ[n]
    M[ortho_center] =
        bond_tensors(ψ)[ortho_center - 1] *
        site_tensors(ψ)[ortho_center] *
        bond_tensors(ψ)[ortho_center]

    for n in (ortho_center + 1):N
        # To the right of the orthocenter: M[n] = Γ[n] Λ[n]
        M[n] = site_tensors(ψ)[n] * bond_tensors(ψ)[n]
    end

    # Restore the standard tags structure of an MPS created by ITensor. The tags resulting
    # from the contractions above might be different from the standard, and this might
    # create some problems for example with the `replaceinds!` function we use in
    # `convert(VidalMPS, ...)` below.
    old_link_inds = [commonind(M[n], M[n + 1]) for n in 1:(N - 1)]
    new_link_inds = [Index(dim(old_link_inds[n]); tags="Link,l=$n") for n in 1:(N - 1)]
    for n in 1:(N - 1)
        M[n] *= delta(old_link_inds[n], new_link_inds[n])
    end
    for n in 2:N
        M[n] *= delta(old_link_inds[n - 1], new_link_inds[n - 1])
    end
    return MPS(M; ortho_lims=ortho_center:ortho_center)
end

# MPS to VidalMPS: orthogonalise the MPS first, then use the SVD to separate the bond
# tensors from the site tensors until we reach the opposite edge of the MPS.
function Base.convert(::Type{VidalMPS}, ψ::MPS; kwargs...)
    # We follow Schollwöck's approach from his 2011 review article ("Conversion A, B → Γ Λ"
    # starting at page 138).

    # Start from a right-normalised MPS.
    ψ = deepcopy(ψ)
    orthogonalize!(ψ, 1)
    # (We need to deepcopy the MPS otherwise the calls to `replacetags!` below wend up
    # modifying the original ψ. I hoped that calling the non-modifying version of
    # `orthogonalize` would be enough to create a copy of the original MPS, but apparently
    # it is not enough.)

    # I don't really know why, but the MPS needs to be normalised for the conversion
    # function to produce a correctly canonicalised MPS. Probably it's because once we
    # extract the bond tensor Λₖ, we also reincorporate it into the site tensor to its right
    # before we continue the SVD on the next bond, i.e. with
    #   M = bond_ts[n - 1] * V * ψ[n]
    # so the norm gets propagated on all bonds.
    # This (silently) creates all sorts of errors, and most importantly breaks the gauge.
    # For example, we get
    #   norm(2v) !≈ norm(convert(VidalMPS, 2v))
    # while, correctly,
    #   norm(2v) ≈ norm(2 * convert(VidalMPS, v))
    # although quite surprisingly we still have
    #   2 * convert(VidalMPS, v) ≈ convert(VidalMPS, 2v)
    # which likely signals that the problem lies in the broken canonical gauge, whose
    # properties are used by the norm function.

    # TODO Maybe we need to rethink the conversion routine? This was taken from Schollwöck,
    # but he assumes the MPS represents a pure state so he doesn't have to worry about a
    # norm which is not one.
    # Anyway, we extract the norm and normalise the MPS, then we'll reintegrate the norm at
    # the end of the procedure with our `scalarmult` function.
    norm_ψ = norm(ψ)
    normalize!(ψ)

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
        ψ[1], uniqueinds(ψ[1], ψ[2]); lefttags="Link,r=1", righttags="Link,l=2", kwargs...
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
            kwargs...,
        )

        # ───A[n]─── = ───Λ[n-1]──────Γ[n]────
        #     │                        │
        #     │                        │

        site_ts[n] = inv.(bond_ts[n - 1]) * A
    end

    # M = bond_ts[N-1] * V * ψ[N]
    # site_ts[N] = inv.(bond_ts[N-1]) * M = inv.(bond_ts[N-1]) * bond_ts[N-1] * V * ψ[N]
    site_ts[N] = V * ψ[N]

    # Add the trivial bond tensors at the edges of the MPS, put the norm back into the MPS,
    # and return.
    return norm_ψ *
           VidalMPS(site_ts, OffsetVector([ITensor(1.0); bond_ts; ITensor(1.0)], 0:N))
end

# InverseCanonicalMPS to MPS: contract Cₖ and Vₖ to get Aₖ, and contract Vₖ and Cₖ₊₁ to
# obtain Bₖ. The remaining site tensor is the orthocentre.
function Base.convert(::Type{MPS}, ψ::InverseCanonicalMPS; ortho_center=1)
    N = nsites(ψ)
    M = Vector{ITensor}(undef, N)

    for n in 1:(ortho_center - 1)
        # To the left of the orthocenter: M[n] = C[n] V[n]
        M[n] = site_tensors(ψ)[n] * bond_tensors(ψ)[n]
    end

    # Orthocenter site: M[n] = C[n]
    M[ortho_center] = site_tensors(ψ)[ortho_center]

    for n in (ortho_center + 1):N
        # To the right of the orthocenter: M[n] = V[n-1] C[n]
        M[n] = bond_tensors(ψ)[n - 1] * site_tensors(ψ)[n]
    end

    # Restore the standard tags structure of an MPS created by ITensor. The tags resulting
    # from the contractions above might be different from the standard, and this might
    # create some problems for example with the `replaceinds!` function we use in
    # `convert(InverseCanonicalMPS, ...)` below.
    old_link_inds = [commonind(M[n], M[n + 1]) for n in 1:(N - 1)]
    new_link_inds = [Index(dim(old_link_inds[n]); tags="Link,l=$n") for n in 1:(N - 1)]
    for n in 1:(N - 1)
        M[n] *= delta(old_link_inds[n], new_link_inds[n])
    end
    for n in 2:N
        M[n] *= delta(old_link_inds[n - 1], new_link_inds[n - 1])
    end
    return MPS(M; ortho_lims=ortho_center:ortho_center)
end

# MPS to InverseCanonicalMPS: orthogonalise the MPS first, then use the SVD to separate the
# bond tensors from the site tensors until we reach the opposite edge of the MPS.
function Base.convert(::Type{InverseCanonicalMPS}, ψ::MPS; kwargs...)
    # Cheap way of doing the conversion: first convert ψ to a VidalMPS, which takes care of
    # performing the SVDs correctly. Then multiply/divide by the bond tensors accordingly.
    return convert(InverseCanonicalMPS, convert(VidalMPS, ψ; kwargs...))
end

function Base.convert(::Type{InverseCanonicalMPS}, ψ::VidalMPS)
    N = nsites(ψ)
    Γ = site_tensors(ψ)
    Λ = bond_tensors(ψ)

    # VidalMPS to InverseCanonicalMPS: Vₖ = Λₖ⁻¹, and
    #   Cₖ = Λₖ Γₖ Λₖ,
    #   C₁ = Γ₁ Λ₁,
    #   Cₙ = Λₙ₋₁Γₙ.
    #
    #        │       │       │         │       │       │
    #    ◇···▧───◆───▧───◆───▧─╶╶  ╶╶╶─▧───◆───▧───◆───▧···◇
    #       C[1]    C[2]    C[3]      C[N-2]  C[N-1]  C[N]
    #   V[0]    V[1]    V[2]             V[N-2]  V[N-1]   V[N]
    #
    # (With the trivial bond tensors Λ₀ = Λₙ = 1, the first equation holds for all k.)

    ic_site_ts = [Λ[n - 1] * Γ[n] * Λ[n] for n in 1:N]
    ic_bond_ts = [inv.(Λⱼ) for Λⱼ in Λ]  # This includes the trivial bond tensors.

    # However, just using the formulas above doesn't work, because the link indices become
    # messed up:
    #
    #   ((dim=2|id=#|"S=1/2,Site,n=1"), (dim=2|id=#|"Link,l=2"))
    #   ((dim=2|id=#|"Link,r=1"), (dim=2|id=#|"Link,l=2"))
    #   ((dim=2|id=#|"Link,r=1"), (dim=2|id=#|"S=1/2,Site,n=2"), (dim=2|id=#|"Link,l=3"))
    #   ((dim=2|id=#|"Link,r=2"), (dim=2|id=#|"Link,l=3"))
    #
    # while for example the first site tensor should have a link index with tags "r=1".
    # This breaks the index structure so that functions such as `apply` do not work as
    # expected.
    # (The order in the bond tensors doesn't really matter since they are symmetric.)
    # We have to restore the standard tags structure of an InverseCanonicalMPS.

    old_rlink_inds = [commonind(ic_site_ts[n], ic_bond_ts[n]) for n in 1:(N - 1)]
    old_llink_inds = [commonind(ic_bond_ts[n], ic_site_ts[n + 1]) for n in 1:(N - 1)]

    new_rlink_inds = [Index(dim(old_rlink_inds[n]); tags="Link,r=$n") for n in 1:(N - 1)]
    new_llink_inds = [
        Index(dim(old_llink_inds[n]); tags="Link,l=$(n+1)") for n in 1:(N - 1)
    ]

    for n in 1:(N - 1)
        ic_site_ts[n] *= delta(old_rlink_inds[n], new_rlink_inds[n])
    end

    for n in 1:(N - 1)
        # Beware: we need to do these index manipulations without changing the type of
        # the bond tensors from NDTensors.Diag to NDTensors.Dense, otherwise when we call
        # `inv.` on the bond tensors we get `Inf`s everywhere.
        # The `delta` tensor is NDTensors.Diag, but the product of two delta tensors isn't,
        # thus we apparently need to perform the multiplication in two steps.
        # Unfortunately, contracting a diagonal tensor with a delta tensor returns an error
        # message saying "Not implemented", which I guess tells us that this kind of
        # operation is not possible in ITensor (yet?).
        # So, as a workaround, we do create a non-diagonal tensor first, by multiplying the
        # deltas together and then with the bond tensor; then, we turn it into a diagonal
        # tensor with the syntax `diag_itensor(vector(diag(t)), inds(t)...)`
        Vₙ =
            ic_bond_ts[n] * (
                delta(old_rlink_inds[n], new_rlink_inds[n]) *
                delta(old_llink_inds[n], new_llink_inds[n])
            )
        ic_bond_ts[n] = diag_itensor(vector(diag(Vₙ)), inds(Vₙ)...)
    end

    for n in 2:N
        ic_site_ts[n] *= delta(old_llink_inds[n - 1], new_llink_inds[n - 1])
    end

    return InverseCanonicalMPS(ic_site_ts, ic_bond_ts)
end

# InverseCanonicalMPS to VidalMPS: Λₖ = Vₖ⁻¹, and
#   Cₖ = Λₖ Γₖ Λₖ,
#   C₁ = Γ₁ Λ₁,
#   Cₙ = Λₙ₋₁ Γₙ.
function Base.convert(::Type{VidalMPS}, ψ::InverseCanonicalMPS)
    N = nsites(ψ)
    C = site_tensors(ψ)
    V = bond_tensors(ψ)

    c_site_ts = [
        C[1] * V[1];
        [V[n - 1] * C[n] * V[n] for n in 2:(N - 1)];
        V[n - 1] * C[n]
    ]
    c_bond_ts = [inv.(Vⱼ) for Vⱼ in V]

    return VidalMPS(c_site_ts, c_bond_ts)
end
