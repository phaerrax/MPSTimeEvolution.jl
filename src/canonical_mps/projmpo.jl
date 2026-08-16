using ITensorMPS: AbstractProjMPO

function ITensorMPS._makeL!(
    P::AbstractProjMPO, psi::InverseCanonicalMPS, k::Int
)::Union{ITensor,Nothing}
    # Construct the left environment for `P` based on the MPS `psi`:
    #
    #   ▒───       ▧───◆───▧─╶╶  ╶╶╶─▧───◆───▧───◆───▧───◆───
    #   ▒          │       │         │       │       │    
    #   ▒───   =   o───────o─╶╶  ╶╶╶─o───────o───────o───────
    #   ▒          │       │         │       │       │    
    #   ▒───       ▧───◆───▧─╶╶  ╶╶╶─▧───◆───▧───◆───▧───◆───
    #    Lⱼ        1       2        j-2     j-1      j

    # Save the last `L` that is made to help with caching for DiskProjMPO
    ll = P.lpos

    if ll ≥ k
        # Special case when nothing has to be done.
        # Still need to change the position if lproj is being moved backward.
        P.lpos = k
        return nothing
    end

    # Make sure ll is at least 0 for the generic logic below.
    ll = max(ll, 0)

    L = lproj(P)
    # L is the array of cached left environments: if ll < k, it means that we are moving to
    # the right, and we need to update the array of left environments with the new ones from
    # ll+1 to k:
    #
    #   ▒───       ▒────▧───◆───
    #   ▒          ▒    │       
    #   ▒───   =   ▒────o───────
    #   ▒          ▒    │       
    #   ▒───       ▒────▧───◆───
    #   Lⱼ₊₁       Lⱼ  j+1

    while ll < k
        L =
            L *
            psi.site_tensors[ll + 1] *
            psi.bond_tensors[ll + 1] *
            P.H[ll + 1] *
            dag(prime(psi.site_tensors[ll + 1] * psi.bond_tensors[ll + 1]))
        P.LR[ll + 1] = L
        ll += 1
    end

    # Needed when moving lproj backward.
    P.lpos = k
    return L
end

function ITensorMPS.makeL!(P::AbstractProjMPO, psi::InverseCanonicalMPS, k::Int)
    ITensorMPS._makeL!(P, psi, k)
    return P
end

function ITensorMPS._makeR!(
    P::AbstractProjMPO, psi::InverseCanonicalMPS, k::Int
)::Union{ITensor,Nothing}
    # Construct the right environment for `P` based on the MPS `psi`:
    #
    #   ───▒        ───◆───▧───◆───▧───◆───▧─╶╶  ╶╶╶─▧───◆───▧
    #      ▒               │       │       │         │       │
    #   ───▒    =   ───────o───────o───────o─╶╶  ╶╶╶─o───────o
    #      ▒               │       │       │         │       │
    #   ───▒        ───◆───▧───◆───▧───◆───▧─╶╶  ╶╶╶─▧───◆───▧
    #    Rⱼ                j      j+1     j+2       N-1      N

    # Save the last `R` that is made to help with caching for DiskProjMPO.
    rl = P.rpos

    if rl ≤ k
        # Special case when nothing has to be done.
        # Still need to change the position if rproj is being moved backward.
        P.rpos = k
        return nothing
    end

    N = length(P.H)
    # Make sure rl is no bigger than `N + 1` for the generic logic below.
    rl = min(rl, N + 1)

    R = rproj(P)
    # L is the array of cached right environments: if rl > k, it means that we are moving to
    # the left, and we need to update the environments with the new ones from rl-1 to k:
    #
    #   ───▒        ───◆───▧────▒
    #      ▒               │    ▒
    #   ───▒    =   ───────o────▒
    #      ▒               │    ▒
    #   ───▒        ───◆───▧────▒
    #   Rⱼ₋₁              j-1   Rⱼ

    while rl > k
        R =
            R *
            psi.bond_tensors[rl - 2] *
            psi.site_tensors[rl - 1] *
            P.H[rl - 1] *
            dag(prime(psi.bond_tensors[rl - 2] * psi.site_tensors[rl - 1]))
        P.LR[rl - 1] = R
        rl -= 1
    end

    P.rpos = k
    return R
end

function ITensorMPS.makeR!(P::AbstractProjMPO, psi::InverseCanonicalMPS, k::Int)
    ITensorMPS._makeR!(P, psi, k)
    return P
end

"""
    position!(P::ProjMPO, psi::InverseCanonicalMPS, pos::Int)

Given an inverse-canonical MPS `psi`, shift the projection of the MPO represented by the
ProjMPO `P` such that the set of unprojected sites begins with site `pos`.
This operation efficiently reuses previous projections of the MPO on sites that have already
been projected.
The InverseCanonicalMPS `psi` must have compatible bond indices with the previous projected
MPO tensors for this operation to succeed.
"""
function ITensorMPS.position!(P::AbstractProjMPO, psi::InverseCanonicalMPS, pos::Int)
    # makeL!, _makeL!, makeR! and _makeR! are not explicitly exported by ITensorMPS, so even
    # if we are defining new methods for them in this very file we still need to prefix them
    # with `ITensorMPS.`. (Either this or we import the names at the beginning.)
    ITensorMPS.makeL!(P, psi, pos - 1)
    ITensorMPS.makeR!(P, psi, pos + nsite(P))
    return P
end
