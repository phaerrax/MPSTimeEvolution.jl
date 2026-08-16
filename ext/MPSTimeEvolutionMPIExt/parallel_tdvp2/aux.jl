# Parallel TDVP2 sweeps
# =====================

using ITensorMPS: position!, AbstractProjMPO, set_nsite!
using MPSTimeEvolution: nsites, site_tensors, bond_tensors, TDVP2
using KrylovKit: exponentiate

# We can't use the full MPS ψ here, because ψ is being modified by other processes while
# this function is running. We need to break up the MPS in chunks and make each process
# unaware of the chunks which aren't assigned to it.
# We will need to `position!` an ProjMPO `PH` in order to move its “gap” on the correct
# site of the MPS. The problem is, we don't have the MPS, so we cannot use this method: we
# must move the projections by hand.  Calling
#
#   position!(PH, ψ, bond)
#
# means calling (we are interested now in the two-site projection):
#
#   makeL!(PH, ψ, bond - 1)
#   makeR!(PH, ψ, bond + 2)
#
# We rewrite these functions so that instead of ψ they accept the vectors of site tensors
# and bond tensors separately.

function ITensorMPS.makeL!(
    P::AbstractProjMPO, site_ts, bond_ts, k::Int
)::Union{ITensor,Nothing}
    ll = P.lpos
    if ll ≥ k
        P.lpos = k
        return nothing
    end
    ll = max(ll, 0)

    L = lproj(P)
    while ll < k
        L =
            L *
            site_ts[ll + 1] *
            bond_ts[ll + 1] *
            P.H[ll + 1] *
            dag(prime(site_ts[ll + 1] * bond_ts[ll + 1]))
        P.LR[ll + 1] = L
        ll += 1
    end

    P.lpos = k
    return L
end

function ITensorMPS.makeR!(
    P::AbstractProjMPO, site_ts, bond_ts, k::Int
)::Union{ITensor,Nothing}
    rl = P.rpos
    if rl ≤ k
        P.rpos = k
        return nothing
    end
    rl = min(rl, length(P.H) + 1)

    R = rproj(P)
    while rl > k
        #   ───▒             ───◆───▧────▒
        #      ▒                    │    ▒
        #   ───▒        =    ───────o────▒
        #      ▒                    │    ▒
        #   ───▒             ───◆───▧────▒
        #  PH.LR[r-1]        Vᵣ₋₂ Cᵣ₋₁ PH.LR[r]
        R =
            R *
            bond_ts[rl - 2] *
            site_ts[rl - 1] *
            P.H[rl - 1] *
            dag(prime(bond_ts[rl - 2] * site_ts[rl - 1]))
        P.LR[rl - 1] = R
        rl -= 1
    end

    P.rpos = k
    return R
end

function ITensorMPS.position!(P::AbstractProjMPO, site_ts, bond_ts, pos::Int)
    ITensorMPS.makeL!(P, site_ts, bond_ts, pos - 1)
    ITensorMPS.makeR!(P, site_ts, bond_ts, pos + nsite(P))
    return P
end

# We can use the following functions when we are updating the ProjMPO at the boundary of the
# partitions, where we would be out of the bounds of the `site_ts` and `bond_ts` arrays.
# Here we just need the new bond and site tensors, instead of the whole array.

function shiftleft!(
    P::AbstractProjMPO, new_site::ITensor, new_bond::ITensor
)::Union{ITensor,Nothing}
    # Shift the ProjMPO left by one site.  Like `makeR!` but with `k = P.rpos-1`.
    rl = min(P.rpos, length(P.H) + 1)

    #   ───▒        ───◆───▧────▒
    #      ▒               │    ▒
    #   ───▒    =   ───────o────▒
    #      ▒               │    ▒
    #   ───▒        ───◆───▧────▒
    #   Rₖ₋₁         Vₖ₋₂ Cₖ₋₁  Rₖ

    # In this diagram, k is `rl`, Vₖ₋₂ is `new_bond` and Cₖ₋₁ is `new_site`.
    R = rproj(P) * new_bond * new_site * P.H[rl - 1] * dag(prime(new_bond * new_site))
    P.LR[rl - 1] = R

    P.rpos -= 1
    return R
end

function shiftright!(
    P::AbstractProjMPO, new_site::ITensor, new_bond::ITensor
)::Union{ITensor,Nothing}
    # Shift the ProjMPO right by one site.  Like `makeL!` but with `k = P.lpos+1`.
    ll = max(P.lpos, 0)

    #   ▒───       ▒────▧───◆───
    #   ▒          ▒    │       
    #   ▒───   =   ▒────o───────
    #   ▒          ▒    │       
    #   ▒───       ▒────▧───◆───
    #   Lₖ₊₁       Lₖ  Cₖ₊₁ Vₖ₊₁

    # In this diagram, k is `ll`, Vₖ₊₁ is `new_bond` and Cₖ₊₁ is `new_site`.
    L = lproj(P) * new_site * new_bond * P.H[ll + 1] * dag(prime(new_site * new_bond))
    P.LR[ll + 1] = L

    P.lpos += 1
    return L
end

function invcanonical_decompose(M::ITensor, bond; linds, lefttags, righttags, kwargs...)
    U, S, V, spectrum = svd(M, linds...; lefttags=lefttags, righttags=righttags, kwargs...)
    S /= sqrt(scalar(S*S))

    # We assign U to the first site tensors within the segment that is being updated.
    # We will frequently multiply the tensors by delta(inds(S)): it is necessary in order to
    # correct the link indices.
    return (U * S) * delta(inds(S)), inv.(S), (S * V) * delta(inds(S)), spectrum.truncerr
end

function twositeupdate!(
    site_ts, bond_ts, PH, bond::Int, dt; maxdim, cutoff, sweepdir, current_time
)
    # Forward two-site evolution.
    set_nsite!(PH, 2)
    position!(PH, site_ts, bond_ts, bond)

    twositeblock = site_ts[bond] * bond_ts[bond] * site_ts[bond + 1]
    updated_twositeblock, info = exponentiate(PH, dt, twositeblock)
    info.converged == 0 && throw("exponentiate did not converge")

    linds = commoninds(twositeblock, site_ts[bond])
    ltags = tags(commonind(site_ts[bond], bond_ts[bond]))
    rtags = tags(commonind(site_ts[bond + 1], bond_ts[bond]))

    site_ts[bond], bond_ts[bond], site_ts[bond + 1], discarded_weight = invcanonical_decompose(
        updated_twositeblock,
        bond;
        linds=linds,
        maxdim=maxdim,
        cutoff=cutoff,
        lefttags=ltags,
        righttags=rtags,
    )

    return discarded_weight
end

function twositeupdate(
    Ψₗ::ITensor,
    V::ITensor,
    Ψᵣ::ITensor,
    PH,
    bond::Int,
    dt;
    maxdim,
    cutoff,
    sweepdir,
    current_time,
)
    # PH must be already “positioned”.
    updated_twositeblock, info = exponentiate(PH, dt, Ψₗ*V*Ψᵣ)
    info.converged == 0 && throw("exponentiate did not converge")

    linds = commoninds(updated_twositeblock, Ψₗ)
    ltags = tags(commonind(Ψₗ, V))
    rtags = tags(commonind(Ψᵣ, V))

    return invcanonical_decompose(
        updated_twositeblock,
        bond;
        linds=linds,
        maxdim=maxdim,
        cutoff=cutoff,
        lefttags=ltags,
        righttags=rtags,
    )
end

function onesiteupdate!(site_ts, bond_ts, PH, site::Int, dt; current_time)
    # Backward one-site evolution
    set_nsite!(PH, 1)
    position!(PH, site_ts, bond_ts, site)

    updated_onesiteblock, info = exponentiate(PH, dt, site_ts[site])
    info.converged == 0 && throw("exponentiate did not converge")

    site_ts[site] = updated_onesiteblock

    return nothing
end

function onesiteupdate(Ψ::ITensor, PH, site::Int, dt; current_time)
    # PH must be already “positioned”.
    updated_onesiteblock, info = exponentiate(PH, dt, Ψ)
    info.converged == 0 && throw("exponentiate did not converge")

    return updated_onesiteblock
end

function fullupdate!(
    site_ts, bond_ts, PH, bond::Int, dt; maxdim, cutoff, sweepdir, current_time
)
    @assert !(sweepdir == "right" && bond+1 == length(PH)) &&
        !(sweepdir == "left" && bond == 1)

    # a) Forward two-site evolution
    discarded_weight = twositeupdate!(
        site_ts,
        bond_ts,
        PH,
        bond,
        dt;
        maxdim=maxdim,
        cutoff=cutoff,
        sweepdir=sweepdir,
        current_time=current_time,
    )

    # b) Backward one-site evolution on the next site
    next_site_n = sweepdir == "right" ? bond+1 : bond
    onesiteupdate!(site_ts, bond_ts, PH, next_site_n, -dt; current_time=current_time)

    return discarded_weight
end

function partsweep_start_msg(partn, site_range, current_time)
    return string(
        "[Partition $partn] Site range: ",
        first(site_range),
        " to ",
        last(site_range),
        ". Beginning time evolution step at t = $current_time",
    )
end
