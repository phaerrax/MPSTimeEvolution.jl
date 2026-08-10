# Sweep sub-processes (1 to 4)
# ============================

# When carrying out a two-site update, the current left and right site tensors are
# labelled Ψₗ and Ψᵣ. (We could call them Ψⱼ and Ψⱼ₊₁, with j sweeping the current
# partition from left to right or vice versa, depending on the process).  At each step,
# βₗ and γₗ are the effective (left and right) environment tensors for Ψₗ, while βᵣ and
# γᵣ are the effective environment tensors for Ψᵣ.

# Process 1
# =========
function tdvp2_parallel_sweep_4p!(
    ::Val{1}, comm::MPI.Comm, site_ts, bond_ts, PH, dt; maxdim, cutoff, current_time
)
    site_range = eachindex(site_ts)

    # • Ψᵣ ⟵  partition’s leftmost site tensor
    bond = first(site_range)-1
    @assert bond == 0
    # This is just temporary: `bond` will be 1 before the first update is performed.

    # • Repeat ... until Ψᵣ is partition’s rightmost site tensor
    while true
        # • Ψₗ ⟵  Ψᵣ and Ψᵣ ⟵  site tensor to the right of Ψₗ
        bond += 1

        # • Perform two-site update on (Ψₗ,Ψᵣ)
        # • Update Ψᵣ, and the environment blocks in PH (create βᵣ, discard γᵣ)
        fullupdate!(
            site_ts,
            bond_ts,
            PH,
            bond,
            -0.5im*dt;
            maxdim=maxdim,
            cutoff=cutoff,
            sweepdir="right",
            current_time=current_time + 0.5dt,
        )

        bond+1 == last(site_range) && break
    end
    @assert bond+1 == last(site_range)
    # Now we have updated the whole `site_range` for dt/2.

    # • Ψₗ ⟵  partition’s rightmost site tensor
    bond = last(site_range)
    Ψₗ = site_ts[bond]
    V = bond_ts[bond]
    set_nsite!(PH, 2)

    # • Send βₗ to process 2
    βₗ = PH.LR[1:(bond - 1)]
    MPI.send(βₗ, comm; dest=1, tag=intcode("betaL"))

    # • Receive γᵣ and Ψᵣ from process 2
    #
    #       process 1 ╎ process 2
    #                 ╎ 
    #        ▒───     ╎     ───▒
    #        ▒    │   ╎   │    ▒
    #    βₗ  ▒────o───╎───o────▒  γᵣ
    #        ▒    │   ╎   │    ▒
    #        ▒───     ╎     ───▒
    #       b-1   b   ╎  b+1  b+2
    #                 ╎ 
    #             Ψₗ  ╎   Ψᵣ

    γᵣ = MPI.recv(comm; source=1, tag=intcode("gammaR"))
    Ψᵣ = MPI.recv(comm; source=1, tag=intcode("PsiR"))
    PH.LR[(bond + 2):end] .= γᵣ
    PH.rpos = bond+2
    PH.lpos = bond-1
    # site_ts[bond+1] = Ψᵣ
    # This Ψᵣ is outside the partition, so we cannot really include it in `site_ts`.
    # We update it together with `site_ts[bond]` and `bond_ts[bond]` in a “detached” way,
    # then we send it back to process 2 and forget about it.

    # • Evolve Ψₗ, V, Ψᵣ forwards by dt.
    Ψₗ, V, Ψᵣ = twositeupdate(
        Ψₗ,
        V,
        Ψᵣ,
        PH,
        bond,
        -im*dt;
        maxdim=maxdim,
        cutoff=cutoff,
        sweepdir="right",
        current_time=current_time + 0.5dt,
    )

    # • Send Ψₗ, V, and Ψᵣ to process 2
    MPI.send(Ψₗ, comm; dest=1, tag=intcode("PsiL"))
    MPI.send(V, comm; dest=1, tag=intcode("V"))
    MPI.send(Ψᵣ, comm; dest=1, tag=intcode("PsiR"))
    site_ts[bond] = Ψₗ
    bond_ts[bond] = V

    # • Create γₗ (aka the right environment for Ψₗ)
    set_nsite!(PH, 1)
    shiftleft!(PH, Ψᵣ, V)

    # • Ψₗ ⟵  the partition's rightmost site
    bond = last(site_range)
    # Next update is a one-site update, so we won't reach into the partition of process 2
    # even if `bond` is on the last site of this partition.

    # Update Ψₗ.
    site_ts[bond] = onesiteupdate(
        site_ts[bond], PH, bond, 0.5im*dt; current_time=current_time+dt
    )

    # • Repeat ... until Ψₗ is partition’s leftmost site tensor
    while true
        bond -= 1

        # • Perform a full two-site update, except when we are at the leftmost site, where
        #   we perform only the first part of the update.

        # a) Forward two-site evolution on (bond, bond+1).
        set_nsite!(PH, 2)
        position!(PH, site_ts, bond_ts, bond)
        twositeupdate!(
            site_ts,
            bond_ts,
            PH,
            bond,
            -0.5im*dt;
            maxdim=maxdim,
            cutoff=cutoff,
            sweepdir="left",
            current_time=current_time+dt,
        )

        bond == first(site_range) && break

        # b) Backward one-site evolution on the next site (which is `bond` because we are
        # sweeping leftwards).
        set_nsite!(PH, 1)
        position!(PH, site_ts, bond_ts, bond)
        onesiteupdate!(site_ts, bond_ts, PH, bond, 0.5im*dt; current_time=current_time+dt)
    end
    @assert bond == first(site_range)

    return site_ts, bond_ts, PH
end
