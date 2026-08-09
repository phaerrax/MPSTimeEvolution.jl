# Sweep sub-processes (1 to 4)
# ============================

# When carrying out a two-site update, the current left and right site tensors are
# labelled Ψₗ and Ψᵣ. (We could call them Ψⱼ and Ψⱼ₊₁, with j sweeping the current
# partition from left to right or vice versa, depending on the process).  At each step,
# βₗ and γₗ are the effective (left and right) environment tensors for Ψₗ, while βᵣ and
# γᵣ are the effective environment tensors for Ψᵣ.

# Process 4
# =========
function tdvp2_parallel_sweep_4p!(
    ::Val{4}, comm::MPI.Comm, site_ts, bond_ts, PH, dt; maxdim, cutoff, current_time
)
    site_range = eachindex(site_ts)
    @debug partsweep_start_msg(4, site_range, current_time)

    bond = last(site_range)

    # • Ψₗ ⟵  the partition's rightmost site
    # • Repeat... until Ψₗ is leftmost site
    while true
        bond -= 1
        # • Perform two-site update
        fullupdate!(
            site_ts,
            bond_ts,
            PH,
            bond,
            -0.5im*dt;
            maxdim=maxdim,
            cutoff=cutoff,
            sweepdir="left",
            current_time=current_time+0.5dt,
        )
        bond == first(site_range) && break
    end
    @assert bond == first(site_range)

    # • Ψᵣ ⟵  partition's leftmost site
    bond = first(site_range)-1
    set_nsite!(PH, 2)
    position!(PH, site_ts, bond_ts, bond)  # ...this should work?
    Ψᵣ=site_ts[bond + 1]

    # • Receive βₗ
    #
    #      process 3 ╎ process 4
    #                ╎      
    #        ▒───   ─╎─   ─ ──▒
    #        ▒    │  ╎  │     ▒
    #    βₗ  ▒────o──╎──o─────▒  γₗ
    #        ▒    │  ╎  │     ▒
    #        ▒───   ─╎─   ─ ──▒
    #       b-1   b  ╎ b+1   b+2
    #                ╎
    #             Ψₗ ╎  Ψᵣ

    βₗ = MPI.recv(comm; source=2, tag=intcode("betaL"))
    PH.LR[1:(bond - 1)] .= βₗ

    # • Send γᵣ, Ψᵣ
    γᵣ=PH.LR[(bond + 2):end]
    MPI.send(γᵣ, comm; dest=2, tag=intcode("gammaR"))
    MPI.send(Ψᵣ, comm; dest=2, tag=intcode("PsiR"))

    # ... wait for process 3 to update site_ts[bond], bond_ts[bond] and site_ts[bond+1]

    # • Receive Ψₗ, V, Ψᵣ
    Ψₗ = MPI.recv(comm; source=2, tag=intcode("PsiL"))
    V = MPI.recv(comm; source=2, tag=intcode("V"))
    Ψᵣ = MPI.recv(comm; source=2, tag=intcode("PsiR"))

    # • Create βᵣ
    shiftright!(PH, Ψₗ, V)

    ### Second half of the sweep

    # • Ψᵣ ⟵  leftmost site, update Ψᵣ
    set_nsite!(PH, 1)
    site_ts[bond + 1] = onesiteupdate(
        Ψᵣ, PH, bond+1, 0.5im*dt; current_time=current_time+dt
    )

    # • Repeat... 
    while true
        bond += 1
        # a) Forward two-site evolution
        twositeupdate!(
            site_ts,
            bond_ts,
            PH,
            bond,
            -0.5im*dt;
            maxdim=maxdim,
            cutoff=cutoff,
            sweepdir="right",
            current_time=current_time+dt,
        )

        # ...until Ψᵣ is the partition's rightmost site
        bond+1 == last(site_range) && break

        # b) Backward one-site evolution on the next site
        onesiteupdate!(site_ts, bond_ts, PH, bond+1, 0.5im*dt; current_time=current_time+dt)
    end
    @assert bond+1 == last(site_range)

    return site_ts, bond_ts, PH
end
