# Sweep sub-processes (1 to 4)
# ============================

# When carrying out a two-site update, the current left and right site tensors are
# labelled Ψₗ and Ψᵣ. (We could call them Ψⱼ and Ψⱼ₊₁, with j sweeping the current
# partition from left to right or vice versa, depending on the process).  At each step,
# βₗ and γₗ are the effective (left and right) environment tensors for Ψₗ, while βᵣ and
# γᵣ are the effective environment tensors for Ψᵣ.

# Process 3
# =========
function tdvp2_parallel_sweep_4p(
    ::Val{3}, comm::MPI.Comm, site_ts, bond_ts, PH, dt; maxdim, cutoff, current_time
)
    site_range = eachindex(site_ts)
    @debug partsweep_start_msg(3, site_range, current_time)

    # • Ψᵣ ⟵  partition’s leftmost site tensor
    bond = first(site_range)-1
    Ψᵣ = site_ts[bond + 1]

    # • Send γᵣ, Ψᵣ to process 2
    #   First, reposition the ProjMPO so that the most recent γᵣ is created.
    set_nsite!(PH, 2)

    # Current PH configuration:
    #
    #       process 2 ╎ process 3
    #                 ╎ 
    #        ▒───     ╎     ───▒
    #        ▒    │   ╎   │    ▒
    #    βₗ  ▒────o───╎───o────▒  γᵣ
    #        ▒    │   ╎   │    ▒
    #        ▒───     ╎     ───▒
    #       b-1   b   ╎  b+1  b+2
    #                 ╎ 
    #             Ψₗ  ╎   Ψᵣ

    position!(PH, site_ts, bond_ts, bond)
    # This updates PH.LR[1:(bond - 1)] (aka βᵣ) and PH.LR[(bond + 2):end] (aka γᵣ).
    γᵣ = PH.LR[(bond + 2):end]
    MPI.send(γᵣ, comm; dest=1, tag=intcode("gammaR"))
    MPI.send(Ψᵣ, comm; dest=1, tag=intcode("PsiR"))

    # • Receive βₗ from process 2
    #   Ψₗ is currently the rightmost site in partition 2, which is on `bond-1`.
    βₗ = MPI.recv(comm; source=1, tag=intcode("betaL"))
    PH.LR[1:(bond - 1)] .= βₗ

    # ... wait for process 2 perform a two-site update on (bond, bond+1) ...

    # • Receive Ψₗ, V, Ψᵣ from process 2
    Ψₗ = MPI.recv(comm; source=1, tag=intcode("PsiL"))
    V = MPI.recv(comm; source=1, tag=intcode("V"))
    Ψᵣ = MPI.recv(comm; source=1, tag=intcode("PsiR"))

    # • Create βᵣ
    #   We create βᵣ from the βₗ we received before, using shiftright!.
    #   If βₗ is PH.LR[1:(bond - 2)], then βᵣ is obtained by multiplying βₗ by V and Ψₗ,
    #   creating PH.LR[1:(bond - 1)].
    shiftright!(PH, Ψₗ, V)

    # • Update Ψᵣ and discard γᵣ
    bond = first(site_range)
    set_nsite!(PH, 1)
    #position!(PH, site_ts, bond_ts, bond)
    site_ts[bond] = onesiteupdate(Ψᵣ, PH, bond, -0.5im*dt; current_time=current_time+0.5dt)

    # • Sweep right, starting from bond = first(site_range)...
    while true
        fullupdate!(
            site_ts,
            bond_ts,
            PH,
            bond,
            -0.5im*dt;
            maxdim=maxdim,
            cutoff=cutoff,
            sweepdir="right",
            current_time=current_time+0.5dt,
        )

        # ...until Ψᵣ is the rightmost site in the partition.
        bond+1 == last(site_range) && break
        bond += 1
    end
    @assert bond+1 == last(site_range)

    # • Ψₗ ⟵  the partition's rightmost site
    bond = last(site_range)
    Ψₗ = site_ts[bond]
    V = bond_ts[bond]

    # • Send βₗ to process 4
    set_nsite!(PH, 2)
    position!(PH, site_ts, bond_ts, bond)  # we only update “from the left” so it should work

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

    βₗ = PH.LR[1:(bond - 1)]
    MPI.send(βₗ, comm; dest=3, tag=intcode("betaL"))

    # • Receive γᵣ and Ψᵣ from process 4
    γᵣ = MPI.recv(comm; source=3, tag=intcode("gammaR"))
    Ψᵣ = MPI.recv(comm; source=3, tag=intcode("PsiR"))
    PH.LR[(bond + 2):end] .= γᵣ

    # • Perform 2-site update on the boundary between process 3 and process 4
    set_nsite!(PH, 2)
    Ψₗ, V, Ψᵣ = twositeupdate(
        Ψₗ,
        V,
        Ψᵣ,
        PH,
        bond,
        -0.5im*dt;
        maxdim=maxdim,
        cutoff=cutoff,
        sweepdir="right",
        current_time=current_time+0.5dt,
    )

    # • Send Ψₗ, V, Ψᵣ to process 4
    MPI.send(Ψₗ, comm; dest=3, tag=intcode("PsiL"))
    MPI.send(V, comm; dest=3, tag=intcode("V"))
    MPI.send(Ψᵣ, comm; dest=3, tag=intcode("PsiR"))
    bond_ts[bond] = V

    # • Create γₗ
    #   Here γₗ = PH.LR[bond+1], we create it from γᵣ
    set_nsite!(PH, 1)
    shiftleft!(PH, Ψᵣ, V)

    ### Second half of the sweep

    # • Ψₗ ⟵  the partition’s rightmost site tensor
    bond = last(site_range)

    # • Update Ψₗ (PH is already in the correct position)
    site_ts[bond] = onesiteupdate(Ψₗ, PH, bond, -0.5im*dt; current_time=current_time+dt)

    # • Sweep left with full two-site updates, from bond = last(site_range)-1
    while true
        bond -= 1
        fullupdate!(
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
    end
    @assert bond == first(site_range)

    # • Ψᵣ ⟵  the partition’s leftmost site tensor
    bond = first(site_range)-1
    Ψᵣ = site_ts[bond + 1]
    set_nsite!(PH, 2)
    position!(PH, site_ts, bond_ts, bond)  # should work, we come from the right

    # • receive βₗ from process 2

    #       process 2 ╎ process 3
    #                 ╎ 
    #        ▒───     ╎     ───▒
    #        ▒    │   ╎   │    ▒
    #    βₗ  ▒────o───╎───o────▒  γᵣ
    #        ▒    │   ╎   │    ▒
    #        ▒───     ╎     ───▒
    #       b-1   b   ╎  b+1  b+2
    #                 ╎ 
    #             Ψₗ  ╎   Ψᵣ

    βₗ = MPI.recv(comm; source=1, tag=intcode("betaL"))
    PH.LR[1:(bond - 1)] .= βₗ

    # • Send γᵣ and Ψᵣ to process 2
    γᵣ=PH.LR[(bond + 2):end]
    MPI.send(γᵣ, comm; dest=1, tag=intcode("gammaR"))
    MPI.send(Ψᵣ, comm; dest=1, tag=intcode("PsiR"))

    # ... wait for process 2 to perform the two site update on (bond, bond+1) ...

    # • Receive Ψₗ , V, and Ψᵣ from process 2
    Ψₗ = MPI.recv(comm; source=1, tag=intcode("PsiL"))
    V = MPI.recv(comm; source=1, tag=intcode("V"))
    Ψᵣ = MPI.recv(comm; source=1, tag=intcode("PsiR"))

    #site_ts[bond] = Ψₗ  # outside bonds
    #bond_ts[bond] = V  # outside bonds
    site_ts[bond + 1] = Ψᵣ

    # • Create βᵣ
    shiftright!(PH, Ψₗ, V)

    return site_ts, bond_ts, PH
end
