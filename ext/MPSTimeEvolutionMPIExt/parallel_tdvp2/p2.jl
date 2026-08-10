# Sweep sub-processes (1 to 4)
# ============================

# When carrying out a two-site update, the current left and right site tensors are
# labelled Ψₗ and Ψᵣ. (We could call them Ψⱼ and Ψⱼ₊₁, with j sweeping the current
# partition from left to right or vice versa, depending on the process).  At each step,
# βₗ and γₗ are the effective (left and right) environment tensors for Ψₗ, while βᵣ and
# γᵣ are the effective environment tensors for Ψᵣ.

# Process 2
# =========
function tdvp2_parallel_sweep_4p!(
    ::Val{2}, comm::MPI.Comm, site_ts, bond_ts, PH, dt; maxdim, cutoff, current_time
)
    site_range = eachindex(site_ts)

    # • Ψₗ ⟵  partition’s rightmost site tensor
    bond = last(site_range)
    Ψₗ = site_ts[bond]
    V = bond_ts[bond]

    # • Receive γᵣ and Ψᵣ from process 3
    #   Ψᵣ is the site tensor on the leftmost site of partition 3.
    γᵣ = MPI.recv(comm; source=2, tag=intcode("gammaR"))
    Ψᵣ = MPI.recv(comm; source=2, tag=intcode("PsiR"))
    # Ψᵣ would be site_ts[bond+1] which of course is outside the bounds.
    PH.LR[(bond + 2):end] .= γᵣ

    # • Send βₗ to process 3
    #   The site tensor Ψₗ is on `bond`, so βₗ = PH.LR[1:bond-1]
    set_nsite!(PH, 2)  # Set PH on (bond, bond+1).
    position!(PH, site_ts, bond_ts, bond)

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
    #
    MPI.send(PH.LR[1:(bond - 1)], comm; dest=2, tag=intcode("betaL"))

    # • Perform a two-site update across the partition border.
    Ψₗ, V, Ψᵣ = twositeupdate(
        Ψₗ,
        V,
        Ψᵣ,
        PH,
        bond,
        -0.5im*dt;
        maxdim=maxdim,
        cutoff=cutoff,
        sweepdir="left",
        current_time=current_time+0.5dt,
    )

    # • Send Ψₗ, V, and Ψᵣ to process 3
    MPI.send(Ψₗ, comm; dest=2, tag=intcode("PsiL"))
    MPI.send(V, comm; dest=2, tag=intcode("V"))
    MPI.send(Ψᵣ, comm; dest=2, tag=intcode("PsiR"))
    site_ts[bond] = Ψₗ
    bond_ts[bond] = V
    # site_ts[bond+1] = Ψᵣ

    # • Create γₗ (using the updated tensors)
    #   Since Ψₗ is on site `bond`, γₗ is PH.LR[bond+1:end]. We cannot call
    #
    #       position!(PH, site_ts, bond_ts, bond)
    #
    #   since it would require site_ts[bond+1], which is outside the bounds of the vector.
    #   Instead, we build on the previously received γᵣ by calling shiftleft!.
    set_nsite!(PH, 1)
    shiftleft!(PH, Ψᵣ, V)
    # We have also put it in the correct configuration for the following single-site update.

    # • Update Ψₗ
    site_ts[bond] = onesiteupdate(Ψₗ, PH, bond, 0.5im*dt; current_time=current_time+0.5dt)

    # • Discard βₗ (this happens automatically once we shift PH)

    # • Repeat ... until Ψₗ is the partition’s leftmost site tensor
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
            current_time=current_time+0.5dt,
        )
        bond == first(site_range) && break
    end
    @assert bond == first(site_range)

    # • Ψᵣ ⟵  the partition’s leftmost site tensor
    bond = first(site_range)-1
    set_nsite!(PH, 2)
    position!(PH, site_ts, bond_ts, bond)  # This should work because we come from the right
    Ψᵣ = site_ts[bond + 1]

    # • Receive βₗ from process 1
    #   Ψₗ is the rightmost site tensor on partition 1, so it is on site bond-1, and
    #   consequently βₗ refers to the environment from 1 to bond-2.
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
    βₗ = MPI.recv(comm; source=0, tag=intcode("betaL"))
    PH.LR[1:(bond - 1)] .= βₗ

    # Ψᵣ is on site `bond` so its right environment γᵣ contains sites from `bond+1` to the
    # end (of the overall system).
    γᵣ = PH.LR[(bond + 2):end]

    # • Send γᵣ and Ψᵣ to process 1
    MPI.send(γᵣ, comm; dest=0, tag=intcode("gammaR"))
    MPI.send(Ψᵣ, comm; dest=0, tag=intcode("PsiR"))

    # ... wait for process 1 to perform the two site update on (bond-1, bond) ...

    # • Receive Ψₗ , V, and Ψᵣ from process 1
    Ψₗ = MPI.recv(comm; source=0, tag=intcode("PsiL"))
    V = MPI.recv(comm; source=0, tag=intcode("V"))
    Ψᵣ = MPI.recv(comm; source=0, tag=intcode("PsiR"))
    # site_ts[bond] = Ψₗ
    # bond_ts[bond] = V
    site_ts[bond + 1] = Ψᵣ

    # • Create βᵣ (by shifting the ProjMPO)
    #   We already obtained PH.LR[1:(bond - 1)] .= βₗ some lines above.
    #   Now we need to shift the left projection one site to the right to create βᵣ, using
    #   the tensors that process 1 just sent us.
    set_nsite!(PH, 1)
    shiftright!(PH, Ψₗ, V)

    # • Update Ψᵣ
    #   We are now sweeping back right.
    site_ts[bond + 1] = onesiteupdate(Ψᵣ, PH, bond, 0.5im*dt; current_time=current_time+dt)

    # • Repeat ...  until Ψᵣ is the partition’s rightmost site tensor
    while true
        bond += 1
        # Ψₗ ⟵  Ψᵣ
        # Ψᵣ ⟵  site tensor to the right of Ψₗ
        # Perform full two-site update
        fullupdate!(
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
        bond+1 == last(site_range) && break
    end
    @assert bond+1 == last(site_range)

    # Now we have come back to the middle of the MPS, where we started! We just need one
    # last two-site update across the boundary from process 2 and process 3.

    # • Ψₗ ⟵  the partition’s rightmost site tensor
    bond = last(site_range)
    Ψₗ = site_ts[bond]
    V = bond_ts[bond]

    # • Send βₗ to process 3
    set_nsite!(PH, 2)
    position!(PH, site_ts, bond_ts, bond)

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

    βₗ = PH.LR[1:(bond - 1)]
    MPI.send(βₗ, comm; dest=2, tag=intcode("betaL"))

    # • Receive γᵣ and Ψᵣ from process 3
    γᵣ = MPI.recv(comm; source=2, tag=intcode("gammaR"))
    Ψᵣ = MPI.recv(comm; source=2, tag=intcode("PsiR"))
    # Incorporate γᵣ into PH
    PH.LR[(bond + 2):end] .= γᵣ

    # • Perform two-site update
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
        current_time=current_time+dt,
    )

    # • Send Ψₗ , V, and Ψᵣ to process 3
    MPI.send(Ψₗ, comm; dest=2, tag=intcode("PsiL"))
    MPI.send(V, comm; dest=2, tag=intcode("V"))
    MPI.send(Ψᵣ, comm; dest=2, tag=intcode("PsiR"))
    site_ts[bond] = Ψₗ
    bond_ts[bond] = V

    # • Create γₗ
    #   Since site_ts[bond] = Ψₗ, γₗ is the right environment on bond+1:end.
    #   site_ts stops at the site `bond`, so we need to use shiftleft!.
    set_nsite!(PH, 1)
    shiftleft!(PH, Ψᵣ, V)

    return site_ts, bond_ts, PH
end
