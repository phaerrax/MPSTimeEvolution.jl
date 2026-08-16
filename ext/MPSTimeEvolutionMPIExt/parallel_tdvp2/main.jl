function rangepart(N)
    k = div(N, 4)
    return [1:k, (k + 1):2k, (2k + 1):3k, (3k + 1):N]
end

function partition(
    ψ::InverseCanonicalMPS, partition_n; site_partitions=rangepart(nsites(ψ))
)
    # Partition the MPS into chunks. The bond tensor between two consecutive chunks is
    # assigned to the chunk on its left.
    r = site_partitions[partition_n]
    st = OffsetVector(site_tensors(ψ)[r], r)

    r = if partition_n < length(site_partitions)
        site_partitions[partition_n]
    else
        site_partitions[partition_n][1:(end - 1)]
        # Maybe we could include site_partitions[partition_n][end] anyway, since that would
        # be the trivial bond tensor at the edge of the IC MPS, but I'm not sure it wouldn't
        # mess up the ProjMPO construction.
    end
    bt = OffsetVector(bond_tensors(ψ)[r], r)

    return st, bt
end

function initialize_envs_on_partition(
    H, ψ::InverseCanonicalMPS, partition_n, comm; site_partitions=rangepart(nsites(ψ))
)::ProjMPO
    N = nsites(ψ)
    procrank = MPI.Comm_rank(comm)
    # We do the computations on worker 0, regardless of whether it is the root process or
    # not (it's easier).

    # Odd partitions (we start counting from one!) start sweeping left to right.
    # This means that they need the environment on the left of the leftmost MPS site of the
    # partition. Then, as they move rightwards, they will progressively create the missing
    # left environments. Right environments, instead, must already be there for all sites.
    # Even partitions work analogously.
    #
    # So, for example, the first partition starts from site 1 and needs the right
    # environments for all sites in the partition, but the left environment for site 1 only.
    # Partition 3 (say we are using 4 partition in total) starts from the middle of the MPS,
    # and needs the left environment for the middle site, and the right environments for all
    # its sites.
    # Clearly, as the environments are created incrementally, we can recycle the results so
    # that each left or right environment is computed only once: to this purpose, we perform
    # one full sweep with a single “base” PH object, then copy the environments created
    # along the way to the actual PHs.
    PH = if procrank == 0
        PH₀ = ProjMPO(H)  # Create a new ProjMPO from scratch.
        set_nsite!(PH₀, 1)
        position!(PH₀, ψ, 1)
        position!(PH₀, ψ, N)  # Sweep to the end.

        # Now all left environments have been created. We can send them to the ProjMPO of
        # the last partition.
        MPI.send(PH₀, comm; dest=3)

        # Continue sweeping leftwards until the middle of the MPS.
        position!(PH₀, ψ, first(site_partitions[3]))
        # Now we have created all right environments from the middle to the end of the MPS.
        # We also have the left environments from the start of the MPS to the middle,
        # created in the sweep from 1 to N above.
        MPI.send(PH₀, comm; dest=2)

        # Continue this way until we exhaust the partitions.
        position!(PH₀, ψ, last(site_partitions[2]))
        MPI.send(PH₀, comm; dest=1)

        position!(PH₀, ψ, first(site_partitions[1]))
        PH₀  # this is for procrank == 0
    else
        MPI.recv(comm; source=0)
    end
    MPI.Barrier(comm)

    return PH
end

function tdvp2_parallel_step!(
    ψ, PH, partition_n, dt, comm; rootrank=0, maxdim, cutoff, current_time
)
    # TODO Make this function not return a new InverseCanonicalMPS object each time, but
    # work on the partition directly.
    # Recombining everything into an InverseCanonicalMPS is needed only when we need to
    # (re-)initialise the ProjMPO environments and when we compute the expectation values
    # (well, not really, but it's easier that way).
    N = nsites(ψ)
    st, bt = partition(ψ, partition_n)
    isroot = (MPI.Comm_rank(comm) == rootrank)

    # Each process updates its assigned partition, and returns the total truncation error
    # accumulated during the evolution, so that we can update the overall norm of the MPS.
    _, _, _, truncerr = tdvp2_parallel_sweep_4p!(
        Val(partition_n), comm, st, bt, PH, dt; maxdim, cutoff, current_time
    )
    MPI.Barrier(comm)  # Wait for everyone to finish...

    isroot && @debug "Sweep complete. Gathering data from workers..."
    sts = MPI.gather(parent(st), comm; root=rootrank)
    bts = MPI.gather(parent(bt), comm; root=rootrank)

    # We compute the norm from the total discarded weights gathered from the partitions.
    # (Note that MPI.Reduce returns `nothing` on non-root processes, so we edit the norm
    # only on root. It's not an issue since the other processes don't actually use the MPS
    # anyway.)
    tot_truncerr = MPI.Reduce(truncerr, +, comm; root=rootrank)
    if isroot
        updated_norm = ψ.norm * sqrt(1 - tot_truncerr)
    end

    # Put the MPS chunks back together so that we can compute expectation values later.
    if isroot
        InverseCanonicalMPS(
            reduce(vcat, sts),
            OffsetVector([ITensor(1.0); reduce(vcat, bts); ITensor(1.0)], 0:N),
            updated_norm,
        )
    else
        InverseCanonicalMPS()
        # We could return nothing, but this way the function is type-stable. (Does it
        # really make a difference here?)
    end
end

function MPSTimeEvolution.tdvp2_parallel!(
    ψ::InverseCanonicalMPS, H::MPO, dt, tmax; kwargs...
)
    return MPSTimeEvolution.tdvp2_parallel!(ψ, ProjMPO(H), dt, tmax; kwargs...)
end

function MPSTimeEvolution.tdvp2_parallel!(
    ψ::InverseCanonicalMPS, PH, dt, tmax; rootrank=0, comm, kwargs...
)
    nsteps = floor(Int, tmax / dt)
    maxdim = get(kwargs, :maxdim, maxlinkdim(ψ))
    cutoff = get(kwargs, :cutoff, 1e-15)
    hermitian = get(kwargs, :hermitian, true)
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    normalize_state = get(kwargs, :normalize, true)
    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)
    store_state0 = get(kwargs, :store_psi0, false)
    norm_threshold = get(kwargs, :norm_threshold, nothing)

    # If a real-valued time step `dt` is given, we compute a real-time evolution
    # given by the operator exp(-itH).
    # Passing an imaginary time step (and `tmax`) as an argument triggers instead an
    # evolution according to the operator exp(-tH), which we don't support at the moment,
    # so we throw an error instead.
    !isreal(dt) && error("imaginary-time evolution not implemented for parallel TDVP2")

    procrank = MPI.Comm_rank(comm)
    isroot = (procrank == rootrank)
    # Rank legend (n = 4):
    #   0 → root process (manager)
    #   1:n-1 → worker processes
    # The manager runs the algorithm too, it's just that it also has other tasks to handle,
    # such as computing the expectation values.

    store_state0 && (ψ₀ = copy(ψ))
    N = nsites(ψ)
    current_time = 0.0

    if isroot
        @debug "Initialising time evolution."
        cb = get(kwargs, :callback, NoTEvoCallback())

        if get(kwargs, :progress, true)
            pbar = Progress(nsteps; desc="Evolving state... ")
        else
            pbar = nothing
        end

        io_handle = MPSTimeEvolution.writeheaders_data(io_file, cb; kwargs...)
        isnothing(io_file) || @debug "Initialising output file $io_file..."
        ranks_handle = MPSTimeEvolution.writeheaders_ranks(ranks_file, nsites(ψ))
        times_handle = MPSTimeEvolution.writeheaders_stime(times_file)

        # Measure everything once in the initial state.
        @debug "Computing expectation values on initial state..."
        MPSTimeEvolution.apply!(cb, ψ, TDVP2(); current_time, sweepend=true)
        MPSTimeEvolution.compute_normalization!(cb, ψ, TDVP2(); current_time)

        if store_state0
            MPSTimeEvolution.printoutput_data(io_handle, cb, ψ; psi0=ψ0, kwargs...)
        else
            MPSTimeEvolution.printoutput_data(io_handle, cb, ψ; kwargs...)
        end
        MPSTimeEvolution.printoutput_ranks(ranks_handle, cb, ψ)
    end

    partn = procrank+1

    # Each worker has its ProjMPO object (each one independent from the others).
    # Initialise the ProjMPOs, while each worker still has the full MPS.
    # We need to do this now because each process will only get a chunk of the MPS tensors
    # inside the tdvp2_parallel_sweep_4p function, and will not be able to re-position the
    # ProjMPO outsite the bounds of its chunk.  At the beginning of the evolution the
    # environments in the ProjMPOs need to be generated from scratch, involving all tensors
    # of the MPS.
    isroot && @debug "Initialising TDVP environments..."
    PH = initialize_envs_on_partition(PH.H, ψ, partn, comm)

    for s in 1:nsteps
        isroot && @debug "Executing step $s of $nsteps..."
        result = @timed tdvp2_parallel_step!(
            ψ, PH, partn, dt, comm; rootrank=0, maxdim, cutoff, current_time
        )
        # result.value -> actual result
        # result.time -> elapsed time

        current_time += dt
        stime = result.time

        # Broadcast the updates MPS to all workers, so that they can use it when the new
        # step begins.
        isroot && @debug "Broadcasting the MPS to all workers..."
        ψ = MPI.bcast(result.value, comm; root=rootrank)

        # Check if norm exceeds the threshold; if it does, recanonicalise the MPS (and
        # consequently recompute the PHs).  The recanonicalisation will also normalise ψ.
        if !isnothing(norm_threshold) && abs(1 - norm(ψ)) > norm_threshold
            ψ = if isroot
                @debug "Re-canonicalising the MPS..."
                canonicalize(ψ; use_absolute_cutoff=true, cutoff=0)
            else
                InverseCanonicalMPS()
            end
            ψ = MPI.bcast(ψ, comm; root=rootrank)
            PH = initialize_envs_on_partition(PH.H, ψ, partn, comm)
            # Don't run this as root only! All workers need to recreate their PHs.
        end

        if isroot
            @debug "Computing expectation values at t = $current_time."
            MPSTimeEvolution.apply!(cb, ψ, TDVP2(); current_time, sweepend=true)
            MPSTimeEvolution.compute_normalization!(cb, ψ, TDVP2(); current_time)

            !isnothing(pbar) && ProgressMeter.next!(
                pbar; showvalues=MPSTimeEvolution.simulationinfo(ψ, current_time, stime)
            )

            if !isempty(measurement_ts(cb)) && current_time ≈ measurement_ts(cb)[end]
                if store_state0
                    MPSTimeEvolution.printoutput_data(io_handle, cb, ψ; psi0=ψ₀, kwargs...)
                else
                    MPSTimeEvolution.printoutput_data(io_handle, cb, ψ; kwargs...)
                end
                MPSTimeEvolution.printoutput_ranks(ranks_handle, cb, ψ)
                MPSTimeEvolution.printoutput_stime(times_handle, stime)
            end

            MPSTimeEvolution.checkdone!(cb) && break
        end
    end

    if isroot
        !isnothing(io_file) && close(io_handle)
        !isnothing(ranks_file) && close(ranks_handle)
        !isnothing(times_file) && close(times_handle)
    end

    return nothing
end
