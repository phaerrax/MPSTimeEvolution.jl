function rangepart(N)
    k = div(N, 4)
    return [1:k, (k + 1):2k, (2k + 1):3k, (3k + 1):N]
end

function tdvp2_parallel_sweep_4p!(
    partition::MPSPartition, dt, comm; maxdim, cutoff, current_time
)
    return tdvp2_parallel_sweep_4p!(
        Val(rank(partition)), partition, dt, comm; maxdim, cutoff, current_time
    )
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

    cb = get(kwargs, :callback, NoTEvoCallback())

    pbar = if get(kwargs, :progress, true)
        Progress(nsteps; desc="Evolving state... ")
    else
        pbar = nothing
    end

    isroot && @debug "Initialising time evolution."

    if isroot && !isnothing(io_file)
        # TODO This should go into `writeheaders_data`...
        @debug "Initialising output file $io_file..."
    end
    io_handle = MPSTimeEvolution.writeheaders_data(
        io_file, comm, cb; root=rootrank, kwargs...
    )
    ranks_handle = MPSTimeEvolution.writeheaders_ranks(ranks_file, comm, N; root=rootrank)
    times_handle = isroot ? MPSTimeEvolution.writeheaders_stime(times_file) : nothing

    if isroot
        # Measure everything once in the initial state.
        @debug "Computing expectation values on initial state..."
        MPSTimeEvolution.apply!(cb, ψ, TDVP2(); current_time, sweepend=true)
        MPSTimeEvolution.compute_normalization!(cb, ψ, TDVP2(); current_time)

        if store_state0
            MPSTimeEvolution.printoutput_data(io_handle, cb, ψ; psi0=ψ₀, kwargs...)
        else
            MPSTimeEvolution.printoutput_data(io_handle, cb, ψ; kwargs...)
        end
        MPSTimeEvolution.printoutput_ranks(ranks_handle, cb, ψ)
    end

    partn = procrank+1

    isroot && @debug "Initialising partitions..."
    partition = MPSPartition(ψ, PH.H, partn, comm)

    for s in 1:nsteps
        isroot && @debug "Executing step $s of $nsteps..."
        # Each process updates its assigned partition and can updates the overall norm of
        # the MPS (saved within each partition struct).
        result = @timed tdvp2_parallel_sweep_4p!(
            partition, dt, comm; maxdim, cutoff, current_time
        )
        current_time += dt
        # result.value -> actual result
        # result.time -> elapsed time
        partition, truncerr = result.value
        stime = result.time

        # Compute the norm from the total discarded weights gathered from the partitions.
        tot_truncerr = MPI.Allreduce(truncerr, +, comm)
        partition.norm *= sqrt(1 - tot_truncerr)

        # Check if norm exceeds the threshold; if it does, recanonicalise the MPS (and
        # consequently recompute the PHs).  The recanonicalisation will also normalise ψ.
        if !isnothing(norm_threshold) && abs(1 - partition.norm) > norm_threshold
            ψ = gather_to_mps(partition, comm; root=rootrank)
            new_ψ = if isroot
                @debug "Re-canonicalising the MPS..."
                canonicalize(ψ; use_absolute_cutoff=true, cutoff=0)
            else
                InverseCanonicalMPS()
            end
            ψ = MPI.bcast(new_ψ, comm; root=rootrank)
            # Recompute partitions with the re-canonicalised MPS.
            # (Don't run this as root only! All workers need to recreate their partitions.)
            partition = MPSPartition(ψ, PH.H, partn, comm)
        end

        MPSTimeEvolution.apply!(
            cb, partition, TDVP2(), comm; current_time, sweepend=true, root=rootrank
        )
        # (The apply! function takes care of computing the norm as well, in this case.)

        if !isnothing(pbar)
            # Make sure that `simulationinfo` gets called by every worker, not just root ---
            # it contains calls to MPI.Allreduce and will deadlock otherwise.
            showvalues_fn=MPSTimeEvolution.simulationinfo(
                partition, current_time, stime, comm
            )
            isroot && ProgressMeter.next!(pbar; showvalues=showvalues_fn)
        end

        # The `printoutput_data` and `printoutput_ranks` functions call MPI methods
        # internally, so don't run them as root only. The print-to-file statements inside
        # them are already conditioned on `isroot` being true. `printoutput_stime` is fine,
        # instead, there's no MPI there.
        if !isempty(measurement_ts(cb)) && current_time ≈ measurement_ts(cb)[end]
            if store_state0
                MPSTimeEvolution.printoutput_data(
                    io_handle,
                    cb,
                    partition,
                    comm;
                    root=rootrank,
                    store_psi0=true,
                    psi0=ψ₀,
                    kwargs...,
                )
            else
                MPSTimeEvolution.printoutput_data(
                    io_handle, cb, partition, comm; root=rootrank, kwargs...
                )
            end
            MPSTimeEvolution.printoutput_ranks(
                ranks_handle, cb, partition, comm; root=rootrank
            )
            isroot && MPSTimeEvolution.printoutput_stime(times_handle, stime)
        end

        MPSTimeEvolution.checkdone!(cb) && break
    end

    if isroot
        !isnothing(io_file) && close(io_handle)
        !isnothing(ranks_file) && close(ranks_handle)
        !isnothing(times_file) && close(times_handle)
    end

    return nothing
end
