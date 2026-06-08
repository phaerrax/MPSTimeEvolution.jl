function rangepart(N)
    k = div(N, 4)
    return [1:k, (k + 1):2k, (2k + 1):3k, (3k + 1):N]
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
    normalize = get(kwargs, :normalize, true)
    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)
    store_state0 = get(kwargs, :store_psi0, false)

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
        MPSTimeEvolution.apply!(cb, ψ, TDVP2(); current_time=current_time)
        MPSTimeEvolution.compute_norm!(cb, ψ, TDVP2(); current_time=current_time)

        if store_state0
            MPSTimeEvolution.printoutput_data(io_handle, cb, ψ; psi0=ψ0, kwargs...)
        else
            MPSTimeEvolution.printoutput_data(io_handle, cb, ψ; kwargs...)
        end
        MPSTimeEvolution.printoutput_ranks(ranks_handle, cb, ψ)
    end

    ranges = rangepart(N)
    @assert reduce(vcat, rangepart(N)) == 1:N
    partn = procrank+1

    # Each worker has its ProjMPO object (each one independent from the others).
    # Initialise the ProjMPOs, while each worker still has the full MPS.
    # We need to do this now because each process will only get a chunk of the MPS tensors
    # inside the tdvp2_parallel_sweep_4p function, and will not be able to re-position the
    # ProjMPO outsite the bounds of its chunk.  At the beginning of the evolution the
    # environments in the ProjMPOs need to be generated from scratch, involving all tensors
    # of the MPS.
    set_nsite!(PH, 1)
    if partn == 1
        position!(PH, ψ, first(ranges[partn]))
    elseif partn == 2
        position!(PH, ψ, 1)
        position!(PH, ψ, last(ranges[partn]))
    elseif partn == 3
        position!(PH, ψ, N)
        position!(PH, ψ, first(ranges[partn]))
    elseif partn == 4
        position!(PH, ψ, last(ranges[partn]))
    end

    for s in 1:nsteps
        ψ = begin  # Maybe this could be a function?
            # Partition the MPS into four chunks. The bond tensor between two chunks gets
            # assigned to the chunk on its left.
            r = ranges[partn]
            st = OffsetVector(site_tensors(ψ)[r], r)

            r = if partn < 4
                ranges[partn]
            else
                ranges[partn][1:(end - 1)]
            end
            # Maybe we'd get the same result by splitting rangepart(N-1)?
            bt = OffsetVector(bond_tensors(ψ)[r], r)

            # Parallel sweeps
            st, bt, PH = tdvp2_parallel_sweep_4p(
                Val(partn),
                comm,
                st,
                bt,
                PH,
                dt;
                maxdim=maxdim,
                cutoff=cutoff,
                current_time=current_time,
            )
            MPI.Barrier(comm)  # Wait for everyone to finish...

            @debug "Step $s complete. Gathering data from workers..."
            sts = MPI.gather(parent(st), comm; root=rootrank)
            bts = MPI.gather(parent(bt), comm; root=rootrank)

            # Recompose the MPS so that we can compute expectation values later.
            # return
            if isroot
                InverseCanonicalMPS(
                    reduce(vcat, sts),
                    OffsetVector([ITensor(1.0); reduce(vcat, bts); ITensor(1.0)], 0:N),
                )
            else
                nothing
            end
        end
        isroot && @debug "Broadcasting the MPS to all workers..."
        ψ = MPI.bcast(ψ, comm; root=rootrank)
        # Broadcast the updates MPS to all workers, so that they can use it when the new
        # step begins.

        current_time += dt
        stime = 0.0  # FIXME

        if isroot
            @debug "Computing expectation values at t = $current_time."
            MPSTimeEvolution.apply!(cb, ψ, TDVP2(); current_time=current_time)
            MPSTimeEvolution.compute_norm!(cb, ψ, TDVP2(); current_time=current_time)

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
                MPSTimeEvolution.printoutput_stime(times_handle, 0.0)
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
