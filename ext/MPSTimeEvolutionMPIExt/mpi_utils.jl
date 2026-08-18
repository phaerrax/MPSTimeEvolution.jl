# A struct representing a chunk of an InverseCanonicaMPS object, used in the parallel TDVP2
# algorithm to group relevant variables together.
mutable struct MPSPartition
    site_tensors::OffsetVector{ITensor}
    bond_tensors::OffsetVector{ITensor}
    PH::ProjMPO
    range::UnitRange{Int}
    rank::Int
    norm::Float64   # this rank's copy, kept in sync via allreduce
end

rank(partition::MPSPartition) = partition.rank

function ITensorMPS.linkdims(partition::MPSPartition)
    return [
        dim(commonind(partition.site_tensors[j], partition.bond_tensors[j])) for
        j in eachindex(partition.bond_tensors)
    ]
end

function ITensorMPS.maxlinkdim(partition::MPSPartition)
    md = 1
    for j in eachindex(partition.bond_tensors)
        l = commonind(partition.site_tensors[j], partition.bond_tensors[j])
        linkdim = isnothing(l) ? 1 : dim(l)
        md = max(md, linkdim)
    end
    return md
end

function MPSPartition(
    ψ::InverseCanonicalMPS, H::MPO, partition_n, comm; site_partitions=rangepart(nsites(ψ))
)
    # We do the computations on worker 0, regardless of whether it is the root process or
    # not (it's easier this way).
    N = nsites(ψ)
    procrank = MPI.Comm_rank(comm)

    # 1. Partition the MPS into chunks.
    # The bond tensor between two consecutive chunks is assigned to the chunk on its left.
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

    # 2. Initialise the ProjMPOs.
    # Each worker will have its own ProjMPO object (independent from the others).
    # We need to create the ProjMPOs now because each process will only get a chunk of the
    # MPS tensors inside the tdvp2_parallel_sweep_4p function, and will not be able to
    # re-position the ProjMPO outsite the bounds of its chunk.  At the beginning of the
    # evolution the environments in the ProjMPOs need to be generated from scratch,
    # involving all tensors of the MPS.

    # Odd partitions (we start counting from one!) start sweeping left to right.
    # This means that they need the environment on the left of the leftmost MPS site of the
    # partition. Then, as they move rightwards, they will progressively create the missing
    # left environments. Right environments, instead, must already be there for all sites.
    # Even partitions work analogously.
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

    return MPSPartition(st, bt, PH, r, partition_n, norm(ψ))
end

function gather_to_mps(partition::MPSPartition, comm; root)
    sts = MPI.gather(parent(partition.site_tensors), comm; root)
    bts = MPI.gather(parent(partition.bond_tensors), comm; root)

    return if MPI.Comm_rank(comm) == root
        @debug "Gathering data from workers..."
        # Put the MPS chunks back together so that we can compute expectation values later.
        site_tensors = reduce(vcat, sts)
        bond_tensors = OffsetVector(
            [ITensor(1.0); reduce(vcat, bts); ITensor(1.0)], 0:length(site_tensors)
        )
        InverseCanonicalMPS(site_tensors, bond_tensors, partition.norm)
    else
        InverseCanonicalMPS()  # for type-stability purposes only
    end
end

# New `apply!` method that uses MPS partitions instead of a whole InverseCanonicaMPS.
function MPSTimeEvolution.apply!(
    cb::ExpValueCallback,
    partition::MPSPartition,
    alg::MPSTimeEvolution.TDVP2,
    comm;
    sweepend,
    current_time,
    root,
)
    # We perform measurements only at the end of a sweep and at measurement steps.
    # Only in this case we collect the partitons back into a proper InverseCanonicaMPS.
    if sweepend
        on_schedule = MPSTimeEvolution.register_time!(cb, current_time)
        if on_schedule
            isroot=(MPI.Comm_rank(comm) == root)
            isroot && @debug "Computing expectation values at t = $current_time"
            ψ = gather_to_mps(partition, comm; root)
            if isroot
                MPSTimeEvolution.apply!(cb, ψ, alg; current_time, sweepend)
                MPSTimeEvolution.compute_normalization!(cb, ψ, alg; current_time)
            end
        end
    end

    return nothing
end

# Ugly hack to use strings as tags for MPI messages.
# TODO I checked that there are no conflicts for the six strings we use:
#
#   julia> strs = ["betaL", "betaR", "gammaL", "gammaR", "PsiL", "PsiR", "V"]
#   julia> intcode.(strs)
#   6-element Vector{Int64}:
#    488
#    494
#    591
#    597
#    376
#    382
#    86
#
# but we need to find something better...
intcode(str)::Int = sum(convert(Int, c) for c in str)

function MPSTimeEvolution.printoutput_data(
    io_handle,
    cb,
    partition::MPSPartition,
    comm;
    root,
    store_psi0=false,
    psi0=nothing,
    kwargs...,
)
    isroot = (MPI.Comm_rank(comm) == root)
    if !isnothing(io_handle)
        if isroot
            results = MPSTimeEvolution.measurements(cb)
            data = [last(MPSTimeEvolution.measurement_ts(cb))]
            for opname in sort(collect(keys(results)))
                x = last(results[opname])
                push!(data, real(x), imag(x))
            end
        end

        if store_psi0 && !isnothing(psi0)
            psi = gather_to_mps(partition, comm; root)
            if isroot
                overlap = dot(psi0, psi)
                push!(data, real(overlap), imag(overlap))
            end
        end

        if isroot
            n = last(MPSTimeEvolution.measurements_norm(cb))
            push!(data, real(n), imag(n))

            println(io_handle, join(data, ","))
            flush(io_handle)
        end
    end

    return nothing
end

function MPSTimeEvolution.printoutput_ranks(
    ranks_handle, cb, partition::MPSPartition, comm; root
)
    isroot = (MPI.Comm_rank(comm) == root)
    if !isnothing(ranks_handle)
        current_time = last(MPSTimeEvolution.measurement_ts(cb))
        bonddims = MPI.reduce(linkdims(partition), vcat, comm; root)

        if isroot
            println(ranks_handle, current_time, ",", join(bonddims, ","))
            flush(ranks_handle)
        end
    end

    return nothing
end

function MPSTimeEvolution.simulationinfo(
    partition::MPSPartition, current_time, stime, comm; digits=3
)
    # The maximum bond dimension and the MPS size are global quantities that involve all
    # partitions, thus we need to invoke MPI to compute them. This means that they can't be
    # computed lazily inside the returned closure (which is only ever invoked by root, via
    # ProgressMeter's `showvalues`). That's not a problem, though, because we are just
    # dealing with a bunch of scalar quantities that are cheap to compute.
    local_maxbonddim = maxlinkdim(partition)
    global_maxbonddim = MPI.Allreduce(local_maxbonddim, max, comm)
    total_size = MPI.Allreduce(Base.summarysize(partition), +, comm)

    return () -> [
        ("t", current_time),
        ("Maximum bond dimension", global_maxbonddim),
        ("Wall time / step", round(stime; digits=digits)),
        ("MPS size / MiB", round(total_size / (2^20); digits=digits)),
        # ↖ amount of memory, in bytes, used by all unique objects reachable from x
        ("GC live / MiB", round(Base.gc_live_bytes() / (2^20); digits=digits)),
        # ↖ total size of objects currently in memory
        ("JIT / MiB", round(Base.jit_total_bytes() / (2^20); digits=digits)),
        # ↖ total amount allocated by the just-in-time compiler
        ("Max. RSS / GiB", round(Sys.maxrss() / (2^30); digits=digits)),
        # ↖ maximum resident set size utilized (i.e. the maximum amount of memory
        # that the job may occupy)
    ]
end

function MPSTimeEvolution.writeheaders_data(io_file, comm::MPI.Comm, cb; root, kwargs...)
    io_handle = nothing
    isroot=(MPI.Comm_rank(comm) == root)
    if !isnothing(io_file)
        io_handle = open(io_file, "w")

        columnheaders = ["time"]

        res = MPSTimeEvolution.measurements(cb)
        for op in sort(collect(keys(res)))
            push!(
                columnheaders,
                MPSTimeEvolution.name(op) * "_re",
                MPSTimeEvolution.name(op) * "_im",
            )
        end

        if get(kwargs, :store_psi0, false)
            push!(columnheaders, "overlap_re", "overlap_im")
        end

        push!(columnheaders, "Norm_re", "Norm_im")

        isroot && println(io_handle, join(columnheaders, ","))
    end

    return io_handle
end

function MPSTimeEvolution.writeheaders_ranks(ranks_file, comm::MPI.Comm, Ns::Int...; root)
    ranks_handle = nothing
    isroot=(MPI.Comm_rank(comm) == root)
    if !isnothing(ranks_file)
        ranks_handle = open(ranks_file, "w")

        columnheaders = ["time"]

        for N in Ns
            append!(columnheaders, string.(1:(N - 1)))
        end

        isroot && println(ranks_handle, join(columnheaders, ","))
    end

    return ranks_handle
end
