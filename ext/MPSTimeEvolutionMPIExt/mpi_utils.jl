function bcast(obj, root::Integer, comm::MPI.Comm)
    isroot = MPI.Comm_rank(comm) == root
    count = Ref{Clong}()
    if isroot
        buf = MPI.serialize(obj)
        count[] = length(buf)
    end
    MPI.Bcast!(count, root, comm)
    if !isroot
        buf = Array{UInt8}(undef, count[])
    end
    MPI.Bcast!(buf, root, comm)
    if !isroot
        obj = MPI.deserialize(buf)
    end
    return obj
end

function gather(obj, root::Integer, comm::MPI.Comm)
    # From ITensorParallel.jl:
    # https://github.com/ITensor/ITensorParallel.jl/blob/f03529b1/src/mpi_extensions.jl#L2
    isroot = MPI.Comm_rank(comm) == root
    count = Ref{Clong}()
    buf = MPI.serialize(obj)
    count[] = length(buf)
    counts = MPI.Gather(count[], root, comm)
    if isroot
        rbuf = Array{UInt8}(undef, reduce(+, counts))
        rbuf = MPI.VBuffer(rbuf, counts)
    else
        rbuf = nothing
    end
    MPI.Gatherv!(buf, rbuf, root, comm)
    if isroot
        objs = []
        for v in 1:length(rbuf.counts)
            endind = v == length(rbuf.counts) ? length(rbuf.data) : rbuf.displs[v + 1] + 1
            startind = rbuf.displs[v] + 1
            push!(objs, MPI.deserialize(rbuf.data[startind:endind]))
        end
    else
        objs = nothing
    end
    return objs
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
