# TDVP2 with MPI example script

using MPSTimeEvolution, ITensors, ITensorMPS
using MPI
MPI.Init()

let comm = MPI.COMM_WORLD
    prank = MPI.Comm_rank(comm)
    isroot = prank == 0

    N = 20

    # Use bcast otherwise the indices in different workers will have different IDs and will
    # not match in contractions.
    s = MPI.bcast(siteinds("S=1/2", N), comm)

    maxdim = 40
    cutoff = 1e-6
    ψ = MPI.bcast(InverseCanonicalMPS(s, n -> n == 1 ? "Up" : "Dn"), comm)
    ψ′ = isroot ? InverseCanonicalMPS(s, n -> n == 1 ? "Up" : "Dn") : nothing

    h = OpSum();
    for n in 1:(N - 1)
        h += -0.5, "σz", n, "σz", n+1
    end

    for n in 1:N
        h += "σx", n
    end

    H = MPI.bcast(MPO(h, s), comm)
    # We rename the link indices of the Hamiltonian MPO so that we can more easily tell them
    # apart from the link indices of the MPS within the ProjMPO objects.
    for j in eachindex(H)
        replacetags!(H[j], "l=$j" => "h=$j")
        replacetags!(H[j], "l=$(j-1)" => "h=$(j-1)")
    end

    dt = 0.01
    tmax = 0.5

    cb = ExpValueCallback("σz(1,2,3,4)", s, dt)
    cb′ = ExpValueCallback("σz(1,2,3,4)", s, dt)

    io_file, _ = mktemp(; cleanup=false)
    io_file′, _ = mktemp(; cleanup=false)

    tdvp2_parallel!(
        ψ,
        H,
        dt,
        tmax;
        rootrank=0,
        comm=comm,
        maxdim=maxdim,
        cutoff=cutoff,
        progress=false,
        io_file=io_file,
        callback=cb,
    )

    if isroot
        tdvp2!(
            ψ′,
            H,
            dt,
            tmax;
            progress=false,
            maxdim=maxdim,
            cutoff=cutoff,
            io_file=io_file′,
            callback=cb′,
        )
    end
    return nothing
end
