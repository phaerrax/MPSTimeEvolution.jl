using IterTools: partition
using CSV

# We define a small SIAM system and evolve it with our different TDVP methods.
# We choose a single fermion interacting with just a single environment: we need to focus
# on the correctness of the TDVP functions, mostly from a syntactic point of view. It
# doesn't have to be a physically interesting, or correct, system.

function siam_tdvp1(; dt, tmax, N)
    N = max(N, 5)
    sites = siteinds("Fermion", N + 1)

    state_0 = MPS(sites, n -> n == 1 ? "Occ" : "Emp")
    maxbonddim = maximum(maxlinkdims(state_0))
    state_0 = enlargelinks(state_0, maxbonddim; ref_state=n -> n == 1 ? "Occ" : "Emp")

    h = spin_chain([0.5; fill(0.25, N)], fill(1, N), sites)
    H = MPO(h, sites)

    operators = [LocalOperator(Dict(1 => "N")), LocalOperator(Dict(5 => "N"))]
    cb = ExpValueCallback(operators, sites, dt)

    tmpfile = tempname()

    tdvp1!(
        state_0,
        H,
        dt,
        tmax;
        hermitian=true,
        normalize=false,
        callback=cb,
        progress=true,
        store_psi0=false,
        io_file=tmpfile,
        io_ranks="/dev/null",
        io_times="/dev/null",
    )

    f = CSV.File(tmpfile)
    return f["time"],
    complex.(f["N{1}_re"], f["N{1}_im"]),
    complex.(f["N{5}_re"], f["N{5}_im"])
end

function siam_tdvp1vec(; dt, tmax, N)
    N = max(N, 5)
    sites = siteinds("vFermion", N + 1)

    state_0 = MPS(sites, n -> n == 1 ? "Occ" : "Emp")
    maxbonddim = maximum(maxlinkdims(state_0))
    state_0 = enlargelinks(state_0, maxbonddim; ref_state=n -> n == 1 ? "Occ" : "Emp")

    ℓ = spin_chain([0.5; fill(0.25, N)], fill(1, N), sites)
    L = MPO(ℓ, sites)

    operators = [LocalOperator(Dict(1 => "vN")), LocalOperator(Dict(5 => "vN"))]
    cb = ExpValueCallback(operators, sites, dt)

    tmpfile = tempname()

    tdvp1vec!(
        state_0,
        L,
        dt,
        tmax,
        sites;
        hermitian=false,
        normalize=false,
        callback=cb,
        progress=true,
        store_psi0=false,
        io_file=tmpfile,
        io_ranks="/dev/null",
        io_times="/dev/null",
    )

    f = CSV.File(tmpfile)
    return f["time"],
    complex.(f["vN{1}_re"], f["vN{1}_im"]) ./ f["Norm"],
    complex.(f["vN{5}_re"], f["vN{5}_im"]) ./ f["Norm"]
end

function siam_adjtdvp1vec(; dt, tmax, N)
    N = max(N, 5)
    sites = siteinds("vFermion", N + 1)

    state_0 = MPS(sites, n -> n == 1 ? "Occ" : "Emp")

    adjℓ = spin_chain′([0.5; fill(0.25, N)], fill(1, N), sites)
    adjL = MPO(adjℓ, sites)

    tmpfile = tempname()

    targetop = MPS(ComplexF64, sites, [n == 1 ? "vN" : "vId" for n in eachindex(sites)])
    maxbonddim = maximum(maxlinkdims(targetop))
    targetop = enlargelinks(
        targetop, maxbonddim; ref_state=[n == 1 ? "vN" : "vId" for n in eachindex(sites)]
    )

    adjtdvp1vec!(
        targetop,
        state_0,
        adjL,
        dt,
        tmax,
        dt,
        sites;
        hermitian=false,
        normalize=false,
        progress=true,
        store_psi0=false,
        io_file=tmpfile,
        io_ranks="/dev/null",
        io_times="/dev/null",
    )

    f = CSV.File(tmpfile)
    n1 = complex.(f["exp_val_real"], f["exp_val_imag"])

    targetop = MPS(ComplexF64, sites, [n == 5 ? "vN" : "vId" for n in eachindex(sites)])
    maxbonddim = maximum(maxlinkdims(targetop))
    targetop = enlargelinks(
        targetop, maxbonddim; ref_state=[n == 5 ? "vN" : "vId" for n in eachindex(sites)]
    )

    adjtdvp1vec!(
        targetop,
        state_0,
        adjL,
        dt,
        tmax,
        dt,
        sites;
        hermitian=false,
        normalize=false,
        progress=true,
        store_psi0=false,
        io_file=tmpfile,
        io_ranks="/dev/null",
        io_times="/dev/null",
    )

    f = CSV.File(tmpfile)
    n5 = complex.(f["exp_val_real"], f["exp_val_imag"])

    return f["time"], n1, n5
end

# This tests pushes the bond dimension to the maximum admitted by the sizes of the system,
# so it's best to keep N relatively low so that the computation doesn't get too heavy.
function siam_compare_tdvp(; dt=0.01, tmax=0.5, N=6)
    res_tdvp1 = siam_tdvp1(; dt=dt, tmax=tmax, N=N)
    res_tdvp1vec = siam_tdvp1vec(; dt=dt, tmax=tmax, N=N)
    res_adjtdvp1vec = siam_adjtdvp1vec(; dt=dt, tmax=tmax, N=N)
    return all(res_tdvp1 .≈ res_tdvp1vec .≈ res_adjtdvp1vec)
end
