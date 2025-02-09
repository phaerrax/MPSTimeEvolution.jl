using CSV

function siam_tdvp1vec_single_mpo(; dt, tmax, N)
    init(n) = isodd(n) ? "Occ" : "Emp"
    N = max(N, 5)
    sites = siteinds("vFermion", N + 1)

    state_0 = MPS(sites, init)
    maxbonddim = maximum(maxlinkdims(state_0))
    state_0 = enlargelinks(state_0, maxbonddim; ref_state=init)

    freqs = [0.5; fill(0.25, N)]
    coups = fill(1, N)
    ℓ = spin_chain(freqs, coups, sites)
    L = MPO(ℓ, sites)

    operators = [LocalOperator(Dict(1 => "N")), LocalOperator(Dict(5 => "N"))]
    cb = ExpValueCallback(operators, sites, dt)

    tmpfile = tempname()

    tdvp1vec!(
        state_0,
        L,
        dt,
        tmax;
        callback=cb,
        progress=false,
        io_file=tmpfile,
        io_ranks="/dev/null",
        io_times="/dev/null",
    )

    f = CSV.File(tmpfile)
    return f["time"],
    complex.(f["N{1}_re"], f["N{1}_im"]) ./ f["Norm"],
    complex.(f["N{5}_re"], f["N{5}_im"]) ./ f["Norm"]
end

function siam_tdvp1vec_two_mpos(; dt, tmax, N)
    init(n) = isodd(n) ? "Occ" : "Emp"
    N = max(N, 5)
    r = div(N, 2)
    sites = siteinds("vFermion", N + 1)

    state_0 = MPS(sites, init)
    maxbonddim = maximum(maxlinkdims(state_0))
    state_0 = enlargelinks(state_0, maxbonddim; ref_state=init)

    freqs1 = [0.25; fill(0.125, N)]
    coups1 = [fill(0.75, N - r); fill(0.25, r)]
    ℓ1 = spin_chain(freqs1, coups1, sites)

    freqs2 = [0.25; fill(0.125, N)]
    coups2 = [fill(0.25, N - r); fill(0.75, r)]
    ℓ2 = spin_chain(freqs2, coups2, sites)

    # ℓ1 + ℓ2 == ℓ from the other function above

    L1 = MPO(ℓ1, sites)
    L2 = MPO(ℓ2, sites)

    operators = [LocalOperator(Dict(1 => "N")), LocalOperator(Dict(5 => "N"))]
    cb = ExpValueCallback(operators, sites, dt)

    tmpfile = tempname()

    tdvp1vec!(
        state_0,
        [L1, L2],
        dt,
        tmax;
        callback=cb,
        progress=false,
        io_file=tmpfile,
        io_ranks="/dev/null",
        io_times="/dev/null",
    )

    f = CSV.File(tmpfile)
    return f["time"],
    complex.(f["N{1}_re"], f["N{1}_im"]) ./ f["Norm"],
    complex.(f["N{5}_re"], f["N{5}_im"]) ./ f["Norm"]
end

# This tests pushes the bond dimension to the maximum admitted by the sizes of the system,
# so it's best to keep N relatively low so that the computation doesn't get too heavy.
function siam_compare_tdvp_with_sum(; dt=0.01, tmax=0.5, N=6)
    res_tdvp1vec = siam_tdvp1vec_single_mpo(; dt=dt, tmax=tmax, N=N)
    res_tdvp1vec_sum = siam_tdvp1vec_two_mpos(; dt=dt, tmax=tmax, N=N)
    return all(res_tdvp1vec .≈ res_tdvp1vec_sum)
end
