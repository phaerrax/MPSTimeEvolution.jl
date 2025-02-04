using CSV

# We define a small SIAM system and evolve it with our different TDVP methods.
# We choose a single fermion interacting with just a single environment: we need to focus
# on the correctness of the TDVP functions, mostly from a syntactic point of view. It
# doesn't have to be a physically interesting, or correct, system.

function siam_tdvp1(; dt, tmax, N, check_sites)
    sites = siteinds("Fermion", N + 1)

    init(n) = n == 1 ? "Occ" : "Emp"
    state_0 = MPS(sites, init)
    maxbonddim = maximum(maxlinkdims(state_0))
    state_0 = enlargelinks(state_0, maxbonddim; ref_state=init)

    h = spin_chain([0.5; fill(0.25, N)], fill(1, N), sites)
    H = MPO(h, sites)

    operators = [LocalOperator(Dict(n => "N")) for n in check_sites]
    cb = ExpValueCallback(operators, sites, dt)

    tmpfile = tempname()

    tdvp1!(
        state_0,
        H,
        dt,
        tmax;
        callback=cb,
        progress=false,
        io_file=tmpfile,
        io_ranks="/dev/null",
        io_times="/dev/null",
    )

    f = CSV.File(tmpfile)
    results = [complex.(f["N{$n}_re"], f["N{$n}_im"]) for n in check_sites]
    return f["time"], results...
end

function siam_tdvp1vec(; dt, tmax, N, check_sites)
    sites = siteinds("vFermion", N + 1)

    state_0 = MPS(sites, n -> n == 1 ? "Occ" : "Emp")
    maxbonddim = maximum(maxlinkdims(state_0))
    state_0 = enlargelinks(state_0, maxbonddim; ref_state=n -> n == 1 ? "Occ" : "Emp")

    ℓ = spin_chain([0.5; fill(0.25, N)], fill(1, N), sites)
    L = MPO(ℓ, sites)

    operators = [LocalOperator(Dict(n => "vN")) for n in check_sites]
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
    results = [complex.(f["vN{$n}_re"], f["vN{$n}_im"]) ./ f["Norm"] for n in check_sites]
    return f["time"], results...
end

function siam_tdvp1vec_superfermions(; dt, tmax, N, check_sites)
    sf_index(n) = 2n - 1
    # (2n-1)-th site with superfermions == n-th site with traditional fermions

    sites = siteinds("Fermion", 2(N + 1))

    init(n) = div(n + 1, 2) == 1 ? "Occ" : "Emp"
    state_0 = MPS(sites, init)
    maxbonddim = maximum(maxlinkdims(state_0))
    state_0 = enlargelinks(state_0, maxbonddim; ref_state=init)

    ℓ =
        spin_chain([0.5; fill(0.25, N)], fill(1, N), sites[1:2:end]) -
        spin_chain([0.5; fill(0.25, N)], fill(1, N), sites[2:2:end])
    L = MPO(-im * ℓ, sites)

    operators = [LocalOperator(Dict(n => "N")) for n in sf_index.(check_sites)]
    cb = SuperfermionCallback(operators, sites, dt)

    tmpfile = tempname()

    tdvp1vec!(
        state_0,
        L,
        dt,
        tmax;
        superfermions=true,
        callback=cb,
        progress=false,
        io_file=tmpfile,
        io_ranks="/dev/null",
        io_times="/dev/null",
    )

    f = CSV.File(tmpfile)
    results = [
        complex.(f["N{$n}_re"], f["N{$n}_im"]) ./ f["Norm"] for n in sf_index.(check_sites)
    ]
    return f["time"], results...
end

function siam_adjtdvp1vec(; dt, tmax, N, check_sites)
    sites = siteinds("vFermion", N + 1)

    state_0 = MPS(sites, n -> n == 1 ? "Occ" : "Emp")

    adjℓ = spin_chain′([0.5; fill(0.25, N)], fill(1, N), sites)
    adjL = MPO(adjℓ, sites)

    tmpfile = tempname()

    results = []
    time = Float64[]
    for n in check_sites
        init = [j == n ? "vN" : "vId" for j in eachindex(sites)]
        targetop = MPS(ComplexF64, sites, init)
        maxbonddim = maximum(maxlinkdims(targetop))
        targetop = enlargelinks(targetop, maxbonddim; ref_state=init)

        adjtdvp1vec!(
            targetop,
            state_0,
            adjL,
            dt,
            tmax,
            dt;
            io_file=tmpfile,
            progress=false,
            io_ranks="/dev/null",
            io_times="/dev/null",
        )

        f = CSV.File(tmpfile)
        if !isempty(time) && time != f["time"]
            error("time series in adjtdvp1vec do not match.")
        else
            time = f["time"]
        end
        push!(results, complex.(f["exp_val_real"], f["exp_val_imag"]))
    end

    return time, results...
end

# This tests pushes the bond dimension to the maximum admitted by the sizes of the system,
# so it's best to keep N relatively low so that the computation doesn't get too heavy.
function siam_compare_tdvp(; dt=0.01, tmax=0.5, N=4)
    s = [1, 2, 3]
    res_tdvp1 = siam_tdvp1(; dt=dt, tmax=tmax, N=N, check_sites=s)
    res_tdvp1vec = siam_tdvp1vec(; dt=dt, tmax=tmax, N=N, check_sites=s)
    res_tdvp1vec_superfermions = siam_tdvp1vec_superfermions(;
        dt=dt, tmax=tmax, N=N, check_sites=s
    )
    res_adjtdvp1vec = siam_adjtdvp1vec(; dt=dt, tmax=tmax, N=N, check_sites=s)
    return all(res_tdvp1 .≈ res_tdvp1vec .≈ res_tdvp1vec_superfermions .≈ res_adjtdvp1vec)
end
