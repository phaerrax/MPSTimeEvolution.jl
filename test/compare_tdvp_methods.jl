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

    site_pairs = [
        (check_sites[j], check_sites[i]) for i in eachindex(check_sites) for j in 1:(i - 1)
    ]
    jws(n1, n2) = (i => "F" for i in (n1 + 1):(n2 - 1))  # Jordan-Wigner string
    operators = [
        [LocalOperator(n => "n") for n in check_sites]
        [
            LocalOperator((n1 => "a", jws(n1, n2)..., n2 => "adag")) for
            (n1, n2) in site_pairs
        ]
        [
            LocalOperator((n1 => "adag", jws(n1, n2)..., n2 => "a")) for
            (n1, n2) in site_pairs
        ]
    ]
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
    results = [
        complex.(f["$(l)_re"], f["$(l)_im"]) for
        l in MPSTimeEvolution.name.(MPSTimeEvolution.ops(cb))
    ]
    return f["time"], results...
end

function siam_tdvp1_with_qns(; dt, tmax, N, check_sites)
    sites = siteinds("Fermion", N + 1; conserve_nf=true)

    init(n) = n == 1 ? "Occ" : "Emp"
    state_0 = MPS(sites, init)
    maxbonddim = maximum(maxlinkdims(state_0))
    state_0 = enlargelinks(state_0, maxbonddim; ref_state=init)

    h = spin_chain([0.5; fill(0.25, N)], fill(1, N), sites)
    H = MPO(h, sites)

    site_pairs = [
        (check_sites[j], check_sites[i]) for i in eachindex(check_sites) for j in 1:(i - 1)
    ]
    jws(n1, n2) = (i => "F" for i in (n1 + 1):(n2 - 1))  # Jordan-Wigner string
    operators = [
        [LocalOperator(n => "n") for n in check_sites]
        [
            LocalOperator((n1 => "a", jws(n1, n2)..., n2 => "adag")) for
            (n1, n2) in site_pairs
        ]
        [
            LocalOperator((n1 => "adag", jws(n1, n2)..., n2 => "a")) for
            (n1, n2) in site_pairs
        ]
    ]
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
    results = [
        complex.(f["$(l)_re"], f["$(l)_im"]) for
        l in MPSTimeEvolution.name.(MPSTimeEvolution.ops(cb))
    ]
    return f["time"], results...
end

function siam_tdvp1vec(; dt, tmax, N, check_sites)
    sites = siteinds("vFermion", N + 1)

    state_0 = MPS(sites, n -> n == 1 ? "Occ" : "Emp")
    maxbonddim = maximum(maxlinkdims(state_0))
    state_0 = enlargelinks(state_0, maxbonddim; ref_state=n -> n == 1 ? "Occ" : "Emp")

    ℓ = spin_chain([0.5; fill(0.25, N)], fill(1, N), sites)
    L = MPO(ℓ, sites)

    site_pairs = [
        (check_sites[j], check_sites[i]) for i in eachindex(check_sites) for j in 1:(i - 1)
    ]
    jws(n1, n2) = (i => "vF" for i in (n1 + 1):(n2 - 1))  # Jordan-Wigner string
    operators = [
        # Remember to use the adjoints! This is a small flaw in the design of the callback
        # object. It should call the adjoints of the operators by itself.
        [LocalOperator(n => "vN") for n in check_sites]
        [
            LocalOperator((n1 => "vAdag", jws(n1, n2)..., n2 => "vA")) for
            (n1, n2) in site_pairs
        ]
        [
            LocalOperator((n1 => "vA", jws(n1, n2)..., n2 => "vAdag")) for
            (n1, n2) in site_pairs
        ]
    ]
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
    results = [
        complex.(f["$(l)_re"], f["$(l)_im"]) for
        l in MPSTimeEvolution.name.(MPSTimeEvolution.ops(cb))
    ]
    return f["time"], results...
end

function siam_tdvp1vec_superfermions(; dt, tmax, N, check_sites)
    sf_index(n) = 2n - 1
    # (2n-1)-th site with superfermions == n-th site with traditional fermions

    sites = siteinds("Fermion", 2(N + 1); conserve_nfparity=true)

    init(n) = div(n + 1, 2) == 1 ? "Occ" : "Emp"
    state_0 = MPS(sites, init)
    maxbonddim = maximum(maxlinkdims(state_0))
    state_0 = enlargelinks(state_0, maxbonddim; ref_state=init)

    ℓ =
        spin_chain([0.5; fill(0.25, N)], fill(1, N), sites[1:2:end]) -
        spin_chain([0.5; fill(0.25, N)], fill(1, N), sites[2:2:end])
    L = MPO(-im * ℓ, sites)

    sf_check_sites = sf_index.(check_sites)
    site_pairs = [
        (check_sites[j], check_sites[i]) for i in eachindex(check_sites) for j in 1:(i - 1)
    ]
    sf_site_pairs = [sf_index.(p) for p in site_pairs]
    jws(n1, n2) = (i => "F" for i in (n1 + 1):(n2 - 1))  # Jordan-Wigner string
    operators = [
        # We don't need to use the adjoints here because SuperfermionCallback is coded
        # properly: it does the adjunction by itself already.
        [LocalOperator(n => "n") for n in sf_check_sites]
        [
            LocalOperator((n1 => "a", jws(n1, n2)..., n2 => "adag")) for
            (n1, n2) in sf_site_pairs
        ]
        [
            LocalOperator((n1 => "adag", jws(n1, n2)..., n2 => "a")) for
            (n1, n2) in sf_site_pairs
        ]
    ]
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
        complex.(f["$(l)_re"], f["$(l)_im"]) for
        l in MPSTimeEvolution.name.(MPSTimeEvolution.ops(cb))
    ]
    return f["time"], results...
end

function siam_adjtdvp1vec(; dt, tmax, N, check_sites)
    sites = siteinds("vFermion", N + 1)

    state_0 = MPS(sites, n -> n == 1 ? "Occ" : "Emp")

    adjℓ = spin_chain′([0.5; fill(0.25, N)], fill(1, N), sites)
    adjL = MPO(adjℓ, sites)

    site_pairs = [
        (check_sites[j], check_sites[i]) for i in eachindex(check_sites) for j in 1:(i - 1)
    ]
    jws(n1, n2) = (i => "vF" for i in (n1 + 1):(n2 - 1))  # Jordan-Wigner string
    operators = [
        [LocalOperator(n => "vN") for n in check_sites]
        [
            LocalOperator((n1 => "vA", jws(n1, n2)..., n2 => "vAdag")) for
            (n1, n2) in site_pairs
        ]
        [
            LocalOperator((n1 => "vAdag", jws(n1, n2)..., n2 => "vA")) for
            (n1, n2) in site_pairs
        ]
    ]
    cb = ExpValueCallback(operators, sites, dt)  # needed for ordering the operators

    tmpfile = tempname()

    results = []
    time = Float64[]
    for l in MPSTimeEvolution.ops(cb)
        targetop = MPSTimeEvolution.mps(sites, l)
        maxbonddim = maximum(maxlinkdims(targetop))
        growMPS!(targetop, maxbonddim)

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
    s = [1, 3]
    res_tdvp1 = siam_tdvp1(; dt=dt, tmax=tmax, N=N, check_sites=s)
    res_tdvp1_with_qns = siam_tdvp1(; dt=dt, tmax=tmax, N=N, check_sites=s)
    res_tdvp1vec = siam_tdvp1vec(; dt=dt, tmax=tmax, N=N, check_sites=s)
    res_tdvp1vec_superfermions = siam_tdvp1vec_superfermions(;
        dt=dt, tmax=tmax, N=N, check_sites=s
    )
    res_adjtdvp1vec = siam_adjtdvp1vec(; dt=dt, tmax=tmax, N=N, check_sites=s)

    return all(
        res_tdvp1 .≈
        res_tdvp1vec .≈
        res_tdvp1_with_qns .≈
        res_tdvp1vec_superfermions .≈
        res_adjtdvp1vec,
    )
end
