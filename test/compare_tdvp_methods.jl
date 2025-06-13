jws(n1, n2) = (i => "F" for i in (n1 + 1):(n2 - 1))  # Jordan-Wigner string

function itensors_tdvp(; dt, tmax, freqs, couplings, check_sites, init)
    N = length(freqs)
    sites = siteinds("Fermion", N)

    state_0 = MPS(sites, init)
    maxbonddim = maximum(maxlinkdims(state_0))
    state_0 = enlargelinks(state_0, maxbonddim; ref_state=init)
    h = spin_chain(freqs, couplings, sites)
    H = MPO(h, sites)

    site_pairs = [
        (check_sites[j], check_sites[i]) for i in eachindex(check_sites) for j in 1:(i - 1)
    ]
    cc_ops = [
        [
            begin
                os = OpSum()
                # In the other test functions below we measure, with our TDVP,
                #
                #   a(n1) ⊗ f(n1+1) ⊗ ⋯ ⊗ f(n2-1) ⊗ adag(n2),
                #   adag(n1) ⊗ f(n1+1) ⊗ ⋯ ⊗ f(n2-1) ⊗ a(n2)
                #
                # for all `n1 < n2` from `check_sites`. With the `c` operators we'd get
                #
                #   cdag(n1) c(n2) = adag(n1)f(n1) ⊗ f(n1+1) ⊗ ⋯ ⊗ f(n2-1) ⊗ adag(n2)
                #                  = adag(n1) ⊗ f(n1+1) ⊗ ⋯ ⊗ f(n2-1) ⊗ adag(n2),
                #   c(n1) cdag(n2) = a(n1)f(n1) ⊗ f(n1+1) ⊗ ⋯ ⊗ f(n2-1) ⊗ adag(n2)
                #                  = -a(n1) ⊗ f(n1+1) ⊗ ⋯ ⊗ f(n2-1) ⊗ adag(n2)
                #
                # so in order to have the same results we need an additional minus sign.
                os += (-1, "c", n1, "cdag", n2)
                MPO(os, sites)
            end for (n1, n2) in site_pairs
        ]
        [
            begin
                os = OpSum()
                os += ("cdag", n1, "c", n2)
                MPO(os, sites)
            end for (n1, n2) in site_pairs
        ]
    ]
    current_time(; current_time) = real(im * current_time)
    measure_n(; state) = expect(state, "n"; sites=check_sites)
    measure_cc(; state) = [inner(state', obs, state) for obs in cc_ops]

    obs = observer("t" => current_time, "n" => measure_n, "cc" => measure_cc)
    ITensorMPS.tdvp(
        H,
        -im * tmax,
        state_0;
        time_step=-im * dt,
        (step_observer!)=obs,
        outputlevel=0,
        updater_kwargs=(; tol=1e-14, krylovdim=30, maxiter=100),  # same as our defaults
    )

    # We also need to add the initial expectation values---ITensors' TDVP doesn't measure
    # them. (Maybe there's a way we can do this via the observer...)
    results = [
        [
            [expect(state_0, "n"; sites=check_sites[n]); [row[n] for row in obs.n]] for
            n in eachindex(check_sites)
        ]
        [
            [inner(state_0', cc_ops[n], state_0); [row[n] for row in obs.cc]] for
            n in eachindex(cc_ops)
        ]
    ]

    return [0.0; obs.t], results...
end

# We define a small SIAM system and evolve it with our different TDVP methods.
# We choose a single fermion interacting with just a single environment: we need to focus
# on the correctness of the TDVP functions, mostly from a syntactic point of view. It
# doesn't have to be a physically interesting, or correct, system.

function siam_tdvp1(; dt, tmax, freqs, couplings, check_sites, init)
    N = length(freqs)
    sites = siteinds("Fermion", N)

    state_0 = MPS(sites, init)
    maxbonddim = maximum(maxlinkdims(state_0))
    state_0 = enlargelinks(state_0, maxbonddim; ref_state=init)

    h = spin_chain(freqs, couplings, sites)
    H = MPO(h, sites)

    site_pairs = [
        (check_sites[j], check_sites[i]) for i in eachindex(check_sites) for j in 1:(i - 1)
    ]
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
        complex.(f["$(l)_re"], f["$(l)_im"]) ./ complex.(f["Norm_re"], f["Norm_im"]) for
        l in MPSTimeEvolution.name.(MPSTimeEvolution.ops(cb))
    ]
    return f["time"], results...
end

function siam_tdvp1_with_qns(; dt, tmax, freqs, couplings, check_sites, init)
    N = length(freqs)
    sites = siteinds("Fermion", N; conserve_nfparity=true)

    state_0 = MPS(sites, init)
    maxbonddim = maximum(maxlinkdims(state_0))
    state_0 = enlargelinks(state_0, maxbonddim; ref_state=init)

    h = spin_chain(freqs, couplings, sites)
    H = MPO(h, sites)

    site_pairs = [
        (check_sites[j], check_sites[i]) for i in eachindex(check_sites) for j in 1:(i - 1)
    ]
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
        complex.(f["$(l)_re"], f["$(l)_im"]) ./ complex.(f["Norm_re"], f["Norm_im"]) for
        l in MPSTimeEvolution.name.(MPSTimeEvolution.ops(cb))
    ]
    return f["time"], results...
end

function siam_tdvp1vec(; dt, tmax, freqs, couplings, check_sites, init)
    N = length(freqs)
    sites = siteinds("vFermion", N)

    state_0 = MPS(sites, init)
    maxbonddim = maximum(maxlinkdims(state_0))
    state_0 = enlargelinks(state_0, maxbonddim; ref_state=init)

    ℓ = spin_chain(freqs, couplings, sites)
    L = MPO(ℓ, sites)

    site_pairs = [
        (check_sites[j], check_sites[i]) for i in eachindex(check_sites) for j in 1:(i - 1)
    ]
    operators = [
        [LocalOperator(n => "N") for n in check_sites]
        [
            LocalOperator((n1 => "A", jws(n1, n2)..., n2 => "Adag")) for
            (n1, n2) in site_pairs
        ]
        [
            LocalOperator((n1 => "Adag", jws(n1, n2)..., n2 => "A")) for
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
        complex.(f["$(l)_re"], f["$(l)_im"]) ./ complex.(f["Norm_re"], f["Norm_im"]) for
        l in MPSTimeEvolution.name.(MPSTimeEvolution.ops(cb))
    ]
    return f["time"], results...
end

function invert_ancillary_states(states)
    if any(s -> s != "Occ" && s != "Emp", states)
        error("initial states must be either \"Occ\" or \"Emp\".")
    end

    @assert iseven(length(states))
    N = div(length(states), 2)

    # Invert initial state of ancillary sites
    for i in 2:2:2N
        if states[i] == "Occ"
            states[i] = "Emp"
        else  # states[i] == "Emp"
            states[i] = "Occ"
        end
    end
    return states
end

function siam_tdvp1vec_superfermions(; dt, tmax, freqs, couplings, check_sites, init)
    N = length(freqs)

    sites = siteinds("Fermion", 2N; conserve_nfparity=true)

    init_states_str = invert_ancillary_states(init.(_sf_translate_sites_inv.(1:2N)))
    state_0 = MPS(sites, init_states_str)
    maxbonddim = maximum(maxlinkdims(state_0))
    state_0 = enlargelinks(state_0, maxbonddim; ref_state=init)

    ℓ = OpSum()
    for j in 1:N
        ℓ += -im*freqs[j], "c† * c", _sf_translate_sites(j)
        ℓ += im*freqs[j], "c * c†", _sf_translate_sites(j)+1
        if _sf_translate_sites(j+1) ≤ 2N
            ℓ += -im*couplings[j],
            "c†", _sf_translate_sites(j), "c",
            _sf_translate_sites(j+1)
            ℓ += -im*couplings[j],
            "c†", _sf_translate_sites(j+1), "c",
            _sf_translate_sites(j)
        end
        if _sf_translate_sites(j+1)+1 ≤ 2N
            ℓ += im*couplings[j],
            "c", _sf_translate_sites(j)+1, "c†",
            _sf_translate_sites(j+1)+1
            ℓ += im*couplings[j],
            "c", _sf_translate_sites(j+1)+1, "c†",
            _sf_translate_sites(j)+1
        end
    end
    L = MPO(ℓ, sites)

    sf_check_sites = _sf_translate_sites.(check_sites)
    site_pairs = [
        (check_sites[j], check_sites[i]) for i in eachindex(check_sites) for j in 1:(i - 1)
    ]
    sf_site_pairs = [_sf_translate_sites.(p) for p in site_pairs]
    operators = [
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
        complex.(f["$(l)_re"], f["$(l)_im"]) ./ complex.(f["Norm_re"], f["Norm_im"]) for
        l in MPSTimeEvolution.name.(MPSTimeEvolution.ops(cb))
    ]
    return f["time"], results...
end

function siam_adjtdvp1vec(; dt, tmax, freqs, couplings, check_sites, init)
    N = length(freqs)
    sites = siteinds("vFermion", N)

    state_0 = MPS(sites, init)

    adjℓ = spin_chain′(freqs, couplings, sites)
    adjL = MPO(adjℓ, sites)

    site_pairs = [
        (check_sites[j], check_sites[i]) for i in eachindex(check_sites) for j in 1:(i - 1)
    ]
    operators = [
        [LocalOperator(n => "N") for n in check_sites]
        [
            LocalOperator((n1 => "A", jws(n1, n2)..., n2 => "Adag")) for
            (n1, n2) in site_pairs
        ]
        [
            LocalOperator((n1 => "Adag", jws(n1, n2)..., n2 => "A")) for
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

function siam_adaptivetdvp1(;
    dt, tmax, freqs, couplings, check_sites, init, convergence_factor_bonddims=1e-12
)
    N = length(freqs)
    sites = siteinds("Fermion", N)

    state_0 = MPS(sites, init)
    state_0 = enlargelinks(state_0, 2; ref_state=init)

    h = spin_chain(freqs, couplings, sites)
    H = MPO(h, sites)

    site_pairs = [
        (check_sites[j], check_sites[i]) for i in eachindex(check_sites) for j in 1:(i - 1)
    ]
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

    adaptivetdvp1!(
        state_0,
        H,
        dt,
        tmax;
        callback=cb,
        progress=true,
        io_file=tmpfile,
        io_ranks="/dev/null",
        io_times="/dev/null",
        convergence_factor_bonddims=convergence_factor_bonddims,
    )

    f = CSV.File(tmpfile)
    results = [
        complex.(f["$(l)_re"], f["$(l)_im"]) ./ complex.(f["Norm_re"], f["Norm_im"]) for
        l in MPSTimeEvolution.name.(MPSTimeEvolution.ops(cb))
    ]
    return f["time"], results...
end
