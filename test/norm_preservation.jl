function allapproxequal(itr; kwargs...)
    if Base.haslength(itr)
        length(itr) <= 1 && return true
    end
    pl = Iterators.peel(itr)
    isnothing(pl) && return true
    a, rest = pl
    return all(isapprox(a; kwargs...), rest)
end

function tdvp1_preserves_norm(; dt, tmax, N)
    sites = siteinds("Fermion", N)

    maxbonddim = maximum(maxlinkdims("Fermion", N))
    state_0 = random_mps(sites; linkdims=maxbonddim)

    h = spin_chain([0.2; fill(0.1, N - 1)], fill(0.3, N - 1), sites)
    H = MPO(h, sites)

    cb = ExpValueCallback(LocalOperator[], sites, dt)

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
    return allapproxequal(complex.(f["Norm_re"], f["Norm_im"]))
end

function tdvp1vec_preserves_trace(; dt, tmax, N)
    sites = siteinds("vFermion", N)

    maxbonddim = maximum(maxlinkdims("vFermion", N))
    state_0 = random_mps(sites; linkdims=maxbonddim)

    l = spin_chain([0.2; fill(0.1, N - 1)], fill(0.3, N - 1), sites)
    L = MPO(l, sites)

    cb = ExpValueCallback(LocalOperator[], sites, dt)

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
    return allapproxequal(complex.(f["Norm_re"], f["Norm_im"]))
end
