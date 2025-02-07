using CSV

function siam_tdvp1_with_phase(; phase, dt, tmax, N)
    N = max(N, 5)
    sites = siteinds("Fermion", N + 1)

    init(n) = isodd(n) ? "Occ" : "Emp"
    state_0 = MPS(sites, init)
    maxbonddim = maximum(maxlinkdims(state_0))
    state_0 = enlargelinks(state_0, maxbonddim; ref_state=init)

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
        callback=cb,
        progress=false,
        io_file=tmpfile,
        io_ranks="/dev/null",
        io_times="/dev/null",
    )

    f = CSV.File(tmpfile)
    return f["time"],
    exp(-im * phase) .* complex.(f["N{1}_re"], f["N{1}_im"]),
    exp(-im * phase) .* complex.(f["N{5}_re"], f["N{5}_im"])
end

function siam_jointtdvp1_with_phase(; phase, dt, tmax, N)
    N = max(N, 5)
    sites = siteinds("Fermion", N + 1)

    init(n) = isodd(n) ? "Occ" : "Emp"
    state_R = MPS(sites, init)
    state_L = exp(im * phase) * MPS(sites, init)

    maxbonddim = maximum(maxlinkdims(state_L))
    state_L = enlargelinks(state_L, maxbonddim; ref_state=init)
    state_R = enlargelinks(state_R, maxbonddim; ref_state=init)

    h = spin_chain([0.5; fill(0.25, N)], fill(1, N), sites)
    H = MPO(h, sites)

    operators = [LocalOperator(Dict(1 => "N")), LocalOperator(Dict(5 => "N"))]
    cb = ExpValueCallback(operators, sites, dt)

    tmpfile = tempname()

    jointtdvp1!(
        (state_L, state_R),
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
    return f["time"],
    complex.(f["N{1}_re"], f["N{1}_im"]),
    complex.(f["N{5}_re"], f["N{5}_im"])
end

# This tests pushes the bond dimension to the maximum admitted by the sizes of the system,
# so it's best to keep N relatively low so that the computation doesn't get too heavy.
function siam_check_jointtdvp1(; phase=pi / 6, dt=0.01, tmax=0.5, N=6)
    t, res_tdvp1_n1, res_tdvp1_n5 = siam_tdvp1_with_phase(;
        phase=phase, dt=dt, tmax=tmax, N=N
    )
    _, res_joint_tdvp1_n1, res_joint_tdvp1_n5 = siam_jointtdvp1_with_phase(;
        phase=phase, dt=dt, tmax=tmax, N=N
    )
    # Be careful to check the equality on expectation values that do not get too close to
    # zero---so, avoid occupation numbers of initially empty sites. The `isapprox` function
    # struggles (and requires manual intervention) when comparing tiny values: for example
    # here I tried making site n° 5 start from the empty state, and after the very first
    # step we had an absolute difference of 1e-27 but a relative difference of 1e-8, since
    # the denominator in the formula of the relative difference is very very small, causing
    # numerical instability. 1e-8 is rejected by `isapprox`, returning false in this case.
    return all(res_tdvp1_n1 .≈ res_joint_tdvp1_n1) &&
           all(res_tdvp1_n5 .≈ res_joint_tdvp1_n5)
end
