function expval_smart_contract(; N=10)
    N = min(8, N)
    # Check the correct behaviour of the `_expval_while_sweeping` function.
    s = siteinds("Boson", N; dim=4, conserve_number=true)
    x = random_mps(s, n -> isodd(n) ? "1" : "0"; linkdims=4)

    onesite = [LocalOperator(1 => "n"), LocalOperator(3 => "n")]
    twosite = [
        LocalOperator((1 => "adag", 8 => "a")), LocalOperator((4 => "a", 6 => "adag"))
    ]

    ev1_smart = [MPSTimeEvolution._expval_while_sweeping(x, l) for l in onesite]
    ev1_inner = [inner(x', MPSTimeEvolution.mpo(s, l), x) for l in onesite]
    ev_expect = [
        expect(x, opname; sites=opsite) for (opsite, opname) in zip(
            first.(MPSTimeEvolution.domain.(onesite)),
            first.(MPSTimeEvolution.factors.(onesite)),
        )
    ]

    ev2_smart = [MPSTimeEvolution._expval_while_sweeping(x, l) for l in twosite]
    ev2_inner = [inner(x', MPSTimeEvolution.mpo(s, l), x) for l in twosite]

    return all(ev1_smart .≈ ev1_inner .≈ ev_expect) && all(ev2_smart .≈ ev2_inner)
end

function expval_vec(; N=10)
    N = min(8, N)
    # Check the correct behaviour of the `measure_localops!(cb, ψ, alg::TDVP1vec)` method.
    s = siteinds("vFermion", N)
    x = random_mps(s; linkdims=4)

    operators = [
        LocalOperator(1 => "N"),
        LocalOperator(3 => "N"),
        LocalOperator((1 => "Adag", 3 => "F", 8 => "A")),
        LocalOperator((4 => "A", 6 => "Adag")),
    ]

    cb = ExpValueCallback(operators, s, 0.1)

    MPSTimeEvolution.apply!(cb, x, MPSTimeEvolution.TDVP1vec(); t=0, sweepend=true)
    ev_cb = [MPSTimeEvolution.measurements(cb)[l][end] for l in MPSTimeEvolution.ops(cb)]

    ev_inner = [inner(
        # We need to use `dag` on the operator MPS here because ⟨A,ρ⟩ = tr(A*ρ).
        dag(MPSTimeEvolution.mps(s, l)),
        x,
    ) for l in MPSTimeEvolution.ops(cb)]

    return all(ev_cb .≈ ev_inner)
end

function expval_vec_sf(; N=10)
    N = min(8, N)
    jws(n1, n2) = (i => "F" for i in (n1 + 1):(n2 - 1))  # Jordan-Wigner string

    # "Traditional" vectorisation
    s = siteinds("vFermion", N)
    x = MPS(s, n -> isodd(n) ? "Occ" : "Emp")

    operators = [
        LocalOperator(1 => "N"),
        LocalOperator(3 => "N"),
        LocalOperator((1 => "Adag", jws(1, 8)..., 8 => "A")),
        LocalOperator((4 => "A", jws(4, 6)..., 6 => "Adag")),
    ]
    cb = ExpValueCallback(operators, s, 0.1)

    MPSTimeEvolution.apply!(cb, x, MPSTimeEvolution.TDVP1vec(); t=0, sweepend=true)
    ev_trad = [MPSTimeEvolution.measurements(cb)[l][end] for l in MPSTimeEvolution.ops(cb)]

    # Check the correct behaviour of the `measure_localops!(cb, ψ, alg::TDVP1vec)` method.
    s = siteinds("Fermion", 2N; conserve_nfparity=true)
    init(n) = isodd(n) ? "Occ" : "Emp"
    init_states_sf = invert_ancillary_states(init.(_sf_translate_sites_inv.(1:2N)))
    x = MPS(s, init_states_sf)

    operators = [
        LocalOperator(_sf_translate_sites(1) => "N"),
        LocalOperator(_sf_translate_sites(3) => "N"),
        LocalOperator((
            _sf_translate_sites(1) => "Adag",
            jws(_sf_translate_sites(1), _sf_translate_sites(8))...,
            _sf_translate_sites(8) => "A",
        )),
        LocalOperator((
            _sf_translate_sites(4) => "A",
            jws(_sf_translate_sites(4), _sf_translate_sites(6))...,
            _sf_translate_sites(6) => "Adag",
        )),
    ]
    cb = SuperfermionCallback(operators, s, 0.1)

    MPSTimeEvolution.apply!(cb, x, MPSTimeEvolution.TDVP1vec(); t=0, sweepend=true)
    ev_sf = [MPSTimeEvolution.measurements(cb)[l][end] for l in MPSTimeEvolution.ops(cb)]

    return all(ev_trad .≈ ev_sf)
end
