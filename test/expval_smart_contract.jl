function expval_smart_contract(; N=10)
    N = min(8, N)
    # Check the correct behaviour of the `_expval_while_sweeping` function.
    s = siteinds("Boson", N; dim=4)
    x = random_mps(s; linkdims=4)

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
