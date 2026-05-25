using Test, Documenter, MPSTimeEvolution
using LinearAlgebra, ITensors, ITensorMPS, Observers, CSV, LindbladVectorizedTensors

using MPSTimeEvolution:
    _sf_translate_sites,
    _sf_translate_sites_inv,
    check_vidal_form,
    check_inverse_canonical_form

include("testset_skip.jl")

@testset "Documentation examples" begin
    doctest(MPSTimeEvolution; manual=false)
end

@testset "Operator parsing" begin
    @test parseoperators("a(2,3,4)") ==
        [LocalOperator(2 => "a"), LocalOperator(3 => "a"), LocalOperator(4 => "a")]
    @test parseoperators("a(1),b(4)") == [LocalOperator(1 => "a"), LocalOperator(4 => "b")]
    @test parseoperators("a(1)b(3)") == [LocalOperator(Dict(1 => "a", 3 => "b"))]
    @test_throws ArgumentError LocalOperator("a(1),b(3)")

    N = 5
    dt = 0.1
    cb = ExpValueCallback("σx(1),σy(4)", siteinds("S=1/2", N), dt)
    @test MPSTimeEvolution.ops(cb) == [LocalOperator(1 => "σx"), LocalOperator(4 => "σy")]
    @test length(MPSTimeEvolution.sites(cb)) == 5
    @test MPSTimeEvolution.callback_dt(cb) == dt
end

include("norm_preservation.jl")
include("compare_tdvp_methods.jl")

@testset verbose=true "TDVP1 methods" begin
    @testset verbose=true "Norm/trace preservation" begin
        dt = 0.01
        tmax = 0.5
        N = 5
        @testset "Standard TDVP1" begin
            @test tdvp1_preserves_norm(; dt=dt, tmax=tmax, N=N)
        end
        @testset "Vectorised TDVP1" begin
            @test tdvp1vec_preserves_trace(; dt=dt, tmax=tmax, N=N)
        end
        @testset "Adaptive TDVP1" begin
            @test adaptivetdvp1_preserves_trace(; dt=dt, tmax=tmax, N=N)
        end
    end

    # These tests push the bond dimension to the maximum admitted by the sizes of the
    # system, so it's best to keep N relatively low so that the computation doesn't get too
    # heavy.
    dt = 0.01
    tmax = 0.5
    N = 6
    freqs = [isodd(n) ? 1 / 2 : 1 / 4 for n in 1:N]
    couplings = fill(1 / 16, N - 1)
    alternate(n) = isodd(n) ? "Occ" : "Emp"
    sites = [1, 3]

    res_tdvp1 = siam_tdvp1(;
        dt=dt,
        tmax=tmax,
        freqs=freqs,
        couplings=couplings,
        check_sites=sites,
        init=alternate,
    )

    atol = 1e-8
    @testset "tdvp1! method against ITensor's TDVP1 (atol=$atol)" begin
        itensors_result = itensors_tdvp(;  # Result from ITensor's in-house TDVP.
            dt=dt,
            tmax=tmax,
            freqs=freqs,
            couplings=couplings,
            check_sites=sites,
            init=alternate,
        )

        # A vanilla `isapprox` test with ours and ITensor's TDVP functions usually fails, as
        # it imposes a too stringent condition. The two functions do give 𝑎𝑝𝑝𝑟𝑜𝑥𝑖𝑚𝑎𝑡𝑒𝑙𝑦
        # equal results but not with the default rtol/atol set in `isapprox`.
        @test all(
            all(isapprox.(r1, r2; atol=atol)) for
            (r1, r2) in zip(itensors_result, res_tdvp1)
        )
    end

    @testset verbose=true "Compare different TDVP methods" begin
        # It's best to choose sites that start from an occupied state, to avoid slight
        # numerical instabilities that could make `isapprox` fail.
        @testset "TDVP1 with quantum numbers" begin
            res_tdvp1_with_qns = siam_tdvp1_with_qns(;
                dt=dt,
                tmax=tmax,
                freqs=freqs,
                couplings=couplings,
                check_sites=sites,
                init=alternate,
            )
            @test_skip all(
                all(isapprox.(r1, r2)) for (r1, r2) in zip(res_tdvp1, res_tdvp1_with_qns)
            )
        end

        tdvp2_atol=1e-8
        @testset "TDVP2 with ordinary MPS (atol=$tdvp2_atol)" begin
            res_tdvp2 = siam_tdvp2(;
                dt=dt,
                tmax=tmax,
                freqs=freqs,
                couplings=couplings,
                check_sites=sites,
                init=alternate,
                cutoff=1e-16,
            )

            @test all(
                all(isapprox.(r1, r2; atol=tdvp2_atol)) for
                (r1, r2) in zip(res_tdvp1, res_tdvp2)
            )
        end

        @testset "TDVP2 with inverse-canonical MPS (atol=$tdvp2_atol)" begin
            res_tdvp2_ic = siam_tdvp2_ic(;
                dt=dt,
                tmax=tmax,
                freqs=freqs,
                couplings=couplings,
                check_sites=sites,
                init=alternate,
                cutoff=1e-16,
            )

            @test all(
                all(isapprox.(r1, r2; atol=tdvp2_atol)) for
                (r1, r2) in zip(res_tdvp1, res_tdvp2_ic)
            )
        end

        @testset "TDVP1 with superfermion states" begin
            res_tdvp1vec_sf = siam_tdvp1vec_superfermions(;
                dt=dt,
                tmax=tmax,
                freqs=freqs,
                couplings=couplings,
                check_sites=sites,
                init=alternate,
            )

            @test all(
                all(isapprox.(r1, r2)) for (r1, r2) in zip(res_tdvp1, res_tdvp1vec_sf)
            )
        end

        @testset "Vectorized TDVP1" begin
            res_tdvp1vec = siam_tdvp1vec(;
                dt=dt,
                tmax=tmax,
                freqs=freqs,
                couplings=couplings,
                check_sites=sites,
                init=alternate,
            )
            @test all(all(isapprox.(r1, r2)) for (r1, r2) in zip(res_tdvp1, res_tdvp1vec))
        end

        @testset "Adjoint vectorized TDVP1" begin
            res_adjtdvp1vec = siam_adjtdvp1vec(;
                dt=dt,
                tmax=tmax,
                freqs=freqs,
                couplings=couplings,
                check_sites=sites,
                init=alternate,
            )
            @test all(
                all(isapprox.(r1, r2)) for (r1, r2) in zip(res_tdvp1, res_adjtdvp1vec)
            )
        end

        adaptive_tdvp1_atol=1e-6
        @testset "Adaptive TDVP1 (atol=$adaptive_tdvp1_atol)" begin
            res_adaptivetdvp1 = siam_adaptivetdvp1(;
                dt=dt,
                tmax=tmax,
                freqs=freqs,
                couplings=couplings,
                check_sites=sites,
                init=alternate,
                maxbonddim=10,
            )
            @test all(
                all(isapprox.(r1, r2; atol=adaptive_tdvp1_atol)) for
                (r1, r2) in zip(res_tdvp1, res_adaptivetdvp1)
            )
        end
    end
end

include("joint_tdvp.jl")
@testset "Joint TDVP1 method" begin
    @test siam_check_jointtdvp1()
end

include("tdvp_sum_mpos.jl")
@testset "Vectorised TDVP1 method with a sum of MPOs" begin
    @test siam_compare_tdvp_with_sum()
end

include("expval_smart_contract.jl")
@testset "Expectation value with smart contractions" begin
    @test expval_smart_contract()
    @test expval_vec()
    @test expval_vec_sf()
end

@testset verbose=true "Vidal MPSs" begin
    N = 8
    s = siteinds("S=1/2", N)
    x = random_mps(ComplexF64, s; linkdims=4)
    y = random_mps(ComplexF64, s; linkdims=4)
    x_vidal = convert(VidalMPS, x)
    y_vidal = convert(VidalMPS, y)

    @testset "Conversion to MPS" begin
        @test all(1:N) do n
            v = convert(MPS, x_vidal; ortho_center=n)
            norm(v) ≈ inner(v[n], v[n])
        end
    end

    @testset "Truncation" begin
        s′ = siteinds("Boson", N; dim=6)
        # We need bigger site indices so that we have more room for truncation.
        z = random_mps(ComplexF64, s′; linkdims=4)
        z_vidal = convert(VidalMPS, z)
        maxdim = 10
        cutoff = 1e-8
        @test truncate(z; maxdim=maxdim, site_range=3:4) ≈
            convert(MPS, truncate(z_vidal; maxdim=maxdim, site_range=3:4))
        @test truncate(z; cutoff=cutoff) ≈ convert(MPS, truncate(z_vidal; cutoff=cutoff))
        @test truncate(z; maxdim=maxdim) ≈ convert(MPS, truncate(z_vidal; maxdim=maxdim))
    end

    @testset "Arithmetic operations" begin
        xy_vidal_sum_dm = +(x_vidal, y_vidal; alg="densitymatrix")
        xy_vidal_sum_ds = +(x_vidal, y_vidal; alg="directsum")
        # The two sum methods produce equivalent MPSs, in the sense that both are valid
        # decompositions of the  vector x + y, and will yield the same results when they are
        # used in functions that fully contract them.
        # However, the direct-sum method doesn't return a valid Vidal-form MPS, i.e. the
        # canonical properties are broken, which e.g. `norm` relies upon.
        @test xy_vidal_sum_dm ≈ xy_vidal_sum_ds
        @test check_vidal_form(xy_vidal_sum_dm)
        @test check_vidal_form(2 * y_vidal)
        @test_broken check_vidal_form(xy_vidal_sum_ds; verbose=false)
        @test check_vidal_form(canonicalize(xy_vidal_sum_ds))  # This shoud work

        # Here we use the default density-matrix approach for the sum, so everything's okay.
        @test 2x_vidal ≈ x_vidal + x_vidal
        @test x_vidal + y_vidal - x_vidal ≈ y_vidal
        @test y_vidal / 4 ≈ 0.25 * y_vidal
    end

    @testset "Inner product and norm" begin
        @test dot(x_vidal, y_vidal) ≈ conj(dot(y_vidal, x_vidal))
        @test dot(x_vidal, y_vidal) ≈ dot(x, y)
        @test norm(x_vidal) ≈ norm(x)
        @test norm(-x_vidal) ≈ norm(x_vidal)

        λ = 3.5im
        @test dot(λ * x_vidal, y_vidal) ≈ conj(λ) * dot(x_vidal, y_vidal)
        @test dot(x_vidal, x_vidal - y_vidal) ≈
            dot(x_vidal, x_vidal) - dot(x_vidal, y_vidal)

        @test norm(normalize(convert(InverseCanonicalMPS, 2x))) ≈ 1
    end

    @testset "Application of one-site unitary operators" begin
        a = op("RandomUnitary", s, 1)
        @test check_vidal_form(apply(a, x_vidal))
        @test convert(MPS, apply(a, x_vidal)) ≈ apply(a, x)

        b = op("RandomUnitary", s, 3)
        @test check_vidal_form(apply(b, x_vidal))
        @test convert(MPS, apply(b, x_vidal)) ≈ apply(b, x)

        c = op("RandomUnitary", s, N)
        @test check_vidal_form(apply(c, x_vidal))
        @test convert(MPS, apply(c, x_vidal)) ≈ apply(c, x)
    end

    @testset "Application of two-site unitary operators" begin
        a = op("RandomUnitary", s, 1, 2)
        @test check_vidal_form(apply(a, x_vidal))
        @test convert(MPS, apply(a, x_vidal)) ≈ apply(a, x)

        b = op("RandomUnitary", s, 2, 3)
        @test check_vidal_form(apply(b, x_vidal))
        @test convert(MPS, apply(b, x_vidal)) ≈ apply(b, x)

        c = op("RandomUnitary", s, N-1, N)
        @test check_vidal_form(apply(c, x_vidal))
        @test convert(MPS, apply(c, x_vidal)) ≈ apply(c, x)

        d = op("RandomUnitary", s, 2, 4)
        # The tensor indices are not contiguous site indices. The apply function should
        # throw an error in this case.
        @test_throws ErrorException apply(d, x_vidal)
    end

    @testset "Application of three-site unitary operators" begin
        a = op("RandomUnitary", s, 1, 2, 3)
        @test check_vidal_form(apply(a, x_vidal))
        @test convert(MPS, apply(a, x_vidal)) ≈ apply(a, x)

        b = op("RandomUnitary", s, 2, 3, 4)
        @test check_vidal_form(apply(b, x_vidal))
        @test convert(MPS, apply(b, x_vidal)) ≈ apply(b, x)

        c = op("RandomUnitary", s, N-2, N-1, N)
        @test check_vidal_form(apply(c, x_vidal))
        @test convert(MPS, apply(c, x_vidal)) ≈ apply(c, x)
    end

    @testset "Expectation values" begin
        @test expect(x, "S²") ≈ expect(x_vidal, "S²")

        m = [1 im; 1+1im 0]
        @test expect(x, m) ≈ expect(x_vidal, m)

        @test expect(y, "ProjDn"; sites=3:N) ≈ expect(y_vidal, "ProjDn"; sites=3:N)
        @test expect(x, ["Sx", "Sy", "Sz"]) ≈ expect(x_vidal, ["Sx", "Sy", "Sz"])
    end
end

@testset verbose=true "TEBD" begin
    N = 5
    ε = rand(N)
    λ = rand(ComplexF64, N-1)
    s = siteinds("S=1/2", N)

    h = OpSum()
    for n in 1:N
        h += ε[n], "Sz", n
    end
    for n in 1:(N - 1)
        h += λ[n], "S+", n, "S-", n+1
        h += conj(λ[n]), "S-", n, "S+", n+1
    end

    # First operator in the odd subsequence
    h₁₂ =
        ε[1] * op("Sz", s, 1) * op(I, s, 2) +
        0.5ε[2] * op(I, s, 1) * op("Sz", s, 2) +
        λ[1] * op("S+", s, 1) * op("S-", s, 2) +
        conj(λ[1]) * op("S-", s, 1) * op("S+", s, 2)
    # First operator in the even subsequence
    h₂₃ =
        0.5ε[2] * op("Sz", s, 2) * op(I, s, 3) +
        0.5ε[3] * op(I, s, 2) * op("Sz", s, 3) +
        λ[2] * op("S+", s, 2) * op("S-", s, 3) +
        conj(λ[2]) * op("S-", s, 2) * op("S+", s, 3)
    # Second operator in the odd subsequence
    h₃₄ =  # (this holds if N > 4!)
        0.5ε[3] * op("Sz", s, 3) * op(I, s, 4) +
        0.5ε[4] * op(I, s, 3) * op("Sz", s, 4) +
        λ[3] * op("S+", s, 3) * op("S-", s, 4) +
        conj(λ[3]) * op("S-", s, 3) * op("S+", s, 4)

    @test prod(MPO(h, s)) ≈
        MPSTimeEvolution.full_tensor(MPSTimeEvolution.tebdsequence(h, s), s)

    dt = 0.1

    @testset "1st order Trotter-Suzuki" begin
        t1odd, t1even = MPSTimeEvolution.trotter1(h, s, -im*dt)
        if isodd(N)
            @test length(t1odd) == length(t1even) == div(N, 2)
        else
            @test length(t1odd) == length(t1even) + 1 == div(N, 2)
        end

        # Check that all operators returned by `trotter1` are unitary.
        @test all([t1odd; t1even]) do u
            u_inds = findall(in(inds(u)), s)
            apply(MPSTimeEvolution.adj(u), u) ≈ op("Id", s[u_inds])
        end

        @test t1odd[div(1, 2) + 1] ≈ exp(-im * dt * h₁₂)
        @test t1odd[div(3, 2) + 1] ≈ exp(-im * dt * h₃₄)
        @test t1even[div(2, 2)] ≈ exp(-im * dt * h₂₃)
    end

    @testset "2nd order Trotter-Suzuki" begin
        t1odd, t1even = MPSTimeEvolution.trotter1(h, s, -im*dt)
        t2odd, t2even, t2odd_again = MPSTimeEvolution.trotter2(h, s, -im*dt)

        @test all(t2odd .≈ t2odd_again)
        @test all([apply(u, u) for u in t2odd] .≈ t1odd)
        @test all(t2even .≈ t1even)

        if isodd(N)
            @test length(t2odd) == length(t2even) == div(N, 2)
        else
            @test length(t2odd) == length(t2even) + 1 == div(N, 2)
        end

        # Check that all operators returned by `trotter2` are unitary.
        @test all([t2odd; t2even]) do u
            u_inds = findall(in(inds(u)), s)
            apply(MPSTimeEvolution.adj(u), u) ≈ op("Id", s[u_inds])
        end

        @test t2odd[div(1, 2) + 1] ≈ exp(-im * dt/2 * h₁₂)
        @test t2odd[div(3, 2) + 1] ≈ exp(-im * dt/2 * h₃₄)
        @test t2even[div(2, 2)] ≈ exp(-im * dt * h₂₃)
    end

    @testset "TEBD1 evolution" begin
        dt = 0.01
        tmax = 0.5
        cb = ExpValueCallback("Sz(" * join(1:N, ",") * ")", s, dt)

        v = MPS(s, n -> n == 1 ? "Up" : "Dn")
        vv = convert(VidalMPS, v)
        norm_init = norm(vv)

        tebd1!(vv, h, dt, tmax; cutoff=1e-12, maxdim=10, progress=false, callback=cb);
        @test norm(vv) ≈ norm_init
    end

    @testset "TEBD2 evolution" begin
        dt = 0.01
        tmax = 0.5
        cb = ExpValueCallback("Sz(" * join(1:N, ",") * ")", s, 10dt)
        # We choose 10dt as measurement step in `cb` so as to use the “combined steps”
        # feature of TEBD2.

        v = MPS(s, n -> n == 1 ? "Up" : "Dn")
        vv = convert(VidalMPS, v)
        norm_init = norm(vv)
        tebd2!(vv, h, dt, tmax; cutoff=1e-12, maxdim=10, progress=false, callback=cb);
        @test norm(vv) ≈ norm_init
    end
end

@testset verbose=true "Inverse-canonical MPSs" begin
    N = 8
    s = siteinds("S=1/2", N)
    x = random_mps(ComplexF64, s; linkdims=4)
    x_ican = convert(InverseCanonicalMPS, x)
    @test MPSTimeEvolution.check_inverse_canonical_form(x_ican)
    @test MPSTimeEvolution.check_inverse_canonical_form(InverseCanonicalMPS(s, "Up"))

    @testset "Conversion to/from MPS" begin
        @test all(1:N) do n
            v = convert(MPS, x_ican; ortho_center=n)
            norm(v) ≈ inner(v[n], v[n])
        end
    end

    @testset "Truncation" begin
        s′ = siteinds("Boson", N; dim=6)
        # We need bigger site indices so that we have more room for truncation.
        z = random_mps(ComplexF64, s′; linkdims=4)
        z_ican = convert(InverseCanonicalMPS, z)
        maxdim = 10
        cutoff = 1e-8
        @test truncate(z; maxdim=maxdim, site_range=3:4) ≈
            convert(MPS, truncate(z_ican; maxdim=maxdim, site_range=3:4))
        @test truncate(z; cutoff=cutoff) ≈ convert(MPS, truncate(z_ican; cutoff=cutoff))
        @test truncate(z; maxdim=maxdim) ≈ convert(MPS, truncate(z_ican; maxdim=maxdim))
    end

    y = random_mps(ComplexF64, s; linkdims=4)
    y_ican = convert(InverseCanonicalMPS, y)
    @testset "Arithmetic operations" begin
        @test x_ican + y_ican - x_ican ≈ y_ican

        xy_ican_sum_dm = +(x_ican, y_ican; alg="densitymatrix")
        xy_ican_sum_ds = +(x_ican, y_ican; alg="directsum")
        @test xy_ican_sum_dm ≈ xy_ican_sum_ds
        @test check_inverse_canonical_form(xy_ican_sum_dm)

        @test_broken check_inverse_canonical_form(xy_ican_sum_ds; verbose=false)
        @test check_inverse_canonical_form(canonicalize(xy_ican_sum_ds))

        yy_ican = deepcopy(y_ican)
        yy_ican.site_tensors[2] *= 2
        # We cannot multiply `y_ican` by two directly (it is not allowed by the * operation)
        # but we can do what we want with the individual tensors. This should break the IC
        # gauge.
        @test_broken check_inverse_canonical_form(yy_ican; verbose=false)
        @test check_inverse_canonical_form(canonicalize(yy_ican))
    end

    @testset "Inner product and norm" begin
        @test dot(x_ican, y_ican) ≈ conj(dot(y_ican, x_ican))
        @test dot(x_ican, y_ican) ≈ dot(x, y)
        @test norm(x_ican) ≈ norm(x)
        @test norm(-x_ican) ≈ norm(x_ican)

        λ = cis(rand())
        @test dot(λ * x_ican, y_ican) ≈ conj(λ) * dot(x_ican, y_ican)
        @test dot(x_ican, x_ican - y_ican) ≈ dot(x_ican, x_ican) - dot(x_ican, y_ican)

        @test norm(normalize(convert(InverseCanonicalMPS, 2x))) ≈ 1
    end

    @testset "Application of one-site unitary operators" begin
        a = op("RandomUnitary", s, 1)
        @test check_inverse_canonical_form(apply(a, x_ican))
        @test convert(MPS, apply(a, x_ican)) ≈ apply(a, x)

        b = op("RandomUnitary", s, 3)
        @test check_inverse_canonical_form(apply(b, x_ican))
        @test convert(MPS, apply(b, x_ican)) ≈ apply(b, x)

        c = op("RandomUnitary", s, N)
        @test check_inverse_canonical_form(apply(c, x_ican))
        @test convert(MPS, apply(c, x_ican)) ≈ apply(c, x)
    end

    @testset "Application of two-site unitary operators" begin
        a = op("RandomUnitary", s, 1, 2)
        @test check_inverse_canonical_form(apply(a, x_ican))
        @test convert(MPS, apply(a, x_ican)) ≈ apply(a, x)

        b = op("RandomUnitary", s, 2, 3)
        @test check_inverse_canonical_form(apply(b, x_ican))
        @test convert(MPS, apply(b, x_ican)) ≈ apply(b, x)

        c = op("RandomUnitary", s, N-1, N)
        @test check_inverse_canonical_form(apply(c, x_ican))
        @test convert(MPS, apply(c, x_ican)) ≈ apply(c, x)

        d = op("RandomUnitary", s, 2, 4)
        # The tensor indices are not contiguous site indices. The apply function should
        # throw an error in this case.
        @test_throws ErrorException apply(d, x_ican)
    end

    @testset "Application of three-site unitary operators" begin
        a = op("RandomUnitary", s, 1, 2, 3)
        @test check_inverse_canonical_form(apply(a, x_ican))
        @test convert(MPS, apply(a, x_ican)) ≈ apply(a, x)

        b = op("RandomUnitary", s, 2, 3, 4)
        @test check_inverse_canonical_form(apply(b, x_ican))
        @test convert(MPS, apply(b, x_ican)) ≈ apply(b, x)

        c = op("RandomUnitary", s, N-2, N-1, N)
        @test check_inverse_canonical_form(apply(c, x_ican))
        @test convert(MPS, apply(c, x_ican)) ≈ apply(c, x)
    end

    @testset "Expectation values" begin
        @test expect(x, "S²") ≈ expect(x_ican, "S²")

        m = [1 im; 1+1im 0]
        @test expect(x, m) ≈ expect(x_ican, m)

        @test expect(y, "ProjDn"; sites=3:N) ≈ expect(y_ican, "ProjDn"; sites=3:N)
        @test expect(x, ["Sx", "Sy", "Sz"]) ≈ expect(x_ican, ["Sx", "Sy", "Sz"])
    end
end
