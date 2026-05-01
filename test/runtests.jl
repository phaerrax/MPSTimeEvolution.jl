using Test, Documenter, MPSTimeEvolution
using ITensors, ITensorMPS, LindbladVectorizedTensors, Observers, CSV

using MPSTimeEvolution: _sf_translate_sites, _sf_translate_sites_inv

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

include("compare_tdvp_methods.jl")
@testset verbose = true "TDVP1 methods" begin
    # These tests push the bond dimension to the maximum admitted by the sizes of the system,
    # so it's best to keep N relatively low so that the computation doesn't get too heavy.
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

    @testset "Inner product and norm" begin
        @test dot(x_vidal, y_vidal) ≈ conj(dot(y_vidal, x_vidal))
        @test dot(x_vidal, y_vidal) ≈ dot(x, y)
        @test norm(x_vidal) ≈ norm(x)
    end

    @testset "Application of one-site operators" begin
        a = random_itensor(s[1], s[1]')
        @test convert(MPS, apply(a, x_vidal)) ≈ apply(a, x)

        b = random_itensor(s[3], s[3]')
        @test convert(MPS, apply(b, x_vidal)) ≈ apply(b, x)

        c = random_itensor(s[N], s[N]')
        @test convert(MPS, apply(c, x_vidal)) ≈ apply(c, x)
    end

    @testset "Application of two-site operators" begin
        a = random_itensor(s[1], s[2], s[1]', s[2]')
        @test convert(MPS, apply(a, x_vidal)) ≈ apply(a, x)

        b = random_itensor(s[2], s[3], s[2]', s[3]')
        @test convert(MPS, apply(b, x_vidal)) ≈ apply(b, x)

        c = random_itensor(s[N - 1], s[N], s[N - 1]', s[N]')
        @test convert(MPS, apply(c, x_vidal)) ≈ apply(c, x)

        d = random_itensor(s[2], s[4], s[2]', s[4]')
        # The tensor indices are not contiguous site indices. The apply function should
        # throw an error in this case.
        @test_throws ErrorException apply(d, x_vidal)
    end

    @testset "Application of three-site operators" begin
        a = random_itensor(s[1], s[2], s[3], s[1]', s[2]', s[3]')
        @test convert(MPS, apply(a, x_vidal)) ≈ apply(a, x)

        b = random_itensor(s[2], s[3], s[4], s[2]', s[3]', s[4]')
        @test convert(MPS, apply(b, x_vidal)) ≈ apply(b, x)

        c = random_itensor(s[N - 2], s[N - 1], s[N], s[N - 2]', s[N - 1]', s[N]')
        @test convert(MPS, apply(c, x_vidal)) ≈ apply(c, x)
    end

    @testset "Arithmetic operations" begin
        @test dot(x_vidal, x_vidal + y_vidal) ≈
            dot(x_vidal, x_vidal) + dot(x_vidal, y_vidal)
    end
end
