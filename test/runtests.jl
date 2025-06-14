using MPSTimeEvolution
using Test
using ITensors, ITensorMPS, LindbladVectorizedTensors, Observers, CSV

using MPSTimeEvolution: _sf_translate_sites, _sf_translate_sites_inv

include("testset_skip.jl")

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

        # A vanilla `isapprox` test with ours and ITensor's TDVP functions usually fails, as it
        # imposes a too stringent condition. The two functions do give 𝑎𝑝𝑝𝑟𝑜𝑥𝑖𝑚𝑎𝑡𝑒𝑙𝑦 equal
        # results but not with the default rtol/atol set in `isapprox`.
        @test all(
            all(isapprox.(r1, r2; atol=atol)) for
            (r1, r2) in zip(itensors_result, res_tdvp1)
        )
    end

    @testset verbose=true "Compare different TDVP methods" begin
        # It's best to choose sites that start from an occupied state, to avoid slight numerical
        # instabilities that could make `isapprox` fail.
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
