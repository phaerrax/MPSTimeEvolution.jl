using MPSTimeEvolution
using Test
using ITensors, ITensorMPS, LindbladVectorizedTensors, Observers, CSV

include("compare_tdvp_methods.jl")
# This tests pushes the bond dimension to the maximum admitted by the sizes of the system,
# so it's best to keep N relatively low so that the computation doesn't get too heavy.
atol = 1e-8
@testset "Compare different TDVP methods at atol=$atol" begin
    dt = 0.01
    tmax = 0.5
    N = 6
    freqs = [isodd(n) ? 1 / 2 : 1 / 4 for n in 1:N]
    couplings = fill(1 / 16, N - 1)
    alternate(n) = isodd(n) ? "Occ" : "Emp"
    sites = [1, 3]
    # It's best to choose sites that start from an occupied state, to avoid slight numerical
    # instabilities that could make `isapprox` fail.

    reference_result = itensors_tdvp(;  # Reference result from ITensor's in-house TDVP.
        dt=dt,
        tmax=tmax,
        freqs=freqs,
        couplings=couplings,
        check_sites=sites,
        init=alternate,
    )

    res_tdvp1 = siam_tdvp1(;
        dt=dt,
        tmax=tmax,
        freqs=freqs,
        couplings=couplings,
        check_sites=sites,
        init=alternate,
    )
    @test all(
        all(isapprox.(r1, r2; atol=atol)) for (r1, r2) in zip(reference_result, res_tdvp1)
    )

    res_tdvp1_with_qns = siam_tdvp1_with_qns(;
        dt=dt,
        tmax=tmax,
        freqs=freqs,
        couplings=couplings,
        check_sites=sites,
        init=alternate,
    )
    @test_skip all(
        all(isapprox.(r1, r2; atol=atol)) for
        (r1, r2) in zip(reference_result, res_tdvp1_with_qns)
    )

    res_tdvp1vec_sf = siam_tdvp1vec_superfermions(;
        dt=dt,
        tmax=tmax,
        freqs=freqs,
        couplings=couplings,
        check_sites=sites,
        init=alternate,
    )
    @test all(
        all(isapprox.(r1, r2; atol=atol)) for
        (r1, r2) in zip(reference_result, res_tdvp1vec_sf)
    )

    res_tdvp1vec = siam_tdvp1vec(;
        dt=dt,
        tmax=tmax,
        freqs=freqs,
        couplings=couplings,
        check_sites=sites,
        init=alternate,
    )
    @test all(
        all(isapprox.(r1, r2; atol=atol)) for
        (r1, r2) in zip(reference_result, res_tdvp1vec)
    )

    res_adjtdvp1vec = siam_adjtdvp1vec(;
        dt=dt,
        tmax=tmax,
        freqs=freqs,
        couplings=couplings,
        check_sites=sites,
        init=alternate,
    )
    @test all(
        all(isapprox.(r1, r2; atol=atol)) for
        (r1, r2) in zip(reference_result, res_adjtdvp1vec)
    )
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
