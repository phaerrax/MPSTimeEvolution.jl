using MPSTimeEvolution
using Test
using ITensors, ITensorMPS, LindbladVectorizedTensors

include("compare_tdvp_methods.jl")
# This tests pushes the bond dimension to the maximum admitted by the sizes of the system,
# so it's best to keep N relatively low so that the computation doesn't get too heavy.
@testset "Compare different TDVP methods" begin
    dt = 0.01
    tmax = 0.5
    N = 4
    s = [1, 3]
    res_tdvp1 = siam_tdvp1(; dt=dt, tmax=tmax, N=N, check_sites=s)  # reference

    res_tdvp1vec_sf = siam_tdvp1vec_superfermions(; dt=dt, tmax=tmax, N=N, check_sites=s)
    @test all(res_tdvp1 .≈ res_tdvp1vec_sf)
    res_tdvp1vec = siam_tdvp1vec(; dt=dt, tmax=tmax, N=N, check_sites=s)
    @test all(res_tdvp1 .≈ res_tdvp1vec)
    res_adjtdvp1vec = siam_adjtdvp1vec(; dt=dt, tmax=tmax, N=N, check_sites=s)
    @test all(res_tdvp1 .≈ res_adjtdvp1vec)
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
