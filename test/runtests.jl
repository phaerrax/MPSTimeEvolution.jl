using MPSTimeEvolution
using Test
using ITensors, ITensorMPS, LindbladVectorizedTensors

include("compare_tdvp_methods.jl")
@testset "Compare different TDVP methods" begin
    @test siam_compare_tdvp()
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
@testset "Expectation value with smart contraction" begin
    @test expval_smart_contract()
end
