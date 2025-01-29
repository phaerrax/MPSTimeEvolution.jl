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
