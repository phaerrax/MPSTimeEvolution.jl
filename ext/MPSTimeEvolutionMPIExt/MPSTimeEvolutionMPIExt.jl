module MPSTimeEvolutionMPIExt

using MPSTimeEvolution  # base package
using MPI               # package that triggers this extension
using ITensors,         # other packages
    ITensorMPS,
    OffsetArrays,
    ProgressMeter

include("mpi_utils.jl")

include("parallel_tdvp2/aux.jl")
include("parallel_tdvp2/p1.jl")
include("parallel_tdvp2/p2.jl")
include("parallel_tdvp2/p3.jl")
include("parallel_tdvp2/p4.jl")
include("parallel_tdvp2/main.jl")

end # module MPSTimeEvolutionMPIExt
