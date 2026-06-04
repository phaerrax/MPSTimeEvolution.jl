module MPSTimeEvolution

using Adapt
using TypeParameterAccessors
using NDTensors
using ITensors
using ITensorMPS
using IterTools
using IsApprox
using LinearAlgebra
using OffsetArrays
using OrderedCollections
using Memoize
using KrylovKit: exponentiate
using ProgressMeter
using JSON
using DelimitedFiles
using Permutations

abstract type TDVP end
struct TDVP1 <: TDVP end
struct TDVP1vec <: TDVP end
struct TDVP2 <: TDVP end

struct TEBD end

# Vidal-form and inverse-canonical MPS
include("canonical_mps/definition.jl")
include("canonical_mps/indices.jl")
include("canonical_mps/to_mps.jl")
include("canonical_mps/truncation.jl")
include("canonical_mps/algebra.jl")
include("canonical_mps/apply.jl")
include("canonical_mps/projmpo.jl")

include("itensor.jl")
include("callback.jl")
include("localoperator.jl")
include("expvalue_callback.jl")
include("superfermion_callback.jl")

include("utils.jl")

# TDVP base functions
include("timedependentsum.jl")
include("adaptivetdvp.jl")
include("tdvp_step.jl")

# TDVP variants
include("tdvp_variants/tdvp1.jl")
include("tdvp_variants/tdvp1vec.jl")
include("tdvp_variants/tdvp2vec.jl")
include("tdvp_variants/adjtdvp1vec.jl")
include("tdvp_variants/tdvp_other.jl")
include("tdvp_variants/jointtdvp1.jl")
include("tdvp_variants/tdvp2.jl")
include("tdvp_variants/parallel_tdvp2.jl")  # placeholder

# TEBD
include("opsum_to_tebdsequence.jl")
include("tebd.jl")

include("physical_systems.jl")

end # module
