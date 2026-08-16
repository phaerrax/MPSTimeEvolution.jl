# Ugly hack to use strings as tags for MPI messages.
# TODO I checked that there are no conflicts for the six strings we use:
#
#   julia> strs = ["betaL", "betaR", "gammaL", "gammaR", "PsiL", "PsiR", "V"]
#   julia> intcode.(strs)
#   6-element Vector{Int64}:
#    488
#    494
#    591
#    597
#    376
#    382
#    86
#
# but we need to find something better...
intcode(str)::Int = sum(convert(Int, c) for c in str)
