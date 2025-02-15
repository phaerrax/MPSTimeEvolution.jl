export LocalOperator

"""
    LocalOperator(terms::OrderedDict{Int,AbstractString})

A LocalOperator represents a product of local operators, whose names (strings, as recognized
by ITensors) are specified by `factors` and acting on sites which are not necessarily
consecutive. For example, the operator ``id ⊗ A ⊗ id ⊗ C`` would be represented as
`{2 => "A", 4 => "C"}`.
"""
struct LocalOperator
    terms::OrderedDict{Int,AbstractString}
    # OrderedDict{Int,AbstractString} and _not_ OrderedDict{AbstractString,Int}: the
    # integers are unique, there can be at most one operator per site, but the same
    # operator (i.e. the same string) can be repeated on more sites.
    # The dictionary is sorted for later convenience.
    LocalOperator(d) = new(sort(OrderedDict(d)))
end

factors(op::LocalOperator) = values(op.terms)
Base.length(op::LocalOperator) = Base.length(op.terms)
domain(op::LocalOperator) = collect(keys(op.terms))
connecteddomain(op::LocalOperator) = first(domain(op)):last(domain(op))
# Since LocalOperator structs are dictionaries sorted by their keys, `domain` and
# `connecteddomain` are guaranteed to return sorted lists of numbers.
name(op::LocalOperator) = *(["$val{$key}" for (key, val) in op.terms]...)

# Sorting utilities
Base.:(==)(a::LocalOperator, b::LocalOperator) = (a.terms == b.terms)
Base.hash(a::LocalOperator) = hash(a.terms)
function Base.isless(a::LocalOperator, b::LocalOperator)
    # Compare domains first (lexicographycally)
    return if !isequal(domain(a), domain(b))
        isless(domain(a), domain(b))
    else  # the names of the factors
        collect(factors(a)) < collect(factors(b))
    end
end

Base.getindex(op::LocalOperator, key) = Base.getindex(op.terms, key)

Base.show(io::IO, op::LocalOperator) = print(io, name(op))

"""
    mpo(sites::Vector{<:Index}, l::LocalOperator)

Return an MPO that represents the operator described by `l`.
"""
@memoize function mpo(sites::Vector{<:Index}, l::LocalOperator)
    return MPO(sites, [i in domain(l) ? l[i] : "Id" for i in eachindex(sites)])
end

"""
    mps(sites::Vector{<:Index}, l::LocalOperator)

Return an MPS that represents the operator described by `l` in a vectorised form.
The MPS will have the factors in `l` prefixed by `v` on the domain of the operator, or
`vId` outside of the domain.
"""
@memoize function mps(sites::Vector{<:Index}, l::LocalOperator)
    return MPS(
        ComplexF64, sites, [i in domain(l) ? "v" * l[i] : "vId" for i in eachindex(sites)]
    )
    # The MPS needs to be complex, in general, since we can have the vectorized form of
    # non-Hermitian operator such as A or Adag. The coefficients on the Gell-Mann basis
    # of non-Hermitian operator are complex, in general.
    #
    # Since we are using Memoize for the `mpo` function we might as well memoize this
    # method too. With
    #   s = siteinds("vOsc", 400; dim=4)
    #   o = LocalOperator(Dict(20 => "vA", 19 => "vAdag"))
    # the non-memoized function takes 6.710 ms (97932 allocations: 30.36 MiB), while the
    # memoized one takes only 45.232 ns (1 allocation: 32 bytes).
end
