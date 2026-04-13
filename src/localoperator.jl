export LocalOperator, parseoperators

"""
    LocalOperator(terms::OrderedDict{Int,AbstractString})

A LocalOperator represents a product of local operators, whose names (strings, as recognized
by ITensors) are specified by `factors` and acting on sites which are not necessarily
consecutive. For example, the operator ``1 ⊗ A ⊗ 1 ⊗ C`` would be represented as
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
name(op::LocalOperator) = reduce(*, "$val($key)" for (key, val) in op.terms)

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
The MPS will have the factors in `l` on the domain of the operator, or `Id` outside of the
domain.
"""
@memoize function mps(sites::Vector{<:Index}, l::LocalOperator)
    return MPS(ComplexF64, sites, [i in domain(l) ? l[i] : "Id" for i in eachindex(sites)])
    # The MPS needs to be complex, in general, since we can have the vectorized form of
    # non-Hermitian operator such as A or Adag. The coefficients on the Gell-Mann basis
    # of non-Hermitian operator are complex, in general.
    #
    # Since we are using Memoize for the `mpo` function we might as well memoize this
    # method too. With
    #   s = siteinds("vBoson", 400; dim=4)
    #   o = LocalOperator("Adag(19)A(20)")
    # the non-memoized function takes 6.710 ms (97932 allocations: 30.36 MiB), while the
    # memoized one takes only 45.232 ns (1 allocation: 32 bytes).
end

# Easy input of LocalOperators

function expandsequence(seq)
    re = match(r"(?<name>\w+)\((?<sites>.+)\)", seq)
    sites = parse.(Int, split(re["sites"], ","))
    return [string(re["name"], "(", j, ")") for j in sites]
end

"""
    parseoperators(s::AbstractString)

Parse the string `s` as a list of `LocalOperator` objects, with the following rules:

- operators are written as products of local operators of the form `name(site)` where
    `name` is a string and `site` is an integer;
- operators on different sites can be multiplied together by writing them one after the
    other, i.e. `a(1)b(2)`;
- for convenience, writing a comma-separated list of numbers in the parentheses expands to a
    list of operators with the same name on each of the sites in the list, i.e. `a(1,2,3)`
    is interpreted as `a(1),a(2),a(3)`.

# Example

```julia-repl
julia> parseoperators("x(1)y(3),y(4),z(1,2,3)")
5-element Vector{LocalOperator}:
 x(1)y(3)
 y(4)
 z(1)
 z(2)
 z(3)
```
"""
function parseoperators(s::AbstractString)
    s *= ","  # add extra delimiter at the end (needed for regex below)
    # Split each occurrence made by anything between a word character and "),",
    # matching as few characters as possible between them.
    opstrings = Base.chop.([r.match for r in eachmatch(r"\w+\(.+?\),", s * ",")])
    ops = LocalOperator[]
    i = 1
    while i <= length(opstrings)
        # Replace each sequence, i.e. an item like x(1,2,4), by its expansion.
        if contains(opstrings[i], ",")  # it is a sequence
            # ↖ Replace `opstrings[i]` with the expanded sequence, shifting the following
            # elements in the array to the right to make space for it. Then, restart the
            # loop iteration from the same index, which will be the first operator of the
            # just expanded sequence.
            splice!(opstrings, i:i, expandsequence(opstrings[i]))
            continue
        end
        # Decide whether we have a product of operators, such as x(1)y(2), or a
        # sequence such as x(1,2,3)
        d = Dict{Int,String}()
        foreach(
            re -> push!(d, parse(Int, re["site"]) => re["name"]),
            eachmatch(r"(?<name>\w+?)\((?<site>\d+?)\)", opstrings[i]),
        )
        push!(ops, LocalOperator(d))
        i += 1
    end

    return ops
end

"""
    parseoperators(d::Dict)

Translate a dictionary into a list of `LocalOperator` objects, where each `(key, value)`
pair is interpreted as the operator `key` (a string) on each of the sites contained in
`value` (a list of integers), separately.
This type of input does not support products of operators on different sites.

# Example

```julia-repl
julia> ops = Dict("x" => [1, 2, 3], "y" => [2]);

julia> parseoperators(ops)
4-element Vector{LocalOperator}:
 x{1}
 x{2}
 x{3}
 y{2}
```
"""
function parseoperators(d::Dict)
    ops = LocalOperator[]
    for (k, v) in d
        for n in v
            push!(ops, LocalOperator(Dict(n => k)))
        end
    end
    return ops
end

# We reuse the `parseoperators` function to create a `LocalOperator` constructor: we parse
# the string, and then ensure that there is only one operator (`only` throws an error if the
# collection has more than one element).
LocalOperator(s::AbstractString) = only(parseoperators(s))
