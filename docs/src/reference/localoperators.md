# Local operators

The `LocalOperator` type is the way many-body operators are encoded in
MPSTimeEvolution. It represents tensor products of single-site operators, i.e.
operators of the form \\(A\sb1 \otimes A\sb2 \otimes \dotsb \otimes A\sb{N}\\)
for an \\(N\\)-body system.

```@docs; canonical=false
LocalOperator
```

The struct is essentially a wrapper around a dictionary whose keys are integers
and whose values are strings, representing operator names: the integers
represent the site on which the associated operator acts.  In other words, the
\\(A\sb1 \otimes A\sb2 \otimes \dotsb \otimes A\sb{N}\\) operator, assuming the
operator names are given in the list `A`, is represented by the association `n
=> A[n] for n in 1:N`.

In order to create a `LocalOperator`, we can pass a dictionary defined this way
to the constructor, such as `LocalOperator(Dict(n => A[n] for n in 1:N))`.
Identity factors are implied and do not need to be explicitly included.
Moreover, the total number of sites in the many-body system is not included in
the description, so that for example `LocalOperator(Dict(1 => "Sx"))` represents
the `"Sx"` operator on the first site no matter how many sites there are in the
system.  Here are more practical examples:

* \\(S_x \otimes \id \otimes \id\\)
* \\(\id \otimes A \otimes \id \otimes \adj{A} \otimes \id\\)
* \\(Z \otimes Z \otimes Z \otimes Z\\)

are respectively

```jldoctest localoperators; setup = :(using MPSTimeEvolution)
julia> LocalOperator(Dict(1 => "Sx"))
Sx(1)

julia> LocalOperator(Dict(2 => "A", 4 => "Adag"))
A(2)Adag(4)

julia> LocalOperator(Dict(n => "Z" for n in 1:4))
Z(1)Z(2)Z(3)Z(4)

```

The strings are supposed to be valid ITensor operator names, as in this package
we will create ITensors, MPSs and MPOs out of these `LocalOperator`s; however,
as far as only `LocalOperator`s are concerned, the strings can be anything, and
no check is performed.

Instead of explicitly creating a `Dict` in order to define a `LocalOperator`, we
can also pass a string to the constructor where we concatenate "name(site)"
terms, exactly as in the output of the previous examples.

```jldoctest localoperators
julia> LocalOperator("Sx(1)")
Sx(1)

julia> LocalOperator("A(2)Adag(4)")
A(2)Adag(4)

julia> LocalOperator("Z(1)Z(2)Z(3)Z(4)")
Z(1)Z(2)Z(3)Z(4)

```

The order in which the operators are specified in the constructor doesn't matter:

```jldoctest localoperators
julia> LocalOperator("Y(3)Y(4)") == LocalOperator("Y(4)Y(3)")
true

```

!!! warning "Repeated sites"
    Due to how dictionaries work, if a site is repeated then the already present
    value gets overwritten (the key-value pairs are read left to right):

    ```jldoctest localoperators
    julia> LocalOperator("a(1)b(2)c(1)")
    c(1)b(2)

    ```

## Lists of local operators

The `parseoperators` function provides a convenient way to create a list of
`LocalOperator` objects (the same method is used for example in the constructors
of [Callback objects](@ref)).
It parses a string using the rules for the `LocalOperator` constructor, where
each term is separated by a comma, and additionally:

* writing a comma-separated list of numbers in the parentheses expands to a
  list of operators with the same name on each of the sites in the list, i.e.
  `a(1,2,3)` is interpreted as `a(1),a(2),a(3)`.

```jldoctest localoperators
julia> parseoperators("x(1)y(3),y(4),z(1,2,3)")
5-element Vector{LocalOperator}:
 x(1)y(3)
 y(4)
 z(1)
 z(2)
 z(3)

julia> parseoperators("g(1,2,3,4)")
4-element Vector{LocalOperator}:
 g(1)
 g(2)
 g(3)
 g(4)

```
