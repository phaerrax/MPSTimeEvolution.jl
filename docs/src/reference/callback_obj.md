# Callback objects

A _callback_ is an object passed to the time-evolution methods, intended to be
“called back” after a single step or a sweep of the evolution completes.

Generally speaking, a callback object describes:

* which operators we are interested in computing expectation values of,
* how frequently we want to calculate them, and
* how we should compute them.

The first two points are usually handled by the user at the moment of creating
the object, while the “how” is already handled by the library, at least for the
already defined callbacks.

More complicated callbacks can be defined: for example, we could define a
callback that computes the entanglement entropy relative to a bipartition by
computing first the singular values of the MPS over one of its bonds.

## Basic callback

```@docs; canonical=false
ExpValueCallback
```

`ExpValueCallback` is a basic callback object, already available in this
package. It keeps track of the expectation values of a series of operators.
To construct this object, we need to provide:

* a list of `LocalOperator`s,
* the ITensor site indices over which the MPS we evolve is defined,
* the time step for measurements.

The list of operators can be an actual list of `LocalOperator` objects, but it
can also be a string where `LocalOperator`s are described more compactly, as
explained in [Local operators](@ref "Local operators").
Each operator must be an existing ITensor operator. Note that no check is
performed at the moment the callback is created: if the operator does not exist,
an error will be thrown during the time evolution, when the callback object
attempts to compute its expectation value.

### Example

Say we have an MPS `v` defined on a set of `N = 10` site indices of the “Boson”
site type, and we want to measure the average occupation number on each site.
We also want to compute these values not after every time step, but only after
ten time steps.  This is how we need to define the callback object.

```jldoctest callback_obj
julia> using ITensorMPS, MPSTimeEvolution

julia> dt = 0.1;  # the time step for the time evolution

julia> N = 10; sites = siteinds("Boson", N; dim=3);

julia> list = LocalOperator.([n => "N" for n in 1:N])
10-element Vector{LocalOperator}:
 N(1)
 N(2)
 N(3)
 N(4)
 N(5)
 N(6)
 N(7)
 N(8)
 N(9)
 N(10)

julia> cb = ExpValueCallback(list, sites, 10dt)
ExpValueCallback
Operators: N(1), N(2), N(3), N(4), N(5), N(6), N(7), N(8), N(9) and N(10)
No measurements performed

```

Alternatively, `cb` can be defined with the more compact (and readable) syntax
of the [parseoperators](@ref) function.  If we wanted to compute, for example,
the average occupation number on the first four sites, we would write

```jldoctest callback_obj
julia> cb = ExpValueCallback("N(1,2,3,4)", sites, 10dt)
ExpValueCallback
Operators: N(1), N(2), N(3) and N(4)
No measurements performed

```

### Custom operators

If the operator we would like to use is not already provided by ITensor,
creating it is very simple. For example, let's consider the operator
\\(X=\frac{1}{\sqrt{2}}(a+\adj{a})\\): we simply have to define a new ITensor
`op` as

```jldoctest callback_obj
julia> using ITensors

julia> ITensors.op(::OpName"X", ::SiteType"Boson", s::Index) = 1/sqrt(2) * op("a† + a", s)

```

Now we can use it in our callback by adding it to the list, for example:

```jldoctest callback_obj
julia> cb = ExpValueCallback("N(1,2,3,4),X(1,2,3,4)", sites, 10dt);

```

### Results readout

In this section we will see how to read the expectation values collected in a
callback object. We will use the callback we defined in the last section, with
the `"N(1,2,3,4),X(1,2,3,4)"` operators.
Let's set up a Hamiltonian operator and run a quick simulation with TDVP1 (see
the [TDVP1 tutorial](@ref "Standard TDVP1")):

```jldoctest callback_obj
julia> v = MPS(sites, n -> n == 1 ? "1" : "0");

julia> v = enlargelinks(v, 4);

julia> h = OpSum();

julia> for n in 1:N
           h += "N", n
       end

julia> for n in 1:N-1
           h += "a†", n, "a", n+1
           h += "a", n, "a†", n+1
       end

julia> H = MPO(h, sites);

julia> tdvp1!(v, H, dt, 10; callback=cb, progress=false);
```

By invoking `cb` we can see that there is new information:

```jldoctest callback_obj
julia> cb
ExpValueCallback
Operators: N(1), N(2), N(3), N(4), X(1), X(2), X(3) and X(4)
Measured times:
  from 0.0
  to 9.99999999999998
  each 1.0

```

The computed expectation values can be accessed by calling the `expvalues`
method on the callback: the result is a dictionary where each local operator is
assigned the series of its expectation values.

```jldoctest callback_obj
julia> expvalues(cb)
OrderedCollections.OrderedDict{LocalOperator, Vector{ComplexF64}} with 8 entries:
  N(1) => [1.0+0.0im, 0.332612+0.0im, 0.00109072+0.0im, 0.00850228+0.0im, 0.003…
  X(1) => [0.0+0.0im, -1.13453e-6-4.60006e-23im, -3.93457e-6-2.03123e-22im, -2.…
  N(2) => [0.0+0.0im, 0.497967+0.0im, 0.132586+0.0im, 0.0262118+0.0im, 0.003191…
  X(2) => [0.0+0.0im, -1.61758e-6+6.61744e-23im, -2.2168e-6-2.60208e-18im, -2.1…
  N(3) => [0.0+0.0im, 0.149637+0.0im, 0.416359+0.0im, 0.0131695+0.0im, 0.047656…
  X(3) => [0.0+0.0im, -1.14256e-6+3.43722e-22im, -2.63261e-5-4.06683e-20im, -2.…
  N(4) => [0.0+0.0im, 0.0184514+0.0im, 0.316095+0.0im, 0.227534+0.0im, 0.011087…
  X(4) => [0.0+0.0im, 4.27215e-5-2.05802e-21im, 0.000124825-1.03762e-20im, 5.01…

```

The time series for individual operators can be accessed via the `expvalues(cb,
lop)` syntax, where `lop` is either a `LocalOperator` or a string that defines a
single `LocalOperator`.

```jldoctest callback_obj
julia> expvalues(cb, LocalOperator(3 => "N"))
11-element Vector{ComplexF64}:
                   0.0 + 0.0im
    0.1496371087008667 + 0.0im
    0.4163587833281586 + 0.0im
  0.013169546650199981 + 0.0im
  0.047656092490155594 + 0.0im
 0.0012094869867415171 + 0.0im
  0.009031865524793884 + 0.0im
  0.008905435038874998 + 0.0im
  0.010261216883988691 + 0.0im
  0.032619559874800697 + 0.0im
   0.22093690525361828 + 0.0im

julia> expvalues(cb, "X(2)")
11-element Vector{ComplexF64}:
                     0.0 + 0.0im
   -1.617576049847552e-6 + 6.617439891157473e-23im
   -2.216802827443102e-6 - 2.602082008249062e-18im
  -2.1754485086580383e-5 + 5.293471330973909e-22im
    4.696594721193055e-5 + 6.511993073259912e-21im
   -6.763856924121541e-5 + 3.4410971976147374e-21im
   -3.686889866871368e-5 - 3.599741391814458e-21im
    2.176614750375809e-5 + 2.6487551060169705e-22im
   -6.419493464380101e-5 - 2.8434333556510326e-25im
 -0.00010698407746021099 + 6.775843525379591e-21im
    4.065995056214414e-5 + 6.776282561183226e-21im

```

Finally, we can access the time series relative to the state norm (which is the
actual 2-norm of the state, the trace, or another relevant quantity depending on
the chosen time-evolution algorithm).

```jldoctest callback_obj
julia> measurements_norm(cb)
11-element Vector{ComplexF64}:
                1.0 + 0.0im
 1.0000000000000262 + 0.0im
 1.0000000000000322 + 0.0im
 1.0000000000000813 + 0.0im
 1.0000000000001033 + 0.0im
 1.0000000000001088 + 0.0im
 1.0000000000001144 + 0.0im
 1.0000000000001192 + 0.0im
  1.000000000000159 + 0.0im
 1.0000000000001599 + 0.0im
 1.0000000000001545 + 0.0im

```
