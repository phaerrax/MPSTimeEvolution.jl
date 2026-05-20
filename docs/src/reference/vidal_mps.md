# Matrix-product states in Vidal form

The *Vidal form* is another way of representing a many-body state as a
matrix-product state, specifically a more explicit form that exposes the
singular values of all the possible bipartitionings of the system.  Introduced
by Vidal [Vidal2003:slightly_entangled](@cite), this kind of MPS takes the form

```math
\begin{equation}
    \psi = \sum_{i_1,\dotsc,i_n}
    \Gamma^{(1,i_1)}
    \Lambda^{(1)}
    \Gamma^{(2,i_2)}
    \Lambda^{(2)}
    \dotsm
    \Gamma^{(n-1,i_{n-1})}
    \Lambda^{(n-1)}
    \Gamma^{(n,i_n)}
    e_{i_1}^{(1)} \otimes \dotsb \otimes e_{i_n}^{(n)}
\label{eq:vidal_mps}
\end{equation}
```

in which we have a *site tensor* \\(\Gamma^{(k)}\\) for each degree of
freedom of the system and a *bond tensor* \\(\Lambda^{(k)}\\), namely a
non-negative diagonal matrix, associated to each bond.
In this package, a Vidal-form MPS is represented by the `VidalMPS` type, which
is little more than a container of the two arrays of *site* and *bond*
tensors.

An important feature of Vidal-form MPSs is their orthonormality rules, that
allow simplifying many calculations. They are

```math
\begin{equation}
    \begin{aligned}
        \sum_{i_k} \Gamma^{(k,i_k)\dagger} \Lambda^{(k-1)\dagger}
            \Lambda^{(k-1)} \Gamma^{(k,i_k)} &= \tr(\Lambda^{(k-1)2}),\\
        \sum_{i_k} \Gamma^{(k,i_k)} \Lambda^{(k)} \adj{(\Lambda^{(k)})}
            \Gamma^{(k,i_k)\dagger} &= \tr(\Lambda^{(k)2}),
    \end{aligned}
\label{eq:vidalmps_cancellation_rules}
\end{equation}
```

where for convenience we define \\(\Lambda^{(0)}\\) and \\(\Lambda^{(n)}\\)
equal to the identity.

## Basic features

This package offers some basic functionality to work with Vidal-form MPS, in
particular:

- conversion to/from ordinary MPSs,
- addition and multiplication with scalars,
- bond dimension truncation,
- application of (local) operators,
- inner products and expectation values.

The type is designed to follow the same interface as ordinary MPSs from ITensor.

### Constructors

You can construct a trivial (i.e. a tensor-product) Vidal-form MPS with the same
syntax as ordinary MPSs, that is, by specifying a list of site indices and a
list of strings denoting the states, or a function of the site number, and so
on.

```jldoctest vidalmps; setup = :(using ITensorMPS, MPSTimeEvolution), filter = r"id=\d+" => "id=###"
julia> s = siteinds("S=1/2", 4);

julia> VidalMPS(s, ["Up", "Dn", "Dn", "Up"]);

julia> VidalMPS(s, n -> n == 1 ? "Up" : "Dn")
4-site VidalMPS:
 ((dim=2|id=822|"S=1/2,Site,n=1"), (dim=1|id=16|"Link,r=1"))
 ((dim=1|id=16|"Link,r=1"), (dim=1|id=624|"Link,l=1"))
 ((dim=1|id=624|"Link,l=1"), (dim=2|id=229|"S=1/2,Site,n=2"), (dim=1|id=792|"Link,r=2"))
 ((dim=1|id=792|"Link,r=2"), (dim=1|id=183|"Link,l=2"))
 ((dim=1|id=183|"Link,l=2"), (dim=2|id=161|"S=1/2,Site,n=3"), (dim=1|id=475|"Link,r=3"))
 ((dim=1|id=475|"Link,r=3"), (dim=1|id=851|"Link,l=3"))
 ((dim=1|id=851|"Link,l=3"), (dim=2|id=643|"S=1/2,Site,n=4"))

```

A `VidalMPS` is displayed by listing the indices of its tensors in the order
in which they appear in \eqref{eq:vidal_mps}.
Note that there are two “types” of link indices, called *left* and *right* link
indices. They are denoted by `Link,l=...` and `Link,r=...`, respectively, in the
display output above. Simply put, the site tensors have two link indices (or
just one, in case they are at the edge of the MPS): the link index connecting
\\(\Gamma^{(k)}\\) with \\(\Lambda^{(k-1)}\\) is the \\(k\\)-th *left* index,
while the one connecting \\(\Gamma^{(k)}\\) with \\(\Lambda^{(k)}\\) is the
\\(k\\)-th *right* index.

There is currently no way to generate a random Vidal-form MPS. Should you need
one, just generate a random ordinary MPS with the desired properties and then
convert it to the `VidalMPS` type, as explained below.

### Conversion to/from ordinary MPSs

The `convert` function can be used to transform an `MPS` into a `VidalMPS`, with
the algorithm outlined in [Schollwoeck2011:dmrg_in_the_age_of_mps; Sec.
4.6](@cite).

```jldoctest vidalmps; filter = r"id=\d+" => "id=###"
julia> v = random_mps(s; linkdims=3);

julia> vv = convert(VidalMPS, v)
4-site VidalMPS:
 ((dim=2|id=822|"S=1/2,Site,n=1"), (dim=2|id=968|"Link,r=1"))
 ((dim=2|id=968|"Link,r=1"), (dim=2|id=479|"Link,l=2"))
 ((dim=2|id=479|"Link,l=2"), (dim=2|id=229|"S=1/2,Site,n=2"), (dim=3|id=436|"Link,r=2"))
 ((dim=3|id=436|"Link,r=2"), (dim=3|id=287|"Link,l=3"))
 ((dim=3|id=287|"Link,l=3"), (dim=2|id=161|"S=1/2,Site,n=3"), (dim=2|id=217|"Link,r=3"))
 ((dim=2|id=217|"Link,r=3"), (dim=2|id=679|"Link,l=4"))
 ((dim=2|id=679|"Link,l=4"), (dim=2|id=643|"S=1/2,Site,n=4"))

```

Vice versa, a `VidalMPS` can be converted to an ordinary `MPS`: you can choose
where to place the orthogonality center of the resulting MPS via the
`ortho_center` keyword argument.

```jldoctest vidalmps; filter = r"id=\d+" => "id=###"
julia> convert(MPS, vv; ortho_center=3)
4-element MPS:
 ((dim=2|id=822|"S=1/2,Site,n=1"), (dim=2|id=793|"Link,l=1"))
 ((dim=2|id=229|"S=1/2,Site,n=2"), (dim=3|id=260|"Link,l=2"), (dim=2|id=793|"Link,l=1"))
 ((dim=2|id=161|"S=1/2,Site,n=3"), (dim=2|id=826|"Link,l=3"), (dim=3|id=260|"Link,l=2"))
 ((dim=2|id=643|"S=1/2,Site,n=4"), (dim=2|id=826|"Link,l=3"))

```

### Vector operations

In order to multiply a `VidalMPS` by a scalar (complex) number, or to add two or
more `VidalMPS`s together, use the `*` and `+` operators, respectively.

```jldoctest vidalmps
julia> (2 + 3im) * vv;

julia> ww = convert(VidalMPS, random_mps(s));

julia> vv + 2ww;

```

ITensor provides two algorithms to sum MPSs: the “direct sum”, and the
“density matrix” approaches, that can be chosen by passing the `alg` keyword
argument to the sum function.  In MPSTimeEvolution, only the direct-sum
algorithm is natively implemented for the `VidalMPS` type, while the
density-matrix one is executed by converting the summands to ordinary `MPS`s,
running the ITensor `+` function and then converting the result back to a
`VidalMPS`.  The density-matrix approach is the default one, as it provides the
most accurate results.

```jldoctest vidalmps
julia> +(vv, ww; alg="directsum");

julia> +(vv, ww; alg="densitymatrix");

```

!!! warning "Restoring the canonical gauge"
    The result of the direct-sum addition algorithm, in general, doesn't satisfy
    the canonical gauge, i.e. \eqref{eq:vidalmps_cancellation_rules} don't hold
    anymore.  You can check whether a `VidalMPS` object `v` satisfies
    \eqref{eq:vidalmps_cancellation_rules}, and therefore is _actually_ a
    Vidal-form MPS, with the (internal) `check_vidal_form` function.
    If the MPS doesn't satisfy the requirements of the Vidal form, its behaviour
    in several functions (which assume the cancellation rules hold) becomes
    undefined, and may lead to wrong results, e.g. `norm(v)^2` may not equal to
    `inner(v, v)`.  In such cases, the canonical gauge can be restored by
    calling the `canonicalize` function on the offending `VidalMPS` object.

### Bond dimension truncation

The Vidal-form MPS can be compressed at a specified bond \\(k\\) by deleting the
singular values in \\(\Lambda^{(k)}\\) under a certain cutoff, or beyond a
certain number, through a singular-value decomposition.  This can be done by
calling the `truncate` function (or its in-place version `truncate!`).

```jldoctest vidalmps
julia> truncate!(vv; cutoff=1e-12, site_range=3:3);

```

The truncation can be performed on multiple bonds in a single call, by providing
a `site_range` argument which spans more than one bond.

!!! info "Truncation order"
    When truncating multiple bonds, the operation is performed spanning the
    range from right to left. This is to ensure consistency between the
    `truncate` method for `VidalMPS`s and the ITensor implementation, so that
    `truncate(v::MPS; ...) ≈ truncate(convert(VidalMPS, v); ...)` for the same
    `site_range`, `cutoff` and `maxdim` on both sides.

### Application of operators

The `apply` function allows multiplying a `VidalMPS` by an ITensor operator,
exactly as for ordinary MPSs:

```jldoctest vidalmps
julia> Sx1 = op("Sx", s, 1);

julia> Sy2Sy3 = op("Sy", s, 2) * op("Sy", s, 3);

julia> apply(Sx1, vv);

julia> apply(Sy2Sy3, vv; cutoff=1e-14);

```

The result can be truncated by using the `cutoff` and `maxdim` keyword
arguments, in the same manner as with the `truncate` function.

### Inner products and expectation values

The `inner` (or `dot`) function implements the inner product between two
Vidal-form MPS, and the norm of an MPS can be retrieved with `norm`.  The `norm`
function, in practice, doesn't compute the full inner product between a
`VidalMPS` and itself, but uses the cancellation rules in
\eqref{eq:vidalmps_cancellation_rules} to obtain the norm just by computing the
trace of the square of the bond tensors:

```math
\norm{\psi}^2 = \prod_{k=1}^{n-1} \tr(\Lambda^{(k)2}).
```

Expectation values of single-site operators can be computed via the `expect`
function. For a detailed explanation of its options, see the documentation for
the [`ITensorMPS.expect`](@extref) function.
Thanks to the cancellation rules, the `expect` function, like `norm`, computes
the result in an efficient way, by contracting only the sites affected by the
operator.
