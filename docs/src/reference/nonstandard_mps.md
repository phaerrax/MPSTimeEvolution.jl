# Non-standard matrix-product-state gauges

## Definitions

This package features two less common ways of writing a many-body quantum state
as a matrix-product state, that expose the singular values of all the possible
bipartitions of the system: the *Vidal*, or *canonical*, form, and the *inverse
canonical* form.

### Vidal form

Introduced by Vidal [Vidal2003:slightly_entangled](@cite), this MPS form is
written as

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
\label{eq:vidal-mps}
\end{equation}
```

in which we have a *site tensor* \\(\Gamma^{(k)}\\) for each degree of
freedom of the system and a *bond tensor* \\(\Lambda^{(k)}\\), namely a
non-negative diagonal matrix, associated to each bond.  Each \\(\Lambda^{(k)}\\)
contains the weights of the Schmidt decomposition at bond \\(k\\).

In this package, a Vidal-form MPS is represented by the `VidalMPS` type, which
is little more than a container of the two arrays of *site* and *bond*
tensors.

### Inverse canonical form

The inverse canonical gauge, introduced in [Stoudenmire2013:real_space_parallel_dmrg](@cite), allows us to write an MPS in such a way that each site tensor can be considered an orthogonality centre.

This gauge has a similar structure to the one of the canonical gauge,

```math
\begin{equation}
    \psi = \sum_{i_1,\dotsc,i_n}
    \Psi^{(1,i_1)}
    V^{(1)}
    \Psi^{(2,i_2)}
    V^{(2)}
    \dotsm
    \Psi^{(n-1,i_{n-1})}
    V^{(n-1)}
    \Psi^{(n,i_n)}
    e_{i_1}^{(1)} \otimes \dotsb \otimes e_{i_n}^{(n)}
\label{eq:ic-mps}
\end{equation}
```

but in this case each bond tensor \\(V^{(k)}\\) is a diagonal matrix containing
the *inverse* of the Schmidt weights.
One can directly map between the two gauges with \\(V^{(k)} =
(\Lambda^{(k)})^{−1}\\) and \\(\Psi^{(k,i\sb{k})} = \Lambda^{(k-1)}
\Gamma^{(k,i\sb{k})} \Lambda^{(k)}\\).

In this package, a inverse-canonical-form MPS is represented by the
`InverseCanonicalMPS` type.

## Properties

An important feature of Vidal-form and inverse-canonical MPSs is their
orthonormality rules, that allow simplifying many calculations. They are

```math
\begin{equation}
    \begin{aligned}
        \sum_{i_k} \Gamma^{(k,i_k)\dagger} \Lambda^{(k-1)\dagger}
            \Lambda^{(k-1)} \Gamma^{(k,i_k)} &= \tr(\Lambda^{(k-1)2}),\\
        \sum_{i_k} \Gamma^{(k,i_k)} \Lambda^{(k)} \adj{(\Lambda^{(k)})}
            \Gamma^{(k,i_k)\dagger} &= \tr(\Lambda^{(k)2}),
    \end{aligned}
\label{eq:vidalmps-cancellation-rules}
\end{equation}
```

for the former type, and

```math
\begin{equation}
    \begin{aligned}
        \sum_{i_k} V^{(k)\dagger} \Psi^{(k,i_k)\dagger}
            \Psi^{(k,i_k)} V^{(k)} &=
            \tr\bigl((V^{(k-1)})^{-2}\bigr),\\
        \sum_{i_k} V^{(k-1)} \Psi^{(k,i_k)}
            \Psi^{(k,i_k)\dagger} \adj{(V^{(k-1)})} &=
            \tr\bigl((V^{(k)})^{-2}\bigr),
    \end{aligned}
\label{eq:icmps-cancellation-rules}
\end{equation}
```

for the latter, where for convenience we define \\(\Lambda^{(0)}\\),
\\(\Lambda^{(n)}\\), \\(V^{(0)}\\) and \\(V^{(n)}\\) equal to the identity.

!!! warning "Preserving the gauge conditions"
    You can check whether a `VidalMPS` object `v` satisfies
    \eqref{eq:vidalmps-cancellation-rules}, and therefore is _actually_ a
    Vidal-form MPS, with the (internal) `check_vidal_form` function.
    For an `InverseCanonicalMPS` object, use the `check_inverse_canonical_form`
    instead to verify if \eqref{eq:vidalmps-cancellation-rules} are satisfied.
    If the MPS doesn't satisfy the requirements of its gauge, its behaviour
    in several functions (which assume the cancellation rules hold) becomes
    undefined, and may lead to wrong results, e.g. `norm(v)^2` may not be equal
    to `inner(v, v)`.  In such cases, the (inverse-) canonical gauge can be
    restored by calling the `canonicalize` function on the offending `VidalMPS`
    or `InverseCanonicalMPS` object.

## Basic features

This package offers some basic functionality to work with Vidal-form and
inverse-canonical MPS, in particular:

- conversion to/from ordinary MPSs,
- addition and multiplication with scalars,
- bond dimension truncation,
- application of (local) operators,
- inner products and expectation values.

The new types are designed to follow the same interface of ordinary MPSs from
ITensor.

!!! info
    In this tutorial we will mostly showcase examples with the `VidalMPS` type,
    but the same code works with `InverseCanonicalMPS` objects as well (except
    when stated otherwise).

### Constructors

You can construct a trivial (i.e. a tensor-product) Vidal-form or
inverse-canonical MPS with the same syntax as ordinary MPSs, that is, by
specifying a list of site indices and a list of strings denoting the states, or
a function of the site number, and so on.
We will show an example for the `VidalMPS` type; the `InverseCanonicalMPS`
type follows the same structure.

```jldoctest vidalmps; setup = :(using ITensorMPS, MPSTimeEvolution), filter = r"id=\d+" => "id=###"
julia> s = siteinds("S=1/2", 4);

julia> VidalMPS(s, ["Up", "Dn", "Dn", "Up"]);

julia> VidalMPS(s, n -> n == 1 ? "Up" : "Dn")
4-site VidalMPS:
 ((dim=2|id=822|"S=1/2,Site,n=1"), (dim=1|id=16|"Link,r=1"))
 ((dim=1|id=16|"Link,r=1"), (dim=1|id=624|"Link,l=2"))
 ((dim=1|id=624|"Link,l=2"), (dim=2|id=229|"S=1/2,Site,n=2"), (dim=1|id=792|"Link,r=2"))
 ((dim=1|id=792|"Link,r=2"), (dim=1|id=183|"Link,l=3"))
 ((dim=1|id=183|"Link,l=3"), (dim=2|id=161|"S=1/2,Site,n=3"), (dim=1|id=475|"Link,r=3"))
 ((dim=1|id=475|"Link,r=3"), (dim=1|id=851|"Link,l=4"))
 ((dim=1|id=851|"Link,l=4"), (dim=2|id=643|"S=1/2,Site,n=4"))

```

A `VidalMPS` is displayed by listing the indices of its tensors in the order
in which they appear in \eqref{eq:vidal-mps} (or \eqref{eq:ic-mps} for
`InverseCanonicalMPS` objects).
Note that there are two “types” of link indices, which we call *left* and
*right* link indices. They are denoted by `Link,l=...` and `Link,r=...`,
respectively, in the display output above. Simply put, the site tensors have two
link indices (or just one, in case they are at the edge of the MPS): the link
index connecting \\(\Gamma^{(k)}\\) with \\(\Lambda^{(k-1)}\\) is the \\(k\\)-th
*left* index, while the one connecting \\(\Gamma^{(k)}\\) with
\\(\Lambda^{(k)}\\) is the \\(k\\)-th *right* index.

There is currently no way to generate random MPS of these new types. Should you
need one, just generate a random ordinary MPS with the desired properties and
then convert it to the appropriate type, as explained below.

### Conversion to/from ordinary MPSs

The `convert` function can be used to transform an `MPS` into a `VidalMPS` or
`InverseCanonicalMPS`, with the algorithm outlined in
[Schollwoeck2011:dmrg_in_the_age_of_mps; Sec. 4.6](@cite).
The `cutoff` and `maxdim` keyword arguments can be adjusted to control how small
and how many the singular values in the bond tensors can be.

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

Vice versa, a `VidalMPS` or an `InverseCanonicalMPS` can be converted to an
ordinary `MPS`: you can choose where to place the orthogonality center of the
resulting MPS via the `ortho_center` keyword argument.

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

For inverse-canonical MPSs we define the scalar multiplication only when the
number has unit absolute value.

```jldoctest vidalmps
julia> v_ic = convert(InverseCanonicalMPS, random_mps(s));

julia> w_ic = convert(InverseCanonicalMPS, random_mps(s));

julia> v_ic - im*w_ic;

julia> 2v_ic
ERROR: scalar factor multiplying the MPS does not have unit absolute value.
[...]
```

ITensor provides two algorithms to sum MPSs: the “direct sum”, and the “density
matrix” approaches, that can be chosen by passing the `alg` keyword argument to
the sum function.  In MPSTimeEvolution, only the direct-sum algorithm is
natively implemented for the `VidalMPS` and `InverseCanonicalMPS` types, while
the density-matrix one is executed by converting the summands to ordinary
`MPS`s, calling the ITensor `+` function and then converting the result back to
the original type.  The density-matrix approach is the default one, as it
provides the most accurate results.

```jldoctest vidalmps
julia> +(vv, ww; alg="directsum");

julia> +(vv, ww; alg="densitymatrix");

```

!!! warning "Loss of the canonical gauge"
    The result of the direct-sum addition algorithm, in general, doesn't satisfy
    the canonical gauge, i.e. \eqref{eq:vidalmps-cancellation-rules} and
    \eqref{eq:icmps-cancellation-rules} don't hold anymore.

### Bond dimension truncation

An MPS can be compressed at a specified bond \\(k\\) by deleting the
singular values in \\(\Lambda^{(k)}\\) (or their inverse in \\(V^{(k)}\\)) under
a certain cutoff, or beyond a certain number, through a singular-value
decomposition.  This can be done by calling the `truncate` function (or its
in-place version `truncate!`).

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

The `apply` function allows multiplying a `VidalMPS` or an `InverseCanonicalMPS`
by an ITensor operator, like with ordinary MPSs:

```jldoctest vidalmps
julia> Sx1 = op("Sx", s, 1);

julia> Sy2Sy3 = op("Sy", s, 2) * op("Sy", s, 3);

julia> apply(Sx1, vv);

julia> apply(Sy2Sy3, vv; cutoff=1e-14);

```

The result can be truncated by using the `cutoff` and `maxdim` keyword
arguments, in the same manner as with the `truncate` function.
Note that if the operator is not unitary, in general, the result might not
satisfy the (inverse-) canonical gauge anymore.

!!! info "Limited implementation"
    The `apply` function currently allows multiplying an MPS by a single
    operator which is defined on consecutive sites. The application of operators
    with non-consecutive sites, or of a vector of operators, both supported by
    ITensor's `apply` function, are not currently implemented in this version
    for the `VidalMPS` and `InverseCanonicalMPS` types.

### Inner products and expectation values

The `inner` (or `dot`) function implements the inner product between two
MPS, and the norm of an MPS can be retrieved with `norm`.  The `norm`
function, in practice, doesn't compute the full inner product between a
`VidalMPS` or an `InverseCanonicalMPS` and itself, but uses the cancellation
rules in \eqref{eq:vidalmps-cancellation-rules} and
\eqref{eq:icmps-cancellation-rules} to obtain the norm just by computing the
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
