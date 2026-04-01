# Time-dependent TDVP1

In this tutorial we will see how to solve the Schrödinger equation with a
time-dependent Hamiltonian using the TDVP1 method.

!!! warning "Why'd you have to go and make things so complicated?"
    There isn't a nice interface to define and solve a time-dependent ODE
    problem with TDVP yet. As a consequence, we will need to cobble together
    some methods and types from ITensorMPS which will allow us to use a
    time-dependent Hamiltonian in an efficient way. However, these methods are
    rather obscure as they are not documented, and they are probably still in a
    development phase. For this reason, this tutorial will be quite technical,
    and there is no guarantee that it will keep working in the future, because
    development in ITensorMPS might break some of the definitions we will make
    here.

Our Hamiltonian is composed of two parts \\(H\sb0\\) and \\(H\sb1\\) such that

```math
\begin{equation}
    H(t) = H_0 + f(t) H_1,
    \qquad
    f(t) = \min\{t/T, 1\},
\label{eq:time-dependent-hamiltonian}
\end{equation}
```

where \\(H\sb0\\) is the Hamiltonian of a collection of free electrons

```math
H_0 =
\sum_{k=1}^{N} \sum_{s\in\{\spinup,\spindown\}}
    \varepsilon \adj{c_{k,s}} c_{k,s}\phantomadj +
\sum_{k=1}^{N-1} \sum_{s\in\{\spinup,\spindown\}}
    \lambda (\adj{c_{k,s}} c_{k+1,s}\phantomadj +
             \adj{c_{k+1,s}} c_{k,s}\phantomadj)
```

and \\(H\sb1\\) represents on-site Coulomb repulsion,

```math
H_1 = \sum_{k=1}^{N} U n_{k,\spinup} n_{k,\spindown}.
```

Let's define these operators with ITensor.
We also choose as the initial condition a state where all sites contain an
electron in the “down” state.

```julia-repl
julia> using ITensorMPS, MPSTimeEvolution

julia> N = 6; s = siteinds("Electron", N);

julia> ψ₀ = MPS(s, "↓");

julia> ε = 0.1; λ = 0.2; u = -2ε;

julia> h₀ = OpSum(); h₁ = OpSum();

julia> for k in 1:N
           h₀ += ε, "Ntot", k
       end

julia> for k in 1:N-1
           h₀ += λ, "Cdagup", k, "Cup", k+1
           h₀ += λ, "Cdagdn", k, "Cdn", k+1
           h₀ += λ, "Cup", k, "Cdagup", k+1
           h₀ += λ, "Cdn", k, "Cdagdn", k+1
       end

julia> H₀ = MPO(h₀, s);

julia> for k in 1:N
           h₁ += u, "Nup * Ndn", k
       end

julia> H₁ = MPO(h₁, s);
```

In order to implement the time-dependent ODE problem, we will need to use a
custom `solver` argument as the first argument of `tdvp1!`. This argument is
normally implicitly set to the default solver, the `exponentiate` method from
`KrylovKit`.
First, we define a time-dependent version of `exponentiate` (with the same
structure as `exponentiate`):

```julia-repl
julia> using KrylovKit: exponentiate

julia> using ITensorMPS: TimeDependentSum

julia> function time_dependent_exp(
           H::TimeDependentSum, time_step, ψ₀; current_time=0.0, outputlevel=0, kwargs...
       )
           ψₜ, info = exponentiate(H(current_time), time_step, ψ₀; kwargs...)
           return ψₜ, info
       end
```

The first argument `TimeDependentSum` is a struct, defined in `ITensorMPS` (but
not exported), that contains two members called `coefficients` and `terms`.
When a `TimeDependentSum` object `S` is called on a number `t`, it yields
something like `sum(c(t) * s for (c, s) in zip(coefficients, terms))`: this
means that it can be used to define a function like \\(H(t)\\).
In our case, we can write \\(H(t)\\) from \eqref{eq:time-dependent-hamiltonian}
as follows, once we choose a value for \\(T\\):

```julia-repl
julia> T = 10;

julia> ramp(t) = min(t / T, one(t));

julia> fs = [one, ramp]; Hs = [H₀, H₁];
```

Here `one` is the function that returns the multiplicative identity for its
argument, i.e. `one(t)` returns `1.0` if `t` is a `Float64`.
We define the time-dependent solver as follows:

```julia-repl
julia> function time_dependent_solver(PHs::ProjMPOSum, time_step, ψ₀; kwargs...)
           return time_dependent_exp(
               TimeDependentSum(fs, PHs), time_step, ψ₀; ishermitian=true, kwargs...
           )
       end
```

This function will be used inside `tdvp1!`, precisely in the `tdvp_site_update!`
method that computes the update of a single site during each sweep. We also add
the `ishermitian=true` flag, that will be forwarded to the inner `exponentiate`
method.

!!! details "More technical information"
    During the execution of `tdvp_site_update!` the solver function, which in
    this case will be `time_dependent_solver`, is given the arguments `(H, dt,
    ψ; current_time)`: `H` will end up in the `TimeDependentSum` object inside
    the previous definition of `time_dependent_exp`.
    Note that while the time-dependent Hamiltonian could be described by
    `TimeDependentSum(fs, Hs)`, it is not what we actually use in the `tdvp1!`
    method. Instead, we use a `ProjMPOSum` object from `ITensorMPS`: `ProjMPO`
    objects, and their variant `ProjMPOSum`, are the fundamental object used in
    the TDVP algorithm. They store the Hamiltonian (or equivalent) operator
    together with its partial projections on the left and right parts of the
    MPS.

Now we set the variables related to the time and the output files, and the
callback operator:

```julia-repl
julia> dt = 0.01; tmax = 2T;

julia> meas_file = "measurements.csv";

julia> bdim_file = "bond_dimensions.csv";

julia> time_file = "wallclock_time.csv";

julia> cb = ExpValueCallback("Nup(1,5),Ndn(1,5)", s, dt);
```

One last thing: during the execution of `time_dependent_exp`, inside the stack
of function calls, the `iterate` method is called on the `PHs` object.
Unfortunately, at the time this tutorial is written, no such function is
defined by ITensorMPS, so we need to do a bit of “plumbing” ourselves.
Specifically, we need to extend the two `Base.iterate` methods to the
`ProjMPOSum` type, so that Julia can use them:

```julia-repl
julia> Base.iterate(PH::ProjMPOSum, state) = iterate(PH.terms, state)

julia> Base.iterate(PH::ProjMPOSum) = iterate(PH.terms)
```

!!! warning "Type piracy"
    With these definitions, we are extending methods in Base on a type from
    a package we do not “own”, ITensorMPS. In Julia this is usually called
    [type
    piracy](https://docs.julialang.org/en/v1/manual/style-guide/#avoid-type-piracy).
    Here we are doing in an isolated interactive session, so no problems should
    arise. Be careful when doing this elsewhere.

Now that `iterate` is properly extended, we can finally call `tdvp1!` with
everything we defined:

```julia-repl
julia> tdvp1!(
           time_dependent_solver,
           ψ₀,
           Hs,
           dt,
           tmax,
           callback=cb,
           ishermitian=true,
           io_file=meas_file,
           io_ranks=bdim_file,
           io_times=time_file
       )
```

!!! tip "A more efficient solution"
    Since for \\(t \gt T\\) the Hamiltonian operator is constant, we could
    elaborate a more efficient solution by running the time-dependent TDVP until
    \\(t=T\\), then continuing it for the rest of the evolution with a standard,
    time-independent TDVP.

!!! todo
    The list `fs` of coefficients of the time-dependent Hamiltonian remains
    somewhat hidden in the call to `tdvp1!`; it is only used in the definition
    of `time_dependent_solver`. It would be preferrable to define a
    self-contained and generic time-dependent solver (unlike ours, which “grabs”
    `fs` from outside its scope) so that `fs` would need to be given in `tdvp1!`
    together with `Hs`, in an explicit way.
