# Parallel TDVP2

## Overview

The parallel two-site time-dependent variational principle (TDVP2) algorithm was
introduced in [Secular2020:parallel_tdvp](@cite), and distributes the workload
by parallelising the sweeps over different parts of the MPS, so that each
process acts only on a subregion of the whole system.
This package implements the four-process variant of the parallel algorithm in
the `parallel_tdvp2!` method, distributing the sites of the MPS equally between
the workers.  For this reason, we suggest it with at least 12 MPS sites, so that
each process works with a partition no smaller than 3 sites.

The algorithm uses the MPI protocol for exchanging data between the parallel
processes, which is provided by the MPI.jl package.
Since MPI features are used only in this instance, the parallel TDVP2 is
not directly implemented in the MPSTimeEvolution package, but in an extension,
which gets loaded (and possibly precompiled) only if the MPI package is loaded,
too.
This means that the `tdvp2_parallel!` function is not available by loading
MPSTimeEvolution only; you will need to load MPSTimeEvolution *and* MPI in order
to activate the extension.

The first thing we need to do, therefore, is to initialise the MPI environment.
For example, we can use the `mpiexecjl` command to launch Julia with MPI, as

```bash
mpiexecjl -np 4 julia --project julia_script.jl
```

See the [MPI.jl](https://juliaparallel.github.io/MPI.jl/stable/) documentation
for instructions on how to configure and use Julia with MPI.

!!! info "Interactive sessions and MPI"
    There is currently no way to open an interactive session with Julia and MPI.
    For this reason, you will not be able to run the code blocks of this
    tutorial into the Julia REPL. For convenience, we will provide at the end a
    single script file collecting all the step-by-step instructions, that you
    can run with the `mpiexecjl` executable as explained above.

We set up the MPI environment with the following instructions.

```julia-repl
julia> using MPSTimeEvolution, MPI

julia> MPI.Init()

julia> comm = MPI.COMM_WORLD
```

## Physical system example

We will test this algorithm on the same model we used for adaptive TDVP1: a spin
coupled to a bosonic bath,

```math
\begin{gather*}
H = H\Sys + H\Env + H\Int,\\
H\Sys = \frac{\omega_0}{2} \sigma_z,\\
H\Env = \int_0^{\omega\cutoff} \adj{a_\omega} a_\omega\phantomadj\,\dd\omega,\\
H\Int = \sigma_x \otimes \int_0^{\omega\cutoff}
        (a_\omega\phantomadj + \adj{a_\omega}) \sqrt{J(\omega)} \,\dd\omega,
\end{gather*}
```

with \\(J(\omega) = 2\pi \alpha \omega\\) on \\([0,\omega\cutoff]\\), and the
initial state

```math
\rho_0 = \proj{\spinup} \otimes \frac{1}{\exp(-\beta H\Env)} \exp(-\beta H\Env).
```

We transform this system with T-TEDOPA, into a discrete chain of bosonic
modes: we obtain a new Hamiltonian

```math
\begin{gather*}
H' = H\Sys + H'\Env + H'\Int,\\
H'\Env = \sum_{n=1}^{+\infty} \adj{A_n} A_n\phantomadj
         +\sum_{n=1}^{+\infty} (
            \adj{A_n} A_{n+1}\phantomadj
            +\adj{A_{n+1}} A_n\phantomadj
         ),\\
H'\Int = \sigma_x \otimes (A_1 + \adj{A_1})
\end{gather*}
```

and the vacuum as the initial state.
We'll use the [TEDOPA](https://github.com/phaerrax/TEDOPA.jl) package to compute
the chain mapping, and truncate the infinite chain to \\(N=20\\) sites.

```julia-repl
julia> using TEDOPA

julia> N = 20;

julia> envdict = Dict(
             "environment" => Dict(
                 "spectral_density_parameters" => [],
                 "spectral_density_function" => "0.2 * x",
                 "domain" => [0, 1],
                 "temperature" => 1,
             ),
             "chain_length" => N,
             "PolyChaos_nquad" => 200,
         );

julia> env = chainmapping_ttedopa(envdict);
```

Now we define the Hamiltonian operators and the initial state.

```julia-repl
julia> s = MPI.bcast([siteind("S=1/2"); siteinds("Boson", N; dim=6)], comm);

julia> h = OpSum();

julia> h += 0.1, "σz", 1;

julia> h += couplings(env)[1], "σx", 1, "A + Adag", 2;

julia> for n in 1:N
           h += frequencies(env)[n], "N", n+1
       end

julia> for n in 1:N-1
           h += couplings(env)[n+1], "Adag", n+1, "A", n+2
           h += couplings(env)[n+1], "Adag", n+2, "A", n+1
       end

julia> H = MPI.bcast(MPO(h, s), comm);

julia> ρₜ = MPI.bcast(InverseCanonicalMPS(s, n -> n == 1 ? "Up" : "0"), comm);
```

We also set the time step and the total evolution time:

```julia-repl
julia> dt = 0.01; tmax = 5;
```

We want to observe the evolution of the magnetisation on the spin, and the heat
flow between the spin and the heat bath:

```julia-repl
julia> cb = ExpValueCallback("σz(1),σy(1)A + Adag(2)", s, dt)
ExpValueCallback
Operators: σz(1) and σy(1)A + Adag(2)
No measurements performed

```

The parallel TDVP2 algorithm is provided by the `parallel_tdvp2!` function. It
works similarly to the `tdvp2!` function, but we need to provide two additional
arguments: the rank of the root process (that otherwise defaults to `0`) and the
`Comm` object we defined at the beginning with `MPI.COMM_WORLD`.

```julia-repl
julia> maxdim = 20; cutoff = 1e-8;

julia> parallel_tdvp2!(
    v, H, dt, tmax;
    comm=comm, callback=cb, cutoff=cutoff, maxdim=maxdim, progress=false
)
```

For comparison, we also run a standard TDVP2 evolution in the inverse canonical
gauge, with the same bond dimension and cutoff constraints as before.

```julia-repl
julia> ρₜ′ = InverseCanonicalMPS(s, n -> n == 1 ? "Up" : "0");

julia> cb′ = ExpValueCallback("σz(1),σy(1)A + Adag(2)", s, dt);

julia> tdvp2!(
    ρₜ′, H, dt, tmax;
    callback=cb′, cutoff=cutoff, maxdim=maxdim, progress=false
);
```

## Complete tutorial

```julia
using MPSTimeEvolution, ITensorMPS, MPI, TEDOPA

MPI.Init()
comm = MPI.COMM_WORLD

N = 20
envdict = Dict(
    "environment" => Dict(
        "spectral_density_parameters" => [],
        "spectral_density_function" => "0.2 * x",
        "domain" => [0, 1],
        "temperature" => 1,
        ),
    "chain_length" => N,
    "PolyChaos_nquad" => 200,
)
env = chainmapping_ttedopa(envdict)

s = MPI.bcast([siteind("S=1/2"); siteinds("Boson", N; dim=6)], comm)

h = OpSum()
h += 0.1, "σz", 1
h += couplings(env)[1], "σx", 1, "A + Adag", 2
for n in 1:N
    h += frequencies(env)[n], "N", n+1
end
for n in 1:N-1
    h += couplings(env)[n+1], "Adag", n+1, "A", n+2
    h += couplings(env)[n+1], "Adag", n+2, "A", n+1
end
H = MPI.bcast(MPO(h, s), comm)

dt = 0.01
tmax = 5

maxdim = 20
cutoff = 1e-8

ρₜ = MPI.bcast(InverseCanonicalMPS(s, n -> n == 1 ? "Up" : "0"), comm)
cb = ExpValueCallback("σz(1),σy(1)A + Adag(2)", s, dt)
parallel_tdvp2!(
    v, H, dt, tmax;
    comm=comm, callback=cb, cutoff=cutoff, maxdim=maxdim, progress=false
)

ρₜ′ = InverseCanonicalMPS(s, n -> n == 1 ? "Up" : "0")
cb′ = ExpValueCallback("σz(1),σy(1)A + Adag(2)", s, dt)
tdvp2!(
    ρₜ′, H, dt, tmax; callback=cb′, cutoff=cutoff, maxdim=maxdim, progress=false
)
```
