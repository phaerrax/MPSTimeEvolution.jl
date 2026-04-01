# Standard TDVP1

Let's see how this package works by looking at the ordinary TDVP1 method,
that is, the time-evolution algorithm for the Schrödinger equation.
This is the algorithm described for example in [Lubich2015:tdvp_evolution,Haegeman2016:unifying_time_evolution_optimization_mps](@cite) and [Paeckel2019:time_evolution_methods](@cite).

```@docs; canonical=false, collapsed=true
tdvp1!
```

We need an initial (pure) state and a Hamiltonian. We will take a simple system,
a Heisenberg model given by

```math
H = -\frac12 \sum_{n=1}^{N-1} \pauliz[n] \pauliz[n+1] +\sum_{n=1}^{N} \paulix[n]
```

starting with a state with alternating magnetisation,

```math
\ket{\psi_0} = \ket{\spinup} \otimes \ket{\spindown} \otimes \ket{\spinup} \otimes \ket{\spindown} \otimes \dotsb {}
```

Let's review what we need to set up in order to use the `tdvp1!` method.  First
of all, we define the state and Hamiltonian objects in Julia, with ITensor.

```jldoctest tdvp1
julia> using ITensorMPS, MPSTimeEvolution

julia> N = 10; s = siteinds("S=1/2", N);

julia> ψₜ = MPS(s, n -> isodd(n) ? "↑" : "↓");

julia> h = OpSum();

julia> for n in 1:N
           h += "σx", n
       end

julia> for n in 1:N-1
           h += -0.5, "σz", n, "σz", n+1
       end

julia> H = MPO(h, s);
```

Here we constructed the Hamiltonian with the OpSum feature of ITensor, but any
method is fine as long as, in the end, we have an MPO.
We will also choose the time step and the total evolution time

```jldoctest tdvp1
julia> dt = 0.1; tmax = 1;
```

Let's review the relevant keyword arguments we need to set.
The first one is the _callback object_, which contains information about which
operators we want to track, and how frequently.
Callback objects are reviewed [here](@ref "Callback objects"); in this example,
we define a callback object to track the \\(z\\)-axis magnetisation on the first
three sites.

```jldoctest tdvp1
julia> cb = ExpValueCallback("Sz(1,2,3)", s, dt)
ExpValueCallback
Operators: Sz{1}, Sz{2} and Sz{3}
No measurements performed
```

The last argument of `ExpValueCallback` determines how frequently the
expectation values should be computed. Here we set it equal to `dt`, meaning
that the expectation values will be computed after each time step. If we set it,
for example, to `5dt`, then they would be computed only every fifth time step.
This can be useful if we the expectation values are expensive to compute and we
don't need them at every time step.

We also want to provide three filenames, `io_file`, `io_ranks` and `io_times`,
in which the output of the simulation will be printed step-by-step,
respectively:

* the expectation values of the operators specified in the callback object,
  together with the norm of the state,
* the bond dimensions of the state MPS,
* the wall-clock time needed for a complete sweep of the time-evolution
  algorithm.

For the other keyword arguments, the default value is already enough.
Let's create three temporary files in which to store the results:

```jldoctest tdvp1
julia> meas_file = mktemp();

julia> bdim_file = mktemp();

julia> time_file = mktemp();
```

!!! warning "Preserving the initial state"
    The `tdvp1!` method, as the exclamation point in the name suggests, modifies
    its arguments, in this case the state MPS. If you want to preserve the
    initial state, you need to explicitly copy it before the evolution begins,
    for example by calling `ψ₀ = deepcopy(ψₜ)`.

Now we're all set! Let's call the `tdvp1!` method and begin the time evolution.

```jldoctest tdvp1
julia> tdvp1!(ψₜ, H, dt, tmax; callback=cb, io_file=meas_file, io_ranks=bdim_file, io_times=time_file, progress=false);
```

Some information about the simulation, such as the total memory used, is printed
with the progress bar and updated with every step.

!!! tip "Partial results"
    The output files are updated progressively as the time evolution goes on:
    once the TDVP algorithm completes a time step, the expectation values of the
    callback operators are computed and the results are printed on the output
    file. This means that you can check the output of the evolution “live” while
    the algorithm is still running, and that even if the program were to crash,
    the results computed up to that point remain available in the external
    files.

Here is how the output file `meas_file` looks:

```csv
time,Sz{1}_re,Sz{1}_im,Sz{2}_re,Sz{2}_im,Sz{3}_re,Sz{3}_im,Norm_re,Norm_im
0.0,0.5,0.0,-0.5,0.0,0.5,0.0,1.0,0.0
0.1,0.4900414332914067,5.3791631483433606e-18,-0.4900659962056321,-1.377536144717721e-20,0.49006599630768083,-1.6417627115633278e-20,0.9999999999999994,0.0
0.2,0.4606555451703479,-8.146889331862475e-18,-0.46102931085129484,-7.089461369668159e-20,0.4610293732495007,4.612226493880881e-20,1.0000000000000002,0.0
0.30000000000000004,0.41325597697502137,1.1458579823129588e-17,-0.41500128632300826,4.832769491957939e-19,0.415002665510601,-8.3435393759901015e-19,1.0000000000000002,0.0
0.4,0.35003041948068225,1.1893222271364655e-17,-0.3549633507219295,1.0491784051123786e-18,0.3549748137016917,9.823346134202877e-19,1.0000000000000002,0.0
0.5,0.2737403828182838,-6.352627987291848e-18,-0.28418784834982486,3.255000900727609e-20,0.2842433069354116,1.077647677853708e-18,1.0000000000000002,0.0
0.6,0.18751903974596493,1.4475777098662835e-18,-0.20575298359674477,-2.6867981267790545e-18,0.20594116624070716,-1.0967194168065504e-19,1.0000000000000002,0.0
0.7,0.09470203724582948,1.5053865161571372e-18,-0.1222753891444935,5.106765686808509e-18,0.12277011595120922,-3.4507629265319122e-18,1.0000000000000004,0.0
0.7999999999999999,-0.0012947785980338977,3.0757672918058667e-18,-0.03588626945783665,1.855438881605692e-18,0.03695544978746935,-2.1079528501511206e-18,1.0000000000000002,0.0
0.8999999999999999,-0.09705465554511021,-6.392373957922197e-18,0.05159589677448195,6.803831510665656e-18,-0.04962469003523777,3.5816523246518e-18,0.9999999999999994,0.0
0.9999999999999999,-0.18921090489833187,-2.297309303060014e-18,0.13837867822101355,-6.363169223469177e-18,-0.13520574981975464,1.0456234814271835e-17,1.0,0.0
```

It is a CSV file whose columns represent:

* the current evolution time;
* the expectation values of the operators in the callback object, with their
  real and imaginary parts (useful when the operator is not Hermitian);
* The norm of the MPS.

The `bdim_file` file looks like this:

```csv
time,1,2,3,4,5,6,7,8,9
0.0,1,1,1,1,1,1,1,1,1
0.1,1,1,1,1,1,1,1,1,1
0.2,1,1,1,1,1,1,1,1,1
0.30000000000000004,1,1,1,1,1,1,1,1,1
0.4,1,1,1,1,1,1,1,1,1
0.5,1,1,1,1,1,1,1,1,1
0.6,1,1,1,1,1,1,1,1,1
0.7,1,1,1,1,1,1,1,1,1
0.7999999999999999,1,1,1,1,1,1,1,1,1
0.8999999999999999,1,1,1,1,1,1,1,1,1
0.9999999999999999,1,1,1,1,1,1,1,1,1
```

The first column is again the current time, while the column `n` contains the
dimension of the bond index between site `n` and site `n+1`. Here,
unsurprisingly, these dimensions are all 1, and that is because we started from
an MPS whose bond dimension was 1, and the TDVP1 algorithm does not alter the
bond dimensions.

!!! tip "Increasing the bond dimensions"
    Since with the TDVP1 algorithm the bond dimensions of the MPS remain fixed
    throughout the evolution, they cannot expand to accommodate long-range
    correlations that may arise. To remedy this, the MPS can be manually
    enlarged before the evolution begins: this functionality is provided by the
    [`enlargelinks`](@ref) method. For example, if we wanted to start with an
    MPS with a bond dimension of 5 everywhere, we would call `enlargelinks(ψₜ,
    5)`.

Finally, the `time_file` file is a simple list, that prints the
real-world time (in seconds) that the evolution algorithm took for each step. It
should have one row less than the previous two files, as they also include data
on the initial state.

```csv
walltime/s
0.013851264
0.01222606
0.011079769
0.010687288
0.040394264
0.012894465
0.011599275
0.01163784
0.01080914
0.011201621
```
