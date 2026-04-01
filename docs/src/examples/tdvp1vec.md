# Non-unitary TDVP1

The non-unitary, or _vectorised_, TDVP1 algorithm works in an analogous way as
its unitary counterpart, except that it can be used to solve a more general
ODE problem

```math
\begin{cases}
    \dot f_t = L(f_t),\\
    f_0 = u
\end{cases}
```

where \\(L\\) is a linear operator. In particular, we will use it to solve the
GKSL equation \\(\dot\rho\sb{t} = \lindblad(\rho\sb{t})\\) where \\(\lindblad\\)
is the Lindblad operator.
The structure of the time-evolution part is actually the same as the unitary
TDVP algorithm; what changes is that the expectation values of operators are
computed differently, since with a mixed state \\(\avg{A}\sb{t} = \tr(A
\rho\sb{t})\\) instead of \\(\bra{\psi\sb{t}} A \ket{\psi\sb{t}}\\).

```@docs; canonical=false
tdvp1vec!
```

In order to define mixed states and operators acting on it, we rely on the
[LindbladVectorizedTensors](https://github.com/phaerrax/LindbladVectorizedTensors.jl)
package, which provides new site types that can be used in this scenario.
Our Hamiltonian operator will be

```math
H = -\frac12 \sum_{n=1}^{N-1} \pauliz[n] \pauliz[n+1] +\sum_{n=1}^{N} \paulix[n]
```

and we will start with a state with alternating magnetisation,

```math
\rho_0 = \outp{\spinup}{\spinup} \otimes \outp{\spindown}{\spindown} \otimes \outp{\spinup}{\spinup} \otimes \outp{\spindown}{\spindown} \otimes \dotsb {}
```

just as in the tutorial for the unitary TDVP1 algorithm.

!!! tip "Unitary part of the GKSL equation"
    LindbladVectorizedTensors provides some utility functions that allow us
    to create the \\(-\iu[H,\blank]\\) part of the GKSL equation easily: the
    `gkslcommutator` takes a list of strings and integers `s1, n1, s2, n2, ...` 
    and returns an OpSum object representing \\(-\iu[S,\blank]\\) where \\(S\\)
    is the product of each `sk` on site `nk`.
    
    ```jldoctest; setup = :(using LindbladVectorizedTensors)
    julia> gkslcommutator("A", 1, "B", 3)
    sum(
      0.0 - 1.0im A⋅(1,) B⋅(3,)
      0.0 + 1.0im ⋅A(1,) ⋅B(3,)
    )
    ```

    In this notation, `A⋅` is an operator that multiplies by `A` on the left,
    while `⋅A` is the multiplication on the right.

```jldoctest tdvp1vec
julia> using ITensorMPS, MPSTimeEvolution, LindbladVectorizedTensors

julia> N = 10; s = siteinds("vS=1/2", N);

julia> ρₜ = MPS(s, n -> isodd(n) ? "↑" : "↓");

julia> ℓ = OpSum();

julia> for n in 1:N
           ℓ += gkslcommutator("σx", n)
       end

julia> for n in 1:N-1
           ℓ += -0.5 * gkslcommutator("σz", n, "σz", n+1)
       end
```

Here the strings `"↑"` and `"↓"` define pure states in the up and down spin
positions respectively; this time, unlike with what we had in the unitary TDVP1
tutorial, the pure state is e.g. \\(\outp{\spinup}{\spinup}\\) instead of just
\\(\ket{\spinup}\\).

We also add some non unitary terms:

```math
\dissipator_n(\rho_t) = \paulim[n] \rho_t \paulip[n] - \tfrac12
(\paulip[n] \paulim[n] \rho_t + \rho_t \paulip[n] \paulim[n])
```

on the first and last sites, `n = 1` and `n = N`.

```jldoctest tdvp1vec
julia> for n in [1,N]
           ℓ += "σ-⋅ * ⋅σ+", n
           ℓ += -0.5, "σ+⋅ * σ-⋅", n
           ℓ += -0.5, "⋅σ- * ⋅σ+", n
       end
```

(Note the inverted order in `ℓ += 0.5, "σ-⋅ * σ+⋅", n` with respect to the
definition of \\(\dissipator\sb{n}\\): first the state is multiplied on the
right by \\(\sigma\sb+\\), then by \\(\sigma\sb-\\).)

Finally we construct the MPO:

```jldoctest tdvp1vec
julia> L = MPO(ℓ, s);
```

We set the time step and the total evolution time to

```jldoctest tdvp1vec
julia> dt = 0.1; tmax = 1;
```

and we define a callback object to track the \\(z\\)-axis magnetisation on the
first three sites:

```jldoctest tdvp1vec
julia> cb = ExpValueCallback("Sz(1,2,3)", s, dt)
ExpValueCallback
Operators: Sz{1}, Sz{2} and Sz{3}
No measurements performed
```

For the other keyword arguments, the default value is already enough. Note that
here `hermitian=false` by default, which is what we want.

!!! tip "Normalisation of the state"
    In the mixed-state scenario, normalising the state \\(\rho\sb{t}\\) after
    each time step would mean dividing it by its trace, so that
    \\(\tr\rho\sb{t}\\) is always 1. Note that in this non-unitary case the
    TDVP1 algorithm does not preserve the trace of the state, as the norm of the
    MPS (which _is_ preserved) corresponds to \\(\tr(\rho\sb{t}^2)\\) and not to
    \\(\tr\rho\sb{t}\\).
    As a consequence of the approximations made for the solution of the ODE,
    and especially of the truncation of the bond dimensions, the trace of the
    state is likely to decrease over time. A very slow, linear decrease is still
    acceptable in a “healthy” simulation.  Faster (for example quadratic or
    exponential) decreases usually signal instead that the bond dimension is
    not large enough to contain the time evolution.
    For this reason, we suggest not to normalise the state so that the trace can
    be monitored during the evolution; the expectation values will then need to
    be divided by \\(\tr\rho\sb{t}\\).

Let's assign some temporary files to the output arguments:

```jldoctest tdvp1vec
julia> meas_file = mktemp();

julia> bdim_file = mktemp();

julia> time_file = mktemp();
```

and finally start the evolution, by calling the `tdvp1vec!` method.

```jldoctest tdvp1vec
julia> tdvp1vec!(ρₜ, L, dt, tmax; callback=cb, io_file=meas_file, io_ranks=bdim_file, io_times=time_file, progress=false);
```

The output files look like the ones in [Standard TDVP1](@ref): `meas_file` is

```csv
time,Sz{1}_re,Sz{1}_im,Sz{2}_re,Sz{2}_im,Sz{3}_re,Sz{3}_im,Norm_re,Norm_im
0.0,0.4999999999999991,0.0,-0.499999999999999,0.0,0.499999999999999,0.0,0.9999999999999978,0.0
0.1,0.39598707437112335,6.0527724732136165e-19,-0.490065811967275,-1.353844415229414e-19,0.49006599630691594,-4.7762507132213846e-20,0.9999999999999987,1.4812772427663158e-18
0.2,0.2877548464126332,1.4009187515124302e-18,-0.4610129673309902,-3.39292416943428e-18,0.4610293716309977,4.1677181825687184e-19,0.9999999999999984,4.366378241259551e-18
0.30000000000000004,0.1805706668022498,2.627580179970803e-18,-0.41482129814200247,-8.044545668147055e-18,0.41500258039544197,3.425397109753093e-18,0.9999999999999983,1.2777889291707695e-17
0.4,0.07892708986289437,2.0587241596330907e-18,-0.35404509807228407,-9.130826148772616e-18,0.354973559452091,5.992406651507898e-18,0.9999999999999972,2.011387971123154e-17
0.5,-0.013614234850474383,3.095356401259337e-19,-0.2811888370142225,-8.282561319457782e-18,0.2842340305584332,5.9875985430418044e-18,0.9999999999999971,2.2722776268818143e-17
0.6,-0.09447754435453204,-1.3403375717698688e-18,-0.19849330517525163,-5.349265383030454e-18,0.20589747180678913,3.553570973393531e-18,0.9999999999999952,2.221330462678641e-17
0.7,-0.16203717679461033,-2.9113702579930808e-18,-0.10815122586884929,-3.0467668830231345e-18,0.12262136410700923,2.0259789689511548e-18,0.9999999999999941,2.3307297228216787e-17
0.7999999999999999,-0.21551796652772393,-3.3631311974582865e-18,-0.012692986276901385,-5.432595679719254e-19,0.036560457480609536,2.9417533665856377e-19,0.9999999999999929,2.326474779623259e-17
0.8999999999999999,-0.2548640708794142,-3.7663616396505436e-18,0.08471373566560535,2.127981002920741e-18,-0.05048360438262194,-1.2797687400172327e-18,0.9999999999999921,2.3281912526203648e-17
0.9999999999999999,-0.28060544974450025,-4.333339075314165e-18,0.18017089528582053,4.8776144871940434e-18,-0.13678392049062973,-2.87385447186911e-18,0.9999999999999908,2.4079354343448627e-17
```

While the `tdvp1!` method prints the 2-norm of the pure state under the
`Norm_re` and `Norm_im` columns, the `tdvp1vec!` method prints the trace of the
mixed state instead.  In this case, we see that \\(\tr\rho\sb{t}\\) is
approximately 1 for the whole evolution, which is good.
The other two files `bdim_file` and `time_file` are
completely analogous to the ones in the unitary case.
