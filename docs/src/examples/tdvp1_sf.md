# TDVP1 with superfermions

In this tutorial we will simulate the evolution of a fermionic \\(N\\)-body
system (without spin degrees of freedom) under a dissipative dynamics, that is,
the master equation

```math
\dot\rho_t = \lindblad(\rho_t) = -\iu [H, \rho_t] + \dissipator(\rho_t),
```

with the unitary part given by

```math
H = \sum_{n=1}^{N} \varepsilon_n\phantomadj \adj{c_n} c_n\phantomadj +
    \sum_{n=1}^{N-1} \lambda_n\phantomadj (\adj{c_n} c_{n+1}\phantomadj +
    \adj{c_{n+1}} c_n\phantomadj)
```

and the dissipation part by

```math
\dissipator(\rho_t) =
\sum_{n=1}^{N} \gamma_n\phantomadj (
    c_n\phantomadj \rho_t \adj{c_n}
    -\tfrac12 \adj{c_n} c_n\phantomadj \rho_t
    -\tfrac12 \rho_t \adj{c_n} c_n\phantomadj
).
```

We will do so using the superfermion formalism, where we represent a mixed state
over \\(N\\) fermions as a pure state over \\(2N\\) fermions.

!!! info "Advanced topics"
    In this page we will assume a certain degree of familiarity with the
    formalism, and we will skip the intermediate results of the
    calculations that allow us to rewrite the master equation in the final form.
    See [Brenes2020:TN_thermal_machines](@cite) for a more detailed example on
    how to use the superfermion formalism in practice.

## Time-evolution operator

In the superfermion formalism, let's call \\(a\sb{n}\\) and \\(\anc a\sb{n}\\)
the annihilation operators of the physical and ancillary modes, respectively,
and \\(\vec{\rho\sb{t}}\\) the transformed mixed state.  After the
fermion-to-superfermion mapping, we also swap particles and holes on the
ancillary sites via the \\(\anc a\sb{n} \leftrightarrow \adj{\anc a\sb{n}}\\)
canonical transformation. This allows us to rewrite the master equation in
such a way that the particle number is conserved.
With these transformations, the terms in the master equation are translated to

```math
[H, \blank] =
\sum_{n=1}^{N} \varepsilon_n\phantomadj (
     \adj{a_n}           a_n\phantomadj
    -\anc a_n\phantomadj \adj{\anc a_n}
) +
\sum_{n=1}^{N-1} \lambda_n\phantomadj (
     \adj{a_n}     a_{n+1}\phantomadj
    +\adj{a_{n+1}} a_n\phantomadj
    -\anc a_{n+1}\phantomadj \adj{\anc a_n}
    -\anc a_n\phantomadj     \adj{\anc a_{n+1}}
)
```

and

```math
\dissipator =
\sum_{n=1}^{N} \gamma_n (
    -a_n\phantomadj \adj{\anc a_n}
    -\tfrac12 \adj{a_n} a_n\phantomadj
    -\tfrac12 \anc a_n\phantomadj \adj{\anc a_n}
).
```

Both these operators act on the state by multiplying \\(\vec{\rho\sb{t}}\\) on
the left.
Let's start by creating the generator of the time-evolution operator that we
will plug in the TDVP function. We can use the OpSum method by ITensor, which
thankfully means that Jordan-Wigner strings will be automatically inserted, as
needed.
Remember that odd sites correspond to physical modes, and even sites to
ancillary modes.

```jldoctest tdvp1_sf; setup = :(using MPSTimeEvolution, ITensors, ITensorMPS), filter = r"id=\d+" => "id=###"
julia> N = 4; s = siteinds("Fermion", 2N);

julia> ε = 0.1; ꟛ = 0.5; γ = 0.2;

julia> ℓ = OpSum();

julia> for n in 1:N
           ℓ += -im * ε, "cdag * c", 2n-1
           ℓ += im * ε, "c * cdag", 2n
       end

julia> for n in 1:N-1
           ℓ += -im * ꟛ, "cdag", 2n-1, "c", 2n+1
           ℓ += -im * ꟛ, "cdag", 2n+1, "c", 2n-1
           ℓ += im * ꟛ, "c", 2n, "cdag", 2n+2
           ℓ += im * ꟛ, "c", 2n+2, "cdag", 2n
       end

julia> for n in 1:N
           ℓ += -γ, "c", 2n-1, "cdag", 2n
           ℓ += -γ/2, "cdag", 2n-1, "c", 2n-1
           ℓ += -γ/2, "c", 2n, "cdag", 2n
       end

julia> L = MPO(ℓ, s)
8-element MPO:
 ((dim=4|id=582|"Link,l=1"), (dim=2|id=444|"Fermion,Site,n=1")', (dim=2|id=444|"Fermion,Site,n=1"))
 ((dim=4|id=582|"Link,l=1"), (dim=6|id=796|"Link,l=2"), (dim=2|id=902|"Fermion,Site,n=2")', (dim=2|id=902|"Fermion,Site,n=2"))
 ((dim=6|id=796|"Link,l=2"), (dim=6|id=899|"Link,l=3"), (dim=2|id=138|"Fermion,Site,n=3")', (dim=2|id=138|"Fermion,Site,n=3"))
 ((dim=6|id=899|"Link,l=3"), (dim=6|id=481|"Link,l=4"), (dim=2|id=971|"Fermion,Site,n=4")', (dim=2|id=971|"Fermion,Site,n=4"))
 ((dim=6|id=481|"Link,l=4"), (dim=6|id=127|"Link,l=5"), (dim=2|id=325|"Fermion,Site,n=5")', (dim=2|id=325|"Fermion,Site,n=5"))
 ((dim=6|id=127|"Link,l=5"), (dim=6|id=664|"Link,l=6"), (dim=2|id=159|"Fermion,Site,n=6")', (dim=2|id=159|"Fermion,Site,n=6"))
 ((dim=6|id=664|"Link,l=6"), (dim=4|id=534|"Link,l=7"), (dim=2|id=92|"Fermion,Site,n=7")', (dim=2|id=92|"Fermion,Site,n=7"))
 ((dim=4|id=534|"Link,l=7"), (dim=2|id=934|"Fermion,Site,n=8")', (dim=2|id=934|"Fermion,Site,n=8"))

```

## Initial state

Now we create the initial site. Let's start with the first mode in the occupied
state, and the other modes in the empty state.
This means that we need to initialise the physical sites this way and the
ancillary sites in the opposite way (due to the particle-hole inversion we
performed).
We also enlarge a bit the link indices of the state, so that there's more room
for the dynamics.

```jldoctest tdvp1_sf; filter = r"id=\d+" => "id=###"
julia> sitenames = Vector{String}(undef, 2N);

julia> sitenames[1] = "Occ"; sitenames[2] = "Emp";

julia> for n in 2:N
           sitenames[2n-1] = "Emp"
           sitenames[2n] = "Occ"
       end

julia> Vρₜ = enlargelinks(MPS(s, sitenames), 5)
8-element MPS:
 ((dim=2|id=444|"Fermion,Site,n=1"), (dim=5|id=107|"Link,l=1"))
 ((dim=2|id=902|"Fermion,Site,n=2"), (dim=5|id=107|"Link,l=1"), (dim=5|id=95|"Link,l=2"))
 ((dim=2|id=138|"Fermion,Site,n=3"), (dim=5|id=95|"Link,l=2"), (dim=5|id=969|"Link,l=3"))
 ((dim=2|id=971|"Fermion,Site,n=4"), (dim=5|id=969|"Link,l=3"), (dim=5|id=897|"Link,l=4"))
 ((dim=2|id=325|"Fermion,Site,n=5"), (dim=5|id=897|"Link,l=4"), (dim=5|id=362|"Link,l=5"))
 ((dim=2|id=159|"Fermion,Site,n=6"), (dim=5|id=362|"Link,l=5"), (dim=5|id=620|"Link,l=6"))
 ((dim=2|id=92|"Fermion,Site,n=7"), (dim=5|id=620|"Link,l=6"), (dim=3|id=489|"Link,l=7"))
 ((dim=2|id=934|"Fermion,Site,n=8"), (dim=3|id=489|"Link,l=7"))

```

## Callback operator

Then, we create the callback operator. We set the time step and the operators we
would like to track: let's take the number operator on the first site
(corresponding to site 1 of the MPS), but also the two-body operators
\\(\adj{c\sb2} c\sb3\phantomadj\\) and \\(\adj{c\sb3} c\sb2\phantomadj\\) (which
act on sites 3 to 5 of the MPS).
We need to add Jordan-Wigner strings (manually, this time) since the 2nd and
3rd physical modes are not adjacent: the 2nd ancillary mode sits between them.

```jldoctest tdvp1_sf
julia> dt = 0.1; tmax = 1;

julia> cb = SuperfermionCallback("N(1,2,3,4,5,6,7,8),Adag(3)F(4)A(5),A(3)F(4)Adag(5)", s, dt)
SuperfermionCallback
Operators: N(1), N(2), N(3), N(4), N(5), N(6), N(7), N(8), Adag(3)F(4)A(5) and A(3)F(4)Adag(5)
No measurements performed

```

Everything is ready for the time evolution.

```jldoctest tdvp1_sf
julia> tdvp1vec!(Vρₜ, L, dt, tmax; callback=cb, progress=false);
```

Let's check that the simulation went well. First, let's have a look at the trace
of the state:

```jldoctest tdvp1_sf; filter = r"^(\s+)?[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)? [-+] [-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?im"m => "### + ###im"
julia> measurements_norm(cb)
11-element Vector{ComplexF64}:
                1.0 + 0.0im
 0.9999976513290173 - 8.977259492595898e-7im
 0.9999948101350274 - 3.163081857403324e-6im
 0.9999951242590129 - 4.223380661553636e-6im
 0.9999953256476534 - 7.513087658280052e-6im
 0.9999969286604952 - 1.3626527300478344e-5im
 0.9999966498093151 - 1.476742086858933e-5im
 0.9999959355740604 - 1.4754443837100815e-5im
 0.9999955439946122 - 1.4815943552764734e-5im
 0.9999953010430082 - 1.4917206316730056e-5im
 0.9999951466402556 - 1.5043968837182187e-5im

```

We have in the end a \\(10^{-5}\\) deviation from one, which is not bad
considering we didn't pay too much attention to how we chose the numerical
parameters.

When reading out the expectation values, we divide them first by the trace of
the state: we get

```jldoctest tdvp1_sf; filter = r"^(\s+)?[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)? [-+] [-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?im"m => "### + ###im"
julia> expvalues(cb, "N(1)") ./ measurements_norm(cb)
11-element Vector{ComplexF64}:
                1.0 + 0.0im
 0.9777533155972749 + 9.408225602693399e-7im
 0.9512279385080097 + 3.063819209978666e-6im
 0.9207807528037154 + 4.000141905586285e-6im
 0.8868126447959417 + 7.10344513852771e-6im
 0.8497502908352624 + 1.2740157612670506e-5im
 0.8100477003023183 + 1.338809349130348e-5im
 0.7681698188095493 + 1.285251128655659e-5im
 0.7245898612298369 + 1.2364375506595535e-5im
 0.6797821443239995 + 1.1902610329399669e-5im
 0.6342140100619231 + 1.1462799241583946e-5im

```

which is a sensible result; the total occupation number is decreasing,

```jldoctest tdvp1_sf; filter = r"^(\s+)?[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)? [-+] [-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?im"m => "### + ###im"
julia> sum(expvalues(cb, "N($n)") for n in 1:2:2N) ./ measurements_norm(cb)
11-element Vector{ComplexF64}:
                1.0 + 0.0im
 0.9802018352373681 + 2.023294349033268e-6im
 0.9607953111125953 + 3.541664026343726e-7im
 0.9417703816078047 + 1.803510234288367e-7im
 0.9231220007367444 + 2.4824011214396457e-7im
 0.9048423807287197 + 1.3948843279879256e-7im
 0.8869233043316959 - 6.757077633315075e-8im
  0.869360071398266 - 1.890793175562648e-7im
  0.852145124344158 - 3.6222539732203834e-7im
 0.8352712349011422 - 5.654417529076439e-7im
 0.8187315653670031 - 7.884578175869884e-7im

```

which is what we expect from a dissipative dynamics. (Note that the particle
number conservation is not broken: the conservation happens at the level of the
physical _and ancillary_ sites, while here we are looking only at the physical
ones.)

## Time-evolution with quantum numbers

As we've said before, the time evolution preserves the total number of fermions,
in the superfermion version.
We can use this property to speed up our calculations by using ITensor's quantum
number conservation feature, as follows.

First, we need to define a new set of indices with QN conservation active. With
the Fermion site type and the number of particles, we need to set the
`conserve_nf` keyword argument to `true` in the `siteinds` function.

```jldoctest tdvp1_sf; filter = r"id=\d+" => "id=###"
julia> s_qn = siteinds("Fermion", 2N; conserve_nf=true)
8-element Vector{Index{Vector{Pair{QN, Int64}}}}:
 (dim=2|id=425|"Fermion,Site,n=1") <Out>
 1: QN("Nf",0,-1) => 1
 2: QN("Nf",1,-1) => 1
 (dim=2|id=520|"Fermion,Site,n=2") <Out>
 1: QN("Nf",0,-1) => 1
 2: QN("Nf",1,-1) => 1
 (dim=2|id=73|"Fermion,Site,n=3") <Out>
 1: QN("Nf",0,-1) => 1
 2: QN("Nf",1,-1) => 1
 (dim=2|id=677|"Fermion,Site,n=4") <Out>
 1: QN("Nf",0,-1) => 1
 2: QN("Nf",1,-1) => 1
 (dim=2|id=806|"Fermion,Site,n=5") <Out>
 1: QN("Nf",0,-1) => 1
 2: QN("Nf",1,-1) => 1
 (dim=2|id=278|"Fermion,Site,n=6") <Out>
 1: QN("Nf",0,-1) => 1
 2: QN("Nf",1,-1) => 1
 (dim=2|id=321|"Fermion,Site,n=7") <Out>
 1: QN("Nf",0,-1) => 1
 2: QN("Nf",1,-1) => 1
 (dim=2|id=633|"Fermion,Site,n=8") <Out>
 1: QN("Nf",0,-1) => 1
 2: QN("Nf",1,-1) => 1

```

Then, we redefine everything on top of these new site indices.

```jldoctest tdvp1_sf; filter = r"id=\d+" => "id=###"
julia> L_qn = MPO(ℓ, s_qn);

julia> Vρₜ_qn = enlargelinks(MPS(s_qn, sitenames), 5; ref_state=sitenames);

julia> cb_qn = SuperfermionCallback("N(1),Adag(3)F(4)A(5),A(3)F(4)Adag(5)", s_qn, dt)
SuperfermionCallback
Operators: N(1), Adag(3)F(4)A(5) and A(3)F(4)Adag(5)
No measurements performed
```

Finally, we run the time evolution again. The function we need to call is still
`tdvp1vec!`, exactly as before. Nothing changes here, except that we use the new
QN-conserving objects.

```jldoctest tdvp1_sf
julia> tdvp1vec!(Vρₜ_qn, L_qn, dt, tmax; callback=cb_qn, progress=false);
```

The results are not dramatically different from the previous, non-QN-conserving
case:

```jldoctest tdvp1_sf; filter = r"^(\s+)?[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)? [-+] [-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?im"m => "### + ###im"
julia> measurements_norm(cb_qn)
11-element Vector{ComplexF64}:
                1.0 + 0.0im
 1.0000007697435818 + 3.2286371663490507e-6im
 1.0000007427336053 + 2.9921122286667975e-6im
 1.0000010844937897 + 3.15357514420747e-6im
 1.0000012168824521 + 3.5175397591778605e-6im
 1.0000009773367482 + 4.005154016085866e-6im
 1.0000003978223262 + 4.53047339013793e-6im
 0.9999995928493433 + 5.056645850125124e-6im
 0.9999986491689713 + 5.570968404683099e-6im
 0.9999976183658691 + 6.066318813722024e-6im
 0.9999965322448163 + 6.534766671983195e-6im

julia> expvalues(cb_qn, "N(1)") ./ measurements_norm(cb_qn)
11-element Vector{ComplexF64}:
                1.0 + 0.0im
 0.9777500503027625 - 2.58452718946664e-6im
 0.9512219601550865 - 2.5881967524595883e-6im
 0.9207744474824726 - 2.9355763496950214e-6im
 0.8868057607160773 - 3.446751948491078e-6im
 0.8497440916657568 - 4.020306469595358e-6im
 0.8100403683220311 - 4.560347432640881e-6im
 0.7681607524371609 - 5.0232705946423395e-6im
 0.7245791385952755 - 5.390485864829719e-6im
 0.6797697161251433 - 5.650949979663973e-6im
 0.6341998097851084 - 5.795888235658188e-6im

```

and so on.
