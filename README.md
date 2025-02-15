# MPSTimeEvolution

[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

(Forked from `https://github.com/orialb/TimeEvoMPS.jl`)

Implementations of time-evolution algorithms for matrix-product states using
[ITensors.jl](https://github.com/ITensor/ITensors.jl). 

## Installation

This package isn't included in Julia's general registry, but in [this
one](https://github.com/phaerrax/TensorNetworkSimulations),
so you need to add this registry first in order to download the package
automatically (you can skip this step if you already added the registry
previously: you need to do it just once per Julia installation). Open a
Julia interactive session and enter the Pkg REPL by hitting `]` (see the
[Getting Started with Environments](https://pkgdocs.julialang.org/v1/getting-started/#Getting-Started-with-Environments)
guide) then run `registry add https://github.com/phaerrax/TensorNetworkSimulations.git`:

```julia-repl
(@v1.11) pkg> registry add https://github.com/phaerrax/TensorNetworkSimulations.git
     Cloning registry from "https://github.com/phaerrax/TensorNetworkSimulations.git"
       Added registry `TensorNetworkSimulations` to `~/.julia/registries/TensorNetworkSimulations`
```

(If this is the first time using Julia then run `registry add General` too
otherwise the `General` registry won't be automatically added.)
You can then install this package by running `add MPSTimeEvolution` from the
Pkg REPL.
