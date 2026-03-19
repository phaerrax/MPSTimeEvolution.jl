# MPSTimeEvolution

[![Code Style:
Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://phaerrax.github.io/MPSTimeEvolution.jl/dev/)
(Forked from `https://github.com/orialb/TimeEvoMPS.jl`)

Implementations of time-evolution algorithms for matrix-product states.
Based on the [ITensor](https://itensor.org/) library.

## Installation

### From a registry

This package is registered in the
[TensorNetworkSimulations](https://github.com/phaerrax/TensorNetworkSimulations)
registry. By first adding this registry, with

```julia
using Pkg
pkg"registry add https://github.com/phaerrax/TensorNetworkSimulations.git"
```

(this must be done just once per Julia installation) the package can then be
installed as a normal one:

```julia
using Pkg
pkg"add MPSTimeEvolution"
```

### From GitHub

Alternatively, straight installation from GitHub is also possible:

```julia
using Pkg
pkg "add https://github.com/phaerrax/MPSTimeEvolution.jl"
```
