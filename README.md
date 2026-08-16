# MPSTimeEvolution

[![Code Style:
Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://phaerrax.github.io/MPSTimeEvolution.jl/dev/)

(Originally forked from `https://github.com/orialb/TimeEvoMPS.jl`)

Implementations of time-evolution algorithms for matrix-product states, based on
the [ITensor](https://itensor.org/) library:

* one-site time-dependent variational principle (TDVP1) in its standard version
  as well as
   * its variant with adaptive bond dimensions, and
   * a non-unitary version (for vectorised mixed states);
* matrix-product states in the *Vidal* (or *canonical*) and *inverse-canonical*
  gauges,
* the time-evolving block-decimation algorithm (TEBD) for Vidal-form MPS,
  together with automatic 1st and 2nd order Suzuki-Trotter decompositions,
* the two-site time-dependent variational principle (TDVP2), in a serial and
  a parallel version.

See the [package
documentation](https://phaerrax.github.io/MPSTimeEvolution.jl/dev/) for a
complete list of features, references, a description of the available
methods, and some tutorials.

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
