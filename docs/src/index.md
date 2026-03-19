# MPSTimeEvolution.jl

*This is the documentation for the MPSTimeEvolution.jl package.*

This package implements time-evolution algorithms for tensor networks.
It is based on the [ITensor](https://itensor.org/) library.

## Package features

The package contains an implementation of the one-site time-dependent variational principle (TDVP1) [Lubich2015:tdvp_evolution,Haegeman2016:unifying_time_evolution_optimization_mps,Paeckel2019:time_evolution_methods](@cite) in its standard version, together with:

* the variant with adaptive bond dimensions [Dunnett2021:adaptive_tdvp1](@cite),
* the non-unitary version (for vectorised mixed states).

See [Reference](@ref) for a complete list of features, and a description of the
available methods.

## Bibliography

```@bibliography
```
