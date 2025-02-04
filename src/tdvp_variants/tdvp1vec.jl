export tdvp1vec!, adaptivetdvp1vec!

using ITensors: permute
using ITensorMPS: position!, set_nsite!, check_hascommoninds

"""
    tdvp1vec!([solver,] state::MPS, L::MPO, dt, tmax; kwargs...)
    tdvp1vec!([solver,] state::MPS, L::Vector{MPO}, dt, tmax; kwargs...)

Integrate the equation of motion ``d/dt ρₜ = L(ρₜ)`` using the one-site TDVP algorithm, from
`0` to `tmax` in time steps of `dt`. 
The MPS `state` represents ``ρ`` in a vectorised form.
The evolution operator can be given either as a single MPO or as a vector of MPOs, in the
latter case the evolution operator is taken to be the sum of the elements in the vector.

# Other arguments

* `solver`: a function which takes three arguments `A`, `t`, `B` (and possibly other keyword
    arguments) where `t` is a time step, `B` an ITensor and `A` a linear operator on `B`,
    returning the time-evolved `B`. It defaults to `KrylovKit.exponentiate`.
* `dt`: time step of the evolution.
* `tmax`: end time of the evolution.

# Optional keyword arguments

* `cb`: a callback object describing the observables.
* `hermitian` (default: `false`): whether `L` is an Hermitian operator.
* `exp_tol` (default: `1e-14`): accuracy per unit time for `KrylovKit.exponentiate`.
* `krylovdim` (default: `30`): maximum dimension of the Krylov subspace that will be
    constructed.
* `maxiter` (default: `100`): number of times the Krylov subspace can be rebuilt.
* `normalize` (default: `false`): whether `state` is renormalised after each step.
* `io_file` (default: `nothing`): output file for step-by-step measurements.
* `io_ranks` (default: `nothing`): output file for step-by-step bond dimensions.
* `io_times` (default: `nothing`): output file for simulation wall-clock times.
* `store_psi0` (default: `false`): whether to keep information about the initial state.
* `which_decomp` (default: `"qr"`): name of the decomposition method for the sweeps.
* `progress` (default: `true`): whether to display a progress bar during the evolution.
"""
function tdvp1vec! end

function tdvp1vec!(solver, state::MPS, Ls::Vector{MPO}, dt, tmax; kwargs...)
    # This is copied from ITensorsTDVP. Not sure why it's useful...
    for L in Ls
        check_hascommoninds(siteinds, L, state)
        check_hascommoninds(siteinds, L, state')
    end
    Ls .= permute.(Ls, Ref((linkind, siteinds, linkind)))
    PLs = ProjMPOSum(Ls)
    return tdvp1vec!(solver, state, PLs, dt, tmax; kwargs...)
end

function tdvp1vec!(solver, state::MPS, L::MPO, dt, tmax; kwargs...)
    return tdvp1vec!(solver, state, ProjMPO(L), dt, tmax; kwargs...)
end

function tdvp1vec!(solver, state::MPS, PH, dt, tmax; kwargs...)
    nsteps = Int(tmax / dt)
    cb = get(kwargs, :callback, NoTEvoCallback())
    hermitian = get(kwargs, :hermitian, false)
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    normalize = get(kwargs, :normalize, false)
    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)
    store_state0 = get(kwargs, :store_psi0, false)
    decomp = get(kwargs, :which_decomp, "qr")

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving state... ")
    else
        pbar = nothing
    end

    # Vectorized equations of motion usually are not defined by an anti-Hermitian operator
    # such as -im H in Schrödinger's equation, so we do not bother here with "unitary" or
    # "imaginary-time" evolution types. We just have a generic equation of the form
    # v'(t) = L v(t)  ==>  v(t) = exp(tL) v(0).

    store_state0 && (state0 = copy(state))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(state))
    times_handle = writeheaders_stime(times_file)

    N = length(state)

    # Measure everthing once in the initial state.
    current_time = 0.0
    apply!(cb, state, TDVP1vec(); t=current_time, sweepend=true)

    if store_state0
        printoutput_data(io_handle, cb, state; psi0=state0, vectorized=true, kwargs...)
    else
        printoutput_data(io_handle, cb, state; vectorized=true, kwargs...)
    end
    printoutput_ranks(ranks_handle, cb, state)

    # Prepare for first iteration.
    orthogonalize!(state, 1)
    set_nsite!(PH, 1)
    position!(PH, state, 1)

    for s in 1:nsteps
        # In TDVP1 only one site at a time is modified, so we iterate on the sites
        # of the state's MPS, not the bonds.
        stime = @elapsed for (site, ha) in sweepnext(N; ncenter=1)
            # sweepnext(N) is an iterable object that evaluates to tuples of the form
            # (bond, ha) where bond is the bond number and ha is the half-sweep number.
            # The kwarg ncenter determines the end and turning points of the loop: if
            # it equals 1, then we perform a sweep on each single site.
            sweepdir = (ha == 1 ? "right" : "left")
            tdvp_site_update!(
                solver,
                PH,
                state,
                site,
                0.5dt;
                current_time=(ha == 1 ? current_time + 0.5dt : current_time + dt),
                sweepdir=sweepdir,
                which_decomp=decomp,
                hermitian=hermitian,
                exp_tol=exp_tol,
                krylovdim=krylovdim,
                maxiter=maxiter,
            )
        end

        current_time += dt

        # Now the backwards sweep has ended, so the whole MPS of the state is up-to-date.
        # We can then calculate the expectation values of the observables within cb.
        mtime = @elapsed apply!(cb, state, TDVP1vec(); t=current_time, sweepend=true)

        @debug "Time spent on time-evolution step: $stime s" *
            "\nTime spent on computing expectation values: $mtime s"

        !isnothing(pbar) &&
            ProgressMeter.next!(pbar; showvalues=simulationinfo(state, current_time, stime))

        if !isempty(measurement_ts(cb)) && current_time ≈ measurement_ts(cb)[end]
            if store_state0
                printoutput_data(
                    io_handle, cb, state; psi0=state0, vectorized=true, kwargs...
                )
            else
                printoutput_data(io_handle, cb, state; vectorized=true, kwargs...)
            end
            printoutput_ranks(ranks_handle, cb, state)
            printoutput_stime(times_handle, stime)
        end

        checkdone!(cb) && break
    end

    !isnothing(io_file) && close(io_handle)
    !isnothing(ranks_file) && close(ranks_handle)
    !isnothing(times_file) && close(times_handle)

    return nothing
end

"""
    adaptivetdvp1vec!(solver, state, L, dt, tmax; kwargs...)

Like `tdvp1vec!`, but grows the bond dimensions of the MPS along the time evolution until
a certain convergence criterium is met.
The keyword argument `convergence_factor_bonddims`, which defaults to `1e-4`, controls the
convergence of the adaptation algorithm.

For an explanation of the other arguments, see [`tdvp1vec!`](@ref).
"""
function adaptivetdvp1vec!(
    solver, psi0::MPS, Ls::Vector{MPO}, time_step::Number, tmax::Number; kwargs...
)
    # (Copied from ITensorsTDVP)
    for H in Hs
        check_hascommoninds(siteinds, H, psi0)
        check_hascommoninds(siteinds, H, psi0')
    end
    Hs .= permute.(Hs, Ref((linkind, siteinds, linkind)))
    PHs = ProjMPOSum(Hs)
    return tdvp1vec!(solver, psi0, PHs, time_step, tmax; kwargs...)
end

function adaptivetdvp1vec!(solver, state::MPS, L::MPO, dt::Number, tmax::Number; kwargs...)
    return adaptivetdvp1vec!(solver, state, ProjMPO(L), dt, tmax; kwargs...)
end

function adaptivetdvp1vec!(solver, state::MPS, PH, dt::Number, tmax::Number; kwargs...)
    nsteps = Int(tmax / dt)
    cb = get(kwargs, :callback, NoTEvoCallback())
    hermitian = get(kwargs, :hermitian, true)
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    normalize = get(kwargs, :normalize, false)
    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)
    store_state0 = get(kwargs, :store_psi0, false)
    convergence_factor_bonddims = get(kwargs, :convergence_factor_bonddims, 1e-4)
    max_bond = get(kwargs, :max_bond, maxlinkdim(state))
    decomp = get(kwargs, :which_decomp, "qr")

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving state... ")
    else
        pbar = nothing
    end

    # Vectorized equations of motion usually are not defined by an anti-Hermitian operator
    # such as -im H in Schrödinger's equation, so we do not bother here with "unitary" or
    # "imaginary-time" evolution types. We just have a generic equation of the form
    # v'(t) = L v(t).

    store_state0 && (state0 = copy(state))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(state))
    times_handle = writeheaders_stime(times_file)

    N = length(state)

    # Measure everthing once in the initial state.
    current_time = 0.0
    apply!(cb, state, TDVP1vec(); t=current_time, sweepend=true)

    if store_state0
        printoutput_data(io_handle, cb, state; psi0=state0, vectorized=true, kwargs...)
    else
        printoutput_data(io_handle, cb, state; vectorized=true, kwargs...)
    end
    printoutput_ranks(ranks_handle, cb, state)

    for s in 1:nsteps
        orthogonalize!(state, 1)
        set_nsite!(PH, 1)
        position!(PH, state, 1)

        # Before each sweep, we grow the bond dimensions a bit.
        # See Dunnett and Chin, 2020 [arXiv:2007.13528v2].
        @debug "[Step $s] Attempting to grow the bond dimensions."
        adaptbonddimensions!(state, PH, max_bond, convergence_factor_bonddims)

        stime = @elapsed for (site, ha) in sweepnext(N; ncenter=1)
            # sweepnext(N) is an iterable object that evaluates to tuples of the form
            # (bond, ha) where bond is the bond number and ha is the half-sweep number.
            # The kwarg ncenter determines the end and turning points of the loop: if
            # it equals 1, then we perform a sweep on each single site.
            sweepdir = (ha == 1 ? "right" : "left")
            tdvp_site_update!(
                solver,
                PH,
                state,
                site,
                0.5dt;
                current_time=(ha == 1 ? current_time + 0.5dt : current_time + dt),
                sweepdir=sweepdir,
                which_decomp=decomp,
                hermitian=hermitian,
                exp_tol=exp_tol,
                krylovdim=krylovdim,
                maxiter=maxiter,
            )
        end

        current_time += dt

        # Now the backwards sweep has ended, so the whole MPS of the state is up-to-date.
        # We can then calculate the expectation values of the observables within cb.
        mtime = @elapsed apply!(cb, state, TDVP1vec(); t=current_time, sweepend=true)

        @debug "Time spent on time-evolution step: $stime s" *
            "\nTime spent on computing expectation values: $mtime s"

        !isnothing(pbar) &&
            ProgressMeter.next!(pbar; showvalues=simulationinfo(state, current_time, stime))

        if !isempty(measurement_ts(cb)) && current_time ≈ measurement_ts(cb)[end]
            if store_state0
                printoutput_data(
                    io_handle, cb, state; psi0=state0, vectorized=true, kwargs...
                )
            else
                printoutput_data(io_handle, cb, state; vectorized=true, kwargs...)
            end
            printoutput_ranks(ranks_handle, cb, state)
            printoutput_stime(times_handle, stime)
        end

        checkdone!(cb) && break
    end

    !isnothing(io_file) && close(io_handle)
    !isnothing(ranks_file) && close(ranks_handle)
    !isnothing(times_file) && close(times_handle)

    return nothing
end
