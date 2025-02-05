export tdvp1!, adaptivetdvp1!

using ITensors: permute
using ITensorMPS: position!, set_nsite!, check_hascommoninds

"""
    tdvp1!([solver,] state::MPS, H::Vector{MPO}, dt, tmax; kwargs...)
    tdvp1!([solver,] state::MPS, H::MPO, dt, tmax; kwargs...)

Integrate the Schrödinger equation ``d/dt ψₜ = -i H ψₜ`` using the one-site TDVP algorithm,
where `state` is an MPS representing the state of the system.
The Hamiltonian `H` can be given either as a single MPO or as a vector of MPOs, in the
latter case the total Hamiltonian is taken to be the sum of the elements in the vector.

# Other arguments

* `solver`: a function which takes three arguments `A`, `t`, `B` (and possibly other keyword
    arguments) where `t` is a time step, `B` an ITensor and `A` a linear operator on `B`,
    returning the time-evolved `B`. It defaults to `KrylovKit.exponentiate`.
* `dt`: time step of the evolution.
* `tmax`: end time of the evolution.

# Optional keyword arguments

* `cb`: a callback object describing the observables.
* `hermitian` (default: `true`): whether `H` is an Hermitian operator.
* `exp_tol` (default: `1e-14`): accuracy per unit time for `KrylovKit.exponentiate`.
* `krylovdim` (default: `30`): maximum dimension of the Krylov subspace that will be
    constructed.
* `maxiter` (default: `100`): number of times the Krylov subspace can be rebuilt.
* `normalize` (default: `true`): whether `state` is renormalised after each step.
* `io_file` (default: `nothing`): output file for step-by-step measurements.
* `io_ranks` (default: `nothing`): output file for step-by-step bond dimensions.
* `io_times` (default: `nothing`): output file for simulation wall-clock times.
* `store_psi0` (default: `false`): whether to keep information about the initial state.
* `which_decomp` (default: `"qr"`): name of the decomposition method for the sweeps.
* `progress` (default: `true`): whether to display a progress bar during the evolution.
"""
function tdvp1! end

function tdvp1!(solver, state::MPS, Hs::Vector{MPO}, dt, tmax; kwargs...)
    # (Copied from ITensorsTDVP)
    for H in Hs
        check_hascommoninds(siteinds, H, state)
        check_hascommoninds(siteinds, H, state')
    end
    Hs .= permute.(Hs, Ref((linkind, siteinds, linkind)))
    PHs = ProjMPOSum(Hs)
    return tdvp1!(solver, state, PHs, dt, tmax; kwargs...)
end

function tdvp1!(solver, state::MPS, H::MPO, dt, tmax; kwargs...)
    return tdvp1!(solver, state, ProjMPO(H), dt, tmax; kwargs...)
end

function tdvp1!(solver, state::MPS, PH, dt, tmax; kwargs...)
    nsteps = floor(Int, tmax / dt)
    cb = get(kwargs, :callback, NoTEvoCallback())
    hermitian = get(kwargs, :hermitian, true)
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    normalize = get(kwargs, :normalize, true)
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

    # Usually TDVP is used for ordinary time evolution, according to a Hamiltonian given
    # by `H`: if a real-valued time step `dt` is given, we assume this scenario and set up
    # an evolution given by the operator exp(-itH).
    # Passing an imaginary time step (and `tmax`) as an argument triggers instead an
    # evolution according to the operator exp(-tH), useful for thermalization processes.
    evol_dt = im * dt
    # Discard the imaginary part if time step is real.
    imag(evol_dt) == 0 && (evol_dt = real(evol_dt))

    store_state0 && (state0 = copy(state))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(state))
    times_handle = writeheaders_stime(times_file)

    N = length(state)

    # Measure everything once in the initial state.
    current_time = 0.0
    for j in reverse(eachindex(state))
        apply!(cb, state, TDVP1(); t=current_time, site=j, sweepend=true, sweepdir="left")
    end

    if store_state0
        printoutput_data(io_handle, cb, state; psi0=state0, kwargs...)
    else
        printoutput_data(io_handle, cb, state; kwargs...)
    end
    printoutput_ranks(ranks_handle, cb, state)

    # Prepare for first iteration.
    orthogonalize!(state, 1)
    set_nsite!(PH, 1)
    position!(PH, state, 1)

    for s in 1:nsteps
        stime = @elapsed begin
            # In TDVP1 only one site at a time is modified, so we iterate on the sites
            # of the state's MPS, not the bonds.
            for (site, ha) in sweepnext(N; ncenter=1)
                # sweepnext(N) is an iterable object that evaluates to tuples of the form
                # (bond, ha) where bond is the bond number and ha is the half-sweep number.
                # The kwarg ncenter determines the end and turning points of the loop: if
                # it equals 1, then we perform a sweep on each single site.

                # ha == 1  =>  left-to-right sweep
                # ha == 2  =>  right-to-left sweep
                sweepdir = (ha == 1 ? "right" : "left")
                tdvp_site_update!(
                    solver,
                    PH,
                    state,
                    site,
                    -0.5evol_dt; # forward by -im*dt/2, backwards by im*dt/2.
                    current_time=(ha == 1 ? current_time + 0.5dt : current_time + dt),
                    sweepdir=sweepdir,
                    which_decomp=decomp,
                    hermitian=hermitian,
                    exp_tol=exp_tol,
                    krylovdim=krylovdim,
                    maxiter=maxiter,
                )
                # At least with TDVP1, `tdvp_site_update!` updates the site at `site`, and
                # leaves the MPS with orthocenter at `site+1` or `site-1` if it sweeping
                # rightwards
                apply!(
                    cb,
                    state,
                    TDVP1();
                    t=current_time + dt,
                    site=site,
                    sweepend=(ha == 2),
                    sweepdir=sweepdir,
                )
            end
        end

        current_time += dt

        !isnothing(pbar) &&
            ProgressMeter.next!(pbar; showvalues=simulationinfo(state, current_time, stime))

        if !isempty(measurement_ts(cb)) && current_time ≈ measurement_ts(cb)[end]
            if store_state0
                printoutput_data(io_handle, cb, state; psi0=state0, kwargs...)
            else
                printoutput_data(io_handle, cb, state; kwargs...)
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
    adaptivetdvp1!([solver,] state::MPS, H::MPO, dt, tmax; kwargs...)
    adaptivetdvp1!([solver,] state::MPS, H::Vector{MPO}, dt, tmax; kwargs...)

Like `tdvp1!`, but grows the bond dimensions of the MPS along the time evolution until
a certain convergence criterium is met.
The keyword argument `convergence_factor_bonddims`, which defaults to `1e-4`, controls the
convergence of the adaptation algorithm.

For an explanation of the other arguments, see [`tdvp1!`](@ref).
"""
function adaptivetdvp1!(solver, state::MPS, Hs::Vector{MPO}, dt, tmax; kwargs...)
    # (Copied from ITensorsTDVP)
    for H in Hs
        check_hascommoninds(siteinds, H, state)
        check_hascommoninds(siteinds, H, state')
    end
    Hs .= permute.(Hs, Ref((linkind, siteinds, linkind)))
    PHs = ProjMPOSum(Hs)
    return adaptivetdvp1!(solver, state, PHs, dt, tmax; kwargs...)
end

function adaptivetdvp1!(solver, state::MPS, H::MPO, dt, tmax; kwargs...)
    return adaptivetdvp1!(solver, state::MPS, ProjMPO(H), dt, tmax; kwargs...)
end

function adaptivetdvp1!(solver, state::MPS, PH, dt, tmax; kwargs...)
    nsteps = floor(Int, tmax / dt)
    cb = get(kwargs, :callback, NoTEvoCallback())
    hermitian = get(kwargs, :hermitian, true)
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    normalize = get(kwargs, :normalize, true)
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

    evol_dt = im * dt
    imag(evol_dt) == 0 && (evol_dt = real(evol_dt))

    store_state0 && (state0 = copy(state))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(state))
    times_handle = writeheaders_stime(times_file)

    N = length(state)

    # Measure everything once in the initial state.
    current_time = 0.0
    for j in reverse(eachindex(state))
        apply!(cb, state, TDVP1(); t=current_time, site=j, sweepend=true, sweepdir="left")
    end

    if store_state0
        printoutput_data(io_handle, cb, state; psi0=state0, kwargs...)
    else
        printoutput_data(io_handle, cb, state; kwargs...)
    end
    printoutput_ranks(ranks_handle, cb, state)

    # Prepare for first iteration
    orthogonalize!(state, 1)
    set_nsite!(PH, 1)
    position!(PH, state, 1)

    current_time = 0.0
    for s in 1:nsteps
        orthogonalize!(state, 1)
        set_nsite!(PH, 1)
        position!(PH, state, 1)

        # Before each sweep, we grow the bond dimensions a bit.
        # See Dunnett and Chin, 2020 [arXiv:2007.13528v2].
        @debug "[Step $s] Attempting to grow the bond dimensions."
        adaptbonddimensions!(state, PH, max_bond, convergence_factor_bonddims)

        stime = @elapsed begin
            for (site, ha) in sweepnext(N; ncenter=1)
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
                    -0.5evol_dt;
                    current_time=(ha == 1 ? current_time + 0.5dt : current_time + dt),
                    sweepdir=sweepdir,
                    which_decomp=decomp,
                    hermitian=hermitian,
                    exp_tol=exp_tol,
                    krylovdim=krylovdim,
                    maxiter=maxiter,
                )
                apply!(
                    cb,
                    state,
                    TDVP1();
                    t=current_time,
                    bond=site,
                    sweepend=(ha == 2),
                    sweepdir=sweepdir,
                )
            end
        end

        !isnothing(pbar) &&
            ProgressMeter.next!(pbar; showvalues=simulationinfo(state, current_time, stime))

        current_time += dt

        if !isempty(measurement_ts(cb)) && current_time ≈ measurement_ts(cb)[end]
            if store_state0
                printoutput_data(io_handle, cb, state; psi0=state0, kwargs...)
            else
                printoutput_data(io_handle, cb, state; kwargs...)
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
