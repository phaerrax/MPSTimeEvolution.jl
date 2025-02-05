export jointtdvp1!

using ITensors: permute
using ITensorMPS: position!, set_nsite!, check_hascommoninds

"""
    jointtdvp1!([solver,] states::Tuple{MPS, MPS}, H::MPO, dt, tmax; kwargs...)
    jointtdvp1!([solver,] states::Tuple{MPS, MPS}, H::Vector{MPO}, dt, tmax; kwargs...)

Integrate the Schrödinger equation ``d/dt ψⁱₜ = -i H ψⁱₜ`` using the one-site TDVP
algorithm, for both states ``ψ¹`` and ``ψ²`` in `states` in parallel, from `0` to `tmax` in
time steps of `dt`.
The evolution operator can be given either as a single MPO or as a vector of MPOs, in the
latter case the evolution operator is taken to be the sum of the elements in the vector.

Instead of returning the step-by-step expectation value of operators on a single state, it
returns quantities of the form ``⟨ψ¹ₜ,Aψ²ₜ⟩`` for each operator ``A`` specified in the given
callback object, together with the overlap ``⟨ψ¹ₜ,ψ²ₜ⟩``.

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
* `normalize` (default: `true`): whether the states are renormalised after each step.
* `io_file` (default: `nothing`): output file for step-by-step measurements.
* `io_ranks` (default: `nothing`): output file for step-by-step bond dimensions.
* `io_times` (default: `nothing`): output file for simulation wall-clock times.
* `store_psi0` (default: `false`): whether to keep information about the initial state.
* `which_decomp` (default: `"qr"`): name of the decomposition method for the sweeps.
* `progress` (default: `true`): whether to display a progress bar during the evolution.

"""
function jointtdvp1! end

function jointtdvp1!(solver, states::Tuple{MPS,MPS}, Hs::Vector{MPO}, dt, tmax; kwargs...)
    # (Copied from ITensorsTDVP)
    for H in Hs
        for psi in states
            check_hascommoninds(siteinds, H, psi)
            check_hascommoninds(siteinds, H, psi')
        end
    end
    Hs .= permute.(Hs, Ref((linkind, siteinds, linkind)))
    PHs = ProjMPOSum(Hs)
    return jointtdvp1!(solver, states, PHs, dt, tmax; kwargs...)
end

function jointtdvp1!(solver, states::Tuple{MPS,MPS}, H::MPO, dt, tmax; kwargs...)
    return jointtdvp1!(solver, states, ProjMPO(H), dt, tmax; kwargs...)
end

function jointtdvp1!(solver, states::Tuple{MPS,MPS}, PH, dt, tmax; kwargs...)
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

    # Usually TDVP is used for ordinary time evolution according to a Hamiltonian given
    # by `H`, and a real-valued time step `dt`, combined in the evolution operator
    # U(-itH).
    # Passing an imaginary time step iτ (and `tmax`) as an argument results in an evolution
    # according to the operator U(-τH), useful for thermalization processes.
    evol_dt = im * dt
    # Discard imaginary part if time step is real.
    imag(evol_dt) == 0 && (evol_dt = real(evol_dt))

    store_state0 && (states0 = copy.(states))

    io_handle = writeheaders_data_double(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length.(states)...)
    times_handle = writeheaders_stime(times_file)

    if length(states[1]) != length(states[2])
        error("Lengths of the two given MPSs do not match!")
    else
        N = length(states[1])
    end

    # Measure everything once in the initial state.
    current_time = 0.0
    for j in reverse(eachindex(first(states)))
        apply!(
            cb, states..., TDVP1(); t=current_time, site=j, sweepend=true, sweepdir="left"
        )
    end

    printoutput_data(io_handle, cb, states...; kwargs...)
    printoutput_ranks(ranks_handle, cb, states...)

    states = [states...]  # Convert tuple into vector so that we can mutate its elements
    projections = [PH, deepcopy(PH)]  # The projection on different states will be different

    # Prepare for first iteration.
    for (state, PH) in zip(states, projections)
        orthogonalize!(state, 1)
        set_nsite!(PH, 1)
        position!(PH, state, 1)
    end

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
                for (state, PH) in zip(states, projections)
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
                end
                # At least with TDVP1, `tdvp_site_update!` updates the site at `site`, and
                # leaves the MPS with orthocenter at `site+1` or `site-1` if it sweeping
                # rightwards
                apply!(
                    cb,
                    states...,
                    TDVP1();
                    t=current_time + dt,
                    site=site,
                    sweepend=(ha == 2),
                    sweepdir=sweepdir,
                )
            end
        end

        current_time += dt

        !isnothing(pbar) && ProgressMeter.next!(
            pbar; showvalues=simulationinfo(states, current_time, stime)
        )

        if !isempty(measurement_ts(cb)) && current_time ≈ measurement_ts(cb)[end]
            printoutput_data(io_handle, cb, states...; kwargs...)
            printoutput_ranks(ranks_handle, cb, states...)
            printoutput_stime(times_handle, stime)
        end

        checkdone!(cb) && break
    end

    !isnothing(io_file) && close(io_handle)
    !isnothing(ranks_file) && close(ranks_handle)
    !isnothing(times_file) && close(times_handle)

    return nothing
end
