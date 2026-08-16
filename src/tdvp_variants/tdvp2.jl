export tdvp2!

"""
    tdvp2!([solver,] state::MPS, H::MPO, dt, tmax; kwargs...)
    tdvp2!([solver,] state::InverseCanonicalMPS, H::MPO, dt, tmax; kwargs...)
    tdvp2!([solver,] state::MPS, H::Vector{MPO}, dt, tmax; kwargs...)

Integrate the Schrödinger equation ``d/dt ψₜ = -i H ψₜ`` using the two-site TDVP algorithm,
where `state` is either an ordinary MPS or an MPS in the inverse-canonical gauge
representing the state of the system.
The Hamiltonian `H` can be given either as a single MPO or as a vector of MPOs, in the
latter case the total Hamiltonian is taken to be the sum of the elements in the vector.

# Other arguments

* `solver`: a function which takes three arguments `A`, `t`, `B` (and possibly other keyword
    arguments) where `t` is a time step, `B` an ITensor and `A` a linear operator on `B`,
    returning the time-evolved `B`. It defaults to `KrylovKit.exponentiate`.
* `dt`: time step of the evolution.
* `tmax`: end time of the evolution.

# Truncation keyword arguments

All keyword arguments controlling truncation which are accepted by `ITensors.replacebond!`
are available, namely:

* `maxdim::Int`: if specified, keep only `maxdim` largest singular values after
  applying the gate.
* `mindim::Int`: minimal number of singular values to keep if truncation is performed
  according to value specified by `cutoff`.
* `cutoff::Float`: if specified, keep the minimal number of singular values such that the
  discarded weight is smaller than `cutoff` (but the bond dimension will be kept smaller
  than `maxdim`).
* `absoluteCutoff::Bool`: if `true` truncate all singular-values whose square is smaller
  than `cutoff`.

# Other optional keyword arguments

* `callback`: a callback object describing the observables.
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
* `progress` (default: `true`): whether to display a progress bar during the evolution.

# References

[1] Haegeman, J., Lubich, C., Oseledets, I., Vandereycken, B., & Verstraete, F. (2016).
“Unifying time evolution and optimization with matrix product states”
Physical Review B, 94(16).
https://doi.org/10.1103/PhysRevB.94.165116
"""
function tdvp2! end

function tdvp2!(ψ::MPS, H::MPO, timestep, endtime; kwargs...)
    nsteps = floor(Int, endtime / timestep)
    cb = get(kwargs, :callback, NoTEvoCallback())
    hermitian = get(kwargs, :hermitian, true)
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    normalize = get(kwargs, :normalize, true)

    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving state... ")
    else
        pbar = nothing
    end

    replacebond_allowed_kwargs = [
        :normalize,
        :swapsites,
        :ortho,
        :which_decomp,
        :mindim,
        :maxdim,
        :cutoff,
        :eigen_perturbation,
        :svd_alg,
        :use_absolute_cutoff,
        :use_relative_cutoff,
        :min_blockdim,
    ]
    replacebond_kwargs = filter(p -> first(p) in replacebond_allowed_kwargs, kwargs)

    dt = im * timestep
    # If `timestep` is imaginary and imag(timestep) > 0, this gives us an evolution operator
    # of the form U(dt) = exp(-dt H) which denotes an "imaginary-time" evolution.
    imag(dt) == 0 && (dt = real(dt))
    # A unitary evolution is associated to a real `timestep`. In this case, dt is purely
    # imaginary, as it should.
    # Otherwise, with an imaginary-time evolution, dt is real, but the Type of the variable
    # is Complex, so we truncate any imaginary part away.
    # (`real` doesn't just chop off the imaginary part, it also converts the type from
    # Complex{T} to T.)

    store_psi0 = get(kwargs, :store_psi0, false)
    store_psi0 && (ψ0 = copy(ψ))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(ψ))
    times_handle = writeheaders_stime(times_file)

    # Measure everything once in the initial state.
    current_time = 0.0
    for j in reverse(eachindex(ψ)[1:(end - 1)])
        orthogonalize!(ψ, j)
        apply!(cb, ψ, TDVP2(); current_time, site=j, sweepend=true, sweepdir="left")
    end
    compute_normalization!(cb, ψ, TDVP2(); current_time)

    if store_psi0
        printoutput_data(io_handle, cb, ψ; psi0=ψ0, kwargs...)
    else
        printoutput_data(io_handle, cb, ψ; kwargs...)
    end
    printoutput_ranks(ranks_handle, cb, ψ)

    N = length(ψ)
    orthogonalize!(ψ, 1)
    # Move the orthogonality centre to site 1, i.e. make ψ right-canonical.
    PH = ProjMPO(H)
    position!(PH, ψ, 1)

    for s in 1:nsteps
        stime = @elapsed begin
            for (b, ha) in sweepnext(N; ncenter=2)
                # 1. Evolve with two-site Hamiltonian.
                #    ---------------------------------
                #    We project the Hamiltonian on the current bond b and the next one,
                #    then we evolve the (b, b+1) block for half a time step.
                #    The sweepnext iterator takes care of the correct indices: the pair
                #    (b, ha) here takes the values
                #       (1, 1), …, (N-1, 1), (N-1, 2), …, (1, 2)
                #    so that the correct pair of bonds is always (b, b+1).
                twosite!(PH)
                position!(PH, ψ, b)
                wf = ψ[b] * ψ[b + 1]
                wf, info = exponentiate(
                    PH, -0.5dt, wf; ishermitian=hermitian, tol=exp_tol, krylovdim=krylovdim
                )

                info.converged == 0 && throw("exponentiate did not converge")
                # Replace the ITensors of the MPS `ψ` at sites b and b+1 with `wf`,
                # which is factorized according to the orthogonalization specification
                # given by `ortho` (left for a left-to-right sweep, right otherwise).
                # (replacebond! normalizes the result, since we pass :normalize="true"
                # within the kwargs, and also updates the information about the
                # orthocentre).
                spec = replacebond!(
                    ψ,
                    b,
                    wf;
                    normalize=normalize,
                    ortho=(ha == 1 ? "left" : "right"),
                    replacebond_kwargs...,
                )
                # spec is the spectrum (aka the singular values?) of the SVD.
                # Some types of callback objects might need it later, in order to compute
                # the entropy or other related quantities.

                # 2. Measure the observables.
                #    ------------------------
                #    When we are sweeping right-to-left, once the block at sites (b, b+1)
                #    has been evolved, the tensor at ψ[b + 1] has completed its evolution
                #    within the time step dt.
                #    The MPS is
                #    • left-orthogonal from ψ[1] to ψ[b - 1]
                #    • right-orthogonal from ψ[b] to ψ[end]
                #    so this is a good time to measure observables that are local to
                #    site b+1: when contracting in inner(ψ', A(n), ψ), all the sites
                #    left of ψ[b] (excluded) give the identity, and so do all those
                #    right of ψ[b + 1].
                #    The measurement can then be performed using the tensor composed by
                #    only ψ[b] and ψ[b + 1].
                apply!(
                    cb,
                    ψ,
                    TDVP2();
                    current_time=current_time+timestep,
                    # This is only for storage purposes; we need the original `timestep`.
                    site=b,
                    sweepend=(ha == 2), # apply! is skipped if ha == 1
                    sweepdir=(ha == 1 ? "right" : "left"),
                    spec=spec,
                )

                # 3. Evolve with single-site Hamiltonian backward in time.
                #    -----------------------------------------------------
                #    Evolve the "next" block backwards for half a time step.
                #    Which block is the "next" block depends on the direction of the sweep:
                #    • when sweeping left-to-right, the pivot is on site `b`, and the next
                #      tensor is the one to the right, so `b+1`;
                #    • when sweeping right-to-left, the pivot is on site `b+1`, and the
                #      next tensor is the one to the left, so `b`.
                #    This step is not necessary in the case of imaginary time-evolution [1].
                i = ha == 1 ? b + 1 : b
                if 1 < i < N
                    set_nsite!(PH, 1)
                    position!(PH, ψ, i)
                    ψ[i], info = exponentiate(
                        PH,
                        0.5dt,
                        ψ[i];
                        ishermitian=hermitian,
                        tol=exp_tol,
                        krylovdim=krylovdim,
                        maxiter=maxiter,
                    )
                    info.converged == 0 && throw("exponentiate did not converge")
                end
            end
        end

        current_time += timestep
        compute_normalization!(cb, ψ, TDVP2(); current_time)

        !isnothing(pbar) &&
            ProgressMeter.next!(pbar; showvalues=simulationinfo(ψ, current_time, stime))

        if !isempty(measurement_ts(cb)) && current_time ≈ measurement_ts(cb)[end]
            if store_psi0
                printoutput_data(io_handle, cb, ψ; psi0=ψ0, kwargs...)
            else
                printoutput_data(io_handle, cb, ψ; kwargs...)
            end
            printoutput_ranks(ranks_handle, cb, ψ)
            printoutput_stime(times_handle, stime)
        end

        checkdone!(cb) && break
    end

    !isnothing(io_file) && close(io_handle)
    !isnothing(ranks_file) && close(ranks_handle)
    !isnothing(times_file) && close(times_handle)

    return nothing
end

#= ProjMPOSum methods with InverseCanonicalMPS arguments not yet implemented.

function tdvp2!(solver, ψ::InverseCanonicalMPS, Hs::Vector{MPO}, dt, tmax; kwargs...)
    # (Copied from ITensorsTDVP)
    for H in Hs
        check_hascommoninds(siteinds, H, ψ)
        check_hascommoninds(siteinds, H, ψ')
    end
    Hs .= permute.(Hs, Ref((linkind, siteinds, linkind)))
    PHs = ProjMPOSum(Hs)
    return tdvp2!(solver, ψ, PHs, dt, tmax; kwargs...)
end
=#

function tdvp2!(solver, ψ::InverseCanonicalMPS, H::MPO, dt, tmax; kwargs...)
    return tdvp2!(ψ, ProjMPO(H), dt, tmax; kwargs...)
end

function tdvp2!(solver, ψ::InverseCanonicalMPS, PH, dt, tmax; kwargs...)
    nsteps = floor(Int, tmax / dt)
    maxdim = get(kwargs, :maxdim, maxlinkdim(ψ))
    cutoff = get(kwargs, :cutoff, 1e-15)
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
    # isreal(evol_dt) && (evol_dt = real(evol_dt))
    isreal(evol_dt) &&
        error("imaginary-time evolution not implemented for InverseCanonicalMPS")

    store_state0 && (ψ0 = copy(ψ))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(ψ))
    times_handle = writeheaders_stime(times_file)

    N = nsites(ψ)

    # Measure everything once in the initial state.
    current_time = 0.0
    apply!(cb, ψ, TDVP2(); current_time, sweepend=true)
    compute_normalization!(cb, ψ, TDVP2(); current_time)

    # Prepare for first iteration (TODO do we need this?).
    set_nsite!(PH, 2)
    position!(PH, ψ, 1)

    if store_state0
        printoutput_data(io_handle, cb, ψ; psi0=ψ0, kwargs...)
    else
        printoutput_data(io_handle, cb, ψ; kwargs...)
    end
    printoutput_ranks(ranks_handle, cb, ψ)

    for s in 1:nsteps
        stime = @elapsed for (site, ha) in sweepnext(N; ncenter=2)
            # sweepnext(N) is an iterable object that evaluates to tuples of the form
            # (bond, ha) where bond is the bond number and ha is the half-sweep number.
            # With ncenter=2 we have the following sequence:
            #   1, 1
            #   2, 1
            #    ⋮
            #   N-2, 1
            #   N-1, 1
            #   N-1, 2
            #   N-2, 2
            #    ⋮
            #   2, 2
            #   1, 2

            # ha == 1  =>  left-to-right sweep
            # ha == 2  =>  right-to-left sweep
            sweepdir = (ha == 1 ? "right" : "left")

            set_nsite!(PH, 2)
            position!(PH, ψ, site)

            # Physical index of the next site/bond tensor in the sweep.
            this_site_n = (sweepdir == "right" ? site : site + 1)
            next_bond_n = (sweepdir == "right" ? site : site)
            next_site_n = (sweepdir == "right" ? site + 1 : site)

            # Forward dt/2 evolution half-step.
            current_twosite_block =
                site_tensors(ψ)[this_site_n] *
                bond_tensors(ψ)[next_bond_n] *
                site_tensors(ψ)[next_site_n]
            updated_twosite_block, info = solver(
                PH,
                -0.5evol_dt,
                # TODO change the notation so that we use 0.5evol_dt here.
                current_twosite_block;
                current_time=(
                    sweepdir == "right" ? current_time + 0.5dt : current_time + dt
                ),
            )
            info.converged == 0 && throw("solver did not converge")

            replace_and_decompose!(ψ, updated_twosite_block, maxdim=maxdim, cutoff=cutoff)
            # More explicitly:
            #   linds = commoninds(updated_block, site_ts[site])
            #   U, S, V = svd(updated_block, linds; cutoff=cutoff, maxdim=maxdim)
            #   site_tensors(ψ)[site] = (U * S) * delta(inds(S))
            #   bond_tensors(ψ)[next_bond_n] = inv.(S)
            #   site_tensors(ψ)[next_site_n] = (V * S) * delta(inds(S))

            # If sweepdir == "left", now all blocks from `site` to `N` of the MPS are
            # correctly updated.

            # Backward dt/2 evolution half-step on the next site (only if the next site is
            # not already at an edge of the MPS).
            if (sweepdir == "right" && site != N) || (sweepdir == "left" && site != 1)

                # Prepare the one-site projection.
                set_nsite!(PH, 1)
                position!(PH, ψ, next_site_n)

                # In the TDVP2 for ordinary MPSs, we would first need to incorporate the
                # bond tensors into the next MPS site before the backwards time-evolution
                # step. Here, in the inverse canonical gauge, we've already done it!
                site_tensors(ψ)[next_site_n], info = solver(
                    PH,
                    0.5evol_dt,
                    site_tensors(ψ)[next_site_n];
                    current_time=(ha == 1 ? current_time + 0.5dt : current_time + dt),
                )
            end
        end

        current_time += dt

        # With inverse-canonical MPSs, expectation values can be computed in parallel with
        # no drawbacks, since we don't have to shift the orthocentre each time (as we'd do
        # with ordinary MPSs), so it's more convenient if we call `apply!` just one time at
        # the end of each step, and let it parallelise the computation as best as possible.
        apply!(cb, ψ, TDVP2(); current_time, sweepend=true)
        compute_normalization!(cb, ψ, TDVP2(); current_time)

        !isnothing(pbar) &&
            ProgressMeter.next!(pbar; showvalues=simulationinfo(ψ, current_time, stime))

        if !isempty(measurement_ts(cb)) && current_time ≈ measurement_ts(cb)[end]
            if store_state0
                printoutput_data(io_handle, cb, ψ; psi0=ψ0, kwargs...)
            else
                printoutput_data(io_handle, cb, ψ; kwargs...)
            end
            printoutput_ranks(ranks_handle, cb, ψ)
            printoutput_stime(times_handle, stime)
        end

        checkdone!(cb) && break
    end

    !isnothing(io_file) && close(io_handle)
    !isnothing(ranks_file) && close(ranks_handle)
    !isnothing(times_file) && close(times_handle)

    return nothing
end
