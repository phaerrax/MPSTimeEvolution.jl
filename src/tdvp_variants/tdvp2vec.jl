"""
    tdvp2vec!(state, L::MPO, dt, tf; kwargs...)

Evolve the MPS `state` using the MPO `H` from 0 to `tf` using an integration step `dt`.
"""
function tdvp2vec!(state, L::MPO, dt, tf; kwargs...)
    nsteps = Int(tf / dt)
    cb = get(kwargs, :callback, NoTEvoCallback())
    hermitian = get(kwargs, :hermitian, false) # Lindblad superoperator is not Hermitian
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    normalize = get(kwargs, :normalize, false) # Vectorized states don't need normalization
    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving state... ")
    else
        pbar = nothing
    end

    store_initstate = get(kwargs, :store_psi0, false)

    # Copy the initial state if store_initstate is true
    store_initstate && (initstate = copy(state))

    io_handle = writeheaders_data(io_file, cb; kwargs...)
    ranks_handle = writeheaders_ranks(ranks_file, length(state))
    times_handle = writeheaders_stime(times_file)

    N = length(state)
    orthogonalize!(state, 1)
    PH = ProjMPO(H)
    position!(PH, state, 1)

    for s in 1:nsteps
        stime = @elapsed begin
            for (bond, ha) in sweepnext(N)
                # sweepnext(N) is an iterable object that evaluates to tuples of the form
                # (bond, ha) where bond is the bond number and ha is the half-sweep number.
                #
                # 1st step
                # --------
                # Evolve using two-site Hamiltonian: we extract the tensor on sites (j,j+1)
                # and we solve locally the equations of motion, then we put the result in
                # place of the original tensor.
                #
                # We set PH.nsite to 2, since we want to use a two-site update method,
                # and we shift the projection PH of H such that the set of unprojected
                # sites begins at site bond.
                twosite!(PH)
                position!(PH, state, bond)

                # Completing a single left-to-right sweep is a first-order integrator that
                # produces an updated |ψ⟩ at time t + τ with a local integration error of
                # order O(τ²). Completing the right-to-left sweep is equivalent to
                # composing this integrator with its adjoint, resulting in a second-order
                # symmetric method so that the state at time t + 2τ has a more favourable
                # error of order O(τ³). It is thus natural to set τ → τ/2 and to define
                # the complete sweep (left and right) as a single integration step.
                # [`Unifying time evolution and optimization with matrix product states'.
                # Jutho Haegeman, Christian Lubich, Ivan Oseledets, Bart Vandereycken and
                # Frank Verstraete, Physical Review B 94, 165116, p. 10 October 2016]
                twositeblock = state[bond] * state[bond + 1]
                twositeblock, info = exponentiate(
                    PH,
                    -0.5τ,
                    twositeblock;
                    ishermitian=hermitian,
                    tol=exp_tol,
                    krylovdim=krylovdim,
                )
                info.converged == 0 && throw("exponentiate did not converge")

                # Factorize the twositeblock tensor in two, and replace the ITensors at
                # sites bond and bond + 1 inside the MPS state.
                spec = replacebond!(
                    state,
                    bond,
                    twositeblock;
                    normalize=normalize,
                    ortho=(ha == 1 ? "left" : "right"),
                    kwargs...,
                )
                # normalize && ( state[dir=="left" ? bond+1 : bond] /= sqrt(sum(eigs(spec))) )

                apply!(
                    cb,
                    state;
                    t=s * dt,
                    bond=bond,
                    sweepend=(ha == 2),
                    # apply! does nothing if sweepend is false, so this way we are doing
                    # the measurement only on the second sweep, from right to left.
                    sweepdir=(ha == 1 ? "right" : "left"),
                    spec=spec,
                    alg=TDVP2(),
                )

                # 2nd step
                # --------
                # Evolve the second site of the previous two using the single-site
                # Hamiltonian, backward in time.
                # In the case of imaginary time-evolution this step is not necessary (see
                # Ref. 1).
                i = (ha == 1 ? bond + 1 : bond)
                # The "second site" is the one in the direction of the sweep.
                if 1 < i < N && !(dt isa Complex)
                    set_nsite!(PH, 1)
                    position!(PH, state, i)
                    state[i], info = exponentiate(
                        PH,
                        0.5τ,
                        state[i];
                        ishermitian=hermitian,
                        tol=exp_tol,
                        krylovdim=krylovdim,
                        maxiter=maxiter,
                    )
                    info.converged == 0 && throw("exponentiate did not converge")
                elseif i == 1 && dt isa Complex
                    # Normalization is not necessary (or even wrong...) if the state
                    # is a density matrix.
                    #state[i] /= sqrt(real(scalar(dag(state[i]) * state[i])))
                end
            end
        end

        !isnothing(pbar) &&
            ProgressMeter.next!(pbar; showvalues=simulationinfo(state, dt * s, stime))

        if !isempty(measurement_ts(cb)) && dt * s ≈ measurement_ts(cb)[end]
            if store_initstate
                printoutput_data(io_handle, cb, state; psi0=initstate, kwargs...)
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
