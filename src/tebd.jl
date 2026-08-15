export tebd1!, tebd2!

# Update the MPS by applying a series of operators acting on DISTINCT sites.
# It doesn't matter if the operators act on odd, even or whatever sites, as long as the
# sites are all distinct. This is important, since the operations are done in parallel, if
# possible.
function tebd_step!(ψₜ::VidalMPS, gates; cutoff, maxdim)
    site_ts = site_tensors(ψₜ)
    bond_ts = bond_tensors(ψₜ)

    Threads.@threads :greedy for u in gates
        # The `:greedy` scheduler spawns a certain number N of tasks, each greedily working
        # on the given iterated values as they are produced. This means that (say, for the
        # odd-step routine) task 1 applies u₁₂, task 2 is assigned u₃₄, and so on, until the
        # number of tasks is exhausted. As soon as one task finishes its work, it takes the
        # next value from the iterator, i.e. it goes on to apply the first uⱼ,ⱼ₊₁ operator
        # still waiting to be applied.
        # In contrast, with the default `:dynamic` scheduler each task processes contiguous
        # regions of the iteration space, meaning that the first task is assigned gates 1 to
        # k, task 2 is assigned gates k+1 to 2k, and so on, with k approximately equal to
        # length(gates) / N.
        # This scheduling option is a good choice in our case, because the workload of
        # individual iterations may not be uniform, i.e. in TEDOPA simulations where the
        # sites close to the central system require heavier calculations.
        # With the default scheduler instead, in this case the tasks which are assigned
        # the regions far from the system would finish immediately, whereas task 1 (which is
        # assigned the region with the system) would take a lot longer, effectively
        # stalling the parallel algorithm.
        j1, j2 = findsites(ψₜ, u)
        @assert j2 == j1 + 1
        j = j1

        # Apply two-site gates.
        Mⱼ = bond_ts[j - 1] * site_ts[j] * bond_ts[j] * site_ts[j + 1] * bond_ts[j + 1]
        Mⱼ′ = apply(u, Mⱼ)

        # Update the MPS with the (truncated) updated two-site tensor.
        # This updates the site_ts[j], bond_ts[j] and site_ts[j+1] tensors, except that the
        # bond tensors on the left and right still need to be “extracted”.
        replace_and_decompose!(ψₜ, Mⱼ′; cutoff=cutoff, maxdim=maxdim)

        # Restore the Vidal form by "removing" the singular values we previously
        # incorporated.
        site_ts[j] *= inv.(bond_ts[j - 1])
        site_ts[j + 1] *= inv.(bond_ts[j + 1])
    end

    return nothing
end

tebd_arguments_docstring = """
# Other arguments

* `dt`: time step of the evolution.
* `tmax`: end time of the evolution.

# Optional keyword arguments (with default values)

* `callback`: a callback object describing the observables.
* `maxdim=maxlinkdim(ψₜ)`: the maximum allowed bond dimension of the state during the
  evolution.
* `cutoff=1e-15`: cutoff for truncating small singular values during the evolution.
* `progress=true`: whether to display a progress bar during the evolution.
"""

"""
    tebd1!(ψₜ::VidalMPS, H::OpSum, dt, tmax; kwargs...)

Integrate the Schrödinger equation ``d/dt ψₜ = -i H ψₜ`` using the TEBD algorithm with a
1st-order Trotter-Suzuki decomposition for ``H``, where `ψₜ` is a MPS in the Vidal form,
representing the state of the system.

$tebd_arguments_docstring
"""
function tebd1!(ψₜ::VidalMPS, H::OpSum, dt, tmax; kwargs...)
    nsteps = floor(Int, tmax / dt)
    cb = get(kwargs, :callback, NoTEvoCallback())
    maxdim = get(kwargs, :maxdim, maxlinkdim(ψₜ))
    cutoff = get(kwargs, :cutoff, 1e-15)

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving state... ")
    else
        pbar = nothing
    end

    # Usually TEBD is used for ordinary time evolution, according to a Hamiltonian given
    # by `H`: if a real-valued time step `dt` is given, we assume this scenario and set up
    # an evolution given by the operator exp(-itH).
    # Passing an imaginary time step (and `tmax`) as an argument triggers instead an
    # evolution according to the operator exp(-tH), useful for thermalization processes.
    evol_dt = -im * dt
    # Discard the imaginary part if time step is real.
    isreal(evol_dt) && (evol_dt = real(evol_dt))

    # Compute the 1st-order Trotter-Suzuki decomposition.
    u1odd, u1even = trotter1(H, siteinds(ψₜ), evol_dt)

    # Measure everything once in the initial state.
    current_time = zero(dt)
    apply!(cb, ψₜ, TEBD(); current_time)

    # As we can only apply unitary operators to a Vidal MPS, the norm of the state cannot
    # change during the evolution as a result of applying the operators.
    # It can change, however, as a consequence of the truncation following the evolution
    # step.
    compute_normalization!(cb, ψₜ, TEBD(); current_time)

    for s in 1:nsteps
        stime = @elapsed begin
            tebd_step!(ψₜ, u1odd; cutoff, maxdim)
            tebd_step!(ψₜ, u1even; cutoff, maxdim)
        end
        # The evolution step is done. Compute the expectation values of the observables.
        apply!(cb, ψₜ, TEBD(); current_time=current_time + dt)
        current_time += dt
        compute_normalization!(cb, ψₜ, TEBD(); current_time=current_time)

        !isnothing(pbar) &&
            ProgressMeter.next!(pbar; showvalues=simulationinfo(ψₜ, current_time, stime))

        checkdone!(cb) && break
    end

    return nothing
end

"""
    tebd2!(ψₜ::VidalMPS, H::OpSum, dt, tmax; kwargs...)

Integrate the Schrödinger equation ``d/dt ψₜ = -i H ψₜ`` using the TEBD algorithm with a
2nd-order Trotter-Suzuki decomposition for ``H``, where `ψₜ` is a MPS in the Vidal form,
representing the state of the system.

If a callback is specified, and expectation values are not required at each time step, this
function will combine the final set of gates of a time step with the first set of gates of
the following one, for all steps between two measurements.

$tebd_arguments_docstring
"""
function tebd2!(ψₜ::VidalMPS, H::OpSum, dt, tmax; kwargs...)
    nsteps = floor(Int, tmax / dt)
    cb = get(kwargs, :callback, NoTEvoCallback())
    maxdim = get(kwargs, :maxdim, maxlinkdim(ψₜ))
    cutoff = get(kwargs, :cutoff, 1e-15)

    # Usually TEBD is used for ordinary time evolution, according to a Hamiltonian given
    # by `H`: if a real-valued time step `dt` is given, we assume this scenario and set up
    # an evolution given by the operator exp(-itH).
    # Passing an imaginary time step (and `tmax`) as an argument triggers instead an
    # evolution according to the operator exp(-tH), useful for thermalization processes.
    evol_dt = -im * dt
    # Discard the imaginary part if time step is real.
    isreal(evol_dt) && (evol_dt = real(evol_dt))

    # Compute the 1st-order Trotter-Suzuki decomposition.
    u1odd_halfdt, u1even, _ = trotter2(H, siteinds(ψₜ), evol_dt)

    # The 2nd-order Trotter-Suzuki decomposition yields
    #   U(t) ≈ U_odd(t/2) U_even(t) U_odd(t/2).
    # If a callback is specified and its measurement time step is larger than dt, the
    # odd-step time-evolution operators across time steps can be merged, so for example
    #   U(t) U(t) ≈ U_odd(t/2) U_even(t) U_odd(t) U_even(t) U_odd(t/2)
    # which saves us some gate multiplication and most importantly some SVD decompositions.
    u1odd_fulldt = [apply(u, u) for u in u1odd_halfdt]
    measurement_dt = iszero(callback_dt(cb)) ? dt : callback_dt(cb)
    nsubsteps = floor(Int, measurement_dt / dt)
    # This is the number of steps that occur between a measurement and the following one.
    # We can merge operators inside of them.

    nsteps_grouped = div(nsteps, nsubsteps)
    if get(kwargs, :progress, true)
        pbar = Progress(nsteps_grouped; desc="Evolving state... ")
    else
        pbar = nothing
    end

    # Measure everything once in the initial state.
    current_time = zero(dt)
    apply!(cb, ψₜ, TEBD(); current_time)

    # As we can only apply unitary operators to a Vidal MPS, the norm of the state cannot
    # change during the evolution as a result of applying the operators.
    # It can change, however, as a consequence of the truncation following the evolution
    # step.
    compute_normalization!(cb, ψₜ, TEBD(); current_time)

    for _ in 1:nsteps_grouped
        stime_merged_steps = @elapsed begin
            # nsubsteps = 1:
            # 1. U_odd(t/2)
            # 2. U_even(t)
            # 3. U_odd(t/2)
            #
            # nsubsteps = 2:
            # 1. U_odd(t/2)
            # 2. U_even(t)
            # 3. U_odd(t)  <
            # 4. U_even(t) <
            # 5. U_odd(t/2)
            #
            # nsubsteps = 3:
            # 1. U_odd(t/2)
            # 2. U_even(t)
            # 3. U_odd(t)  <
            # 4. U_even(t) <
            # 5. U_odd(t)  <
            # 6. U_even(t) <
            # 7. U_odd(t/2)
            #
            # Generalising: we always have U_odd(t/2) U_even(t) at the beginning, and we end
            # with U_odd(t/2). Inbetween, we have U_odd(t) U_even(t) a number of times equal
            # to nsubsteps-1.
            tebd_step!(ψₜ, u1odd_halfdt; cutoff, maxdim)
            tebd_step!(ψₜ, u1even; cutoff, maxdim)
            for _ in 1:(nsubsteps - 1)
                tebd_step!(ψₜ, u1odd_fulldt; cutoff, maxdim)
                tebd_step!(ψₜ, u1even; cutoff, maxdim)
            end
            tebd_step!(ψₜ, u1odd_halfdt; cutoff, maxdim)
        end
        stime = stime_merged_steps / nsubsteps  # approx time of single “true” step
        current_time += nsubsteps * dt  # we evolved the state for more than one step!

        # Normally, we would evolve for one time step and then let the callback decide
        # whether it's time to compute the expectation values or not.
        # Here, we have already decided when it's time, so the callback should _always_
        # measure them.
        # FIXME Does something unpleasant happen if measurement_dt is not an integer
        # multiple of dt?
        apply!(cb, ψₜ, TEBD(); current_time)
        compute_normalization!(cb, ψₜ, TEBD(); current_time)

        !isnothing(pbar) &&
            ProgressMeter.next!(pbar; showvalues=simulationinfo(ψₜ, current_time, stime))

        checkdone!(cb) && break
    end

    return nothing
end
