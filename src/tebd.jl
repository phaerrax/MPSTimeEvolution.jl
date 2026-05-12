export tebd1!

"""
    tebd1!(ψₜ::VidalMPS, H::OpSum, dt, tmax; kwargs...)

Integrate the Schrödinger equation ``d/dt ψₜ = -i H ψₜ`` using the TEBD algorithm with a
1st-order Trotter-Suzuki decomposition for ``H``, where `ψₜ` is a MPS in the Vidal form,
representing the state of the system.

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
function tebd1! end

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
    apply!(cb, ψₜ, TEBD(); current_time=current_time)

    # As we can only apply unitary operators to a Vidal MPS, the norm of the state cannot
    # change during the evolution as a result of applying the operators.
    # It can change, however, as a consequence of the truncation following the evolution
    # step.
    compute_norm!(cb, ψₜ, TEBD(); current_time=current_time)

    for s in 1:nsteps
        stime = @elapsed begin
            tebd1_step_odd!(ψₜ, u1odd; cutoff, maxdim)
            tebd1_step_even!(ψₜ, u1even; cutoff, maxdim)
        end
        # The evolution step is done. Compute the expectation values of the observables.
        apply!(cb, ψₜ, TEBD(); current_time=current_time + dt)
        current_time += dt
        compute_norm!(cb, ψₜ, TEBD(); current_time=current_time)

        !isnothing(pbar) &&
            ProgressMeter.next!(pbar; showvalues=simulationinfo(ψₜ, current_time, stime))

        checkdone!(cb) && break
    end

    return nothing
end

function tebd1_step_odd!(ψₜ::VidalMPS, gates; cutoff, maxdim)
    site_ts = site_tensors(ψₜ)
    bond_ts = bond_tensors(ψₜ)

    Threads.@threads for u in gates
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
        indices = inds(Mⱼ′)

        # Restore the Vidal form by "removing" the singular values we previously
        # incorporated.
        site_ts[j] *= inv.(bond_ts[j - 1])
        site_ts[j + 1] *= inv.(bond_ts[j + 1])
    end

    return nothing
end

function tebd1_step_even!(ψₜ::VidalMPS, gates; cutoff, maxdim)
    site_ts = site_tensors(ψₜ)
    bond_ts = bond_tensors(ψₜ)

    Threads.@threads for u in gates
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
