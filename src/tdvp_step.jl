using ITensorMPS: position!, set_nsite!, setleftlim!, setrightlim!

function exponentiate_solver(; kwargs...)
    # Default solver that we provide if no solver is given by the user.
    function solver(H, dt, state_0; kws...)
        solver_kwargs = (;
            tol=get(kwargs, :solver_tol, 1E-12),
            krylovdim=get(kwargs, :solver_krylovdim, 30),
            maxiter=get(kwargs, :solver_maxiter, 100),
            verbosity=get(kwargs, :solver_outputlevel, 0),
            eager=true,
        )
        state_t, info = exponentiate(H, dt, state_0; solver_kwargs...)
        return state_t, info
    end
    return solver
end

function tdvp_solver(; kwargs...)
    # Fallback solver function if no solver is specified when calling tdvp.
    solver_backend = get(kwargs, :solver_backend, "exponentiate")
    if solver_backend == "exponentiate"
        return exponentiate_solver(; kwargs...)
    else
        error(
            "solver_backend=$solver_backend not recognized " *
            "(the only option is \"exponentiate\")",
        )
    end
end

# Fallback functions if no solver is given.
function tdvp1!(state::MPS, H, dt, tmax; kwargs...)
    return tdvp1!(tdvp_solver(; kwargs...), state, H, dt, tmax; kwargs...)
end

function adaptivetdvp1!(state::MPS, H, dt, tmax; kwargs...)
    return adaptivetdvp1!(tdvp_solver(; kwargs...), state, H, dt, tmax; kwargs...)
end

function jointtdvp1!(states::Tuple{MPS,MPS}, H, dt, tmax; kwargs...)
    return jointtdvp1!(tdvp_solver(; kwargs...), states, H, dt, tmax; kwargs...)
end

function tdvp1vec!(state::MPS, L, dt, tmax; kwargs...)
    return tdvp1vec!(tdvp_solver(; kwargs...), state, L, dt, tmax; kwargs...)
end

function adaptivetdvp1vec!(state::MPS, L, dt, tmax; kwargs...)
    return adaptivetdvp1vec!(tdvp_solver(; kwargs...), state, L, dt, tmax; kwargs...)
end

function adjtdvp1vec!(operator::MPS, initialstate, L, dt, tmax, meas_stride; kwargs...)
    return adjtdvp1vec!(
        tdvp_solver(; kwargs...),
        operator,
        initialstate,
        L,
        dt,
        tmax,
        meas_stride;
        kwargs...,
    )
end

function adaptiveadjtdvp1vec!(
    operator::MPS, initialstate, H, dt, tmax, meas_stride; kwargs...
)
    return adaptiveadjtdvp1vec!(
        tdvp_solver(; kwargs...),
        operator,
        initialstate,
        H,
        dt,
        tmax,
        meas_stride;
        kwargs...,
    )
end

"""
    function tdvp_site_update!(
        solver, PH, state::MPS, i::Int, dt;
        sweepdir, current_time, which_decomp="qr", hermitian, exp_tol, krylovdim, maxiter
    )

Update site `i` of the MPS `state` using the 1-site TDVP algorithm with time step `dt`.
The keyword argument `sweepdir` indicates the direction of the current sweep.
"""
function tdvp_site_update!(
    solver,
    PH,
    state::MPS,
    site::Int,
    dt;
    sweepdir,
    current_time,
    which_decomp="qr",
    hermitian,
    exp_tol,
    krylovdim,
    maxiter,
)
    N = length(state)
    set_nsite!(PH, 1)
    position!(PH, state, site)

    # Forward evolution half-step.
    evolved_1site_tensor, info = solver(PH, dt, state[site]; current_time)
    info.converged == 0 && throw("exponentiate did not converge")

    # Backward evolution half-step.
    # (it is necessary only if we're not already at the edges of the MPS)
    if (sweepdir == "right" && (site != N)) || (sweepdir == "left" && site != 1)
        new_proj_base_site = (sweepdir == "right" ? site + 1 : site)
        # When we are sweeping right-to-left and switching from a 1-site projection to a
        # 0-site one, the right-side projection moves one site to the left, but the “base”
        # site of the ProjMPO doesn't move  ==>  new_proj_base_site = site
        # In the other sweep direction, the left-side projection moves one site to the left
        # and so does the “base” site  ==>  new_proj_base_site = site + 1

        next_site = (sweepdir == "right" ? site + 1 : site - 1)
        # This is the physical index of the next site in the sweep.

        if which_decomp == "qr"
            Q, C = factorize(
                evolved_1site_tensor,
                uniqueinds(evolved_1site_tensor, state[next_site]);
                which_decomp="qr",
            )
            state[site] = Q # This is left(right)-orthogonal if ha==1(2).
        elseif which_decomp == "svd"
            U, S, V = svd(
                evolved_1site_tensor, uniqueinds(evolved_1site_tensor, state[next_site])
            )
            state[site] = U # This is left(right)-orthogonal if ha==1(2).
            C = S * V
        else
            error(
                "Decomposition $which_decomp not supported. Please use \"qr\" or \"svd\"."
            )
        end

        if sweepdir == "right"
            setleftlim!(state, site)
        elseif sweepdir == "left"
            setrightlim!(state, site)
        end

        # Prepare the zero-site projection.
        set_nsite!(PH, 0)
        position!(PH, state, new_proj_base_site)

        C, info = solver(PH, -dt, C; current_time)

        # Reunite the backwards-evolved C with the matrix on the next site.
        state[next_site] *= C

        # Now the orthocenter is on `next_site`.
        # Set the new orthogonality limits of the MPS.
        if sweepdir == "right"
            setrightlim!(state, next_site + 1)
        elseif sweepdir == "left"
            setleftlim!(state, next_site - 1)
        else
            throw("Unrecognized sweepdir: $sweepdir")
        end

        # Reset the one-site projection… and we're done!
        set_nsite!(PH, 1)
    else
        # There's nothing to do if the half-sweep is at the last site.
        state[site] = evolved_1site_tensor
    end
end
