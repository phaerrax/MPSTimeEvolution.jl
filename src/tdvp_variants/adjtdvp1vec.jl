export adjtdvp1vec!, adaptiveadjtdvp1vec!

using ITensors: permute
using ITensorMPS: position!, set_nsite!, linkdims, check_hascommoninds

"""
    adjtdvp1vec!(
        [solver,] operator, initialstate::MPS, L, dt, tmax, meas_stride; kwargs...
    )
    adjtdvp1vec!(
        [solver,] operator, initialstates::Vector{MPS}, L, dt, tmax, meas_stride; kwargs...
    )

Integrate the equation of motion ``d/dt Xₜ = φ(Xₜ)`` using the one-site TDVP algorithm, from
`0` to `tmax` in time steps of `dt`.
The MPS `operator` represents ``X₀`` in a vectorised form, while `L` is an MPO representing
the operator ``φ`` that evolves ``X₀`` in time.
The expectation value of the operator is periodically computed (each `meas_stride`) on the
initial state `initialstate` or on a list `initialstates` of initial states.

See [`tdvp1vec!`](@ref) for a list of possible keyword arguments.

# Other keyword arguments (specific to this function)

* `initialstatelabels`: labels for the initial states (to mark the expectation values in the
    output files). Defaults to `["1", "2", …, "N"]` where `N` is the amount of initial
    states provided to the function.
"""
function adjtdvp1vec! end

# Version with a single initial state.
# The lone MPS is put into a vector of MPSs. The evolution operator is forwarded as it is.
# (This is the first function in the dispatch chain if `adjtdvp1vec!` is called with an MPS
# as `initialstate`, because it is the only method that accepts MPSs as 2nd and 3rd
# arguments, or 1st and 2nd if solver is not specified. We wrap the state MPS in a vector
# and leave the evolution operator completely unrestrained: the following methods will take
# care of it.)
function adjtdvp1vec!(
    solver, operator::MPS, initialstate::MPS, PH, dt, tmax, meas_stride; kwargs...
)
    return adjtdvp1vec!(
        solver, operator, [initialstate], PH, dt, tmax, meas_stride; kwargs...
    )
end

# Version with a vector of initial states but an MPO Hamiltonian.
# The MPO is wrapped into a ProjMPO object.
function adjtdvp1vec!(
    solver,
    operator::MPS,
    initialstates::Vector{MPS},
    H::MPO,
    dt,
    tmax,
    meas_stride;
    kwargs...,
)
    return adjtdvp1vec!(
        solver, operator, initialstates, ProjMPO(H), dt, tmax, meas_stride; kwargs...
    )
end

# Complete version (vector of initial states _and_ ProjMPO Hamiltonian)
function adjtdvp1vec!(
    solver, operator::MPS, initialstates::Vector{MPS}, PH, dt, tmax, meas_stride; kwargs...
)
    nsteps = floor(Int, tmax / dt)
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)
    decomp = get(kwargs, :which_decomp, "qr")
    initialstatelabels = get(kwargs, :initialstatelabels, string.(1:length(initialstates)))

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving operator... ")
    else
        pbar = nothing
    end

    # Vectorized equations of motion usually are not defined by an anti-Hermitian operator
    # such as -im H in Schrödinger's equation, so we do not bother here with "unitary" or
    # "imaginary-time" evolution types. We just have a generic equation of the form
    # v'(t) = L v(t).

    io_handle = open(io_file, "w")
    columnheaders = ["time"]
    if length(initialstates) == 1
        push!(columnheaders, "exp_val_real", "exp_val_imag")
    else
        for l in initialstatelabels
            push!(columnheaders, "exp_val_$(l)_real", "exp_val_$(l)_imag")
        end
    end
    println(io_handle, join(columnheaders, ","))

    ranks_handle = writeheaders_ranks(ranks_file, length(operator))
    times_handle = writeheaders_stime(times_file)

    N = length(operator)

    # Prepare for first iteration.
    orthogonalize!(operator, 1)
    set_nsite!(PH, 1)
    position!(PH, operator, 1)

    # Measure everthing once in the initial state.
    current_time = zero(dt)
    prev_t = zero(dt)

    data = [current_time]
    expvals = [inner(s, operator) for s in initialstates]
    for x in expvals
        push!(data, real(x), imag(x))
    end
    println(io_handle, join(data, ","))
    flush(io_handle)

    println(ranks_handle, join([current_time; linkdims(operator)], ","))
    flush(ranks_handle)

    for s in 1:nsteps
        stime = @elapsed begin
            # In TDVP1 only one site at a time is modified, so we iterate on the sites
            # of the operator MPS, not its bonds.
            for (site, ha) in sweepnext(N; ncenter=1)
                # sweepnext(N) is an iterable object that evaluates to tuples of the form
                # (bond, ha) where bond is the bond number and ha is the half-sweep number.
                # The kwarg ncenter determines the end and turning points of the loop: if
                # it equals 1, then we perform a sweep on each single site.
                sweepdir = (ha == 1 ? "right" : "left")
                tdvp_site_update!(
                    solver,
                    PH,
                    operator,
                    site,
                    0.5dt;
                    current_time=(ha == 1 ? current_time + 0.5dt : current_time + dt),
                    sweepdir=sweepdir,
                    which_decomp=decomp,
                    hermitian=false,
                    exp_tol=exp_tol,
                    krylovdim=krylovdim,
                    maxiter=maxiter,
                )
            end
        end
        # The sweep has ended, so we update the current time.
        current_time += dt

        !isnothing(pbar) && ProgressMeter.next!(
            pbar; showvalues=simulationinfo(operator, current_time, stime)
        )

        # We actually want to measure (i.e. contract the MPS) not at each time step, but
        # on each k-th one where k = meas_stride/time_step.
        # We save the previous time at which we computed measurements in prev_t, and each
        # time a sweep and we check if current_time - prev_t = meas_stride.
        # If it is so, then we go on and compute the expectation value (and update prev_t).
        if (current_time - prev_t ≈ meas_stride || current_time == 0)
            data = [current_time]

            expvals = [inner(s, operator) for s in initialstates]
            for x in expvals
                push!(data, real(x), imag(x))
            end

            println(io_handle, join(data, ","))
            flush(io_handle)

            println(ranks_handle, join([current_time; linkdims(operator)], ","))
            flush(ranks_handle)

            printoutput_stime(times_handle, stime)

            prev_t = current_time
        end
    end  # of the time evolution.

    !isnothing(io_file) && close(io_handle)
    !isnothing(ranks_file) && close(ranks_handle)
    !isnothing(times_file) && close(times_handle)

    return nothing
end

"""
    adaptiveadjtdvp1vec!(
        [solver,] operator, initialstate::MPS, L, dt, tmax, meas_stride; kwargs...
    )
    adaptiveadjtdvp1vec!(
        [solver,] operator, initialstates::Vector{MPS}, L, dt, tmax, meas_stride; kwargs...
    )

Like `adjtdvp1vec!`, but grows the bond dimensions of the MPS of the operator along the time
evolution until a certain convergence criterium is met.
The keyword argument `convergence_factor_bonddims`, which defaults to `1e-4`, controls the
convergence of the adaptation algorithm.

For an explanation of the other arguments, see [`adjtdvp1vec!`](@ref).
"""
function adaptiveadjtdvp1vec! end

function adaptiveadjtdvp1vec!(
    solver, operator::MPS, initialstate::MPS, L, dt, tmax, meas_stride; kwargs...
)
    return adaptiveadjtdvp1vec!(
        solver, operator, [initialstate], L, dt, tmax, meas_stride; kwargs...
    )
end
# Transform sum of evolution operator MPOa into a ProjMPOSum
function adaptiveadjtdvp1vec!(
    solver, operator::MPS, initialstates, Ls::Vector{MPO}, dt, tmax, meas_stride; kwargs...
)
    for L in Ls
        check_hascommoninds(siteinds, L, operator)
        check_hascommoninds(siteinds, L, operator')
    end
    Ls .= permute.(Ls, Ref((linkind, siteinds, linkind)))
    PLs = ProjMPOSum(Ls)
    return adaptiveadjtdvp1vec!(
        solver, operator, initialstates, PLs, dt, tmax, meas_stride; kwargs...
    )
end

# Transform evolution operator MPO into a ProjMPO
function adaptiveadjtdvp1vec!(
    solver, operator::MPS, initialstates, L::MPO, dt, tmax, meas_stride; kwargs...
)
    return adaptiveadjtdvp1vec!(
        solver, operator, initialstates, ProjMPO(L), dt, tmax, meas_stride; kwargs...
    )
end

# Most general version: vector of initial states and evolution operator as ProjMPO
function adaptiveadjtdvp1vec!(
    solver, operator::MPS, initialstates::Vector{MPS}, PL, dt, tmax, meas_stride; kwargs...
)
    nsteps = floor(Int, tmax / dt)
    exp_tol = get(kwargs, :exp_tol, 1e-14)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)
    io_file = get(kwargs, :io_file, nothing)
    ranks_file = get(kwargs, :io_ranks, nothing)
    times_file = get(kwargs, :io_times, nothing)
    convergence_factor_bonddims = get(kwargs, :convergence_factor_bonddims, 1e-4)
    max_bond = get(kwargs, :max_bond, maxlinkdim(operator))
    decomp = get(kwargs, :which_decomp, "qr")
    initialstatelabels = get(kwargs, :initialstatelabels, string.(1:length(initialstates)))

    if get(kwargs, :progress, true)
        pbar = Progress(nsteps; desc="Evolving operator... ")
    else
        pbar = nothing
    end

    # Vectorized equations of motion usually are not defined by an anti-Hermitian operator
    # such as -im H in Schrödinger's equation, so we do not bother here with "unitary" or
    # "imaginary-time" evolution types. We just have a generic equation of the form
    # v'(t) = L v(t).

    io_handle = open(io_file, "w")
    columnheaders = ["time"]
    if length(initialstates) == 1
        push!(columnheaders, "exp_val_real", "exp_val_imag")
    else
        for l in initialstatelabels
            push!(columnheaders, "exp_val_$(l)_real", "exp_val_$(l)_imag")
        end
    end
    println(io_handle, join(columnheaders, ","))

    ranks_handle = writeheaders_ranks(ranks_file, length(operator))
    times_handle = writeheaders_stime(times_file)

    N = length(operator)

    current_time = zero(dt)
    prev_t = zero(dt)
    for s in 1:nsteps
        orthogonalize!(operator, 1)
        set_nsite!(PL, 1)
        position!(PL, operator, 1)

        @debug "[Step $s] Attempting to grow the bond dimensions."
        adaptbonddimensions!(operator, PL, max_bond, convergence_factor_bonddims)

        stime = @elapsed begin
            # In TDVP1 only one site at a time is modified, so we iterate on the sites
            # of the operator MPS, not its bonds.
            for (site, ha) in sweepnext(N; ncenter=1)
                # sweepnext(N) is an iterable object that evaluates to tuples of the form
                # (bond, ha) where bond is the bond number and ha is the half-sweep number.
                # The kwarg ncenter determines the end and turning points of the loop: if
                # it equals 1, then we perform a sweep on each single site.
                sweepdir = (ha == 1 ? "right" : "left")
                tdvp_site_update!(
                    solver,
                    PL,
                    operator,
                    site,
                    0.5dt;
                    current_time=(ha == 1 ? current_time + 0.5dt : current_time + dt),
                    sweepdir=sweepdir,
                    which_decomp=decomp,
                    hermitian=false,
                    exp_tol=exp_tol,
                    krylovdim=krylovdim,
                    maxiter=maxiter,
                )
            end
        end
        current_time += dt

        !isnothing(pbar) && ProgressMeter.next!(
            pbar; showvalues=simulationinfo(operator, current_time, stime)
        )

        # We actually want to measure (i.e. contract the MPS) not at each time step, but
        # on each k-th one where k = meas_stride/time_step.
        # We save the previous time at which we computed measurements in prev_t, and each
        # time a sweep and we check if current_time - prev_t = meas_stride.
        # If it is so, then we go on and compute the expectation value (and update prev_t).
        if (current_time - prev_t ≈ meas_stride || current_time == 0)
            data = [current_time]

            expvals = [inner(s, operator) for s in initialstates]
            for x in expvals
                push!(data, real(x), imag(x))
            end

            println(io_handle, join(data, ","))
            flush(io_handle)

            println(ranks_handle, join([current_time; linkdims(operator)], ","))
            flush(ranks_handle)

            printoutput_stime(times_handle, stime)

            prev_t = current_time
        end
    end  # of the time evolution.

    !isnothing(io_file) && close(io_handle)
    !isnothing(ranks_file) && close(ranks_handle)
    !isnothing(times_file) && close(times_handle)

    return nothing
end
