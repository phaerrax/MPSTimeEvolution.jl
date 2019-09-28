using Test, ITensors, TimeEvoMPS

@testset "Basic TDVP tests" begin
    N=10
    sites= spinHalfSites(N)

    #check that evolving forward and backward in time brings us back to the same state
    #(up to numerical errors of course)
    J,h = 1., 0.5
    H = tfi_mpo(J,h,sites)

    psi0 = randomMPS(sites)
    psi = complex!(deepcopy(psi0))
    nsteps = 20
    dt = 1e-1
    tdvp!(psi,H,dt,nsteps*dt, hermitian=true,exp_tol=1e-12/dt,cutoff=1e-14)
    tdvp!(psi,H,-dt,-nsteps*dt, hermitian=true, exp_tol=1e-12/dt, cutoff=1e-14)

    @test inner(psi,psi) ≈ 1
    @test inner(psi0,psi) ≈ 1

    #check that energy is approximately conserved up to time-step error
    #for short time evolution with no truncation
    psi0 = randomMPS(sites)
    psi = complex!(deepcopy(psi0))
    nsteps = 20
    dt = 1e-1
    E0 = inner(psi,H,psi)

    tdvp!(psi,H,dt,nsteps*dt)
    E1 = inner(psi,H,psi)
    @test E0 ≈ E1
end

#TODO: do we want these functions in the package?
function measure!(psi::MPS,opname::String,i::Int)
    orthogonalize!(psi,i)
    return scalar(dag(psi[i])*noprime(op(siteindex(psi,i), opname)*psi[i]))
end

measure!(psi::MPS,opname::String) = map(x->measure!(psi,opname,x), 1:length(psi))

@testset "compare short-time evolution TDVP and TEBD" begin
    N=10
    J,h = 1., 5.87
    dt, tf = 0.01,0.5
    sites= spinHalfSites(N)
    psi_tebd = productMPS(sites,ones(Int,N))
    psi_tdvp = complex!(deepcopy(psi_tebd))

    H = tfi_mpo(J,h,sites)
    tdvp!(psi_tdvp,H,dt,tf, cutoff=1e-12)

    H = tfi_bondop(sites,J,h)
    tebd!(psi_tebd,H,dt,tf,cutoff=1e-12)

    # problem is that the error in TDVP scales as
    # Δt^4 while in TEBD2 it scales as Δ^2
    # so I guess we should be ok with difference which is
    # O(Δt^2) ?
    @test inner(psi_tdvp,psi_tebd) ≈ 1 atol= dt^2

    #measure magnetizations and make sure they are the same
    sz_tebd = measure!(psi_tebd,"Sz")
    sz_tdvp = measure!(psi_tdvp,"Sz")
    @test maximum(abs.(sz_tebd - sz_tdvp)) < dt^2
end

# TODO: compare with ED
# TODO: find some exact results to compare to, if such exist for time evolution

@testset "Imaginary time-evolution TFI model" begin
    N=10
    J = 1.
    h = 0.5
    sites = spinHalfSites(N)
    H = tfi_mpo(J,h,sites)
    psi = complex!(productMPS(sites,ones(Int,N)))

    Es = []
    nsteps = 100
    for dt in [0.1]
        tdvp!(psi,H,-1im*dt,-1im*nsteps*dt ; maxdim=50, hermitian=true, exp_tol = 1e-14/dt,
              verbose = true)
        push!(Es, inner(psi,H,psi))
    end

    # exact expression for ground-state energy
    # at criticality (ref? took it from ITensors.jl tests)

    eexact = 0.25 -0.25/sin(π/(4*N + 2))
    @test Es[end] ≈ eexact atol=1e-4
end


psi = complex!(productMPS(sites,ones(Int,N)))
dt = 0.1
nsteps =1
tdvp!(psi,H,-1im*dt,-1im*nsteps*dt ; maxdim=50, hermitian=true, exp_tol = 1e-14/dt,
              verbose = true)

for i in 1:length(psi)
    orthogonalize!(psi,i)
    println(i, " ", scalar(dag(psi[i])*psi[i]))
end
