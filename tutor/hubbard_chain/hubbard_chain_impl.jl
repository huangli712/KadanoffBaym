#
# Project : Lavender
# Source  : hubbard_chain_impl.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# Open Hubbard chain with an on-site quench in the second Born (2B), GW,
# and particle-particle T-matrix (TPP) approximations. Port of NESSi
# examples/programs/hubbard_chain_{2b,gw,tpp}.cpp and
# hubbard_chain_selfen_impl.cpp. The bubbles of KadanoffBaym contain the
# factor i (bubble1!: i A(t,t') B(t',t), bubble2!: i A(t,t') B(t,t')).
#
#   P(t,t')    = -i G(t,t') G(t',t)
#   Sigma_2B   =  i G(t,t') [U P U](t,t')
#   (1 - P U) * Chi = P,   deltaW = U Chi U
#   Sigma_GW   =  i G(t,t') deltaW(t,t')
#   Phi(t,t')  = -i G(t,t') G(t,t')
#   (1 + Phi U) * TPP = Phi
#   Sigma_TPP  =  i [U TPP U](t,t') G(t',t)
#
# KB time step n = NESSi time step + 1 (n = 0 is the Matsubara branch).
#

using LinearAlgebra
using Printf
using KadanoffBaym

include(joinpath(@__DIR__, "..", "common", "example_config.jl"))
include(joinpath(@__DIR__, "..", "common", "text_output.jl"))
include(joinpath(@__DIR__, "..", "common", "contour_loop.jl"))

const HUBBARD_APPROX = ("2B", "GW", "TPP")
const HUBBARD_MODEL_KEYS = ("Nsites", "HoppingT", "HubbardU", "MuChem", "RampSite", "RampW0")

"""
    HubbardChain

State of the Hubbard chain: the Green's function `G`, the self-energy
`Σ`, the auxiliary contour functions of the approximation, the free
Hamiltonian `eps0` (quenched after the Matsubara solution), the
mean-field Hamiltonian `eps_mf`, the interaction `Ut`, and the chemical
potential `μ`.
"""
mutable struct HubbardChain
    approx :: String
    G      :: ℱ{ComplexF64}
    Σ      :: ℱ{ComplexF64}
    Gold   :: ℱ{ComplexF64}
    P      :: ℱ{ComplexF64}
    X      :: ℱ{ComplexF64}
    XU     :: ℱ{ComplexF64}
    UX     :: ℱ{ComplexF64}
    W      :: ℱ{ComplexF64}
    S      :: ℱ{ComplexF64}
    eps0   :: Matrix{ComplexF64}
    eps_mf :: Cf{ComplexF64}
    Ut     :: Cf{ComplexF64}
    μ      :: Float64
    β      :: Float64
    h      :: Float64
    k      :: Int
    I      :: Integrator
end

"""
    hubbard_chain(approx, cfg::ExampleConfig)

Construct the chain, the noninteracting Green's function of the
unquenched chain, and the chemical potential shifted by the Hartree term
(NESSi hubbard_chain_2b.cpp:158-182).
"""
function hubbard_chain(approx::AbstractString, cfg::ExampleConfig)
    approx in HUBBARD_APPROX || throw(ArgumentError("unknown approximation $approx"))
    c = contour_params(cfg)
    m = cfg.model
    N = m["Nsites"]
    (N isa Integer && N ≥ 1) || throw(ArgumentError("Nsites must be a positive integer"))
    rs = m["RampSite"]
    (rs isa Integer && 1 ≤ rs ≤ N) || throw(ArgumentError("RampSite must be in 1:Nsites"))
    U = Float64(m["HubbardU"])
    C = Cn(c.ntime, c.ntau_points, N, N, (c.ntime - 1) * c.h, c.beta)
    eps0 = zeros(ComplexF64, N, N)
    for i = 1:N-1
        eps0[i,i+1] = -m["HoppingT"]
        eps0[i+1,i] = -m["HoppingT"]
    end
    Ut = Cf(c.ntime, N, N, zero(ComplexF64))
    for i = 0:c.ntime
        Ut[i] = U * Matrix{ComplexF64}(I, N, N)
    end
    G = ℱ(C, FERMI)
    μ = Float64(m["MuChem"])
    green_from_H!(G, μ, eps0, c.beta, c.h)
    npart = real(tr(density_matrix(G, 0)))
    μ += U * npart / N
    B() = ℱ(C, BOSE)
    return HubbardChain(String(approx), G, ℱ(C, FERMI), ℱ(C, FERMI), B(), B(), B(), B(), B(),
                        B(), eps0, Cf(c.ntime, N, N, zero(ComplexF64)), Ut, μ, c.beta, c.h,
                        c.k, c.I)
end

"""
    hartree_hamiltonian!(n, st)

Mean-field Hamiltonian ``h_{MF} = h_0 + \\mathrm{diag}(U n)`` at the KB
time step `n` (NESSi `hubb::Ham_MF`). The occupation is the real part of
the diagonal of the density matrix, i.e. the diagonal of its hermitian
part: during the bootstrapping ``G^<(t,t)`` is anti-hermitian only up to
the discretization error, NESSi uses the complex ``\\rho_{ii}`` (the same
deviation as in tutor/gw, the NESSi reference driver
.nessi/kbref/hubbard_chain_text.cpp applies it as well), and the KB Dyson
solvers require a hermitian Hamiltonian.
"""
function hartree_hamiltonian!(n::Int, st::HubbardChain)
    ρ = density_matrix(st.G, n)
    U = st.Ut[n]
    hmf = copy(st.eps0)
    for i in axes(hmf, 1)
        hmf[i,i] += U[i,i] * real(ρ[i,i])
    end
    st.eps_mf[n] = hmf
    return st
end

"""
    polarization!(n, G, P)

``P(t,t') = -i G(t,t') G(t',t)`` at the KB time step `n` (NESSi
`hubb::Polarization`).
"""
function polarization!(n::Int, G::ℱ, P::ℱ)
    N = getdims(G)[1]
    for i = 1:N, j = 1:N
        bubble1!(n, P, i, j, G, i, j, G, i, j)
    end
    smul!(P, n, -1.0)
    return P
end

"""
    sigma_2b!(n, st)

Second Born self-energy at the KB time step `n` (NESSi `hubb::Sigma_2B`).
The polarization is stored in `st.W`.
"""
function sigma_2b!(n::Int, st::HubbardChain)
    polarization!(n, st.G, st.W)
    smul!(st.W, st.Ut, n)
    smul!(st.Ut, st.W, n)
    N = getdims(st.G)[1]
    for i = 1:N, j = 1:N
        bubble2!(n, st.Σ, i, j, st.G, i, j, st.W, i, j)
    end
    return st
end

# Kernels of the VIE2: XU = s X U, UX = s U X at the KB time step n
function _vie2_kernels!(n::Int, st::HubbardChain, s::Float64)
    memcpy!(st.X, st.XU, n)
    memcpy!(st.X, st.UX, n)
    smul!(st.XU, st.Ut, n)
    smul!(st.Ut, st.UX, n)
    if s != 1.0
        smul!(st.XU, n, s)
        smul!(st.UX, n, s)
    end
end

# Solve the VIE2 for st.S at the KB time step n (Matsubara or time step)
function _vie2_step!(n::Int, st::HubbardChain)
    if n == 0
        vie2_mat!(st.S, st.XU, st.UX, st.X, st.β, st.I)
    else
        vie2_timestep!(n, st.S, st.XU, st.UX, st.X, st.I, st.β, st.h)
    end
end

"""
    gen_chi!(n, st)

Solve ``(1 - P U) * \\chi = P`` at the KB time step `n` (`n = 0` or `n >
k + 1`), the polarization `P = st.X` must be known (NESSi
`hubb::GenChi(tstp, ...)`). The result is stored in `st.S`.
"""
function gen_chi!(n::Int, st::HubbardChain)
    _vie2_kernels!(n, st, -1.0)
    _vie2_step!(n, st)
    return st
end

"""
    gen_chi_start!(st)

Start-up block of ``(1 - P U) * \\chi = P`` (KB time steps `1, ⋯, k + 1`,
NESSi `hubb::GenChi(h, ...)`).
"""
function gen_chi_start!(st::HubbardChain)
    for n = 1:st.k+1
        _vie2_kernels!(n, st, -1.0)
    end
    vie2_start!(st.S, st.XU, st.UX, st.X, st.I, st.β, st.h)
    return st
end

"""
    sigma_gw!(n, st)

GW self-energy ``\\Sigma = i G \\, U \\chi U`` at the KB time step `n`
(NESSi `hubb::Sigma_GW`), `χ = st.S`, `δW` is stored in `st.W`.
"""
function sigma_gw!(n::Int, st::HubbardChain)
    memcpy!(st.S, st.W, n)
    smul!(st.Ut, st.W, n)
    smul!(st.W, st.Ut, n)
    N = getdims(st.G)[1]
    for i = 1:N, j = 1:N
        bubble2!(n, st.Σ, i, j, st.G, i, j, st.W, i, j)
    end
    return st
end

# Φ(t,t') = -i G(t,t') G(t,t') at the KB time step n, stored in st.X
function _phi!(n::Int, st::HubbardChain)
    N = getdims(st.G)[1]
    for i = 1:N, j = 1:N
        bubble2!(n, st.X, i, j, st.G, i, j, st.G, i, j)
    end
    smul!(st.X, n, -1.0)
end

"""
    gen_tpp!(n, st)

``\\Phi = -i G G`` and ``(1 + \\Phi U) * T = \\Phi`` at the KB time step
`n` (`n = 0` or `n > k + 1`, NESSi `hubb::GenTPP(tstp, ...)`). `Φ` is
stored in `st.X`, `T` in `st.S`.
"""
function gen_tpp!(n::Int, st::HubbardChain)
    _phi!(n, st)
    _vie2_kernels!(n, st, 1.0)
    _vie2_step!(n, st)
    return st
end

"""
    gen_tpp_start!(st)

Start-up block of the T-matrix (NESSi `hubb::GenTPP(h, ...)`).
"""
function gen_tpp_start!(st::HubbardChain)
    for n = 1:st.k+1
        _phi!(n, st)
        _vie2_kernels!(n, st, 1.0)
    end
    vie2_start!(st.S, st.XU, st.UX, st.X, st.I, st.β, st.h)
    return st
end

"""
    sigma_tpp!(n, st)

T-matrix self-energy ``\\Sigma = i [U T U](t,t') G(t',t)`` at the KB time
step `n` (NESSi `hubb::Sigma_TPP`), `T = st.S`.
"""
function sigma_tpp!(n::Int, st::HubbardChain)
    memcpy!(st.S, st.W, n)
    smul!(st.Ut, st.W, n)
    smul!(st.W, st.Ut, n)
    N = getdims(st.G)[1]
    for i = 1:N, j = 1:N
        bubble1!(n, st.Σ, i, j, st.W, i, j, st.G, i, j)
    end
    return st
end

"""
    update_sigma!(n, st)

Self-energy of the approximation at the KB time step `n` (`n = 0` or `n >
k + 1`).
"""
function update_sigma!(n::Int, st::HubbardChain)
    if st.approx == "2B"
        sigma_2b!(n, st)
    elseif st.approx == "GW"
        polarization!(n, st.G, st.X)
        gen_chi!(n, st)
        sigma_gw!(n, st)
    else
        gen_tpp!(n, st)
        sigma_tpp!(n, st)
    end
    return st
end

"""
    update_sigma_start!(st)

Self-energy of the start-up time steps `1, ⋯, k + 1`.
"""
function update_sigma_start!(st::HubbardChain)
    r = 1:st.k+1
    if st.approx == "2B"
        foreach(n -> sigma_2b!(n, st), r)
    elseif st.approx == "GW"
        foreach(n -> polarization!(n, st.G, st.X), r)
        gen_chi_start!(st)
        foreach(n -> sigma_gw!(n, st), r)
    else
        gen_tpp_start!(st)
        foreach(n -> sigma_tpp!(n, st), r)
    end
    return st
end

"""
    hubbard_energies(n, st)

``E_{kin} = \\mathrm{Re\\,Tr}(h_0 \\rho)``, ``E_{pot} = \\frac12
\\mathrm{Re\\,Tr}[(h_{MF} - h_0)\\rho] + E_{corr}`` and ``E_{tot}`` at the KB
time step `n` (NESSi hubbard_chain_2b.cpp:20-41).
"""
function hubbard_energies(n::Int, st::HubbardChain)
    ρ = density_matrix(st.G, n)
    ekin = real(tr(st.eps0 * ρ))
    epot = 0.5 * real(tr((st.eps_mf[n] - st.eps0) * ρ)) +
           correlation_energy(n, st.G, st.Σ, st.β, st.h, st.k)
    return ekin, epot, ekin + epot
end

"""
    solve_hubbard!(st, cfg; log = _noop)

Matsubara iteration, quench, bootstrapping (at least three iterations),
and predictor/corrector propagation (NESSi hubbard_chain_*.cpp). Throws a
`ConvergenceError` if the Matsubara or the bootstrap iteration does not
converge.
"""
function solve_hubbard!(st::HubbardChain, cfg::ExampleConfig; log::Function = _noop)
    s = solver_params(cfg)
    c = contour_params(cfg)
    k = st.k
    # Matsubara
    memcpy!(st.G, st.Gold, 0)
    iterate_matsubara!(; max_iter = s.mats_max_iter, tol = s.mats_max_error, log = log) do iter
        hartree_hamiltonian!(0, st)
        update_sigma!(0, st)
        dyson_mat!(st.G, st.Σ, st.μ, st.eps_mf, st.β; solve_order = k)
        err = distance(st.G, st.Gold, 0)
        memcpy!(st.G, st.Gold, 0)
        err
    end
    # Quench of the on-site energy of RampSite
    rs = cfg.model["RampSite"]
    st.eps0[rs,rs] = cfg.model["RampW0"]
    # Bootstrapping
    for n = 1:k+1
        memcpy!(st.G, st.Gold, n)
    end
    iterate_bootstrap!(; max_iter = s.bootstrap_max_iter, tol = s.bootstrap_max_error,
                       min_iterations = 3, log = log) do iter
        for n = 1:k+1
            hartree_hamiltonian!(n, st)
        end
        update_sigma_start!(st)
        dyson_start!(st.G, st.μ, st.eps_mf, st.Σ, st.β, st.h; solve_order = k)
        err = 0.0
        for n = 1:k+1
            err += distance(st.G, st.Gold, n)
            memcpy!(st.G, st.Gold, n)
        end
        err
    end
    # Propagation
    propagate_timesteps!(n -> extrapolate_timestep!(n - 1, st.G, k),
                         (n, iter) -> begin
                             hartree_hamiltonian!(n, st)
                             update_sigma!(n, st)
                             dyson_timestep!(n, st.G, st.μ, st.eps_mf, st.Σ, st.β, st.h;
                                             solve_order = k)
                         end;
                         first_step = k + 2, last_step = c.ntime,
                         corrector_steps = s.corrector_steps, h = st.h, log = log)
    return st
end

"""
    hubbard_observables(st)

Occupations and energies at the real time steps, `t = (n - 1) h`.
"""
function hubbard_observables(st::HubbardChain)
    ntime = getntime(st.G)
    occ = [vcat((n - 1) * st.h, real.(diag(density_matrix(st.G, n)))) for n = 1:ntime]
    en = [vcat((n - 1) * st.h, collect(hubbard_energies(n, st))) for n = 1:ntime]
    return occ, en
end

# The four selected Green's function components (NESSi indices)
function hubbard_selected_green(st::HubbardChain)
    G = st.G
    return [("mat", 0, -1, 0, 0, G.mat[1][1,1]), ("ret", 3, 1, 0, 0, G.ret[4,2][1,1]),
            ("les", 1, 3, 0, 0, G.less[2,4][1,1]), ("tv", 3, 2, 0, 0, G.lmix[4,3][1,1])]
end

"""
    load_hubbard_config(cfgfile, profile)

Read and validate the TOML file of the Hubbard chain.
"""
load_hubbard_config(cfgfile::AbstractString, profile::AbstractString) =
    load_example_config(cfgfile; profile = profile, model_required = HUBBARD_MODEL_KEYS)

"""
    hubbard_main(args, approx; io = stdout)

Entry of the examples: `<config.toml> [--profile quick|release]
[--validate-only]`. Writes `<prefix>_<approx>_occupation.dat` and
`<prefix>_<approx>_energy.dat` (and the selected Green's function
components if `save_green_components = true`). Returns the state and the
observables.
"""
function hubbard_main(args, approx::AbstractString; io::IO = stdout)
    a = parse_example_args(args)
    cfg = load_hubbard_config(a.cfgfile, a.profile)
    solver_params(cfg)
    st = hubbard_chain(approx, cfg)
    a.validate_only && return (config = cfg, state = st)
    logf(e) = e.stage in (:matsubara, :bootstrap) &&
        @printf(io, "%s iteration %d  error = %.6e\n", e.stage, e.iter, e.err)
    solve_hubbard!(st, cfg; log = logf)
    occ, en = hubbard_observables(st)
    dir = cfg.output["dir"]
    dir = isabspath(dir) ? dir : joinpath(cfg.dir, dir)
    mkpath(dir)
    pre = joinpath(dir, cfg.output["prefix"] * "_" * lowercase(approx))
    N = getdims(st.G)[1]
    write_table(pre * "_occupation.dat", ["t"; ["n$i" for i = 1:N]], occ)
    write_table(pre * "_energy.dat", ["t", "Ekin", "Epot", "Etot"], en)
    cfg.output["save_green_components"] &&
        write_green_components(pre * "_green.dat", hubbard_selected_green(st))
    return (config = cfg, state = st, occupation = occ, energy = en)
end

# Run an entry from the command line, a failure gives a nonzero exit code
function hubbard_cli(args, approx)
    try
        hubbard_main(args, approx)
    catch e
        showerror(stderr, e)
        println(stderr)
        exit(1)
    end
end
