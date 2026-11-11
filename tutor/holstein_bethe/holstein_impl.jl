#
# Project : Lavender
# Source  : holstein_impl.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# Shared helpers of the Holstein examples (tutor/holstein_bethe and
# tutor/holstein_impurity): normal-state self-energies in the
# self-consistent (Migdal) and unrenormalized (uMig) Migdal
# approximations, the Dyson equation of the phonon propagator, and the
# phonon displacement. Port of NESSi examples/programs/
# Holstein_impurity_impl.cpp. KB time step n = NESSi time step + 1
# (n = 0 is the Matsubara branch).
#
#   Pi(t,t')    = -2 i g G(t,t') g G(t',t)
#   D           = D0 + D0 * Pi * D        ([1 - D0*Pi] * D = D0)
#   Sigma(t,t') =  i D(t,t') g G(t,t') g   (Migdal, D0 for uMig)
#

using LinearAlgebra
using KadanoffBaym

"""
    HolsteinPhonon

Phonon propagators and work space: the free propagator `D0` (twice the
single-pole XX propagator), the dressed `D`, the polarization `Pi`, the
kernels `D0_Pi = -D0*Pi`, `Pi_D0 = -Pi*D0`, the work spaces `Pol`
(bosonic) and `gGg` (fermionic, the electron size).
"""
struct HolsteinPhonon
    D0    :: ℱ{ComplexF64}
    D     :: ℱ{ComplexF64}
    Pi    :: ℱ{ComplexF64}
    D0_Pi :: ℱ{ComplexF64}
    Pi_D0 :: ℱ{ComplexF64}
    Pol   :: ℱ{ComplexF64}
    gGg   :: ℱ{ComplexF64}
    w0    :: Float64
    β     :: Float64
    h     :: Float64
    k     :: Int
    I     :: Integrator
end

"""
    holstein_phonon(ntime, ntau_points, size, w0, β, h, k)

Allocate the phonon propagators on the contour, `D0 = 2 D_XX` (NESSi
`green_single_pole` followed by `smul(2)`).
"""
function holstein_phonon(ntime::Int, ntau_points::Int, size::Int, w0::Float64, β::Float64,
                         h::Float64, k::Int)
    C1 = Cn(ntime, ntau_points, 1, 1, (ntime - 1) * h, β)
    Cs = Cn(ntime, ntau_points, size, size, (ntime - 1) * h, β)
    B() = ℱ(C1, BOSE)
    D0 = B()
    green_single_pole!(D0, w0, β, h)
    for n = 0:ntime
        smul!(D0, n, 2.0)
    end
    return HolsteinPhonon(D0, B(), B(), B(), B(), B(), ℱ(Cs, FERMI), w0, β, h, k, Integrator(k))
end

"""
    holstein_scalar_cf(ntime, values)

1x1 single-time function from `values[n + 1]` (KB time step `n`).
"""
function holstein_scalar_cf(ntime::Int, values::AbstractVector)
    f = Cf(ntime, 1, 1, zero(ComplexF64))
    for n = 0:ntime
        f[n] = fill(ComplexF64(values[n+1]), 1, 1)
    end
    return f
end

# gGg = g G g at the KB time step n
function _gGg!(n::Int, G::ℱ, g::Cf, ph::HolsteinPhonon)
    memcpy!(G, ph.gGg, n)
    smul!(ph.gGg, g, n)
    smul!(g, ph.gGg, n)
end

# Pi = -2 i gGg(t,t') G(t',t) at the KB time step n
function _pi_normal!(n::Int, G::ℱ, ph::HolsteinPhonon)
    bubble1!(n, ph.Pol, 1, 1, ph.gGg, 1, 1, G, 1, 1)
    memcpy!(ph.Pol, ph.Pi, n)
    smul!(ph.Pi, n, -2.0)
end

# C = A * B at the KB time step n ≥ 1. convolution_time_step accumulates
# the lmix and less components, the time step is cleared first (NESSi
# convolution_timestep overwrites it).
function _conv_step!(n::Int, C::ℱ, A::ℱ, B::ℱ, ph::HolsteinPhonon)
    zeros!(C, n)
    convolution_time_step(n, C, A, B, ph.I, ph.β, ph.h)
end

"""
    step_D!(n, ph; method = :fixpoint)

Solve ``[1 - D_0*\\Pi] * D = D_0`` at the KB time step `n` (`n = 0`:
Matsubara with `method`, `n > k + 1`: time step), NESSi `Hols::step_D`.
"""
function step_D!(n::Int, ph::HolsteinPhonon; method::Symbol = :fixpoint)
    if n == 0
        conv_mat(ph.D0_Pi.mat, ph.D0.mat, ph.Pi.mat, ph.I, ph.β, BOSE)
        conv_mat(ph.Pi_D0.mat, ph.Pi.mat, ph.D0.mat, ph.I, ph.β, BOSE)
    else
        _conv_step!(n, ph.D0_Pi, ph.D0, ph.Pi, ph)
        _conv_step!(n, ph.Pi_D0, ph.Pi, ph.D0, ph)
    end
    smul!(ph.D0_Pi, n, -1.0)
    smul!(ph.Pi_D0, n, -1.0)
    if n == 0
        vie2_mat!(ph.D, ph.D0_Pi, ph.Pi_D0, ph.D0, ph.β, ph.I; method = method)
    else
        vie2_timestep!(n, ph.D, ph.D0_Pi, ph.Pi_D0, ph.D0, ph.I, ph.β, ph.h)
    end
    return ph
end

"""
    start_D!(ph)

Start-up block (KB time steps `1, ⋯, k + 1`) of the phonon Dyson
equation, NESSi `Hols::start_D`.
"""
function start_D!(ph::HolsteinPhonon)
    for n = 1:ph.k+1
        _conv_step!(n, ph.D0_Pi, ph.D0, ph.Pi, ph)
        smul!(ph.D0_Pi, n, -1.0)
        _conv_step!(n, ph.Pi_D0, ph.Pi, ph.D0, ph)
        smul!(ph.Pi_D0, n, -1.0)
    end
    vie2_start!(ph.D, ph.D0_Pi, ph.Pi_D0, ph.D0, ph.I, ph.β, ph.h)
    return ph
end

"""
    sigma_mig!(n, G, Σ, g, ph; method = :fixpoint)

Self-consistent Migdal self-energy at the KB time step `n` (`n = 0` or
`n > k + 1`), NESSi `Hols::Sigma_Mig(tstp, ...)`: `Pi`, the phonon Dyson
equation (Matsubara `method`), and ``\\Sigma = i D \\, gGg``.
"""
function sigma_mig!(n::Int, G::ℱ, Σ::ℱ, g::Cf, ph::HolsteinPhonon; method::Symbol = :fixpoint)
    _gGg!(n, G, g, ph)
    _pi_normal!(n, G, ph)
    step_D!(n, ph; method = method)
    bubble2!(n, Σ, 1, 1, ph.D, 1, 1, ph.gGg, 1, 1)
    return Σ
end

"""
    sigma_mig_start!(G, Σ, g, ph)

Migdal self-energy of the start-up time steps, NESSi
`Hols::Sigma_Mig(G, ...)`.
"""
function sigma_mig_start!(G::ℱ, Σ::ℱ, g::Cf, ph::HolsteinPhonon)
    # gGg of all start-up steps is needed after start_D!
    for n = 1:ph.k+1
        _gGg!(n, G, g, ph)
        _pi_normal!(n, G, ph)
    end
    start_D!(ph)
    for n = 1:ph.k+1
        bubble2!(n, Σ, 1, 1, ph.D, 1, 1, ph.gGg, 1, 1)
    end
    return Σ
end

"""
    sigma_umig!(n, G, Σ, g, ph)

Unrenormalized Migdal self-energy ``\\Sigma = i D_0 \\, gGg`` at the KB
time step `n`, NESSi `Hols::Sigma_uMig(tstp, ...)`.
"""
function sigma_umig!(n::Int, G::ℱ, Σ::ℱ, g::Cf, ph::HolsteinPhonon)
    _gGg!(n, G, g, ph)
    bubble2!(n, Σ, 1, 1, ph.D0, 1, 1, ph.gGg, 1, 1)
    return Σ
end

"""
    sigma_umig_start!(G, Σ, g, ph)

uMig self-energy of the start-up time steps.
"""
function sigma_umig_start!(G::ℱ, Σ::ℱ, g::Cf, ph::HolsteinPhonon)
    for n = 1:ph.k+1
        sigma_umig!(n, G, Σ, g, ph)
    end
    return Σ
end

"""
    gregory_integrate(I, f, n)

``\\sum_j w_{nj} f_j`` with the Gregory weights of `I` (NESSi
`Integrator::integrate(f, n)`); `f[j + 1]` is the value at the index `j`.
"""
function gregory_integrate(I::Integrator, f::AbstractVector, n::Int)
    n == 0 && return zero(eltype(f))
    n₁ = n ≤ I.k ? I.k : n
    s = zero(eltype(f))
    for j = 0:n₁
        s += I.GIW[n,j] * f[j+1]
    end
    return s
end

"""
    phonon_displacement!(n, X, ntot, g, ph)

Phonon displacement at the KB time step `n` (NESSi
`Hols::get_phonon_displace(tstp, ...)`):
``X(t) = X_0 + \\int_0^t D_0^R(t,t') [g n(t') - g n(0^-)] dt'`` with
``X_0 = -2 g n(0^-) / \\omega_0``. `X`, `ntot`, and `g` are indexed by
`n + 1`.
"""
function phonon_displacement!(n::Int, X::Vector{ComplexF64}, ntot::Vector{ComplexF64},
                              g::Vector{ComplexF64}, ph::HolsteinPhonon)
    ng(m) = g[m+1] * ntot[m+1]
    X0 = -2.0 * ng(0) / ph.w0
    if n == 0
        X[1] = X0
    else
        f = [ph.D0.ret[n, it+1][1,1] * (ng(it + 1) - ng(0)) for it = 0:n-1]
        X[n+1] = gregory_integrate(ph.I, f, n - 1) * ph.h + X0
    end
    return X
end

"""
    phonon_displacement_start!(X, ntot, g, ph)

Displacement of the start-up time steps (NESSi
`Hols::get_phonon_displace(X_ph, ...)`): the integral of the time step
`k` with ``g n(t') = g n(0^-)`` for ``t' < 0``.
"""
function phonon_displacement_start!(X::Vector{ComplexF64}, ntot::Vector{ComplexF64},
                                    g::Vector{ComplexF64}, ph::HolsteinPhonon)
    k = ph.k
    ng(m) = g[m+1] * ntot[m+1]
    X0 = -2.0 * ng(0) / ph.w0
    for nn = 0:k
        f = zeros(ComplexF64, k + 1)
        for it = 0:k
            m = it + nn - k
            ngm = m ≥ 0 ? ng(m + 1) : ng(0)
            f[it+1] = ph.D0.ret[k+1, it+1][1,1] * (ngm - ng(0))
        end
        X[nn+2] = gregory_integrate(ph.I, f, k) * ph.h + X0
    end
    return X
end

"""
    holstein_hybridization!(n, Hyb, G, J)

``\\Delta(t,t') = J(t) G(t,t') J(t')`` at the KB time step `n` (right
multiplication first, as in NESSi).
"""
function holstein_hybridization!(n::Int, Hyb::ℱ, G::ℱ, J::Cf)
    memcpy!(G, Hyb, n)
    smul!(Hyb, J, n)
    smul!(J, Hyb, n)
    return Hyb
end

# Hyb_Sig = Hyb + Σ at the KB time step n
function _hyb_sigma!(n::Int, HS::ℱ, Hyb::ℱ, Σ::ℱ)
    memcpy!(Hyb, HS, n)
    incr!(HS, Σ, n, 1.0)
end

"""
    MatsubaraTwoStage

State of the two-stage Matsubara iteration of the Holstein examples:
the FOURIER method until the error is below the threshold, then the
FIXPOINT method until it is below the threshold again.
"""
mutable struct MatsubaraTwoStage
    fixpoint  :: Bool
    converged :: Bool
end

matsubara_method(s::MatsubaraTwoStage) = s.fixpoint ? :fixpoint : :fourier

# Update the stage after an iteration with the error err; returns true if
# the previous iterate must be updated (NESSi: not after convergence)
function matsubara_update!(s::MatsubaraTwoStage, err::Float64, tol::Float64)
    if err < tol && s.fixpoint
        s.converged = true
        return false
    elseif err < tol
        s.fixpoint = true
    end
    return true
end
