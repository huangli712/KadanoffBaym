#
# Project : Lavender
# Source  : holstein_energy.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# Phonon momentum and energies of the Holstein examples, NESSi
# examples/programs/Holstein_utils_impl.cpp.
#

"""
    phonon_momentum(ntot, g, ph)

Phonon momentum ``P(t) = \\int_0^t \\partial_t D_0^R(t-t')/\\omega_0 \\,
[g n(t') - g n(0^-)] dt'`` at all KB time steps (NESSi
`Hols::get_phonon_momentum`), `P[1] = 0` on the Matsubara branch. The
start-up steps use the integral of the time step `k` with ``g n(t') = g
n(0^-)`` for ``t' < 0``.
"""
function phonon_momentum(ntot::Vector{ComplexF64}, g::Vector{ComplexF64}, ph::HolsteinPhonon)
    ntime = getntime(ph.D0)
    Nt = ntime - 1
    k, h, w0 = ph.k, ph.h, ph.w0
    ng(m) = g[m+1] * ntot[m+1]
    d1(m) = -(exp(-im * m * h * w0) + exp(im * m * h * w0))
    P = zeros(ComplexF64, ntime + 1)
    for tstp = 0:k
        f = zeros(ComplexF64, k + 1)
        for it = 0:k
            m = it + tstp - k
            ngm = m ≥ 0 ? ng(m + 1) : ng(0)
            f[it+1] = d1(k - it) * (ngm - ng(0))
        end
        P[tstp+2] = gregory_integrate(ph.I, f, k) * h
    end
    for tstp = k+1:Nt
        f = [d1(tstp - it) * (ng(it + 1) - ng(0)) for it = 0:tstp]
        P[tstp+2] = gregory_integrate(ph.I, f, tstp) * h
    end
    return P
end

"""
    phonon_energy_classical(X, P, w0)

``E_{ph}^{cl} = \\omega_0 (X^2 + P^2) / 4`` (NESSi
`Hols::evaluate_phonon_energy_cl`).
"""
phonon_energy_classical(X::Vector{ComplexF64}, P::Vector{ComplexF64}, w0::Float64) =
    [0.25 * w0 * (X[i] * X[i] + P[i] * P[i]) for i in eachindex(X)]

"""
    phonon_energy_quantum(D, Pi, ph)

Quantum phonon energy ``\\omega_0 (\\langle X^2\\rangle +
\\langle P^2\\rangle)/4`` at all KB time steps (NESSi
`Hols::evaluate_phonon_energy_qu`), from the time derivatives
``D_0^{d1} = \\partial_t D_0/\\omega_0``, ``D_0^{d2} = \\partial_{t'}
D_0/\\omega_0`` and the Dyson equation of `D`.
"""
function phonon_energy_quantum(D::ℱ{ComplexF64}, Pi::ℱ{ComplexF64}, ph::HolsteinPhonon)
    ntime, ntau = getntime(D), getntau(D)
    β, h, w0, I = ph.β, ph.h, ph.w0, ph.I
    C = Cn(ntime, ntau, 1, 1, (ntime - 1) * h, β)
    B() = ℱ(C, BOSE)
    D0_d1, D0_d2, D_d1, D_d2, D0_d1_Pi, Pi_D0_d2 = B(), B(), B(), B(), B(), B()
    Nt = ntime - 1
    dτ = β / (ntau - 1)
    expp = [exp(-im * it * h * w0) for it = 0:Nt]
    fb = bose(β, w0)
    for itau = 0:ntau-1
        τ = itau * dτ
        c1, c2 = exp(-τ * w0), exp(τ * w0)
        v = -(1.0 + fb) * c1 + fb * c2
        D0_d1.mat[itau+1] = fill(ComplexF64(v), 1, 1)
        D0_d2.mat[itau+1] = fill(ComplexF64(-v), 1, 1)
        for it = 0:Nt
            tv = -fb * expp[it+1] * c2 + (1.0 + fb) * conj(expp[it+1]) * c1
            D0_d1.lmix[it+1, itau+1] = fill(tv, 1, 1)
            D0_d2.lmix[it+1, itau+1] = fill(-tv, 1, 1)
        end
    end
    for it1 = 0:Nt, it2 = 0:it1
        e = expp[it1-it2+1]
        r = -(e + conj(e))
        l = -fb * conj(e) + (1.0 + fb) * e
        D0_d1.ret[it1+1, it2+1] = fill(r, 1, 1)
        D0_d2.ret[it1+1, it2+1] = fill(-r, 1, 1)
        D0_d1.less[it2+1, it1+1] = fill(l, 1, 1)
        D0_d2.less[it2+1, it1+1] = fill(-l, 1, 1)
    end
    convolution(D0_d1_Pi, D0_d1, D0_d2, Pi, Pi, I, β, h)
    convolution(Pi_D0_d2, Pi, Pi, D0_d2, D0_d1, I, β, h)
    convolution(D_d1, D0_d1_Pi, Pi_D0_d2, D, D, I, β, h)
    incr!(D_d1, D0_d1, 1.0)
    convolution(D_d2, D, D, Pi_D0_d2, D0_d1_Pi, I, β, h)
    incr!(D_d2, D0_d2, 1.0)
    E = zeros(ComplexF64, ntime + 1)
    for n = 0:ntime
        XX = n == 0 ? -D.mat[1][1,1] : im * D.less[n, n][1,1]
        PP = -convolution_density_matrix(n, D0_d1_Pi, Pi_D0_d2, nothing, D_d2, D_d1, I, β, h)[1,1]
        PP += 1.0 + 2.0 * bose(β, w0)
        E[n+1] = (XX + PP) * w0 / 4.0
    end
    return E
end
