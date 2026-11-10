#
# Project : Lavender
# Source  : gw_selfenergy.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# Self-energies, polarization, and energies of the GW example (Hartree,
# Fock, GW self-energy, bubble polarization, kinetic and interaction
# energy). Port of NESSi examples/programs/gw_selfene_impl.cpp.
#
# The current time step of every k point is read only by the self-energy
# and polarization stages; the results are written into the k point's own
# objects (only the time step `n` is touched), so the k points of one
# stage are independent.
#

using LinearAlgebra
using KadanoffBaym

"""
    sigma_hartree(n, lat, dens)

Hartree self-energy at the KB time step `n` (the same for all k points).
`dens[k]` is the density matrix of the k point `k`. The local interaction
U enters with the density of one spin, the nonlocal one ``V_{q=0} - U``
with the factor 2 (spin).

*References* : NESSi `diag::sigma_Hartree`.
"""
function sigma_hartree(lat::GWLattice, dens::Vector{<:AbstractMatrix})
    d = size(dens[1], 1)
    vloc = lat.U * Matrix{ComplexF64}(I, d, d)
    vnl = interaction_vertex(lat, lat.kpoints[lat.G+1]) * Matrix{ComplexF64}(I, d, d) - vloc
    s = zeros(ComplexF64, d, d)
    for k = 1:lat.nk
        wk = lat.kweights[k]
        ρ = dens[k]
        for i₁ = 1:d, it = 1:d
            s[i₁,i₁] += vloc[i₁,it] * ρ[it,it] * wk
            s[i₁,i₁] += vnl[i₁,it] * ρ[it,it] * wk * 2.0
        end
    end
    return s
end

"""
    sigma_fock(k, lat, dens)

Fock self-energy of the k point `k`, only the nonlocal interaction
contributes, ``-\\sum_q w_q [V_q - U] \\rho_{k-q}``.

*References* : NESSi `diag::sigma_Fock`.
"""
function sigma_fock(k::Int, lat::GWLattice, dens::Vector{<:AbstractMatrix})
    d = size(dens[1], 1)
    s = zeros(ComplexF64, d, d)
    for q = 1:lat.nk
        wq = lat.kweights[q]
        kq = add_kpoints(lat, k, 1, q, -1)
        Vnl = interaction_vertex(lat, lat.kpoints[q]) - lat.U
        ρ = dens[kq]
        for i₁ = 1:d, i₂ = 1:d
            s[i₁,i₂] -= Vnl * ρ[i₁,i₂] * wq
        end
    end
    return s
end

"""
    sigma_gw!(n, k, kps, lat)

GW self-energy ``\\Sigma_k = \\sum_q w_q\\, i G_{k-q} W_q`` (Bubble2) at the
KB time step `n`. The time step `n` of `kps[k].Σ` is overwritten; it is
also used as the work space of the single bubbles.

*References* : NESSi `diag::sigma_GW`.
"""
function sigma_gw!(n::Int, k::Int, kps, lat::GWLattice)
    Σ = kps[k].Σ
    d = getdims(Σ)[1]
    acc = 𝒻(n, getntau(Σ), d, d, FERMI)
    for q = 1:lat.nk
        kq = add_kpoints(lat, k, 1, q, -1)
        memset!(Σ, n, zero(ComplexF64))
        for i₁ = 1:d, i₂ = 1:d
            bubble2!(n, Σ, i₁, i₂, kps[kq].G, i₁, i₂, kps[q].W, i₁, i₂)
        end
        incr!(acc, Σ, n, lat.kweights[q])
    end
    memcpy!(acc, Σ, n)
    return Σ
end

"""
    polarization!(n, q, kps, lat)

Polarization ``P_q = -\\sum_k w_k\\, i G_{k-q}(t,t') G_k(t',t)`` (Bubble1)
at the KB time step `n`. The time step `n` of `kps[q].P` is overwritten.

*References* : NESSi `diag::get_Polarization_Bubble`.
"""
function polarization!(n::Int, q::Int, kps, lat::GWLattice)
    P = kps[q].P
    d = getdims(P)[1]
    acc = 𝒻(n, getntau(P), d, d, BOSE)
    for k = 1:lat.nk
        kq = add_kpoints(lat, k, 1, q, -1)
        memset!(P, n, zero(ComplexF64))
        for i₁ = 1:d, i₂ = 1:d
            bubble1!(n, P, i₁, i₂, kps[kq].G, i₁, i₂, kps[k].G, i₁, i₂)
        end
        incr!(acc, P, n, -lat.kweights[k])
    end
    memcpy!(acc, P, n)
    return P
end

"""
    kinetic_energy_gw(n, lat, dens)

``E_{kin} = \\sum_k w_k \\mathrm{Re\\,Tr}[h_k(t) \\rho_k]`` at the KB time
step `n`, `dens[k]` is the density matrix of the k point `k`.

*References* : NESSi `diag::KineticEnergy`.
"""
function kinetic_energy_gw(n::Int, lat::GWLattice, dens::Vector{<:AbstractMatrix})
    e = 0.0
    for k = 1:lat.nk
        hk = dispersion(lat, lat.kpoints[k], n)
        e += real(tr(hk * dens[k])) * lat.kweights[k]
    end
    return e
end

"""
    interaction_energy_gw(n, kps, lat, β, h, order)

``E_{int} = \\sum_k w_k \\{\\mathrm{Re\\,Tr}[(\\Sigma^H + \\Sigma^F_k)\\rho_k]
+ E_{corr}[G_k, \\Sigma_k]\\}`` at the KB time step `n`. Here `ρ_k` is the
density matrix after the last Dyson step.

*References* : NESSi `diag::CorrelationEnergy`.
"""
function interaction_energy_gw(n::Int, kps, lat::GWLattice, β::Float64, h::Float64,
                               order::Int)
    e = 0.0
    for k = 1:lat.nk
        kp = kps[k]
        wk = lat.kweights[k]
        e += real(tr((kp.SH[n] + kp.SF[n]) * kp.rho[n])) * wk
        e += correlation_energy(n, kp.G, kp.Σ, β, h, order) * wk
    end
    return e
end
