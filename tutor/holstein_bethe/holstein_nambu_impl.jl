#
# Project : Lavender
# Source  : holstein_nambu_impl.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# Nambu (superconducting) self-energies of the Holstein model, NESSi
# examples/programs/Holstein_impurity_impl.cpp (Sigma_Mig_sc,
# Sigma_uMig_sc). The coupling is g σ₃ (2x2):
#
#   Pi(t,t')    = -i Σ_ab gGg_ab(t,t') G_ab(t',t)
#   Sigma_ab    =  i D(t,t') gGg_ab(t,t')   (D0 for uMig)
#

include(joinpath(@__DIR__, "holstein_impl.jl"))

# Pi = -Σ_ab i gGg_ab G_ba at the KB time step n
function _pi_nambu!(n::Int, G::ℱ, ph::HolsteinPhonon)
    first = true
    for a = 1:2, b = 1:2
        bubble1!(n, ph.Pol, 1, 1, ph.gGg, a, b, G, a, b)
        if first
            memcpy!(ph.Pol, ph.Pi, n)
            smul!(ph.Pi, n, -1.0)
            first = false
        else
            incr!(ph.Pi, ph.Pol, n, -1.0)
        end
    end
end

function _sigma_nambu!(n::Int, Σ::ℱ, D::ℱ, ph::HolsteinPhonon)
    for a = 1:2, b = 1:2
        bubble2!(n, Σ, a, b, D, 1, 1, ph.gGg, a, b)
    end
end

"""
    sigma_mig_sc!(n, G, Σ, g, ph; method = :fixpoint)

Nambu Migdal self-energy at the KB time step `n` (NESSi
`Hols::Sigma_Mig_sc(tstp, ...)`).
"""
function sigma_mig_sc!(n::Int, G::ℱ, Σ::ℱ, g::Cf, ph::HolsteinPhonon; method::Symbol = :fixpoint)
    _gGg!(n, G, g, ph)
    _pi_nambu!(n, G, ph)
    step_D!(n, ph; method = method)
    _sigma_nambu!(n, Σ, ph.D, ph)
    return Σ
end

"""
    sigma_mig_sc_start!(G, Σ, g, ph)

Nambu Migdal self-energy of the start-up time steps.
"""
function sigma_mig_sc_start!(G::ℱ, Σ::ℱ, g::Cf, ph::HolsteinPhonon)
    for n = 1:ph.k+1
        _gGg!(n, G, g, ph)
        _pi_nambu!(n, G, ph)
    end
    start_D!(ph)
    for n = 1:ph.k+1
        _sigma_nambu!(n, Σ, ph.D, ph)
    end
    return Σ
end

"""
    sigma_umig_sc!(n, G, Σ, g, ph)

Nambu uMig self-energy at the KB time step `n` (NESSi
`Hols::Sigma_uMig_sc(tstp, ...)`).
"""
function sigma_umig_sc!(n::Int, G::ℱ, Σ::ℱ, g::Cf, ph::HolsteinPhonon)
    _gGg!(n, G, g, ph)
    _sigma_nambu!(n, Σ, ph.D0, ph)
    return Σ
end

"""
    sigma_umig_sc_start!(G, Σ, g, ph)

Nambu uMig self-energy of the start-up time steps.
"""
function sigma_umig_sc_start!(G::ℱ, Σ::ℱ, g::Cf, ph::HolsteinPhonon)
    for n = 1:ph.k+1
        sigma_umig_sc!(n, G, Σ, g, ph)
    end
    return Σ
end
