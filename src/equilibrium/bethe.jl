#
# Project : Lavender
# Source  : bethe.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    _bethe_dos(V::F64, E0::F64)

Return the normalized semicircular density of states of the Bethe
lattice with the bandwidth 4V centered at E0, and its support. It is a
private function.
"""
function _bethe_dos(V::F64, E0::F64)
    V > 0.0 || throw(ArgumentError("V must be positive"))
    d = BetheDOS(E0 - 2.0 * V, E0 + 2.0 * V)
    return d, d.lo, d.hi
end

"""
    green_equilibrium_mat_bethe!(G::ℱ{C64}, β::F64; V::F64 = 1.0,
                                 E0::F64 = 0.0, μ::F64 = 0.0,
                                 limit::I64 = 100, nn::I64 = 20)

Calculate the Matsubara component of the equilibrium Green's function of
the Bethe lattice, ``\\rho(\\epsilon) = \\sqrt{4V^2 - (\\epsilon - E_0)^2}
/ (2\\pi V^2)``.

See also: [`green_equilibrium_mat!`](@ref).
"""
function green_equilibrium_mat_bethe!(G::ℱ{C64}, β::F64; V::F64 = 1.0,
                                      E0::F64 = 0.0, μ::F64 = 0.0,
                                      limit::I64 = 100, nn::I64 = 20)
    ρ, lo, hi = _bethe_dos(V, E0)
    green_equilibrium_mat!(G, ρ, lo, hi, β; μ = μ, limit = limit, nn = nn)
end

"""
    green_equilibrium_bethe!(G::ℱ{C64}, β::F64, h::F64; V::F64 = 1.0,
                             E0::F64 = 0.0, μ::F64 = 0.0,
                             limit::I64 = 100, nn::I64 = 20)

Calculate all the components of the equilibrium Green's function of the
Bethe lattice, ``\\rho(\\epsilon) = \\sqrt{4V^2 - (\\epsilon - E_0)^2} /
(2\\pi V^2)``.

See also: [`green_equilibrium!`](@ref).
"""
function green_equilibrium_bethe!(G::ℱ{C64}, β::F64, h::F64; V::F64 = 1.0,
                                  E0::F64 = 0.0, μ::F64 = 0.0,
                                  limit::I64 = 100, nn::I64 = 20)
    ρ, lo, hi = _bethe_dos(V, E0)
    green_equilibrium!(G, ρ, lo, hi, β, h; μ = μ, limit = limit, nn = nn)
end

