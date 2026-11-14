#
# Project : Lavender
# Source  : dos.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Public Types* : *Density of States*
=#

"""
    BetheDOS

Semicircular density of states on `[lo, hi]`, centered at `E0 = (lo +
hi)/2` with `V = (hi - lo)/4` (bandwidth 4V). It is normalized to 1 and
zero outside of `[lo, hi]`.

*References* :

Please see [`NESSi`] `ness2_equilibrium_impl.hpp`, `bethedos`.
"""
struct BetheDOS
    lo :: F64
    hi :: F64
    V  :: F64
    E0 :: F64
end

"""
    BetheDOS(lo::Real = -2.0, hi::Real = 2.0)

Constructor. The default is the semicircle on `[-2, 2]`.
"""
function BetheDOS(lo::Real = -2.0, hi::Real = 2.0)
    lo < hi || throw(ArgumentError("lo must be smaller than hi"))
    return BetheDOS(F64(lo), F64(hi), (F64(hi) - F64(lo)) / 4.0, (F64(hi) + F64(lo)) / 2.0)
end

function (d::BetheDOS)(x::Real)
    arg = 4.0 * d.V * d.V - (x - d.E0) * (x - d.E0)
    num = d.V * d.V * π * 2
    return arg < 0 ? 0.0 : sqrt(arg) / num
end

"""
    GaussianDOS

Gaussian density of states with the mean `mu` and the standard deviation
`sigma`. The integration window is `[mu - 6 sigma, mu + 6 sigma]`, but as
in NESSi the function itself is not truncated. The normalization constant
`2.506623299999997` (an approximation of √(2π)) is copied from NESSi.

*References* :

Please see [`NESSi`] `ness2_equilibrium_impl.hpp`, `gauss`.
"""
struct GaussianDOS
    lo    :: F64
    hi    :: F64
    mu    :: F64
    sigma :: F64
end

"""
    GaussianDOS(mu::Real = 0.0, sigma::Real = 1.0)

Constructor.
"""
function GaussianDOS(mu::Real = 0.0, sigma::Real = 1.0)
    sigma > 0 || throw(ArgumentError("sigma must be positive, got $sigma"))
    return GaussianDOS(F64(mu) - 6 * F64(sigma), F64(mu) + 6 * F64(sigma),
                       F64(mu), F64(sigma))
end

function (d::GaussianDOS)(x::Real)
    y = (x - d.mu) / d.sigma
    return exp(-0.5 * y * y) / (d.sigma * 2.506623299999997)
end

"""
    OhmicSymDOS

Antisymmetric (bosonic) ohmic spectrum ``A(x) = \\mathrm{sign}(x) |x|
e^{-|x|/\\omega_c}/(2\\omega_c^2)``. The integration window is
`[-20 omegac, 20 omegac]`, the function itself is not truncated, as in
NESSi.

*References* :

Please see [`NESSi`] `ness2_equilibrium_impl.hpp`, `ohmic_sym`.
"""
struct OhmicSymDOS
    lo     :: F64
    hi     :: F64
    omegac :: F64
end

"""
    OhmicSymDOS(omegac::Real = 1.0)

Constructor.
"""
function OhmicSymDOS(omegac::Real = 1.0)
    omegac > 0 || throw(ArgumentError("omegac must be positive, got $omegac"))
    return OhmicSymDOS(-20 * F64(omegac), 20 * F64(omegac), F64(omegac))
end

function (d::OhmicSymDOS)(x::Real)
    ax = abs(x)
    r = ax * exp(-ax / d.omegac) / (2.0 * d.omegac * d.omegac)
    return x ≥ 0 ? r : -r
end

#=
### *Public Functions* : *Equilibrium Steady-state Green's Functions*
=#

"""
    _dos_bound(dos, s::Symbol)

Return the field `lo` or `hi` of a DOS object, or throw an error if the
DOS does not have it. It is a private function.
"""
function _dos_bound(dos, s::Symbol)
    hasproperty(dos, s) ||
        throw(ArgumentError("the DOS has no field $s, pass the keyword `$s`"))
    return getproperty(dos, s)
end

"""
    _bose_integrand(dos, ω::F64, β::F64, μ::F64)

``A(\\omega) n_B(\\omega - \\mu)`` with the cut-offs ``\\pm 100`` and the
DOS-slope limit for ``|\\beta(\\omega - \\mu)| < 10^{-8}``. It is a private
function.

*References* :

Please see [`NESSi`] `ness2_equilibrium_decl.hpp`,
`green_equilibrium_ness_bose`.
"""
function _bose_integrand(dos, ω::F64, β::F64, μ::F64)
    arg = β * (ω - μ)
    arg > 100.0 && return 0.0
    arg < -100.0 && return -F64(dos(ω))
    if abs(arg) < 1.0e-8
        δ = 1.0e-6
        slope = (F64(dos(δ + μ)) - F64(dos(μ))) / δ
        return slope / β
    end
    return F64(dos(ω)) / (exp(arg) - 1.0)
end

