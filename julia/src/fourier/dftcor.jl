#
# Project : Lavender
# Source  : dftcor.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks* : *Cubically Corrected Discrete Fourier Transform*

For a function ``f(t)`` given on an equidistant grid ``t_r = r\delta``
(``r = 0, \cdots, N``), the Fourier integral

```math
\begin{equation}
I(\omega) = \int^{N\delta}_0 dt\ e^{i\omega t} f(t)
\end{equation}
```

is approximated by piecewise cubic interpolation of ``f(t)``, which
yields

```math
\begin{equation}
I(\omega) \approx \delta \left\{
    W(\theta) \sum^{N}_{r=0} e^{i\omega t_r} f_r
    + \sum^{3}_{j=0} \left[
        \alpha_j(\theta) f_j
        + e^{i\omega N\delta} \alpha^{*}_j(\theta) f_{N-j}
    \right]
\right\},
\end{equation}
```

where ``\theta = \omega\delta``, ``W(\theta)`` is the correction factor
and ``\alpha_j(\theta)`` are the boundary terms. For Matsubara frequencies
``e^{i\omega_n \beta} = \xi``.

*References* :

W. H. Press et al., Numerical Recipes 3rd Edition, Section 13.9.

Please see [`NESSi`] `fourier.cpp`, `get_dftcorr_cubic`, and
`cntr_matsubara_impl.hpp`, `matsubara_dft` and `matsubara_ft`.
=#

#=
### *Private Service Functions* : *Corrected DFT*
=#

"""
    _dftcorr_cubic(θ::F64)

Return the correction factor ``W(\\theta)`` and the boundary terms
``\\alpha_j(\\theta)`` (``j = 0, \\cdots, 3``) of the cubically corrected
discrete Fourier transform. For ``|\\theta| < 0.05`` series expansions
are used to avoid the loss of significant digits. It is a private
function.

### Arguments
* θ -> θ = ωδ, where δ is the grid spacing.

### Returns
* corfac -> Correction factor W(θ).
* endcor -> Boundary terms, endcor[j+1] = αⱼ(θ).
"""
function _dftcorr_cubic(θ::F64)
    if abs(θ) < 5.0e-2
        t = θ
        t2 = t * t
        t4 = t2 * t2
        t6 = t4 * t2
        corfac = 1.0 - (11.0 / 720.0) * t4 + (23.0 / 15120.0) * t6
        ar₀ = (-2.0 / 3.0) + t2 / 45.0 + (103.0 / 15120.0) * t4 - (169.0 / 226800.0) * t6
        ar₁ = (7.0 / 24.0) - (7.0 / 180.0) * t2 + (5.0 / 3456.0) * t4 - (7.0 / 259200.0) * t6
        ar₂ = (-1.0 / 6.0) + t2 / 45.0 - (5.0 / 6048.0) * t4 + t6 / 64800.0
        ar₃ = (1.0 / 24.0) - t2 / 180.0 + (5.0 / 24192.0) * t4 - t6 / 259200.0
        ai₀ = t * (2.0 / 45.0 + (2.0 / 105.0) * t2 - (8.0 / 2835.0) * t4 + (86.0 / 467775.0) * t6)
        ai₁ = t * (7.0 / 72.0 - t2 / 168.0 + (11.0 / 72576.0) * t4 - (13.0 / 5987520.0) * t6)
        ai₂ = t * (-7.0 / 90.0 + t2 / 210.0 - (11.0 / 90720.0) * t4 + (13.0 / 7484400.0) * t6)
        ai₃ = t * (7.0 / 360.0 - t2 / 840.0 + (11.0 / 362880.0) * t4 - (13.0 / 29937600.0) * t6)
    else
        cth = cos(θ)
        sth = sin(θ)
        ctth = cth * cth - sth * sth
        stth = 2.0 * sth * cth
        th2 = θ * θ
        th4 = th2 * th2
        tmth2 = 3.0 - th2
        spth2 = 6.0 + th2
        sth4i = 1.0 / (6.0 * th4)
        tth4i = 2.0 * sth4i
        corfac = tth4i * spth2 * (3.0 - 4.0 * cth + ctth)
        ar₀ = sth4i * (-42.0 + 5.0 * th2 + spth2 * (8.0 * cth - ctth))
        ai₀ = sth4i * (θ * (-12.0 + 6.0 * th2) + spth2 * stth)
        ar₁ = sth4i * (14.0 * tmth2 - 7.0 * spth2 * cth)
        ai₁ = sth4i * (30.0 * θ - 5.0 * spth2 * sth)
        ar₂ = tth4i * (-4.0 * tmth2 + 2.0 * spth2 * cth)
        ai₂ = tth4i * (-12.0 * θ + 2.0 * spth2 * sth)
        ar₃ = sth4i * (2.0 * tmth2 - spth2 * cth)
        ai₃ = sth4i * (6.0 * θ - spth2 * sth)
    end

    endcor = (complex(ar₀, ai₀), complex(ar₁, ai₁), complex(ar₂, ai₂), complex(ar₃, ai₃))
    return corfac, endcor
end

"""
    _dftcorr_linear(θ::F64)

Return the correction factor ``W(\\theta)`` and the boundary terms
``\\alpha_0(\\theta)`` and ``\\alpha_1(\\theta) = 0`` of the linearly
corrected discrete Fourier transform. For ``|\\theta| < 0.05`` series
expansions are used. It is a private function.

### Arguments
* θ -> θ = ωδ, where δ is the grid spacing.

### Returns
* corfac -> Correction factor W(θ).
* endcor -> Boundary terms, endcor[j+1] = αⱼ(θ).

*References* :

Please see [`NESSi`] `fourier.cpp`, `get_dftcorr_linear`.
"""
function _dftcorr_linear(θ::F64)
    if abs(θ) < 5.0e-2
        th2 = θ * θ
        th4 = th2 * th2
        th6 = th4 * th2
        corfac = 1.0 - (1.0 / 12.0) * th2 + (1.0 / 360.0) * th4 - (1.0 / 20160.0) * th6
        ar = -0.5 + th2 / 24.0 - (1.0 / 720.0) * th4 + (1.0 / 40320.0) * th6
        ai = θ * (1.0 / 6.0 - (1.0 / 120.0) * th2 + (1.0 / 5040.0) * th4 - (1.0 / 362880.0) * th6)
    else
        cth = cos(θ)
        sth = sin(θ)
        th2 = θ * θ
        corfac = 2.0 * (1.0 - cth) / th2
        ar = (cth - 1.0) / th2
        ai = (θ - sth) / th2
    end

    return corfac, (complex(ar, ai), zero(C64))
end

#=
*Remarks* : *Corrected Fourier Integrals*

The Fourier integral of a function ``f(t)`` on the interval ``[a,b]``,

```math
\begin{equation}
I(\omega) = \int^b_a dt\ e^{i\omega t} f(t),
\end{equation}
```

is approximated by the linearly or cubically corrected discrete Fourier
transform, see *Remarks* : *Cubically Corrected Discrete Fourier
Transform* in `matsubara.jl`, which contains the only
implementation of the correction coefficients. The functions in this
file are public wrappers with argument checks.

The adaptive Fourier integral splits the panel with the largest error
estimate (the difference between the results obtained with all points
and with every second point) until the panel budget is exhausted.

*Remarks* : *Matsubara Fourier Transform*

For a Matsubara function ``G(\tau)`` given at ``\tau_r = r\beta/N``
(``r = 0, \cdots, N``, KB uses ``N + 1`` points), the Fourier
coefficients are

```math
\begin{equation}
G(i\omega_m) = \int^{\beta}_0 d\tau\ e^{i\omega_m\tau} G(\tau),\quad
\omega_m = \frac{\pi}{\beta}(2m + \delta),
\end{equation}
```

where ``\delta = 1`` for fermions and ``\delta = 0`` for bosons.

*References* :

Please see [`NESSi`] `fourier.cpp`, `fourier.hpp`, and
`cntr_matsubara_impl.hpp`.
=#

#=
### *Public Functions* : *Corrected Fourier Integrals*
=#

"""
    get_dftcorr_linear(theta::Real)

Return the correction factor ``W(\\theta)`` and the boundary terms
``(\\alpha_0, \\alpha_1)`` of the linearly corrected DFT. Here
``\\alpha_1 = 0``.

### Arguments
* theta -> θ = ωδ, where δ is the grid spacing.

### Returns
* corfac -> Correction factor, a F64.
* endcor -> Boundary terms, a tuple of two C64.

*References* :

Please see [`NESSi`] `fourier.cpp`, `get_dftcorr_linear`.
"""
function get_dftcorr_linear(theta::Real)
    isfinite(theta) || throw(ArgumentError("theta must be finite"))
    return _dftcorr_linear(F64(theta))
end

"""
    get_dftcorr_cubic(theta::Real)

Return the correction factor ``W(\\theta)`` and the boundary terms
``(\\alpha_0, \\cdots, \\alpha_3)`` of the cubically corrected DFT.

### Arguments
* theta -> θ = ωδ, where δ is the grid spacing.

### Returns
* corfac -> Correction factor, a F64.
* endcor -> Boundary terms, a tuple of four C64.

*References* :

Please see [`NESSi`] `fourier.cpp`, `get_dftcorr_cubic`.
"""
function get_dftcorr_cubic(theta::Real)
    isfinite(theta) || throw(ArgumentError("theta must be finite"))
    return _dftcorr_cubic(F64(theta))
end

"""
    complex_dftcor_cubic(omega::Real, delta::Real, a::Real, b::Real,
                         endpoints::AbstractVector{<:Number})

Boundary correction of the cubically corrected DFT on the interval
`[a, b]`, i.e., ``\\sum^{3}_{j=0} \\alpha_j f_j + e^{i\\omega(b-a)}
\\sum^{3}_{j=0} \\alpha^{*}_j f_{N-j}``.

### Arguments
* omega     -> Frequency ω.
* delta     -> Grid spacing δ.
* a         -> Lower bound.
* b         -> Upper bound.
* endpoints -> `endpoints[j+1]` = ``f_j`` and `endpoints[8-j]` =
               ``f_{N-j}`` (j = 0, 1, 2, 3).

### Returns
* endcor -> Boundary correction, a C64.
* corfac -> Correction factor W(ωδ), a F64.

*References* :

Please see [`NESSi`] `fourier.cpp`, `complex_dftcor_cubic`.
"""
function complex_dftcor_cubic(omega::Real, delta::Real, a::Real, b::Real,
                              endpoints::AbstractVector{<:Number})
    length(endpoints) == 8 ||
        throw(ArgumentError("endpoints must contain 8 points, got $(length(endpoints))"))
    delta > 0 || throw(ArgumentError("delta must be positive"))
    b > a || throw(ArgumentError("b must be larger than a"))
    isfinite(omega) || throw(ArgumentError("omega must be finite"))
    corfac, endcor = _dft_endcor(F64(omega), F64(delta), F64(a), F64(b),
                                 Vector{C64}(endpoints), 1)
    return endcor, corfac
end

