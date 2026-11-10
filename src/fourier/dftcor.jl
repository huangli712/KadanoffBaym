#
# Project : Lavender
# Source  : dftcor.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

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
Transform* in `matsubara_fourier_core.jl`, which contains the only
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


