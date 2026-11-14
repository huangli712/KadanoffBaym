#
# Project : Lavender
# Source  : adft.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Private Struct* : *_AdftPanels*
=#

"""
    _AdftPanels

Panels of the adaptive Fourier integral. `a[i]` and `b[i]` are the
bounds of the i-th panel, and `f[i]` contains the function values at
the `nn + 1` equidistant points of the i-th panel. It is a private
struct.
"""
struct _AdftPanels
    a :: Vector{F64}
    b :: Vector{F64}
    f :: Vector{Vector{C64}}
end

#=
### *Private Service Functions* : *Adaptive Fourier Integral*
=#

"""
    _dft_endcor(w::F64, δ::F64, a::F64, b::F64, f::Vector{C64}, step::I64)

Correction factor and boundary terms of the cubically corrected DFT of
the panel `[a, b]`. Only every `step`-th point of `f` is used, and `δ` is
the corresponding grid spacing. It is a private function.
"""
function _dft_endcor(w::F64, δ::F64, a::F64, b::F64, f::Vector{C64}, step::I64)
    n = length(f) - 1
    corfac, α = _dftcorr_cubic(w * δ)
    cl = zero(C64)
    cr = zero(C64)
    for j = 0:3
        cl += f[step * j + 1] * α[j+1]
        cr += f[n - step * j + 1] * conj(α[j+1])
    end
    cr *= cis(w * (b - a))
    return corfac, cl + cr
end

"""
    _dft_panel(w::F64, a::F64, b::F64, f::Vector{C64})

Cubically corrected DFT of a single panel, i.e., an approximation to
``\\int^b_a dt\\ e^{i\\omega t} f(t)``. The error estimate is the
difference to the result obtained with every second point. It is a
private function.

### Arguments
* w -> Frequency ω.
* a -> Lower bound.
* b -> Upper bound.
* f -> Function values at `n + 1` equidistant points, `n` is even.

### Returns
* res -> Value of the integral.
* err -> Error estimate.

*References* :

Please see [`NESSi`] `fourier.cpp`, `dft_cplx`.
"""
function _dft_panel(w::F64, a::F64, b::F64, f::Vector{C64})
    n = length(f) - 1
    (n > 16 && iseven(n) && b > a) ||
        throw(ArgumentError("_dft_panel: wrong input, n = $n, [$a, $b]"))
    δ = (b - a) / n
    θ = w * δ

    # Corrections using all points, and every second point
    corfac, endcor = _dft_endcor(w, δ, a, b, f, 1)
    corfac₂, endcor₂ = _dft_endcor(w, 2.0 * δ, a, b, f, 2)

    # Discrete Fourier transform, every second point first
    res₂ = zero(C64)
    for j = 0:2:n
        res₂ += cis(j * θ) * f[j+1]
    end
    res = res₂
    for j = 1:2:n-1
        res += cis(j * θ) * f[j+1]
    end

    # Apply the corrections
    expfac = δ * cis(w * a)
    res = (res * corfac + endcor) * expfac
    res₂ = (res₂ * corfac₂ + endcor₂) * (2.0 * expfac)

    return res, res - res₂
end

"""
    _adft_split!(P::_AdftPanels, i::I64, fz)

Bisect the i-th panel. The function values are reused at the old grid
points and evaluated at the new ones. It is a private function.
"""
function _adft_split!(P::_AdftPanels, i::I64, fz)
    a₀, b₀, f₀ = P.a[i], P.b[i], P.f[i]
    n = length(f₀) - 1
    mid = (a₀ + b₀) / 2.0
    δ = (b₀ - a₀) / (2.0 * n)
    m = n ÷ 2

    fl = Vector{C64}(undef, n + 1)
    fr = Vector{C64}(undef, n + 1)
    for j = 0:n
        if iseven(j)
            fl[j+1] = f₀[j ÷ 2 + 1]
            fr[j+1] = f₀[m + j ÷ 2 + 1]
        else
            fl[j+1] = fz(a₀ + δ * j)
            fr[j+1] = fz(mid + δ * j)
        end
    end

    P.b[i] = mid
    P.f[i] = fl
    insert!(P.a, i + 1, mid)
    insert!(P.b, i + 1, b₀)
    insert!(P.f, i + 1, fr)
    return P
end

"""
    _adft_sample(fz, lo::F64, hi::F64, nn::I64, limit::I64;
                 w::F64 = 0.0, nmin::I64 = 20)

Determine the panels of the adaptive Fourier integral of `fz` on the
interval `[lo, hi]`. The panels are selected according to the error
estimates at the Fourier variable `w` (ω = 0 by default), see *Remarks* :
*Equilibrium Green's Functions From DOS*. It is a private function.

### Arguments
* fz    -> Function, fz(x) returns a complex number.
* lo    -> Lower bound.
* hi    -> Upper bound.
* nn    -> Number of subintervals of each panel, even and ≥ nmin.
* limit -> Budget of the panels, `limit ≥ 1`.
* w     -> Fourier variable of the error estimates.
* nmin  -> Lower bound of `nn`.

### Returns
* P -> A _AdftPanels struct.

*References* :

Please see [`NESSi`] `fourier.hpp`, `adft_func::sample`.
"""
function _adft_sample(fz, lo::F64, hi::F64, nn::I64, limit::I64;
                      w::F64 = 0.0, nmin::I64 = 20)
    lo < hi || throw(ArgumentError("lo must be smaller than hi"))
    (iseven(nn) && nn ≥ nmin) ||
        throw(ArgumentError("nn must be even and ≥ $nmin, got $nn"))
    limit ≥ 1 || throw(ArgumentError("limit must be ≥ 1, got $limit"))

    # Sample the first panel
    δ = (hi - lo) / nn
    f = C64[fz(lo + i * δ) for i = 0:nn]
    P = _AdftPanels([lo], [hi], [f])
    limit == 1 && return P

    # Split the first panel
    _adft_split!(P, 1, fz)
    errs = [_dft_panel(w, P.a[i], P.b[i], P.f[i])[2] for i = 1:2]

    # Split the panel with the largest error estimate
    while length(P.a) + 1 < limit
        isplit = 0
        emax = -1.0
        for i in eachindex(errs)
            if abs(errs[i]) > emax
                isplit = i
                emax = abs(errs[i])
            end
        end
        _adft_split!(P, isplit, fz)
        errs[isplit] = _dft_panel(w, P.a[isplit], P.b[isplit], P.f[isplit])[2]
        insert!(errs, isplit + 1,
                _dft_panel(w, P.a[isplit+1], P.b[isplit+1], P.f[isplit+1])[2])
    end

    return P
end

"""
    _adft_eval(P::_AdftPanels, w::F64)

Fourier integral ``\\int d\\epsilon\\ e^{i\\omega\\epsilon} f(\\epsilon)``
on the panels `P`. The error estimate is returned as a diagnostic only.
It is a private function.

### Arguments
* P -> Panels, see `_adft_sample()`.
* w -> Frequency ω.

### Returns
* res -> Value of the integral.
* err -> Error estimate.

*References* :

Please see [`NESSi`] `fourier.hpp`, `adft_func::dft`.
"""
function _adft_eval(P::_AdftPanels, w::F64)
    res = zero(C64)
    err = zero(C64)
    for i in eachindex(P.a)
        rᵢ, eᵢ = _dft_panel(w, P.a[i], P.b[i], P.f[i])
        res += rᵢ
        err += eᵢ
    end
    isfinite(res) || throw(DomainError(res, "the Fourier integral is not finite"))
    return res, err
end
