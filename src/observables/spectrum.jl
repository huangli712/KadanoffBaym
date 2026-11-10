#
# Project : Lavender
# Source  : spectrum.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Public Functions* : *Spectral Functions*
=#

"""
    _linear_fourier(v::AbstractVector{C64}, t::AbstractVector{F64}, ω::F64)

Fourier integral ``\\int dt\\ e^{i\\omega t} v(t)`` of the piecewise linear
interpolation of `v` on the equidistant grid `t`. It is a private function.

*References* :

Please see [`NESSi`] `python3/SpectrumCNTRhdf5.py`, `linear_fourier_t2w`.
At ω = 0 the Python code refers to an undefined variable, here the
trapezoidal rule (the exact integral of the linear interpolation) is used.
"""
function _linear_fourier(v::AbstractVector{C64}, t::AbstractVector{F64}, ω::F64)
    N = length(v) - 1
    dt = t[2] - t[1]
    if ω == 0.0
        res = 0.5 * (v[1] + v[N+1])
        for it = 2:N
            res += v[it]
        end
        return res * dt
    end
    res = zero(C64)
    for it = 1:N
        res += v[it] * cis(ω * t[it])
    end
    s = sin(ω * dt / 2.0)
    res *= 4.0 * s * s / ω / ω / dt
    res += (v[N+1] * cis(ω * t[N+1]) - v[1] * cis(ω * t[1])) *
           (1.0 / (im * ω) - (cis(-ω * dt) - 1.0) / ω / ω / dt)
    return res
end


"""
    evaluate_spectrum_windowed(gret::AbstractArray{<:Number,3},
                               t_rel::AbstractVector{<:Real},
                               omega::AbstractVector{<:Real},
                               method::Symbol, window)

Windowed spectral function of the diagonal elements,

```math
A_{ii}(\\omega) = -\\frac{1}{\\pi}\\mathrm{Im}\\int dt\\ e^{i\\omega t}
W(t) G^R_{ii}(t),
```

on the arbitrary frequencies `omega`. The integral runs over the
equidistant grid `t_rel`.

* `method = :cubic`: the corrected DFT `dft_cplx()`. The number of the
  intervals `length(t_rel) - 1` must be even and larger than 16,
  otherwise an `ArgumentError` is thrown (there is no fallback to
  `:linear`).
* `method = :linear`: the exact integral of the linear interpolation, at
  least two points are needed.

### Arguments
* gret   -> Retarded component, an array of the size `(nt, d, d)`, see
            `read_gf_ret_les_tavtrel()`.
* t_rel  -> Equidistant time grid, length `nt`.
* omega  -> Frequencies.
* method -> `:cubic` or `:linear`.
* window -> Window function, `window(t_rel)` returns a real or complex
            vector of the length `nt`.

### Returns
* A -> Spectral function, a real array of the size `(length(omega), d)`.

*References* :

Please see [`NESSi`] `python3/SpectrumCNTRhdf5.py`,
`evaluate_spectrum_windowed`. The Python `cubic` method integrates a cubic
spline with `scipy.integrate.quad`, here the corrected DFT of the same
order is used.
"""
function evaluate_spectrum_windowed(gret::AbstractArray{<:Number,3},
                                    t_rel::AbstractVector{<:Real},
                                    omega::AbstractVector{<:Real},
                                    method::Symbol, window)
    method in (:cubic, :linear) ||
        throw(ArgumentError("method must be :cubic or :linear, got $method"))
    nt, d₁, d₂ = size(gret)
    d₁ == d₂ || throw(ArgumentError("gret must contain square matrices, got $d₁ × $d₂"))
    length(t_rel) == nt ||
        throw(DimensionMismatch("t_rel must have $nt points, got $(length(t_rel))"))
    nt ≥ 2 || throw(ArgumentError("at least two time points are needed"))
    N = nt - 1
    if method == :cubic
        (N > 16 && iseven(N)) ||
            throw(ArgumentError(":cubic needs an even number (> 16) of intervals, got $N"))
    end
    t = Vector{F64}(t_rel)
    dt = t[2] - t[1]
    dt > 0 || throw(ArgumentError("t_rel must be increasing"))
    all(i -> isapprox(t[i+1] - t[i], dt; rtol = 1e-10), 1:N) ||
        throw(ArgumentError("t_rel must be equidistant"))
    all(isfinite, omega) || throw(ArgumentError("omega must be finite"))

    w = window(t_rel)
    (w isa AbstractVector{<:Number} && length(w) == nt) ||
        throw(DimensionMismatch("window(t_rel) must return a vector of the length $nt"))

    A = zeros(F64, length(omega), d₁)
    v = Vector{C64}(undef, nt)
    for i = 1:d₁
        @. v = gret[:, i, i] * w
        for (k, ω) in enumerate(omega)
            if method == :cubic
                res, _ = dft_cplx(ω, t[1], t[end], v)
            else
                res = _linear_fourier(v, t, F64(ω))
            end
            A[k, i] = -imag(res) / π
        end
    end
    return A
end

