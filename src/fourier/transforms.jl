#
# Project : Lavender
# Source  : transforms.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    FFTIntegralMethod

Method of the integral transforms of the NESS2 data, `FFT_TRAPEZ`
(trapezoidal rule), `FFT_CUBIC` (cubically corrected), or `FFT_ADAPTIVE`
(adaptive Fourier integral, only for the equilibrium Green's functions).
"""
@enum FFTIntegralMethod FFT_TRAPEZ FFT_CUBIC FFT_ADAPTIVE

#=
*Remarks* : *Steady-state Green's Functions (NESS2)*

In a nonequilibrium steady state the Green's functions depend on the
relative time only, ``G(t, t') = G(t - t')``. The retarded and lesser
components are stored on the periodic grid of `Nft` points (see *Remarks* :
*NESS2 Grid* in `types/ness.jl`), both in the time and in the frequency
domain. The Fourier convention is

```math
\begin{equation}
G(\omega) = \int dt\ e^{i\omega t} G(t),\quad
G(t) = \int \frac{d\omega}{2\pi} e^{-i\omega t} G(\omega).
\end{equation}
```

The bare transforms `fft_to_freq!` (FFT_BACKWARD, positive exponent) and
`fft_to_time!` (FFT_FORWARD, negative exponent) are unnormalized. The
physical integral transforms multiply by ``h`` or ``\Delta\omega/(2\pi)``:

* `integral_transform_to_freq!`: the retarded component is integrated over
  ``t \ge 0`` (trapezoidal end correction ``-G^R(0)/2`` or the cubic end
  corrections), the lesser component over all times (the cubic method
  only multiplies by the correction factor ``W(\omega h)``). The Nyquist
  point is set to zero.
* `integral_transform_to_time!`: the retarded component is reconstructed
  from ``G^R(\omega) - G^R(\omega)^\dagger`` and kept at ``t \ge 0`` only.
  The lesser component satisfies ``G^<(-t) = -G^<(t)^\dagger``. The
  Nyquist point is set to zero.

*References* :

Please see [`NESSi`] `ness2_herm_matrix_ness_impl.hpp`.
=#

#=
### *Public Functions* : *Bare Transforms*
=#

"""
    fft_to_freq!(A::NessArray{C64})

Bare (unnormalized) transform of the time data to the frequency data with
the positive exponent (FFT_BACKWARD). Only the frequency data is changed.

*References* :

Please see [`NESSi`] `ness2_fft_impl.hpp`, `fft_array::fft_to_freq`.
"""
function fft_to_freq!(A::NessArray{C64})
    p = plan_fft(getnft(A), FFT_BACKWARD)
    fft_strided!(A.freq, A.time, 3, FFT_BACKWARD; plan = p, scratch = A.scratch)
    return A
end

"""
    fft_to_time!(A::NessArray{C64})

Bare (unnormalized) transform of the frequency data to the time data with
the negative exponent (FFT_FORWARD). Only the time data is changed.

*References* :

Please see [`NESSi`] `ness2_fft_impl.hpp`, `fft_array::fft_to_time`.
"""
function fft_to_time!(A::NessArray{C64})
    p = plan_fft(getnft(A), FFT_FORWARD)
    fft_strided!(A.time, A.freq, 3, FFT_FORWARD; plan = p, scratch = A.scratch)
    return A
end

"""
    fft_to_freq!(G::NessHermMat{C64})

Bare transform of the retarded and lesser components to the frequency
domain.
"""
function fft_to_freq!(G::NessHermMat{C64})
    fft_to_freq!(G.ret)
    fft_to_freq!(G.les)
    return G
end

"""
    fft_to_time!(G::NessHermMat{C64})

Bare transform of the retarded and lesser components to the time domain.
"""
function fft_to_time!(G::NessHermMat{C64})
    fft_to_time!(G.ret)
    fft_to_time!(G.les)
    return G
end

#=
### *Public Functions* : *Physical Integral Transforms*
=#

"""
    _check_ness_method(method::FFTIntegralMethod)

Only `FFT_TRAPEZ` and `FFT_CUBIC` are supported by the integral
transforms. It is a private function.
"""
function _check_ness_method(method::FFTIntegralMethod)
    method in (FFT_TRAPEZ, FFT_CUBIC) ||
        throw(ArgumentError("only FFT_TRAPEZ and FFT_CUBIC are supported, got $method"))
    return nothing
end

"""
    integral_transform_to_freq!(G::NessHermMat{C64},
                                method::FFTIntegralMethod = FFT_TRAPEZ)

Physical Fourier integral ``G(\\omega) = \\int dt\\ e^{i\\omega t} G(t)``
of the retarded and lesser components, see *Remarks* : *Steady-state
Green's Functions (NESS2)*. For the cubic method of the lesser component
only the correction factor is applied (the boundary data is not used, the
functions are assumed to have decayed).

### Arguments
* G      -> A NessHermMat struct, the time data is the input.
* method -> FFT_TRAPEZ or FFT_CUBIC.

### Returns
* G -> The frequency data is updated.

*References* :

Please see [`NESSi`] `ness2_herm_matrix_ness_impl.hpp`,
`integral_transform_to_freq`. NESSi passes uninitialized end points to
`complex_dftcor_cubic` for the lesser component, here only the correction
factor is computed.
"""
function integral_transform_to_freq!(G::NessHermMat{C64},
                                     method::FFTIntegralMethod = FFT_TRAPEZ)
    _check_ness_method(method)
    g = G.grid
    Nft = g.Nft
    h = g.h
    nhalf = Nft ÷ 2
    d₁, d₂ = getdims(G)

    # Retarded component, integrated over t ≥ 0
    fft_to_freq!(G.ret)
    R = G.ret.freq
    if method == FFT_CUBIC
        a = 0.0
        b = (nhalf - 1) * h
        T = G.ret.time
        f₈ = Vector{C64}(undef, 8)
        endcor = Matrix{C64}(undef, d₁, d₂)
        for k = 0:Nft-1
            ω = _freq_mode(g, k)
            corfac = 0.0
            for j = 1:d₂, i = 1:d₁
                for m = 0:3
                    f₈[m+1] = T[i, j, m+1]
                    f₈[8-m] = T[i, j, nhalf-1-m+1]
                end
                corfac, endcor[i, j] = _dft_endcor(ω, h, a, b, f₈, 1)
            end
            Z = view(R, :, :, k + 1)
            @. Z = Z * corfac + endcor * cis(ω * a)
        end
    else
        G₀ = G.ret.time[:, :, 1]
        for k = 1:Nft
            Z = view(R, :, :, k)
            @. Z = Z - 0.5 * G₀
        end
    end
    R .*= h
    R[:, :, nhalf+1] .= 0.0

    # Lesser component, integrated over all times
    fft_to_freq!(G.les)
    L = G.les.freq
    if method == FFT_CUBIC
        for k = 0:Nft-1
            corfac, _ = _dftcorr_cubic(_freq_mode(g, k) * h)
            Z = view(L, :, :, k + 1)
            Z .*= corfac
        end
    end
    L .*= h
    L[:, :, nhalf+1] .= 0.0

    return G
end

"""
    integral_transform_to_time!(G::NessHermMat{C64},
                                method::FFTIntegralMethod = FFT_TRAPEZ)

Physical inverse Fourier integral ``G(t) = \\int \\frac{d\\omega}{2\\pi}
e^{-i\\omega t} G(\\omega)`` of the retarded and lesser components, see
*Remarks* : *Steady-state Green's Functions (NESS2)*. The retarded
component is reconstructed from its anti-hermitian (spectral) part and
kept at ``t \\ge 0``, and ``G^<(-t) = -G^<(t)^\\dagger`` is imposed.

### Arguments
* G      -> A NessHermMat struct, the frequency data is the input.
* method -> FFT_TRAPEZ or FFT_CUBIC.

### Returns
* G -> The time data is updated.

*References* :

Please see [`NESSi`] `ness2_herm_matrix_ness_impl.hpp`,
`integral_transform_to_time`.
"""
function integral_transform_to_time!(G::NessHermMat{C64},
                                     method::FFTIntegralMethod = FFT_TRAPEZ)
    _check_ness_method(method)
    g = G.grid
    Nft = g.Nft
    nhalf = Nft ÷ 2
    dw = g.dw

    # Retarded component from G^R(ω) - G^R(ω)†
    R = G.ret.freq
    tf = similar(R)
    for k = 1:Nft
        tf[:, :, k] .= view(R, :, :, k) .- view(R, :, :, k)'
    end
    tt = similar(R)
    fft_strided!(tt, tf, 3, FFT_FORWARD; scratch = G.ret.scratch)
    if method == FFT_CUBIC
        for n = 0:nhalf-1
            corfac, _ = _dftcorr_cubic(-_time_mode(g, n) * dw)
            view(tt, :, :, n + 1) .*= corfac
        end
    end
    T = G.ret.time
    fill!(T, zero(C64))
    for n = 1:nhalf
        T[:, :, n] .= view(tt, :, :, n) .* (dw * 0.5 / π)
    end

    # Lesser component
    fft_to_time!(G.les)
    L = G.les.time
    if method == FFT_CUBIC
        for n = 0:nhalf-1
            corfac, _ = _dftcorr_cubic(-_time_mode(g, n) * dw)
            view(L, :, :, n + 1) .*= corfac
        end
    end
    L .*= (dw * 0.5 / π)
    for n = 1:nhalf-1
        L[:, :, Nft-n+1] .= .-view(L, :, :, n + 1)'
    end
    L[:, :, nhalf+1] .= 0.0

    return G
end

