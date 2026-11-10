#
# Project : Lavender
# Source  : ness.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    _green_equilibrium_ness_adaptive!(G::HermMatrixNESS{C64}, dos, lo::F64,
                                      hi::F64, nn::I64, limit::I64)

Retarded component ``G^R(t) = -i\\int A(\\omega) e^{-i\\omega t}`` at
``t \\ge 0`` by the adaptive Fourier integral, zero at negative times. It is
a private function.

*References* :

Please see [`NESSi`] `ness2_equilibrium_decl.hpp`,
`green_equilibrium_ret_ness_adaptive`.
"""
function _green_equilibrium_ness_adaptive!(G::HermMatrixNESS{C64}, dos, lo::F64,
                                           hi::F64, nn::I64, limit::I64)
    nt = G.grid.Nft ÷ 2
    h = G.grid.h
    P = _adft_sample(x -> C64(dos(x)), lo, hi, nn, limit; w = 0.0, nmin = 16)
    T = G.ret.time
    fill!(T, zero(C64))
    d = getdims(G)[1]
    for n = 0:nt-1
        res, _ = _adaptive_dos_integral(P, -h * n)
        res *= -im
        for i = 1:d
            T[i, i, n+1] = res
        end
    end
    return G
end

"""
    _compute_lesser_adaptive!(A::FFTArray{C64}, nt::I64, integrand, lo::F64,
                              hi::F64, h::F64, nn::I64, limit::I64)

Lesser component ``-i\\int f(\\omega) e^{-i\\omega t}`` at ``0 \\le t <
N_{ft}/2``, ``G^<(-t) = -G^<(t)^\\dagger``, and zero at the Nyquist point.
The panels are selected at the Fourier variable `10h`, as in NESSi. It is a
private function.

*References* :

Please see [`NESSi`] `ness2_equilibrium_decl.hpp`, `compute_lesser`.
"""
function _compute_lesser_adaptive!(A::FFTArray{C64}, nt::I64, integrand, lo::F64,
                                   hi::F64, h::F64, nn::I64, limit::I64)
    Nft = getnft(A)
    d = getdims(A)[1]
    @assert nt ≤ Nft ÷ 2
    T = A.time
    fill!(T, zero(C64))
    P = _adft_sample(x -> C64(integrand(x)), lo, hi, nn, limit; w = 10 * h, nmin = 16)
    for n = 0:nt-1
        res, _ = _adaptive_dos_integral(P, -n * h)
        res *= -im
        for i = 1:d
            T[i, i, n+1] = res
        end
        if n > 0
            for i = 1:d
                T[i, i, Nft-n+1] = -conj(res)
            end
        end
    end
    T[:, :, Nft÷2+1] .= 0.0
    return A
end

"""
    green_equilibrium_ness!(G::HermMatrixNESS{C64}, sign::Integer, dos,
                            beta::Real, mu::Real; method = FFT_TRAPEZ,
                            lo = dos.lo, hi = dos.hi, limit = 100, nn = 20)

Equilibrium steady-state Green's function of the density of states `dos`
(a callable object, `dos(omega)` returns a real number), diagonal in the
orbital space,

```math
G^R(t) = -i\\int d\\omega\\ A(\\omega) e^{-i\\omega t},\\quad
G^<(t) = -\\xi i\\int d\\omega\\ A(\\omega) f(\\omega - \\mu) e^{-i\\omega t},
```

where ``\\xi`` is `sign` (FERMI or BOSE) and ``f`` is the Fermi or Bose
function. For bosons the limit ``A'(\\mu)/\\beta`` is used at
``\\omega \\approx \\mu`` (the DOS is assumed to vanish linearly there).

* `FFT_TRAPEZ`: ``\\mathrm{Im} G^R(\\omega) = -\\pi A`` and the lesser
  component are set on the frequency grid (the Nyquist mode is zero), and
  transformed back by `integral_transform_to_time!()`. `lo`, `hi`,
  `limit`, and `nn` are not used.
* `FFT_ADAPTIVE`: the time data is computed by the adaptive Fourier
  integral on `[lo, hi]` with `nn` subintervals per panel and the panel
  budget `limit`. The frequency data is not changed.

### Arguments
* G      -> On return, the Green's function.
* sign   -> FERMI or BOSE.
* dos    -> Density of states.
* beta   -> Inverse temperature.
* mu     -> Chemical potential.
* method -> FFT_TRAPEZ or FFT_ADAPTIVE.
* lo, hi -> Integration window of FFT_ADAPTIVE.
* limit  -> Panel budget of FFT_ADAPTIVE.
* nn     -> Subintervals per panel of FFT_ADAPTIVE, even and > 16.

### Returns
* G -> The Green's function.

*References* :

Please see [`NESSi`] `ness2_equilibrium_decl.hpp`, `green_equilibrium_ness`.
"""
function green_equilibrium_ness!(G::HermMatrixNESS{C64}, sign::Integer, dos,
                                 beta::Real, mu::Real;
                                 method::FFTIntegralMethod = FFT_TRAPEZ,
                                 lo::Union{Real,Nothing} = nothing,
                                 hi::Union{Real,Nothing} = nothing,
                                 limit::Integer = 100, nn::Integer = 20)
    sign in (FERMI, BOSE) || throw(ArgumentError("sign must be FERMI or BOSE"))
    beta > 0 || throw(ArgumentError("beta must be positive, got $beta"))
    method in (FFT_TRAPEZ, FFT_ADAPTIVE) ||
        throw(ArgumentError("method must be FFT_TRAPEZ or FFT_ADAPTIVE, got $method"))
    β = F64(beta)
    μ = F64(mu)
    g = G.grid
    Nft = g.Nft
    d = getdims(G)[1]

    if method == FFT_ADAPTIVE
        l = F64(lo === nothing ? _dos_bound(dos, :lo) : lo)
        u = F64(hi === nothing ? _dos_bound(dos, :hi) : hi)
        l < u || throw(ArgumentError("lo must be smaller than hi"))
        _green_equilibrium_ness_adaptive!(G, dos, l, u, I64(nn), I64(limit))
        if sign == FERMI
            integrand = x -> -F64(dos(x)) * _fermi_cut(β * (x - μ))
        else
            integrand = x -> _bose_integrand(dos, F64(x), β, μ)
        end
        _compute_lesser_adaptive!(G.les, Nft ÷ 2, integrand, l, u, g.h, I64(nn), I64(limit))
        return G
    end

    # FFT_TRAPEZ
    fill!(G.ret.freq, zero(C64))
    fill!(G.les.freq, zero(C64))
    nfreq = Nft ÷ 2
    for w = -nfreq+1:nfreq-1
        n = mod(w, Nft) + 1
        ω = _freq_mode(g, w)
        aw = F64(dos(ω))
        r = complex(0.0, -π * aw)
        if sign == FERMI
            l = -2.0 * _fermi_cut(β * (ω - μ)) * r
        else
            l = complex(0.0, -_bose_integrand(dos, ω, β, μ) * 2 * π)
        end
        for i = 1:d
            G.ret.freq[i, i, n] = r
            G.les.freq[i, i, n] = l
        end
    end
    integral_transform_to_time!(G, FFT_TRAPEZ)
    return G
end

"""
    force_equilibrium!(G::HermMatrixNESS{C64}, sign::Integer, beta::Real,
                       mu::Real)

Replace the lesser component by the equilibrium one obtained from the
retarded component by the fluctuation-dissipation relation,

```math
G^<(\\omega) = 2\\xi f(\\omega - \\mu)\\frac{G^R(\\omega) - G^R(\\omega)^\\dagger}{2},
```

where ``\\xi`` is `sign` and ``f`` is the Fermi or Bose function (with the
cut-offs ``\\pm 100``). The retarded component is first transformed to the
frequency domain (trapezoidal rule), the retarded time data is not changed.
The lesser time data satisfies ``G^<(-t) = -G^<(t)^\\dagger`` and is zero at
the Nyquist point.

For bosons and a frequency mode at ``\\omega = \\mu`` (``|\\beta(\\omega -
\\mu)| < 10^{-8}``) the factor ``f`` diverges. If the spectral matrix
``A = -(G^R - G^{R\\dagger})/(2\\pi i)`` vanishes there (relative to its
maximum, tolerance ``10^{-12}``) and the left and right slopes from the
neighbouring modes agree (relative tolerance ``10^{-6}``), the limit
``A'(\\mu)/\\beta`` of ``A n_B`` is used with the averaged slope; otherwise a
`DomainError` is thrown.

The sign differs from NESSi 2.0.0, which returns -1 times the lesser
component of `green_equilibrium_ness` (a negative occupation); here the
result agrees with `green_equilibrium_ness!()`. The bosonic zero mode is an
extension, NESSi produces `NaN` there.

### Arguments
* G    -> Green's function, the retarded time data is the input.
* sign -> FERMI or BOSE.
* beta -> Inverse temperature.
* mu   -> Chemical potential.

### Returns
* G -> The Green's function.

*References* :

Please see [`NESSi`] `ness2_herm_matrix_ness_impl.hpp`, `force_equilibrium`.
"""
function force_equilibrium!(G::HermMatrixNESS{C64}, sign::Integer, beta::Real, mu::Real)
    sign in (FERMI, BOSE) || throw(ArgumentError("sign must be FERMI or BOSE"))
    beta > 0 || throw(ArgumentError("beta must be positive, got $beta"))
    β = F64(beta)
    μ = F64(mu)
    g = G.grid
    Nft = g.Nft

    integral_transform_to_freq!(G, FFT_TRAPEZ)
    R = G.ret.freq
    L = G.les.freq
    fill!(L, zero(C64))

    # Spectral matrix of the mode n
    spec(n) = -(R[:, :, n] .- R[:, :, n]') ./ (2π * im)
    scale = maximum(n -> norm(spec(n)), 1:Nft)

    for k = 0:Nft-1
        n = k + 1
        ω = _freq_mode(g, k)
        M = R[:, :, n]
        ImG = 0.5 .* (M .- M')
        arg = β * (ω - μ)
        if sign == BOSE && abs(arg) < 1.0e-8
            A₀ = spec(n)
            Aₗ = spec(mod(k - 1, Nft) + 1)
            Aᵣ = spec(mod(k + 1, Nft) + 1)
            sₗ = (A₀ .- Aₗ) ./ g.dw
            sᵣ = (Aᵣ .- A₀) ./ g.dw
            ok = norm(A₀) ≤ 1.0e-12 * scale &&
                 all(isfinite, sₗ) && all(isfinite, sᵣ) &&
                 norm(sᵣ .- sₗ) ≤ 1.0e-6 * max(norm(sₗ), norm(sᵣ)) + 1.0e-12 * scale
            ok || throw(DomainError(ω, "force_equilibrium!: the spectral function " *
                                       "does not vanish linearly at omega = mu"))
            slope = 0.5 .* (sₗ .+ sᵣ)
            # A n_B → A'(μ)/β, G^< = -2πi A n_B
            view(L, :, :, n) .= (-2π * im / β) .* slope
            continue
        end
        if sign == BOSE
            f = arg > 100.0 ? 0.0 : (arg < -100.0 ? -1.0 : 1.0 / (exp(arg) - 1.0))
        else
            f = _fermi_cut(arg)
        end
        view(L, :, :, n) .= (2.0 * sign * f) .* ImG
    end

    fft_to_time!(G.les)
    T = G.les.time
    T .*= g.dw / (2π)
    for n = 1:Nft÷2-1
        T[:, :, Nft-n+1] .= .-view(T, :, :, n + 1)'
    end
    T[:, :, Nft÷2+1] .= 0.0
    return G
end

