#
# Project : Lavender
# Source  : ness.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks* : *Steady-state Green's Functions (NESS2)*

In a nonequilibrium steady state the Green's functions depend on the
relative time only, ``G(t, t') = G(t - t')``. The retarded and lesser
components are stored on the periodic grid of `Nft` points (see *Remarks* :
*NESS2 Grid* in `structs.jl`), both in the time and in the frequency
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
    fft_to_freq!(A::FFTArray{C64})

Bare (unnormalized) transform of the time data to the frequency data with
the positive exponent (FFT_BACKWARD). Only the frequency data is changed.

*References* :

Please see [`NESSi`] `ness2_fft_impl.hpp`, `fft_array::fft_to_freq`.
"""
function fft_to_freq!(A::FFTArray{C64})
    p = plan_fft(getnft(A), FFT_BACKWARD)
    fft_strided!(A.freq, A.time, 3, FFT_BACKWARD; plan = p, scratch = A.scratch)
    return A
end

"""
    fft_to_time!(A::FFTArray{C64})

Bare (unnormalized) transform of the frequency data to the time data with
the negative exponent (FFT_FORWARD). Only the time data is changed.

*References* :

Please see [`NESSi`] `ness2_fft_impl.hpp`, `fft_array::fft_to_time`.
"""
function fft_to_time!(A::FFTArray{C64})
    p = plan_fft(getnft(A), FFT_FORWARD)
    fft_strided!(A.time, A.freq, 3, FFT_FORWARD; plan = p, scratch = A.scratch)
    return A
end

"""
    fft_to_freq!(G::HermMatrixNESS{C64})

Bare transform of the retarded and lesser components to the frequency
domain.
"""
function fft_to_freq!(G::HermMatrixNESS{C64})
    fft_to_freq!(G.ret)
    fft_to_freq!(G.les)
    return G
end

"""
    fft_to_time!(G::HermMatrixNESS{C64})

Bare transform of the retarded and lesser components to the time domain.
"""
function fft_to_time!(G::HermMatrixNESS{C64})
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
    integral_transform_to_freq!(G::HermMatrixNESS{C64},
                                method::FFTIntegralMethod = FFT_TRAPEZ)

Physical Fourier integral ``G(\\omega) = \\int dt\\ e^{i\\omega t} G(t)``
of the retarded and lesser components, see *Remarks* : *Steady-state
Green's Functions (NESS2)*. For the cubic method of the lesser component
only the correction factor is applied (the boundary data is not used, the
functions are assumed to have decayed).

### Arguments
* G      -> A HermMatrixNESS struct, the time data is the input.
* method -> FFT_TRAPEZ or FFT_CUBIC.

### Returns
* G -> The frequency data is updated.

*References* :

Please see [`NESSi`] `ness2_herm_matrix_ness_impl.hpp`,
`integral_transform_to_freq`. NESSi passes uninitialized end points to
`complex_dftcor_cubic` for the lesser component, here only the correction
factor is computed.
"""
function integral_transform_to_freq!(G::HermMatrixNESS{C64},
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
    integral_transform_to_time!(G::HermMatrixNESS{C64},
                                method::FFTIntegralMethod = FFT_TRAPEZ)

Physical inverse Fourier integral ``G(t) = \\int \\frac{d\\omega}{2\\pi}
e^{-i\\omega t} G(\\omega)`` of the retarded and lesser components, see
*Remarks* : *Steady-state Green's Functions (NESS2)*. The retarded
component is reconstructed from its anti-hermitian (spectral) part and
kept at ``t \\ge 0``, and ``G^<(-t) = -G^<(t)^\\dagger`` is imposed.

### Arguments
* G      -> A HermMatrixNESS struct, the frequency data is the input.
* method -> FFT_TRAPEZ or FFT_CUBIC.

### Returns
* G -> The time data is updated.

*References* :

Please see [`NESSi`] `ness2_herm_matrix_ness_impl.hpp`,
`integral_transform_to_time`.
"""
function integral_transform_to_time!(G::HermMatrixNESS{C64},
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

#=
### *Public Functions* : *Bath Regularization*
=#

"""
    _fermi_cut(x::F64)

Fermi function ``1/(e^x + 1)`` with the cut-offs ``x > 100`` (0) and
``x < -100`` (1). It is a private function.
"""
function _fermi_cut(x::F64)
    x > 100.0 && return 0.0
    x < -100.0 && return 1.0
    return 1.0 / (1.0 + exp(x))
end

"""
    sigma_bath(omega::Real, h::Real, bath_type::DysonBathType, eta::Real,
               beta::Real, mu::Real)

Retarded and lesser self-energy of the bath that regularizes the
steady-state Dyson equation, at the frequency `omega`:

* `BATH_NONE`  : ``\\Sigma^R = \\Sigma^< = 0``.
* `BATH_CONST` : ``\\Sigma^R = -i\\eta``, ``\\Sigma^< = 2i\\eta f(\\omega-\\mu)``.
* `BATH_GAUSS` : as `BATH_CONST`, multiplied by ``e^{-\\omega^2}``.
* `BATH_OHMIC` : ``\\Sigma^R = -i\\eta\\omega e^{-\\omega^2}``, the lesser
  component uses the Bose factor of ``\\beta\\omega``, with the limit
  ``-2i\\eta e^{-\\omega^2}/\\beta`` for ``|\\beta\\omega| < 10^{-6}``.

NESSi writes the damping factor as `exp(-omega*omega/omegac*omegac)` with
`omegac = π/(10h)`. Since the division and the multiplication are
evaluated from the left, the factor is ``e^{-\\omega^2}`` and does not
depend on `h`; this is reproduced exactly. `h` is only checked.

### Arguments
* omega     -> Frequency ω.
* h         -> Time step.
* bath_type -> Type of the bath.
* eta       -> Coupling η ≥ 0.
* beta      -> Inverse temperature of the bath, β > 0.
* mu        -> Chemical potential of the bath (not used by `BATH_OHMIC`).

### Returns
* ret -> Retarded self-energy, a C64.
* les -> Lesser self-energy, a C64.

*References* :

Please see [`NESSi`] `ness2_dyson_impl.hpp`, `sigma_bath`.
"""
function sigma_bath(omega::Real, h::Real, bath_type::DysonBathType, eta::Real,
                    beta::Real, mu::Real)
    isfinite(omega) || throw(ArgumentError("omega must be finite"))
    bath_type == BATH_NONE && return (zero(C64), zero(C64))
    h > 0 || throw(ArgumentError("h must be positive, got $h"))
    eta ≥ 0 || throw(ArgumentError("eta must be ≥ 0, got $eta"))
    beta > 0 || throw(ArgumentError("beta must be positive, got $beta"))

    ω = F64(omega)
    η = F64(eta)
    β = F64(beta)
    if bath_type == BATH_CONST
        fm = _fermi_cut(β * (ω - F64(mu)))
        return complex(0.0, -η), complex(0.0, 2.0 * fm * η)
    elseif bath_type == BATH_GAUSS
        expfac = exp(-ω * ω)
        fm = _fermi_cut(β * (ω - F64(mu)))
        return complex(0.0, -η * expfac), complex(0.0, 2.0 * fm * η * expfac)
    else
        # BATH_OHMIC
        expfac = exp(-ω * ω)
        ret = complex(0.0, -η * ω * expfac)
        arg = β * ω
        if arg > 100.0
            les = zero(C64)
        elseif arg < -100.0
            les = complex(0.0, 2.0 * η * ω * expfac)
        elseif abs(arg) < 1.0e-6
            les = complex(0.0, -2.0 * η * expfac / β)
        else
            les = complex(0.0, -2.0 * η * ω * expfac / (exp(arg) - 1.0))
        end
        return ret, les
    end
end

#=
### *Public Functions* : *Steady-state Dyson Equation*
=#

"""
    dyson_ness!(G::HermMatrixNESS{C64}, mu::Real, H::AbstractMatrix,
                Sigma::HermMatrixNESS{C64}; method = FFT_TRAPEZ,
                bath_type = BATH_NONE, bath_eta = 0.0, bath_beta = 1.0,
                bath_mu = 0.0)

Solve the steady-state Dyson equation in the frequency domain,

```math
G^R(\\omega) = [\\omega + \\mu - H - \\Sigma^R(\\omega) - \\Sigma^R_b(\\omega)]^{-1},\\quad
G^<(\\omega) = G^R(\\omega)[\\Sigma^<(\\omega) + \\Sigma^<_b(\\omega)]G^R(\\omega)^\\dagger,
```

where ``\\Sigma_b`` is the bath self-energy, see `sigma_bath()`. The
self-energy is transformed by the trapezoidal rule. Only the central band
``|m| < (N_{ft}/2-1)/3`` of the frequency modes is filled, all the other
modes of `G` are set to zero. Finally `G` is transformed back to the time
domain. `Sigma` is not changed (NESSi transforms it in place).

### Arguments
* G         -> On return, the Green's function (both domains).
* mu        -> Chemical potential.
* H         -> Hamiltonian, a square matrix.
* Sigma     -> Self-energy, the time data is the input.
* method    -> Only FFT_TRAPEZ.
* bath_type -> Type of the bath, see `sigma_bath()`.
* bath_eta  -> Bath coupling.
* bath_beta -> Bath inverse temperature.
* bath_mu   -> Bath chemical potential.

### Returns
* G -> The Green's function.

*References* :

Please see [`NESSi`] `ness2_dyson_impl.hpp`, `dyson` and `dyson_fft`.
"""
function dyson_ness!(G::HermMatrixNESS{C64}, mu::Real, H::AbstractMatrix,
                     Sigma::HermMatrixNESS{C64}; method::FFTIntegralMethod = FFT_TRAPEZ,
                     bath_type::DysonBathType = BATH_NONE, bath_eta::Real = 0.0,
                     bath_beta::Real = 1.0, bath_mu::Real = 0.0)
    method == FFT_TRAPEZ ||
        throw(ArgumentError("dyson_ness!: only FFT_TRAPEZ is implemented, got $method"))
    iscompatible(G, Sigma) ||
        throw(DimensionMismatch("G and Sigma must have the same grid and size"))
    G.grid == Sigma.grid || throw(ArgumentError("G and Sigma must have the same grid"))
    size(H) == getdims(G) || throw(DimensionMismatch("H must have the size $(getdims(G))"))
    G === Sigma && throw(ArgumentError("G and Sigma must be different objects"))

    g = G.grid
    Nft = g.Nft
    d = getdims(G)[1]
    Id = Matrix{C64}(I, d, d)
    Hm = Matrix{C64}(H)

    # Local copy, the input self-energy is not changed
    S = deepcopy(Sigma)
    integral_transform_to_freq!(S, FFT_TRAPEZ)

    fill!(G.ret.freq, zero(C64))
    fill!(G.les.freq, zero(C64))
    nfreq = (Nft ÷ 2 - 1) ÷ 3
    ginv = similar(Hm)
    tmp = similar(Hm)
    for w = -nfreq+1:nfreq-1
        n = mod(w, Nft) + 1
        ω = _freq_mode(g, w)
        bret, bles = zero(C64), zero(C64)
        if bath_type != BATH_NONE
            bret, bles = sigma_bath(ω, g.h, bath_type, bath_eta, bath_beta, bath_mu)
        end
        @. ginv = (ω + mu) * Id - Hm - S.ret.freq[:, :, n] - bret * Id
        gret = inv(ginv)
        view(G.ret.freq, :, :, n) .= gret
        @. tmp = S.les.freq[:, :, n] + bles * Id
        view(G.les.freq, :, :, n) .= gret * tmp * gret'
    end

    integral_transform_to_time!(G, FFT_TRAPEZ)
    return G
end

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

#=
### *Public Functions* : *Density Matrices*
=#

"""
    density_matrix(G::HermMatrixNESS{C64}, sign::Integer)

Density matrix ``\\rho = \\xi i G^<(0)`` of a steady-state Green's
function, where ``\\xi`` is `sign` (FERMI or BOSE).

*References* :

Please see [`NESSi`] `ness2_utils_impl.hpp`, `density_matrix`.
"""
function density_matrix(G::HermMatrixNESS{C64}, sign::Integer)
    sign in (FERMI, BOSE) || throw(ArgumentError("sign must be FERMI or BOSE"))
    return G.les.time[:, :, 1] .* complex(0.0, F64(sign))
end

"""
    convolution_density_matrix(A::HermMatrixNESS{C64}, B::HermMatrixNESS{C64},
                               sign::Integer)

Density matrix ``\\xi i C^<(0)`` of the steady-state convolution ``C = A
\\ast B``, with ``C^<(\\omega) = A^<(\\omega) B^A(\\omega) + A^R(\\omega)
B^<(\\omega)``. `A` and `B` are transformed to the frequency domain by the
trapezoidal rule on local copies (NESSi transforms the inputs in place),
the inputs are not changed.

### Arguments
* A    -> Steady-state Green's function, the time data is the input.
* B    -> Steady-state Green's function, the time data is the input.
* sign -> FERMI or BOSE.

### Returns
* ρ -> The density matrix.

*References* :

Please see [`NESSi`] `ness2_utils_impl.hpp`, `convolution_density_matrix`.
NESSi checks the shapes with `&&`, here any mismatch is an error.
"""
function convolution_density_matrix(A::HermMatrixNESS{C64}, B::HermMatrixNESS{C64},
                                    sign::Integer)
    sign in (FERMI, BOSE) || throw(ArgumentError("sign must be FERMI or BOSE"))
    iscompatible(A, B) ||
        throw(DimensionMismatch("convolution_density_matrix: input mismatch"))
    Ac = deepcopy(A)
    Bc = deepcopy(B)
    integral_transform_to_freq!(Ac, FFT_TRAPEZ)
    integral_transform_to_freq!(Bc, FFT_TRAPEZ)
    d = getdims(A)[1]
    C = zeros(C64, d, d)
    for k = 1:getnft(A)
        Al = view(Ac.les.freq, :, :, k)
        Ar = view(Ac.ret.freq, :, :, k)
        Bl = view(Bc.les.freq, :, :, k)
        Br = view(Bc.ret.freq, :, :, k)
        C .+= Al * Br' .+ Ar * Bl
    end
    return C .* complex(0.0, sign * A.grid.dw * 0.5 / π)
end

#=
### *Public Functions* : *Steady-state Bubbles*
=#

"""
    _bubble_ness!(C, c₁, c₂, A, a₁, a₂, B, b₁, b₂, kind::Symbol)

Common driver of `bubble1_ness!()` and `bubble2_ness!()`. The matrix
element `(c₁, c₂)` of `C` is written at the times `0 ≤ t < N_{ft}/2`, the
lesser element `(c₂, c₁)` at the negative times. The retarded element at
the negative times and the lesser elements at the Nyquist point are set to
zero. It is a private function.
"""
function _bubble_ness!(C::HermMatrixNESS{C64}, c₁::I64, c₂::I64,
                       A::HermMatrixNESS{C64}, a₁::I64, a₂::I64,
                       B::HermMatrixNESS{C64}, b₁::I64, b₂::I64, kind::Symbol)
    Nft = getnft(C)
    (getnft(A) == Nft && getnft(B) == Nft) ||
        throw(DimensionMismatch("A, B, and C must have the same number of grid points"))
    dc = getdims(C)[1]
    da = getdims(A)[1]
    db = getdims(B)[1]
    (1 ≤ c₁ ≤ dc && 1 ≤ c₂ ≤ dc) || throw(BoundsError(C.ret.time, (c₁, c₂)))
    (1 ≤ a₁ ≤ da && 1 ≤ a₂ ≤ da) || throw(BoundsError(A.ret.time, (a₁, a₂)))
    (1 ≤ b₁ ≤ db && 1 ≤ b₂ ≤ db) || throw(BoundsError(B.ret.time, (b₁, b₂)))

    CR = C.ret.time
    CL = C.les.time
    for t = 0:Nft÷2-1
        n = t + 1
        aret = A.ret.time[a₁, a₂, n]
        ales = A.les.time[a₁, a₂, n]
        bret = B.ret.time[b₁, b₂, n]
        bles = B.les.time[b₁, b₂, n]
        if kind == :bubble1
            bgtr = bret + bles
            les = im * ales * (-conj(bgtr))
            ret = im * aret * (-conj(bles)) + im * ales * conj(bret)
        else
            les = im * ales * bles
            ret = im * aret * bret + im * ales * bret + im * aret * bles
        end
        CR[c₁, c₂, n] = ret
        CL[c₁, c₂, n] = les
        if t > 0
            CL[c₂, c₁, Nft-t+1] = -conj(les)
        end
    end
    for n = Nft÷2+1:Nft
        CR[c₁, c₂, n] = 0.0
    end
    CL[c₁, c₂, Nft÷2+1] = 0.0
    CL[c₂, c₁, Nft÷2+1] = 0.0
    return C
end

"""
    bubble1_ness!(C::HermMatrixNESS{C64}, c1, c2, A::HermMatrixNESS{C64},
                  a1, a2, B::HermMatrixNESS{C64}, b1, b2)

Steady-state particle-hole bubble ``C_{c_1c_2}(t) = i A_{a_1a_2}(t)
B_{b_2b_1}(-t)``:

```math
C^<_{c_1c_2}(t) = i A^<_{a_1a_2}(t) [-B^>(t)^\\dagger]_{b_2b_1},\\quad
C^R_{c_1c_2}(t) = i A^R_{a_1a_2}(t) [-B^<(t)^\\dagger]_{b_2b_1}
                + i A^<_{a_1a_2}(t) [B^R(t)^\\dagger]_{b_2b_1},
```

for ``0 \\le t < N_{ft}/2`` (time domain), with ``C^<_{c_2c_1}(-t) =
-C^<_{c_1c_2}(t)^*``. The indices are 1-based.

*References* :

Please see [`NESSi`] `ness2_utils_impl.hpp`, `Bubble1_ness`.
"""
function bubble1_ness!(C::HermMatrixNESS{C64}, c1::Integer, c2::Integer,
                       A::HermMatrixNESS{C64}, a1::Integer, a2::Integer,
                       B::HermMatrixNESS{C64}, b1::Integer, b2::Integer)
    return _bubble_ness!(C, I64(c1), I64(c2), A, I64(a1), I64(a2), B, I64(b1), I64(b2), :bubble1)
end

"""
    bubble1_ness!(C::HermMatrixNESS{C64}, A::HermMatrixNESS{C64},
                  B::HermMatrixNESS{C64})

Scalar version of `bubble1_ness!()`, all matrices are 1 × 1. The time data
of `C` is reset to zero first.
"""
function bubble1_ness!(C::HermMatrixNESS{C64}, A::HermMatrixNESS{C64},
                       B::HermMatrixNESS{C64})
    zeros!(C, FFT_TIME)
    return _bubble_ness!(C, 1, 1, A, 1, 1, B, 1, 1, :bubble1)
end

"""
    bubble2_ness!(C::HermMatrixNESS{C64}, c1, c2, A::HermMatrixNESS{C64},
                  a1, a2, B::HermMatrixNESS{C64}, b1, b2)

Steady-state particle-particle bubble ``C_{c_1c_2}(t) = i A_{a_1a_2}(t)
B_{b_1b_2}(t)``:

```math
C^<_{c_1c_2}(t) = i A^<_{a_1a_2}(t) B^<_{b_1b_2}(t),\\quad
C^R_{c_1c_2}(t) = i (A^R B^R + A^< B^R + A^R B^<),
```

for ``0 \\le t < N_{ft}/2`` (time domain), with ``C^<_{c_2c_1}(-t) =
-C^<_{c_1c_2}(t)^*``. The indices are 1-based.

*References* :

Please see [`NESSi`] `ness2_utils_impl.hpp`, `Bubble2_ness`.
"""
function bubble2_ness!(C::HermMatrixNESS{C64}, c1::Integer, c2::Integer,
                       A::HermMatrixNESS{C64}, a1::Integer, a2::Integer,
                       B::HermMatrixNESS{C64}, b1::Integer, b2::Integer)
    return _bubble_ness!(C, I64(c1), I64(c2), A, I64(a1), I64(a2), B, I64(b1), I64(b2), :bubble2)
end

"""
    bubble2_ness!(C::HermMatrixNESS{C64}, A::HermMatrixNESS{C64},
                  B::HermMatrixNESS{C64})

Scalar version of `bubble2_ness!()`, all matrices are 1 × 1. The time data
of `C` is reset to zero first.
"""
function bubble2_ness!(C::HermMatrixNESS{C64}, A::HermMatrixNESS{C64},
                       B::HermMatrixNESS{C64})
    zeros!(C, FFT_TIME)
    return _bubble_ness!(C, 1, 1, A, 1, 1, B, 1, 1, :bubble2)
end

#=
### *Public Functions* : *Resampling*
=#

"""
    downsample(A::FFTArray{C64}, factor::Integer)

Keep every `factor`-th time point and the low frequency modes. `Nft` must
be divisible by `2 factor`. The result has `Nft/factor` points.

*References* :

Please see [`NESSi`] `ness2_utils_impl.hpp`, `downsample`.
"""
function downsample(A::FFTArray{C64}, factor::Integer)
    Nft = getnft(A)
    (factor ≥ 1 && Nft % (2 * factor) == 0) ||
        throw(ArgumentError("downsample: wrong factor $factor"))
    Nft₁ = Nft ÷ factor
    B = FFTArray(Nft₁, getdims(A)[1])
    for t = 0:Nft₁-1
        B.time[:, :, t+1] .= view(A.time, :, :, t * factor + 1)
    end
    for w = 0:Nft₁÷2-1
        B.freq[:, :, w+1] .= view(A.freq, :, :, w + 1)
    end
    for w = 1:Nft₁÷2
        B.freq[:, :, Nft₁-w+1] .= view(A.freq, :, :, Nft - w + 1)
    end
    return B
end

"""
    downsample(G::HermMatrixNESS{C64}, factor::Integer)

Downsample both components, see `downsample(::FFTArray, ::Integer)`. The
new grid has the time step `factor h` and the same frequency step.
"""
function downsample(G::HermMatrixNESS{C64}, factor::Integer)
    Nft = getnft(G)
    (factor ≥ 1 && Nft % (2 * factor) == 0) ||
        throw(ArgumentError("downsample: wrong factor $factor"))
    g = GridInfo(Nft ÷ factor, G.grid.h * factor)
    return HermMatrixNESS{C64}(g, downsample(G.ret, factor), downsample(G.les, factor))
end

"""
    upsample(G::HermMatrixNESS{C64}, factor::Integer)

Increase the number of the grid points by `factor` (the time step becomes
`h/factor`, the frequency step is kept). The frequency data is regenerated
from the time data by the trapezoidal rule, the low frequency modes are
copied, the new modes are zero, and the result is transformed back to the
time domain. `G` is not changed.

*References* :

Please see [`NESSi`] `ness2_utils_impl.hpp`, `upsample`.
"""
function upsample(G::HermMatrixNESS{C64}, factor::Integer)
    factor ≥ 1 || throw(ArgumentError("upsample: wrong factor $factor"))
    Nft = getnft(G)
    Nft₁ = Nft * factor
    Gin = deepcopy(G)
    integral_transform_to_freq!(Gin, FFT_TRAPEZ)
    out = HermMatrixNESS(GridInfo(Nft₁, G.grid.h / factor), getdims(G)[1])
    for (X, Y) in ((out.ret, Gin.ret), (out.les, Gin.les))
        for w = 0:Nft÷2-1
            X.freq[:, :, w+1] .= view(Y.freq, :, :, w + 1)
        end
        for w = 1:Nft÷2
            X.freq[:, :, Nft₁-w+1] .= view(Y.freq, :, :, Nft - w + 1)
        end
    end
    integral_transform_to_time!(out, FFT_TRAPEZ)
    return out
end

"""
    resample(G::HermMatrixNESS{C64}, factor::Integer)

Downsample for `factor > 0`, upsample by `-factor` for `factor < 0`.
`factor = 0` is an error.

*References* :

Please see [`NESSi`] `ness2_utils_impl.hpp`, `resample`.
"""
function resample(G::HermMatrixNESS{C64}, factor::Integer)
    factor == 0 && throw(ArgumentError("resample: factor must not be 0"))
    return factor < 0 ? upsample(G, -factor) : downsample(G, factor)
end

#=
### *Public Functions* : *Conversion With Contour Functions*
=#

"""
    cntr2ness!(Gness::HermMatrixNESS{C64}, G::ℱ{C64}, tstp::Integer = getntime(G))

Sample a contour Green's function at the fixed (KB, 1-based) time step
`tstp`: ``G^R(\\Delta t) = G^R(t_n, t_{n-\\Delta t})`` and ``G^<(-\\Delta t)
= G^<(t_{n-\\Delta t}, t_n)`` for ``\\Delta t = 0, \\cdots, \\min(n - 1,
N_{ft}/2 - 1)``, and ``G^<(\\Delta t) = -G^<(-\\Delta t)^\\dagger``. The other
time data is zero. The frequency data is not changed.

*References* :

Please see [`NESSi`] `ness2_utils_decl.hpp`, `cntr2ness`.
"""
function cntr2ness!(Gness::HermMatrixNESS{C64}, G::ℱ{C64}, tstp::Integer = getntime(G))
    getdims(Gness) == getdims(G) ||
        throw(DimensionMismatch("cntr2ness!: matrix size mismatch"))
    1 ≤ tstp ≤ getntime(G) ||
        throw(ArgumentError("cntr2ness!: tstp must be in 1:$(getntime(G)), got $tstp"))
    n = I64(tstp)
    Nft = getnft(Gness)
    zeros!(Gness, FFT_TIME)
    for dt = 0:min(n - 1, Nft ÷ 2 - 1)
        Gness.ret.time[:, :, dt+1] .= G.ret[n, n-dt]
        M = G.less[n-dt, n]
        if dt == 0
            Gness.les.time[:, :, 1] .= M
        else
            Gness.les.time[:, :, Nft-dt+1] .= M
            Gness.les.time[:, :, dt+1] .= .-M'
        end
    end
    return Gness
end

"""
    ness2cntr!(G::ℱ{C64}, Gness::HermMatrixNESS{C64})

Fill the real-time retarded and lesser components of a contour Green's
function assuming the time translation invariance, ``G^R(t_n, t_m) =
G^R(t_n - t_m)`` and ``G^<(t_m, t_n) = -G^<(t_n - t_m)^\\dagger`` for ``n \\ge
m``. If ``n - m \\ge N_{ft}/2`` the values are zero. The Matsubara and
mixing components are not changed.

*References* :

Please see [`NESSi`] `ness2_utils_decl.hpp`, `ness2cntr`.
"""
function ness2cntr!(G::ℱ{C64}, Gness::HermMatrixNESS{C64})
    getdims(Gness) == getdims(G) ||
        throw(DimensionMismatch("ness2cntr!: matrix size mismatch"))
    Nft = getnft(Gness)
    Z = zeros(C64, getdims(G)...)
    for n = 1:getntime(G)
        for m = 1:n
            dt = n - m
            if dt < Nft ÷ 2
                G.ret[n, m] = Gness.ret.time[:, :, dt+1]
                G.less[m, n] = -Gness.les.time[:, :, dt+1]'
            else
                G.ret[n, m] = Z
                G.less[m, n] = Z
            end
        end
    end
    return G
end
