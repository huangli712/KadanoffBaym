#
# Project : Lavender
# Source  : matsubara.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks* : *Matsubara Seed*

The private function `_matsubara_seed!` solves the integral equation

```math
\begin{equation}
X(\tau) + \int^{\beta}_0 d\tau'\ F(\tau - \tau') X(\tau') = B(\tau)
\end{equation}
```

in the Matsubara frequency domain. The first order tail
``q_{\infty} = \xi B(\beta) - B(0)`` is treated analytically, i.e.,

```math
\begin{equation}
X(\tau) = c_0(\tau) + \frac{1}{\beta} \sum_n e^{-i\omega_n \tau}
    [1 + F(i\omega_n)]^{-1}
    \left\{
        B(i\omega_n) - [1 + F(i\omega_n)]\frac{q_{\infty}}{i\omega_n}
    \right\},
\end{equation}
```

where ``c_0(\tau) = -q_{\infty}/2`` for fermions, and ``c_0(\tau) =
q_{\infty}(\tau/\beta - 1/2)`` for bosons. The frequency sum includes
`2pcf + 1` copies of the basic frequency window (oversampling). For
the bosonic zero frequency the tail term is skipped.

This is not a public Fourier solver. It is the preconditioner used by
the fixpoint and conjugate gradient Matsubara solvers.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_mat_fourier_dispatch`.
=#

"""
    _first_order_tail!(xmat::AbstractArray{C64,3}, coeff::AbstractMatrix{C64},
                       β::F64, sign::I64)

First order tail correction ``c_0(\\tau)`` at the `N + 1` imaginary time
points ``\\tau_r = r\\beta/N``, where `N + 1 = size(xmat, 3)`. It is
``-a_0/2`` for fermions, and ``a_0(\\tau/\\beta - 1/2)`` for bosons,
where ``a_0`` is `coeff`. It is a private function.

### Arguments
* xmat  -> On return, xmat[:,:,r+1] = c₀(τᵣ).
* coeff -> Coefficient a₀ (jump of the Matsubara function).
* β     -> Inverse temperature.
* sign  -> FERMI or BOSE.

### Returns
* xmat -> The tail correction.

*References* :

Please see [`NESSi`] `cntr_matsubara_impl.hpp`, `set_first_order_tail`.
"""
function _first_order_tail!(xmat::AbstractArray{C64,3}, coeff::AbstractMatrix{C64},
                            β::F64, sign::I64)
    N = size(xmat, 3) - 1
    δτ = β / N
    for r = 0:N
        τ = r * δτ
        Z = view(xmat, :, :, r + 1)
        @. Z = -0.5 * coeff
        if sign == BOSE
            @. Z = Z + (τ / β) * coeff
        end
    end
    return xmat
end

"""
    _matsubara_dft(A::Gᵐᵃᵗ{C64}, sign::I64)

Raw discrete Fourier transform (without the factor δτ) of a Matsubara
function, ``\\sum^{N}_{r=0} e^{i\\pi(2m+1)r/N} A(\\tau_r)`` for fermions,
or ``\\sum^{N}_{r=0} e^{2i\\pi m r/N} A(\\tau_r)`` for bosons, where
``m = 0, \\cdots, N``. Here ``N = ntau - 1`` is the number of
imaginary time intervals. It is a private function.

### Arguments
* A    -> Matsubara function.
* sign -> FERMI or BOSE.

### Returns
* mdft -> mdft[:,:,m+1] is the transform for frequency index m.
"""
function _matsubara_dft(A::Gᵐᵃᵗ{C64}, sign::I64)
    @assert sign in (FERMI, BOSE)
    N = getntau(A) - 1
    d₁, d₂ = getdims(A)
    one = (sign == FERMI) ? 1 : 0

    mdft = zeros(C64, d₁, d₂, N + 1)
    for m = 0:N
        Z = view(mdft, :, :, m + 1)
        for r = 0:N
            arg = ((2 * m + one) * r * π) / N
            Z .= Z .+ cis(arg) .* _block(A, r + 1)
        end
    end

    return mdft
end

"""
    _matsubara_ft_direct(m::I64, A::Gᵐᵃᵗ{C64}, mdft::Array{C64,3},
                         sign::I64, β::F64, order::I64)

Corrected Fourier transform of a Matsubara function at the Matsubara
frequency with index `m` (any integer), i.e. an approximation to
``\\int^{\\beta}_0 d\\tau\\ e^{i\\omega_m \\tau} A(\\tau)``. The raw
transform `mdft` is periodically mapped. It is a private function.

### Arguments
* m     -> Index of the Matsubara frequency.
* A     -> Matsubara function.
* mdft  -> Raw transform of `A`, see `_matsubara_dft()`.
* sign  -> FERMI or BOSE.
* β     -> Inverse temperature.
* order -> Order of the boundary correction, 0, 1, or 3.

### Returns
* res -> Fourier transform, a matrix.
"""
function _matsubara_ft_direct(m::I64, A::Gᵐᵃᵗ{C64}, mdft::Array{C64,3},
                              sign::I64, β::F64, order::I64)
    @assert sign in (FERMI, BOSE)
    N = getntau(A) - 1
    δτ = β / N
    one = (sign == FERMI) ? 1 : 0

    # θ = ωₘ δτ
    arg = (2 * m + one) * π / N
    if order == 0
        corfac = 1.0
        bcorr = (zero(C64),)
    elseif order == 1
        corfac, bcorr = _dftcorr_linear(arg)
    elseif order == 3
        corfac, bcorr = _dftcorr_cubic(arg)
    else
        throw(ArgumentError("_matsubara_ft_direct: unsupported order $order"))
    end

    # Shift m to the interval [0, N-1]
    m₁ = mod(m, N)
    res = corfac .* mdft[:, :, m₁ + 1]
    for r = 0:min(order, length(bcorr) - 1)
        b = bcorr[r+1]
        res .= res .+ b .* _block(A, r + 1) .+
                      (sign * conj(b)) .* _block(A, N - r + 1)
    end

    return δτ .* res
end

#=
### *Private Service Functions* : *Matsubara Seed*
=#

"""
    _matsubara_seed!(X::Gᵐᵃᵗ{C64}, F::Gᵐᵃᵗ{C64}, B::Gᵐᵃᵗ{C64}, β::F64,
                     sign::I64; pcf::I64, order::I64 = 3)

Solve ``[1 + F] * X = B`` for Matsubara functions in the frequency
domain, see *Remarks* : *Matsubara Seed*. The number of imaginary time
intervals must be even and at least 4. It is a private function.

### Arguments
* X     -> Matsubara function, the solution.
* F     -> Matsubara function, the kernel.
* B     -> Matsubara function, the source term.
* β     -> Inverse temperature.
* sign  -> FERMI or BOSE.
* pcf   -> Number of the oversampling copies on each side.
* order -> Order of the boundary correction.

### Returns
* X -> The solution.
"""
function _matsubara_seed!(X::Gᵐᵃᵗ{C64}, F::Gᵐᵃᵗ{C64}, B::Gᵐᵃᵗ{C64}, β::F64,
                          sign::I64; pcf::I64, order::I64 = 3)
    # Sanity check
    sign in (FERMI, BOSE) || throw(ArgumentError("sign must be FERMI or BOSE"))
    iscompatible(X, F) || throw(DimensionMismatch("X and F are not compatible"))
    iscompatible(X, B) || throw(DimensionMismatch("X and B are not compatible"))
    d₁, d₂ = getdims(X)
    d₁ == d₂ || throw(DimensionMismatch("square matrices are expected"))
    N = getntau(X) - 1
    (iseven(N) && N ≥ 4) ||
        throw(ArgumentError("the number of imaginary time intervals must be even and ≥ 4, got $N"))
    pcf ≥ 0 || throw(ArgumentError("pcf must be nonnegative"))
    β > 0.0 || throw(ArgumentError("β must be positive"))

    # Parameters
    δτ = β / N
    m₂ = N ÷ 2
    one = (sign == FERMI) ? 1 : 0
    Id = Matrix{C64}(I, d₁, d₁)

    # First order tail correction, q∞ = ξ B(β) - B(0)
    qmasy = sign .* _block(B, N + 1) .- _block(B, 1)
    #
    xmat = zeros(C64, d₁, d₁, N + 1)
    _first_order_tail!(xmat, qmasy, β, sign)

    # Raw discrete Fourier transforms of F and B
    fmdft = _matsubara_dft(F, sign)
    qmdft = _matsubara_dft(B, sign)

    # Oversampling loop over the Matsubara frequencies
    for p = -pcf:pcf
        # For bosons the middle interval includes the zero frequency
        m_low, m_high = 0, 0
        if sign == BOSE
            m_high = p < 0 ? 0 : 1
            m_low = p > 0 ? -1 : 0
        end
        #
        for m = (-m₂ - m_low):(m₂ - 1 + m_high)
            mp = m + p * N
            ωₙ = (2 * mp + one) * π / β
            fiω = _matsubara_ft_direct(mp, F, fmdft, sign, β, order)
            qiω = _matsubara_ft_direct(mp, B, qmdft, sign, β, order)
            #
            # Tail term in frequency space, q∞ / (i ωₙ)
            if sign == BOSE && mp == 0
                z₃ = zeros(C64, d₁, d₁)
            else
                z₃ = (-im / ωₙ) .* qmasy
            end
            #
            # [1 + F]⁻¹ [B - (1 + F) q∞ / (i ωₙ)] / β
            z₁ = Id .+ fiω
            z₂ = qiω .- z₃ .- fiω * z₃
            z₁ = (z₁ \ z₂) ./ β
            #
            for r = 0:N
                Z = view(xmat, :, :, r + 1)
                Z .= Z .+ cis(-ωₙ * r * δτ) .* z₁
            end
        end
    end

    copyto!(X.data, xmat)
    all(isfinite, X.data) || throw(DomainError(X.data, "Matsubara seed is not finite"))

    return X
end

"""
    _matsubara_seed!(X::ℱ{C64}, F::ℱ{C64}, B::ℱ{C64}, β::F64,
                     I::Integrator; pcf::I64, order::I64 = 3)

Solve ``[1 + F] * X = B`` for the Matsubara components of contour
functions. The statistics is taken from `X`. The integration weights in
`I` are not used by the frequency-domain solver, the argument keeps the
interface shared by the Dyson and VIE2 solvers. It is a private function.

See also: [`_matsubara_seed!`](@ref).
"""
function _matsubara_seed!(X::ℱ{C64}, F::ℱ{C64}, B::ℱ{C64}, β::F64,
                          I::Integrator; pcf::I64, order::I64 = 3)
    getsign(X) == getsign(F) == getsign(B) ||
        throw(ArgumentError("X, F, and B must have the same statistics"))
    _matsubara_seed!(X.mat, F.mat, B.mat, β, getsign(X); pcf = pcf, order = order)
    return X
end

"""
    dft_cplx(omega::Real, a::Real, b::Real, values::AbstractVector{<:Number})

Fourier integral ``\\int^b_a dt\\ e^{i\\omega t} f(t)`` by the cubically
corrected DFT. The function is given at `N + 1` equidistant points, `N`
must be even and larger than 16. The error estimate is the difference to
the result obtained with every second point.

### Arguments
* omega  -> Frequency ω.
* a      -> Lower bound.
* b      -> Upper bound.
* values -> Function values at the `N + 1` points.

### Returns
* res -> Fourier integral, a C64.
* err -> Error estimate, a C64.

*References* :

Please see [`NESSi`] `fourier.cpp`, `dft_cplx`.
"""
function dft_cplx(omega::Real, a::Real, b::Real, values::AbstractVector{<:Number})
    N = length(values) - 1
    (N > 16 && iseven(N)) ||
        throw(ArgumentError("N = length(values) - 1 must be even and > 16, got $N"))
    b > a || throw(ArgumentError("b must be larger than a"))
    isfinite(omega) || throw(ArgumentError("omega must be finite"))
    return _dft_panel(F64(omega), F64(a), F64(b), Vector{C64}(values))
end

"""
    adft_func(f, omega::Real, a::Real, b::Real; n::Integer = 18,
              limit::Integer = 100)

Adaptive Fourier integral ``\\int^b_a dt\\ e^{i\\omega t} f(t)``. The
interval is split until `limit - 1` panels (for `limit ≥ 3`) are
reached, every time the panel with the largest error estimate at `omega`
is split. Each panel contains `n` subintervals. Since `dft_cplx()`
requires more than 16 subintervals, `n` must be even and ≥ 18 in
practice (`n = 16` passes the check of the sampler, as in NESSi, but is
rejected by the corrected DFT). NESSi has no default for `n`, the default
`n = 18` is the smallest executable value.

### Arguments
* f     -> Function, f(t) returns a number.
* omega -> Frequency ω.
* a     -> Lower bound.
* b     -> Upper bound.
* n     -> Number of subintervals of each panel.
* limit -> Budget of the panels.

### Returns
* res -> Fourier integral, a C64.
* err -> Error estimate, a C64.

*References* :

Please see [`NESSi`] `fourier.hpp`, `adft_func::sample` and
`adft_func::dft`.
"""
function adft_func(f, omega::Real, a::Real, b::Real; n::Integer = 18,
                   limit::Integer = 100)
    a < b || throw(ArgumentError("a must be smaller than b"))
    limit ≥ 1 || throw(ArgumentError("limit must be ≥ 1, got $limit"))
    isfinite(omega) || throw(ArgumentError("omega must be finite"))
    fz = x -> C64(f(x))
    P = _adft_sample(fz, F64(a), F64(b), I64(n), I64(limit);
                     w = F64(omega), nmin = 16)
    return _adft_eval(P, F64(omega))
end

#=
### *Public Functions* : *Matsubara Fourier Transform*
=#

"""
    get_tau(r::Integer, beta::Real, ntau_points::Integer)

Return the imaginary time ``\\tau_r = r\\beta/N``, where `N =
ntau_points - 1` is the number of imaginary time intervals.

### Arguments
* r           -> Index of the imaginary time (0-based).
* beta        -> Inverse temperature.
* ntau_points -> Number of imaginary time points.

### Returns
* τ -> Imaginary time.

*References* :

Please see [`NESSi`] `cntr_matsubara_impl.hpp`, `get_tau`.
"""
function get_tau(r::Integer, beta::Real, ntau_points::Integer)
    beta > 0 || throw(ArgumentError("beta must be positive"))
    ntau_points ≥ 2 || throw(ArgumentError("at least 2 imaginary time points are required"))
    return r * (F64(beta) / (ntau_points - 1))
end

"""
    get_omega(m::Integer, beta::Real, sign::Integer)

Return the Matsubara frequency ``\\omega_m = \\pi(2m + \\delta)/\\beta``.
Negative indices are allowed.

### Arguments
* m    -> Index of the Matsubara frequency.
* beta -> Inverse temperature.
* sign -> FERMI or BOSE.

### Returns
* ω -> Matsubara frequency.

*References* :

Please see [`NESSi`] `cntr_matsubara_impl.hpp`, `get_omega`.
"""
function get_omega(m::Integer, beta::Real, sign::Integer)
    beta > 0 || throw(ArgumentError("beta must be positive"))
    sign in (FERMI, BOSE) || throw(ArgumentError("sign must be FERMI or BOSE"))
    one = (sign == FERMI) ? 1 : 0
    return π / F64(beta) * (2 * m + one)
end

"""
    set_first_order_tail!(dest::AbstractArray{C64,3},
                          coeff::AbstractMatrix{C64}, beta::Real,
                          sign::Integer)

First order tail correction ``c_0(\\tau_r)``, such that ``G(\\tau) +
c_0(\\tau)`` is continuous. `dest[r+1,:,:]` is ``c_0(\\tau_r)``, and the
number of imaginary time points is `size(dest, 1)`. `coeff` is not
changed.

### Arguments
* dest  -> On return, the tail correction.
* coeff -> Jump coefficient ``a_0``.
* beta  -> Inverse temperature.
* sign  -> FERMI or BOSE.

### Returns
* dest -> The tail correction.

*References* :

Please see [`NESSi`] `cntr_matsubara_impl.hpp`, `set_first_order_tail`.
"""
function set_first_order_tail!(dest::AbstractArray{C64,3},
                               coeff::AbstractMatrix{C64}, beta::Real,
                               sign::Integer)
    beta > 0 || throw(ArgumentError("beta must be positive"))
    sign in (FERMI, BOSE) || throw(ArgumentError("sign must be FERMI or BOSE"))
    np, d₁, d₂ = size(dest)
    np ≥ 2 || throw(ArgumentError("at least 2 imaginary time points are required"))
    size(coeff) == (d₁, d₂) ||
        throw(DimensionMismatch("coeff must be a $d₁ × $d₂ matrix"))
    xmat = zeros(C64, d₁, d₂, np)
    _first_order_tail!(xmat, coeff, F64(beta), I64(sign))
    permutedims!(dest, xmat, (3, 1, 2))
    return dest
end

"""
    matsubara_dft!(dest::AbstractArray{C64,3}, G::Gᵐᵃᵗ{C64}, sign::Integer)

Raw discrete Fourier transform (without the factor δτ) of a Matsubara
function, ``\\sum^{N}_{r=0} e^{i\\pi(2m+\\delta)r/N} G(\\tau_r)`` for
`m = 0, ..., N`. `dest[m+1,:,:]` is the transform for the frequency index
`m`, so that `size(dest) = (N + 1, d₁, d₂)`, where `N + 1 = getntau(G)`.

### Arguments
* dest -> On return, the raw transform.
* G    -> Matsubara function.
* sign -> FERMI or BOSE.

### Returns
* dest -> The raw transform.

*References* :

Please see [`NESSi`] `cntr_matsubara_impl.hpp`, `matsubara_dft`.
"""
function matsubara_dft!(dest::AbstractArray{C64,3}, G::Gᵐᵃᵗ{C64}, sign::Integer)
    sign in (FERMI, BOSE) || throw(ArgumentError("sign must be FERMI or BOSE"))
    getntau(G) ≥ 2 || throw(ArgumentError("at least 2 imaginary time points are required"))
    size(dest) == (getntau(G), getdims(G)...) ||
        throw(DimensionMismatch("dest must have the size $((getntau(G), getdims(G)...))"))
    permutedims!(dest, _matsubara_dft(G, I64(sign)), (3, 1, 2))
    return dest
end

"""
    matsubara_ft!(dest::AbstractMatrix{C64}, m::Integer, G::Gᵐᵃᵗ{C64},
                  raw_dft::AbstractArray{C64,3}, sign::Integer, beta::Real;
                  order::Integer = 3)

Corrected Fourier transform ``\\int^{\\beta}_0 d\\tau\\
e^{i\\omega_m\\tau} G(\\tau)`` for any integer `m`. `raw_dft` is the
result of `matsubara_dft!()`, it is periodically mapped. The boundary
correction is of order 0 (none), 1 (linear), or 3 (cubic).

### Arguments
* dest    -> On return, the Fourier coefficient, a matrix.
* m       -> Index of the Matsubara frequency.
* G       -> Matsubara function.
* raw_dft -> Raw transform of `G`, see `matsubara_dft!()`.
* sign    -> FERMI or BOSE.
* beta    -> Inverse temperature.
* order   -> Order of the boundary correction.

### Returns
* dest -> The Fourier coefficient.

*References* :

Please see [`NESSi`] `cntr_matsubara_impl.hpp`, `matsubara_ft`.
"""
function matsubara_ft!(dest::AbstractMatrix{C64}, m::Integer, G::Gᵐᵃᵗ{C64},
                       raw_dft::AbstractArray{C64,3}, sign::Integer, beta::Real;
                       order::Integer = 3)
    order in (0, 1, 3) || throw(ArgumentError("order must be 0, 1, or 3, got $order"))
    sign in (FERMI, BOSE) || throw(ArgumentError("sign must be FERMI or BOSE"))
    beta > 0 || throw(ArgumentError("beta must be positive"))
    N = getntau(G) - 1
    N ≥ 2 * order + 1 || order == 0 ||
        throw(ArgumentError("too few imaginary time points for order $order"))
    size(raw_dft) == (N + 1, getdims(G)...) ||
        throw(DimensionMismatch("raw_dft must have the size $((N + 1, getdims(G)...))"))
    size(dest) == getdims(G) ||
        throw(DimensionMismatch("dest must have the size $(getdims(G))"))
    mdft = permutedims(raw_dft, (2, 3, 1))
    copyto!(dest, _matsubara_ft_direct(I64(m), G, mdft, I64(sign), F64(beta), I64(order)))
    return dest
end

#=
### *Public API* : *Hermitian Matsubara Component*
=#

"""
    force_matsubara_hermitian!(G::ℱ{T})

Force the Matsubara component of `G` to be hermitian, i.e.,
``G^{M}(\\tau) \\to [G^{M}(\\tau) + G^{M}(\\tau)^{\\dagger}] / 2``.

### Arguments
* G -> A contour-ordered Green's function.

### Returns
N/A
"""
function force_matsubara_hermitian!(G::ℱ{T}) where {T}
    # Sanity check
    @assert equaldims(G)

    for m = 1:getntau(G)
        M = _block(G.mat, m)
        M .= (M .+ copy(M')) ./ 2
    end
end

"""
    force_matsubara_hermitian!(G::𝒻{S})

Force the Matsubara component of `G` to be hermitian, i.e.,
``G^{M}(\\tau) \\to [G^{M}(\\tau) + G^{M}(\\tau)^{\\dagger}] / 2``.

### Arguments
* G -> A contour-ordered Green's function at given time step.

### Returns
N/A
"""
function force_matsubara_hermitian!(G::𝒻{S}) where {S}
    # Sanity check
    @assert equaldims(G)

    for m = 1:getntau(G)
        M = G.mat.data[m]
        M .= (M .+ copy(M')) ./ 2
    end
end

