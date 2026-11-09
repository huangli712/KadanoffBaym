#
# Project : Lavender
# Source  : matsubara_fourier_core.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/09
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
* order -> Order of the boundary correction, 0 or 3.

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
    for r = 0:N
        τ = r * δτ
        Z = view(xmat, :, :, r + 1)
        @. Z = -0.5 * qmasy
        if sign == BOSE
            @. Z = Z + (τ / β) * qmasy
        end
    end

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
