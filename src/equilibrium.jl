#
# Project : Lavender
# Source  : equilibrium.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/09
#

#=
*Remarks* : *Equilibrium Green's Functions From DOS*

For a density of states ``\rho(\epsilon)`` with support ``[lo, hi]``,
``x = \epsilon - \mu``, and the statistics ``\xi = -1`` (``+1``) for
fermions (bosons), the equilibrium Green's function reads

```math
\begin{equation}
G^{R}(t,t') = -i \int d\epsilon\ \rho(\epsilon) e^{-i\epsilon(t-t')},
\end{equation}
```

```math
\begin{equation}
G^{<}(t,t') = -i\xi \int d\epsilon\ \rho(\epsilon) n_{\xi}(x)
    e^{-i\epsilon(t-t')},
\end{equation}
```

```math
\begin{equation}
G^{\rceil}(t,\tau) = -i\xi \int d\epsilon\ \rho(\epsilon)
    e^{\tau x} n_{\xi}(x) e^{-i\epsilon t}.
\end{equation}
```

The Matsubara component is evaluated by the overflow-safe expressions
``\xi \rho(\epsilon) e^{-\tau x} n_{\xi}(-x)`` (``x > 0``) and
``-\rho(\epsilon) e^{(\beta-\tau) x} n_{\xi}(x)`` (``x \le 0``). For
matrix-valued functions, the result is ``G = g\ I_d``.

The Fourier integrals are evaluated by the adaptive, cubically corrected
discrete Fourier transform. The integration interval is divided into
panels of `nn` subintervals. The panels are determined at ω = 0: the
first panel is always bisected (if `limit > 1`), then the panel with the
largest error estimate is bisected, until there are `limit - 1` panels.
There is no tolerance and no early termination. The error estimate is
given by the difference between the results with `nn` and `nn/2`
subintervals.

*References* :

Please see [`NESSi`] `cntr_equilibrium_impl.hpp`, `green_equilibrium`,
and `fourier.hpp`, `adft_func`.
=#

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
    _adaptive_dos_integral(P::_AdftPanels, w::F64)

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
function _adaptive_dos_integral(P::_AdftPanels, w::F64)
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

#=
### *Private Service Functions* : *DOS Wrapper*
=#

"""
    _dos_integrand(dos, ε::F64, comp::Symbol, sign::I64, μ::F64, β::F64, τ::F64)

Integrand of the equilibrium Green's function for the component `comp`
(`:ret`, `:les`, `:tv`, or `:mat`), see *Remarks* : *Equilibrium Green's
Functions From DOS*. The DOS must be finite and nonnegative, and the
bosonic zero mode is rejected. It is a private function.

*References* :

Please see [`NESSi`] `cntr_equilibrium_impl.hpp`, `dos_wrapper`.
"""
function _dos_integrand(dos, ε::F64, comp::Symbol, sign::I64, μ::F64, β::F64, τ::F64)
    a = dos(ε)
    (a isa Real && isfinite(a)) || throw(DomainError(a, "DOS is not finite at ε = $ε"))
    a ≥ 0.0 || throw(DomainError(a, "DOS is negative at ε = $ε"))
    x = ε - μ
    if sign == BOSE && comp != :ret && iszero(x)
        throw(DomainError(ε, "zero mode of the Bose function at ε = $ε"))
    end
    nξ(y) = sign == FERMI ? fermi(β, y) : bose(β, y)

    if comp == :ret
        fm = 1.0
    elseif comp == :les
        fm = sign * nξ(x)
    elseif comp == :tv
        fm = sign * (sign == FERMI ? fermi_exp(β, τ, x) : bose_exp(β, τ, x))
    else # :mat
        if x > 0.0
            fm = sign * exp(-τ * x) * nξ(-x)
        else
            fm = -exp((β - τ) * x) * nξ(x)
        end
    end

    return complex(a * fm)
end

"""
    _check_equilibrium(G::ℱ{C64}, β::F64, limit::I64, nn::I64)

Check the arguments of the equilibrium Green's functions. It is a
private function.
"""
function _check_equilibrium(G::ℱ{C64}, β::F64, limit::I64, nn::I64)
    getsign(G) in (FERMI, BOSE) || throw(ArgumentError("unknown statistics of G"))
    equaldims(G) || throw(DimensionMismatch("G must be square"))
    β > 0.0 || throw(ArgumentError("β must be positive"))
    (iseven(nn) && nn ≥ 20) || throw(ArgumentError("nn must be even and ≥ 20, got $nn"))
    limit ≥ 1 || throw(ArgumentError("limit must be ≥ 1, got $limit"))
end

#=
### *Public API* : *Equilibrium Green's Functions From DOS*
=#

"""
    green_equilibrium_mat!(G::ℱ{C64}, dos, lo::F64, hi::F64, β::F64;
                           μ::F64 = 0.0, limit::I64 = 100, nn::I64 = 20)

Calculate the Matsubara component of the equilibrium Green's function
for the density of states `dos` (a callable object, ε → ρ(ε)) with the
support `[lo, hi]`. See *Remarks* : *Equilibrium Green's Functions From
DOS*.

### Arguments
* G     -> Contour-ordered Green's function.
* dos   -> Density of states.
* lo    -> Lower bound of the support.
* hi    -> Upper bound of the support.
* β     -> Inverse temperature.
* μ     -> Chemical potential.
* limit -> Budget of the panels of the adaptive Fourier integral.
* nn    -> Number of subintervals of each panel.

### Returns
* G -> Contour-ordered Green's function.
"""
function green_equilibrium_mat!(G::ℱ{C64}, dos, lo::F64, hi::F64, β::F64;
                                μ::F64 = 0.0, limit::I64 = 100, nn::I64 = 20)
    _check_equilibrium(G, β, limit, nn)
    sign = getsign(G)
    ntau = getntau(G)
    d = getdims(G)[1]
    δτ = β / (ntau - 1)

    for m = 1:ntau
        τ = (m - 1) * δτ
        fz = ε -> _dos_integrand(dos, ε, :mat, sign, μ, β, τ)
        P = _adft_sample(fz, lo, hi, nn, limit)
        res, _ = _adaptive_dos_integral(P, 0.0)
        _block(G.mat, m) .= res .* Matrix{C64}(I, d, d)
    end

    return G
end

"""
    green_equilibrium!(G::ℱ{C64}, dos, lo::F64, hi::F64, β::F64, h::F64;
                       μ::F64 = 0.0, limit::I64 = 100, nn::I64 = 20)

Calculate all the components of the equilibrium Green's function for
the density of states `dos` (a callable object, ε → ρ(ε)) with the
support `[lo, hi]`. See *Remarks* : *Equilibrium Green's Functions From
DOS*.

### Arguments
* G     -> Contour-ordered Green's function.
* dos   -> Density of states.
* lo    -> Lower bound of the support.
* hi    -> Upper bound of the support.
* β     -> Inverse temperature.
* h     -> Time step.
* μ     -> Chemical potential.
* limit -> Budget of the panels of the adaptive Fourier integral.
* nn    -> Number of subintervals of each panel.

### Returns
* G -> Contour-ordered Green's function.
"""
function green_equilibrium!(G::ℱ{C64}, dos, lo::F64, hi::F64, β::F64, h::F64;
                            μ::F64 = 0.0, limit::I64 = 100, nn::I64 = 20)
    _check_equilibrium(G, β, limit, nn)
    h > 0.0 || throw(ArgumentError("h must be positive"))
    sign = getsign(G)
    ntime = getntime(G)
    ntau = getntau(G)
    d = getdims(G)[1]
    Id = Matrix{C64}(I, d, d)
    δτ = β / (ntau - 1)

    # Matsubara component
    green_equilibrium_mat!(G, dos, lo, hi, β; μ = μ, limit = limit, nn = nn)

    # Retarded component, G^{R}(t,t') with l = t - t'
    fz = ε -> _dos_integrand(dos, ε, :ret, sign, μ, β, 0.0)
    P = _adft_sample(fz, lo, hi, nn, limit)
    for l = 0:ntime-1
        res, _ = _adaptive_dos_integral(P, -h * l)
        res *= -im
        for i = l+1:ntime
            _block(G.ret, i, i - l) .= res .* Id
        end
    end

    # Left-mixing component
    for m = 1:ntau
        τ = (m - 1) * δτ
        fz = ε -> _dos_integrand(dos, ε, :tv, sign, μ, β, τ)
        P = _adft_sample(fz, lo, hi, nn, limit)
        for n = 1:ntime
            res, _ = _adaptive_dos_integral(P, -(n - 1) * h)
            _block(G.lmix, n, m) .= (-im * res) .* Id
        end
    end

    # Lesser component, G^{<}(t',t) with l = t - t'
    fz = ε -> _dos_integrand(dos, ε, :les, sign, μ, β, 0.0)
    P = _adft_sample(fz, lo, hi, nn, limit)
    for l = 0:ntime-1
        res, _ = _adaptive_dos_integral(P, h * l)
        res *= -im
        for i = l+1:ntime
            _block(G.less, i - l, i) .= res .* Id
        end
    end

    return G
end

"""
    _bethe_dos(V::F64, E0::F64)

Return the normalized semicircular density of states of the Bethe
lattice with the bandwidth 4V centered at E0, and its support. It is a
private function.
"""
function _bethe_dos(V::F64, E0::F64)
    V > 0.0 || throw(ArgumentError("V must be positive"))
    ρ(x) = (arg = 4.0 * V * V - (x - E0)^2; arg < 0.0 ? 0.0 : sqrt(arg) / (2π * V * V))
    return ρ, E0 - 2.0 * V, E0 + 2.0 * V
end

"""
    green_equilibrium_mat_bethe!(G::ℱ{C64}, β::F64; V::F64 = 1.0,
                                 E0::F64 = 0.0, μ::F64 = 0.0,
                                 limit::I64 = 100, nn::I64 = 20)

Calculate the Matsubara component of the equilibrium Green's function of
the Bethe lattice, ``\\rho(\\epsilon) = \\sqrt{4V^2 - (\\epsilon - E_0)^2}
/ (2\\pi V^2)``.

See also: [`green_equilibrium_mat!`](@ref).
"""
function green_equilibrium_mat_bethe!(G::ℱ{C64}, β::F64; V::F64 = 1.0,
                                      E0::F64 = 0.0, μ::F64 = 0.0,
                                      limit::I64 = 100, nn::I64 = 20)
    ρ, lo, hi = _bethe_dos(V, E0)
    green_equilibrium_mat!(G, ρ, lo, hi, β; μ = μ, limit = limit, nn = nn)
end

"""
    green_equilibrium_bethe!(G::ℱ{C64}, β::F64, h::F64; V::F64 = 1.0,
                             E0::F64 = 0.0, μ::F64 = 0.0,
                             limit::I64 = 100, nn::I64 = 20)

Calculate all the components of the equilibrium Green's function of the
Bethe lattice, ``\\rho(\\epsilon) = \\sqrt{4V^2 - (\\epsilon - E_0)^2} /
(2\\pi V^2)``.

See also: [`green_equilibrium!`](@ref).
"""
function green_equilibrium_bethe!(G::ℱ{C64}, β::F64, h::F64; V::F64 = 1.0,
                                  E0::F64 = 0.0, μ::F64 = 0.0,
                                  limit::I64 = 100, nn::I64 = 20)
    ρ, lo, hi = _bethe_dos(V, E0)
    green_equilibrium!(G, ρ, lo, hi, β, h; μ = μ, limit = limit, nn = nn)
end

#=
*Remarks* : *Free Green's Functions From Hamiltonian*

Let ``K = \mu - H(-i\beta) = R\ \mathrm{diag}(\lambda) R^{\dagger}``, and
``U_n`` be the propagator from 0 to ``t_n``, including the phase
``e^{i\mu t_n}``. For fermions,

```math
\begin{equation}
G^{M}(\tau) = -R\ \mathrm{diag}[e^{\tau\lambda}n_F(\lambda)] R^{\dagger},
\quad
G^{\rceil}(t_n,\tau) = iU_n R\ \mathrm{diag}[e^{-\tau\lambda}n_F(-\lambda)]
R^{\dagger},
\end{equation}
```

```math
\begin{equation}
G^{R}(t_n,t_j) = -iU_n U^{\dagger}_j,
\quad
G^{<}(t_j,t_n) = iU_j N U^{\dagger}_n,
\quad
N = R\ \mathrm{diag}[n_F(-\lambda)] R^{\dagger}.
\end{equation}
```

For bosons, ``G^{M}`` and ``N`` change sign, ``n_F`` is replaced by
``n_B``, and ``G^{\rceil}`` is multiplied by -1.

For a constant Hamiltonian, ``U_n = e^{ihnK}``. For a time-dependent
Hamiltonian, the commutator-free expansions of order 2 and 4 are used,

```math
\begin{equation}
U_n = e^{-ih H(t_{n-1/2})} U_{n-1}, \quad
U_n = e^{-ih(a_1 H_1 + a_2 H_2)} e^{-ih(a_2 H_1 + a_1 H_2)} U_{n-1},
\end{equation}
```

where ``H_{1,2} = H(t_{n-1} + c_{1,2} h)``, ``c_{1,2} = 1/2 \mp
\sqrt{3}/6``, ``a_{1,2} = (3 \mp 2\sqrt{3})/12``. The Hamiltonian at the
fractional time points is obtained by polynomial interpolation of order
`solve_order`. Finally ``U_n \to U_n e^{i\mu t_n}``.

*References* :

Please see [`NESSi`] `cntr_equilibrium_impl.hpp`, `green_from_H`,
`propagator_exp`.
=#

#=
### *Private Service Functions* : *Free Green's Functions*
=#

"""
    _hamiltonian_cf(nkb::I64, H::Cf{C64}, kt::I64, fix::Bool)

Prepare the Hamiltonian for the propagation from the KB time step
`nkb - 1` to `nkb`. If `fix = false`, `H[nkb]` is extrapolated from the
previous time steps when `nkb - 1 > kt`. It returns the KB index of the
last grid point of the interpolation interval. It is a private function.
"""
function _hamiltonian_cf(nkb::I64, H::Cf{C64}, kt::I64, fix::Bool)
    tstp = nkb - 1 # NESSi index
    if !fix && tstp > kt
        extrapolate_timestep!(nkb - 1, H, Integrator(kt))
    end
    n₁ = tstp ≤ kt ? kt : tstp
    return n₁ + 1
end

"""
    _propagator_cf2!(U::Vector{Matrix{C64}}, nkb::I64, H::Cf{C64},
                     h::F64, kt::I64, fix::Bool)

Second order commutator-free propagation, ``U_n = e^{-ihH(t_{n-1/2})}
U_{n-1}``, where `n = nkb - 1`. It is a private function.

*References* :

Please see [`NESSi`] `cntr_equilibrium_impl.hpp`, `propagator_exp`.
"""
function _propagator_cf2!(U::Vector{Matrix{C64}}, nkb::I64, H::Cf{C64},
                          h::F64, kt::I64, fix::Bool)
    n₁ = _hamiltonian_cf(nkb, H, kt, fix)
    Hm = interpolation(n₁, nkb - 0.5, H, Integrator(kt))
    U[nkb] = exp((-im * h) .* Hm) * U[nkb-1]
    return U
end

"""
    _propagator_cf4!(U::Vector{Matrix{C64}}, nkb::I64, H::Cf{C64},
                     h::F64, kt::I64, fix::Bool)

Fourth order commutator-free propagation, ``U_n = e^{-ih(a_1 H_1 + a_2
H_2)} e^{-ih(a_2 H_1 + a_1 H_2)} U_{n-1}``, where `n = nkb - 1`. It is a
private function.

*References* :

Please see [`NESSi`] `cntr_equilibrium_impl.hpp`, `propagator_exp`.
"""
function _propagator_cf4!(U::Vector{Matrix{C64}}, nkb::I64, H::Cf{C64},
                          h::F64, kt::I64, fix::Bool)
    n₁ = _hamiltonian_cf(nkb, H, kt, fix)
    H₁ = interpolation(n₁, nkb - 0.5 - sqrt(3) / 6.0, H, Integrator(kt))
    H₂ = interpolation(n₁, nkb - 0.5 + sqrt(3) / 6.0, H, Integrator(kt))
    a₁ = (3.0 - 2.0 * sqrt(3.0)) / 12.0
    a₂ = (3.0 + 2.0 * sqrt(3.0)) / 12.0
    arg₁ = (-im * h) .* (a₁ .* H₁ .+ a₂ .* H₂)
    arg₂ = (-im * h) .* (a₂ .* H₁ .+ a₁ .* H₂)
    U[nkb] = exp(arg₁) * exp(arg₂) * U[nkb-1]
    return U
end

"""
    _fill_green_from_propagator!(G::ℱ{C64}, μ::F64, H0::AbstractMatrix{C64},
                                 β::F64, U::Vector{Matrix{C64}})

Fill all the components of the free Green's function `G` from the
Hamiltonian `H0` at the Matsubara axis and the propagators `U[n]`
(including the phase ``e^{i\\mu t}``). See *Remarks* : *Free Green's
Functions From Hamiltonian*. It is a private function.
"""
function _fill_green_from_propagator!(G::ℱ{C64}, μ::F64, H0::AbstractMatrix{C64},
                                      β::F64, U::Vector{Matrix{C64}})
    sign = getsign(G)
    ntime = getntime(G)
    ntau = getntau(G)
    d = getdims(G)[1]
    δτ = β / (ntau - 1)

    K = Hermitian(μ * Matrix{C64}(I, d, d) - H0)
    λ, R = eigen(K)

    # Matsubara and left-mixing components
    for m = 1:ntau
        τ = (m - 1) * δτ
        if sign == FERMI
            _block(G.mat, m) .= -R * Diagonal(fermi_exp(β, τ, λ)) * R'
            X = im * R * Diagonal(fermi_exp(β, τ, -λ)) * R'
        else
            _block(G.mat, m) .= R * Diagonal(bose_exp(β, τ, λ)) * R'
            X = -im * R * Diagonal(bose_exp(β, τ, -λ)) * R'
        end
        for n = 1:ntime
            mul!(_block(G.lmix, n, m), U[n], X)
        end
    end

    # Retarded and lesser components
    if sign == FERMI
        N = R * Diagonal(fermi(β, -λ)) * R'
    else
        N = -R * Diagonal(bose(β, -λ)) * R'
    end
    for m = 1:ntime, n = 1:m
        _block(G.ret, m, n) .= -im .* (U[m] * U[n]')
        _block(G.less, n, m) .= im .* (U[n] * N * U[m]')
    end

    return G
end

"""
    _check_free_hamiltonian(G::ℱ{C64}, H0::AbstractMatrix{C64})

Check the statistics and the shapes for the free Green's functions. It
is a private function.
"""
function _check_free_hamiltonian(G::ℱ{C64}, H0::AbstractMatrix{C64})
    getsign(G) in (FERMI, BOSE) || throw(ArgumentError("unknown statistics of G"))
    d₁, d₂ = getdims(G)
    d₁ == d₂ || throw(DimensionMismatch("G must be square"))
    size(H0) == (d₁, d₂) || throw(DimensionMismatch("H must be $d₁ × $d₂"))
    all(isfinite, H0) || throw(DomainError(H0, "H is not finite"))
    if norm(H0 - H0') > 1e-12 * max(1.0, norm(H0))
        throw(ArgumentError("H is not hermitian"))
    end
end

#=
### *Public API* : *Free Green's Functions From Hamiltonian*
=#

"""
    green_from_H!(G::ℱ{C64}, μ::F64, H0::Matrix{C64}, β::F64, h::F64)

Calculate the free Green's function of the constant hermitian
Hamiltonian `H0`. See *Remarks* : *Free Green's Functions From
Hamiltonian*.

### Arguments
* G  -> Contour-ordered Green's function.
* μ  -> Chemical potential.
* H0 -> Hamiltonian.
* β  -> Inverse temperature.
* h  -> Time step.

### Returns
* G -> Contour-ordered Green's function.
"""
function green_from_H!(G::ℱ{C64}, μ::F64, H0::Matrix{C64}, β::F64, h::F64)
    _check_free_hamiltonian(G, H0)
    β > 0.0 || throw(ArgumentError("β must be positive"))
    d = getdims(G)[1]
    ntime = getntime(G)

    # U_n = exp(i h (μ - H0))^n
    Udt = exp((im * h) .* (μ * Matrix{C64}(I, d, d) - H0))
    U = Vector{Matrix{C64}}(undef, ntime)
    U[1] = Matrix{C64}(I, d, d)
    for n = 2:ntime
        U[n] = U[n-1] * Udt
    end

    _fill_green_from_propagator!(G, μ, H0, β, U)
end

"""
    green_from_H!(G::ℱ{C64}, μ::F64, H::Cf{C64}, β::F64, h::F64;
                  fix_hamiltonian::Bool = true, solve_order::I64 = 5,
                  cf_order::I64 = 4)

Calculate the free Green's function of the time-dependent hermitian
Hamiltonian `H`. The Matsubara component is determined by `H[0]`. The
propagators are obtained by the commutator-free expansion of order
`cf_order` (2 or 4), and the Hamiltonian at the fractional time points
is interpolated with the order `solve_order`. If `fix_hamiltonian =
false`, the values of `H` at `t > solve_order` are extrapolated from the
previous time steps and written into `H`.

### Arguments
* G               -> Contour-ordered Green's function.
* μ               -> Chemical potential.
* H               -> Hamiltonian.
* β               -> Inverse temperature.
* h               -> Time step.
* fix_hamiltonian -> Whether `H` is given at all the time steps.
* solve_order     -> Order of the interpolation, 1 ≤ solve_order ≤ 5.
* cf_order        -> Order of the commutator-free expansion, 2 or 4.

### Returns
* G -> Contour-ordered Green's function.
"""
function green_from_H!(G::ℱ{C64}, μ::F64, H::Cf{C64}, β::F64, h::F64;
                       fix_hamiltonian::Bool = true, solve_order::I64 = 5,
                       cf_order::I64 = 4)
    # Sanity check
    solve_order in 1:5 ||
        throw(ArgumentError("solve_order must be in 1:5, got $solve_order"))
    cf_order in (2, 4) || throw(ArgumentError("cf_order must be 2 or 4, got $cf_order"))
    β > 0.0 || throw(ArgumentError("β must be positive"))
    h > 0.0 || throw(ArgumentError("h must be positive"))
    _check_free_hamiltonian(G, H[0])
    d = getdims(G)[1]
    ntime = getntime(G)
    kt = solve_order
    #
    # The Hamiltonian is needed at the KB time steps 1, ⋯, max(kt, nt) + 1
    # (fix_hamiltonian = true), or 1, ⋯, kt + 1 (fix_hamiltonian = false).
    nmax = fix_hamiltonian ? max(kt + 1, ntime) : kt + 1
    H.ntime ≥ max(nmax, ntime) ||
        throw(ArgumentError("H must have at least $(max(nmax, ntime)) real time steps"))
    _check_hamiltonian(H, 1:nmax, d)

    # Propagators without the phase
    U = Vector{Matrix{C64}}(undef, ntime)
    U[1] = Matrix{C64}(I, d, d)
    for nkb = 2:ntime
        if cf_order == 2
            _propagator_cf2!(U, nkb, H, h, kt, fix_hamiltonian)
        else
            _propagator_cf4!(U, nkb, H, h, kt, fix_hamiltonian)
        end
    end

    # Phase exp(i μ t)
    for nkb = 1:ntime
        U[nkb] = U[nkb] * cis(μ * h * (nkb - 1))
    end

    _fill_green_from_propagator!(G, μ, H[0], β, U)
end

#=
*Remarks* : *Bosonic Single Pole*

For the displacement ``X = (b + b^{\dagger})/\sqrt{2}`` of a harmonic
oscillator with frequency ``w``, ``n_B = n_B(w)``,

```math
\begin{equation}
D^{M}(\tau) = -\frac{1}{2}[e^{\tau w} n_B(w) - e^{-\tau w} n_B(-w)],
\end{equation}
```

```math
\begin{equation}
D^{<}(t',t) = -\frac{i}{2}[e^{-iw(t-t')} + 2n_B\cos w(t-t')],
\quad
D^{R}(t,t') = -\sin w(t-t'),
\end{equation}
```

```math
\begin{equation}
D^{\rceil}(t,\tau) = -\frac{i}{2}\{e^{iwt}e^{-w\tau}
    + n_B[e^{iwt}e^{-w\tau} + e^{-iwt}e^{w\tau}]\}.
\end{equation}
```

Only the (1,1) element is written.

*References* :

Please see [`NESSi`] `cntr_equilibrium_impl.hpp`, `green_single_pole_XX`.
=#

#=
### *Public API* : *Bosonic Single Pole*
=#

"""
    _check_single_pole(D0::ℱ{C64}, w::F64, β::F64)

Check the arguments of the bosonic single pole Green's function. It is a
private function.
"""
function _check_single_pole(D0::ℱ{C64}, w::F64, β::F64)
    getsign(D0) == BOSE || throw(ArgumentError("D0 must be bosonic"))
    w * β > 0.0 || throw(ArgumentError("w β must be positive"))
end

"""
    green_single_pole_XX_mat!(D0::ℱ{C64}, w::F64, β::F64)

Calculate the Matsubara component of the bosonic single pole Green's
function, see *Remarks* : *Bosonic Single Pole*.

### Arguments
* D0 -> Contour-ordered Green's function (bosonic).
* w  -> Frequency.
* β  -> Inverse temperature.

### Returns
* D0 -> Contour-ordered Green's function.
"""
function green_single_pole_XX_mat!(D0::ℱ{C64}, w::F64, β::F64)
    _check_single_pole(D0, w, β)
    ntau = getntau(D0)
    δτ = β / (ntau - 1)
    for v = 1:ntau
        τ = (v - 1) * δτ
        ewv = bose_exp(β, τ, w)
        emwv = -bose_exp(β, τ, -w)
        _block(D0.mat, v)[1,1] = -0.5 * (ewv + emwv)
    end
    return D0
end

"""
    green_single_pole_XX_timestep!(nkb::I64, D0::ℱ{C64}, w::F64, β::F64, h::F64)

Calculate the real time components of the bosonic single pole Green's
function at the time step `nkb`, see *Remarks* : *Bosonic Single Pole*.

### Arguments
* nkb -> Index of the time step.
* D0  -> Contour-ordered Green's function (bosonic).
* w   -> Frequency.
* β   -> Inverse temperature.
* h   -> Time step.

### Returns
* D0 -> Contour-ordered Green's function.
"""
function green_single_pole_XX_timestep!(nkb::I64, D0::ℱ{C64}, w::F64, β::F64, h::F64)
    _check_single_pole(D0, w, β)
    1 ≤ nkb ≤ getntime(D0) || throw(ArgumentError("wrong time step $nkb"))
    tstp = nkb - 1
    ntau = getntau(D0)
    δτ = β / (ntau - 1)
    bw = bose(β, w)

    for tp = 0:tstp
        c = cos(w * (tstp - tp) * h)
        s = sin(w * (tstp - tp) * h)
        d_les = -im * ((c - im * s) + 2.0 * c * bw)
        d_gtr = -im * ((c - im * s) + 2.0 * c * bw)
        d_ret = d_gtr + conj(d_les)
        _block(D0.less, tp + 1, nkb)[1,1] = 0.5 * d_les
        _block(D0.ret, nkb, tp + 1)[1,1] = 0.5 * d_ret
    end

    c = cos(w * tstp * h)
    s = sin(w * tstp * h)
    for v = 0:ntau-1
        ewv = exp(w * v * δτ)
        emwv = exp(-w * v * δτ)
        d_tv = -im * ((c + im * s) * emwv + ((c + im * s) * emwv + (c - im * s) * ewv) * bw)
        _block(D0.lmix, nkb, v + 1)[1,1] = 0.5 * d_tv
    end

    return D0
end

"""
    green_single_pole_XX!(D0::ℱ{C64}, w::F64, β::F64, h::F64)

Calculate all the components of the bosonic single pole Green's
function, see *Remarks* : *Bosonic Single Pole*.

### Arguments
* D0 -> Contour-ordered Green's function (bosonic).
* w  -> Frequency.
* β  -> Inverse temperature.
* h  -> Time step.

### Returns
* D0 -> Contour-ordered Green's function.
"""
function green_single_pole_XX!(D0::ℱ{C64}, w::F64, β::F64, h::F64)
    green_single_pole_XX_mat!(D0, w, β)
    for nkb = 1:getntime(D0)
        green_single_pole_XX_timestep!(nkb, D0, w, β, h)
    end
    return D0
end
