#
# Project : Lavender
# Source  : from_dos.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
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

"""
    _bethe_dos(V::F64, E0::F64)

Return the normalized semicircular density of states of the Bethe
lattice with the bandwidth 4V centered at E0, and its support. It is a
private function.
"""
function _bethe_dos(V::F64, E0::F64)
    V > 0.0 || throw(ArgumentError("V must be positive"))
    d = BetheDOS(E0 - 2.0 * V, E0 + 2.0 * V)
    return d, d.lo, d.hi
end

#=
### *Public API* : *Equilibrium Green's Functions From DOS*
=#

"""
    from_dos_mat!(G::ℱ{C64}, dos, lo::F64, hi::F64, β::F64;
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
function from_dos_mat!(G::ℱ{C64}, dos, lo::F64, hi::F64, β::F64;
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
        res, _ = _adft_eval(P, 0.0)
        _block(G.mat, m) .= res .* Matrix{C64}(I, d, d)
    end

    return G
end

"""
    from_dos!(G::ℱ{C64}, dos, lo::F64, hi::F64, β::F64, h::F64;
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
function from_dos!(G::ℱ{C64}, dos, lo::F64, hi::F64, β::F64, h::F64;
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
    from_dos_mat!(G, dos, lo, hi, β; μ = μ, limit = limit, nn = nn)

    # Retarded component, G^{R}(t,t') with l = t - t'
    fz = ε -> _dos_integrand(dos, ε, :ret, sign, μ, β, 0.0)
    P = _adft_sample(fz, lo, hi, nn, limit)
    for l = 0:ntime-1
        res, _ = _adft_eval(P, -h * l)
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
            res, _ = _adft_eval(P, -(n - 1) * h)
            _block(G.lmix, n, m) .= (-im * res) .* Id
        end
    end

    # Lesser component, G^{<}(t',t) with l = t - t'
    fz = ε -> _dos_integrand(dos, ε, :les, sign, μ, β, 0.0)
    P = _adft_sample(fz, lo, hi, nn, limit)
    for l = 0:ntime-1
        res, _ = _adft_eval(P, h * l)
        res *= -im
        for i = l+1:ntime
            _block(G.less, i - l, i) .= res .* Id
        end
    end

    return G
end

"""
    from_dos_mat_bethe!(G::ℱ{C64}, β::F64; V::F64 = 1.0,
                                 E0::F64 = 0.0, μ::F64 = 0.0,
                                 limit::I64 = 100, nn::I64 = 20)

Calculate the Matsubara component of the equilibrium Green's function of
the Bethe lattice, ``\\rho(\\epsilon) = \\sqrt{4V^2 - (\\epsilon - E_0)^2}
/ (2\\pi V^2)``.

See also: [`from_dos_mat!`](@ref).
"""
function from_dos_mat_bethe!(G::ℱ{C64}, β::F64; V::F64 = 1.0,
                                      E0::F64 = 0.0, μ::F64 = 0.0,
                                      limit::I64 = 100, nn::I64 = 20)
    ρ, lo, hi = _bethe_dos(V, E0)
    from_dos_mat!(G, ρ, lo, hi, β; μ = μ, limit = limit, nn = nn)
end

"""
    from_dos_bethe!(G::ℱ{C64}, β::F64, h::F64; V::F64 = 1.0,
                             E0::F64 = 0.0, μ::F64 = 0.0,
                             limit::I64 = 100, nn::I64 = 20)

Calculate all the components of the equilibrium Green's function of the
Bethe lattice, ``\\rho(\\epsilon) = \\sqrt{4V^2 - (\\epsilon - E_0)^2} /
(2\\pi V^2)``.

See also: [`from_dos!`](@ref).
"""
function from_dos_bethe!(G::ℱ{C64}, β::F64, h::F64; V::F64 = 1.0,
                                  E0::F64 = 0.0, μ::F64 = 0.0,
                                  limit::I64 = 100, nn::I64 = 20)
    ρ, lo, hi = _bethe_dos(V, E0)
    from_dos!(G, ρ, lo, hi, β, h; μ = μ, limit = limit, nn = nn)
end
