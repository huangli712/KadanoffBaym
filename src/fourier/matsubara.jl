#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

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
    return _adaptive_dos_integral(P, F64(omega))
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


