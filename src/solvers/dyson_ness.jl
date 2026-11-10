#
# Project : Lavender
# Source  : dyson_ness.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    DysonBathType

Type of the bath regularization of the steady-state Dyson equation,
`BATH_NONE`, `BATH_CONST`, `BATH_GAUSS`, or `BATH_OHMIC`.
"""
@enum DysonBathType BATH_NONE BATH_CONST BATH_GAUSS BATH_OHMIC

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

