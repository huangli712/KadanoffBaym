#
# Project : Lavender
# Source  : vide.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/09
#

#=
*Remarks* : *Dyson Equation*

The Dyson equation for the contour-ordered Green's function reads

```math
\begin{equation}
[i\partial_t + \mu - H(t)] G(t,t') - [\Sigma * G](t,t')
    = \delta_{\mathcal{C}}(t,t').
\end{equation}
```

It is a Volterra integro-differential equation (VIDE). It is solved in
three steps:

1. Matsubara component, `dyson_mat!()`.

2. Start-up of the real time components at the first 𝑘 + 1 time steps,
   `dyson_start!()`.

3. Time stepping of the real time components, `dyson_timestep!()`.

The full solution is obtained by `dyson!()`.

*References* :

Please see [`NESSi`] Section `7` and `cntr_dyson_impl.hpp`.
=#

#=
*Remarks* : *Matsubara Dyson Equation*

Let ``G_0`` be the free Matsubara Green's function of ``H(-i\beta)``.
The Matsubara Dyson equation is written as

```math
\begin{equation}
[1 + F] * G = Q, \quad F = -G_0 * \Sigma, \quad Q = G_0.
\end{equation}
```

Two methods are supported:

* `:fixpoint` -> Start from the frequency-domain solution (`pcf = 5`),
    then apply the preconditioned residual corrections ``R = F*G + G - Q``,
    ``G \to G - P^{-1}R`` for `maxiter` (default 6) times.

* `:cg` -> Start from the frequency-domain solution (`pcf = 3`), then
    apply the Fletcher-Reeves conjugate gradient method to the hermitian
    operator ``A(X) = X + [F*X + X*F^{cc}]/2``, where ``F^{cc} = -\Sigma
    * G_0``. The inner product is ``\langle X,Y \rangle = \sum_r w_r
    \sum_{ab} \mathrm{Re}[(X^{\dagger}_r Y_r)_{ab}]``, where ``w_r`` are
    the Gregory end weights.

The method `:fourier` solves the equation in the frequency domain with
`pcf = 10` oversampling copies and the cubically corrected transform.

*References* :

Please see [`NESSi`] `cntr_dyson_impl.hpp`, `dyson_mat_fixpoint_dispatch`,
`dyson_mat_steep_dispatch`, and `cntr_vie2_impl.hpp`, `vie2_mat_steep_dispatch`.
=#

#=
### *Private Service Functions* : *Sanity Checks*
=#

"""
    _check_hamiltonian(H::Cf{C64}, tstps, d::I64)

Check that `H` is a finite and hermitian 𝑑 × 𝑑 matrix at all the time
steps in `tstps` (0 for the Matsubara axis). It is a private function.
"""
function _check_hamiltonian(H::Cf{C64}, tstps, d::I64)
    getdims(H) == (d, d) || throw(DimensionMismatch("H must be $d × $d"))
    for n in tstps
        0 ≤ n ≤ H.ntime || throw(ArgumentError("H is not defined at time step $n"))
        Hₙ = H[n]
        all(isfinite, Hₙ) || throw(DomainError(n, "H is not finite at time step $n"))
        # ‖H - H†‖ ≤ tol max(1, ‖H‖), evaluated without temporary arrays
        r = 0.0
        for b = 1:d, a = 1:d
            r += abs2(Hₙ[a,b] - conj(Hₙ[b,a]))
        end
        if sqrt(r) > 1e-12 * max(1.0, norm(Hₙ))
            throw(ArgumentError("H is not hermitian at time step $n"))
        end
    end
end

"""
    _check_dyson(G::ℱ{C64}, Σ::ℱ{C64}, H::Cf{C64}, solve_order::I64)

Check the compatibility of the arguments of the Dyson solvers. It is a
private function.
"""
function _check_dyson(G::ℱ{C64}, Σ::ℱ{C64}, H::Cf{C64}, solve_order::I64)
    solve_order in 1:5 ||
        throw(ArgumentError("solve_order must be in 1:5, got $solve_order"))
    getsign(G) in (FERMI, BOSE) || throw(ArgumentError("unknown statistics of G"))
    getsign(G) == getsign(Σ) ||
        throw(ArgumentError("G and Σ must have the same statistics"))
    d₁, d₂ = getdims(G)
    d₁ == d₂ || throw(DimensionMismatch("G must be square"))
    getdims(Σ) == (d₁, d₂) || throw(DimensionMismatch("G and Σ have different sizes"))
    getntau(G) == getntau(Σ) ||
        throw(DimensionMismatch("G and Σ have different imaginary time grids"))
    getntime(G) == getntime(Σ) ||
        throw(DimensionMismatch("G and Σ have different real time grids"))
    getdims(H) == (d₁, d₂) || throw(DimensionMismatch("H must be $d₁ × $d₂"))
end

#=
### *Private Service Functions* : *Matsubara Dyson*
=#

"""
    _free_matsubara_from_H!(G0::Gᵐᵃᵗ{C64}, μ::F64, H::Cf{C64}, β::F64,
                            sign::I64)

Calculate the free Matsubara Green's function of the hermitian matrix
`H[0]` (Hamiltonian at the Matsubara axis), i.e.,
``G_0(\\tau) = -R\\ \\mathrm{diag}[e^{\\tau\\lambda}n_F(\\lambda)]R^{\\dagger}``
for fermions and
``G_0(\\tau) = R\\ \\mathrm{diag}[e^{\\tau\\lambda}n_B(\\lambda)]R^{\\dagger}``
for bosons, where ``\\mu - H = R\\ \\mathrm{diag}(\\lambda)R^{\\dagger}``.
It is a private function.

### Arguments
* G0   -> Matsubara Green's function.
* μ    -> Chemical potential.
* H    -> Hamiltonian.
* β    -> Inverse temperature.
* sign -> FERMI or BOSE.

### Returns
* G0 -> Matsubara Green's function.
"""
function _free_matsubara_from_H!(G0::Gᵐᵃᵗ{C64}, μ::F64, H::Cf{C64}, β::F64,
                                 sign::I64)
    ntau = getntau(G0)
    d = getdims(G0)[1]
    _check_hamiltonian(H, 0, d)

    K = Hermitian(μ * Matrix{C64}(I, d, d) - H[0])
    λ, R = eigen(K)
    δτ = β / (ntau - 1)
    for m = 1:ntau
        τ = (m - 1) * δτ
        if sign == FERMI
            _block(G0, m) .= -R * Diagonal(fermi_exp(β, τ, λ)) * R'
        else
            _block(G0, m) .= R * Diagonal(bose_exp(β, τ, λ)) * R'
        end
    end

    return G0
end

"""
    _mat_inner(X::Gᵐᵃᵗ{C64}, Y::Gᵐᵃᵗ{C64}, I::Integrator)

Inner product of two Matsubara functions,
``\\sum_r w_r \\sum_{ab} \\mathrm{Re}[(X^{\\dagger}_r Y_r)_{ab}]``,
where ``w_r`` are the Gregory end weights. Note that it is neither the
trace nor the Frobenius inner product. It is a private function.
"""
function _mat_inner(X::Gᵐᵃᵗ{C64}, Y::Gᵐᵃᵗ{C64}, I::Integrator)
    N = getntau(X) - 1
    k = I.k
    s = 0.0
    for r = 0:N
        if r ≤ k
            w = I.GIW.ω[r + 1]
        elseif r ≥ N - k
            w = I.GIW.ω[N - r + 1]
        else
            w = 1.0
        end
        s += w * sum(real, _block(X, r + 1)' * _block(Y, r + 1))
    end
    return s
end

"""
    _dyson_mat_kernel(G::ℱ{C64}, Σ::ℱ{C64}, μ::F64, H::Cf{C64},
                      β::F64, I::Integrator)

Return the free Matsubara Green's function ``G_0`` and the kernel
``F = -G_0 * \\Sigma`` of the Matsubara Dyson equation. It is a private
function.
"""
function _dyson_mat_kernel(G::ℱ{C64}, Σ::ℱ{C64}, μ::F64, H::Cf{C64},
                           β::F64, I::Integrator)
    sign = getsign(G)
    ntau = getntau(G)
    d = getdims(G)[1]

    G0 = Gᵐᵃᵗ(ntau, d, d)
    _free_matsubara_from_H!(G0, μ, H, β, sign)
    #
    F = Gᵐᵃᵗ(ntau, d, d)
    conv_mat(F, G0, Σ.mat, I, β, sign)
    smul!(F, -one(C64))

    return G0, F
end

"""
    _dyson_mat_fixpoint!(G::ℱ{C64}, Σ::ℱ{C64}, μ::F64, H::Cf{C64},
                         β::F64, I::Integrator; maxiter::I64 = 6)

Solve the Matsubara Dyson equation with the fixpoint method. See
*Remarks* : *Matsubara Dyson Equation*. It is a private function.

### Arguments
* G       -> Contour-ordered Green's function.
* Σ       -> Self-energy.
* μ       -> Chemical potential.
* H       -> Hamiltonian.
* β       -> Inverse temperature.
* I       -> An Integrator struct.
* maxiter -> Number of the residual corrections.

### Returns
* G -> Contour-ordered Green's function.
"""
function _dyson_mat_fixpoint!(G::ℱ{C64}, Σ::ℱ{C64}, μ::F64, H::Cf{C64},
                              β::F64, I::Integrator; maxiter::I64 = 6)
    sign = getsign(G)
    ntau = getntau(G)
    d = getdims(G)[1]

    G0, F = _dyson_mat_kernel(G, Σ, μ, H, β, I)
    _matsubara_seed!(G.mat, F, G0, β, sign; pcf = 5, order = 3)

    R = Gᵐᵃᵗ(ntau, d, d)
    dG = Gᵐᵃᵗ(ntau, d, d)
    for _ = 1:maxiter
        # R = F * G + G - Q
        conv_mat(R, F, G.mat, I, β, sign)
        R.data .= R.data .+ G.mat.data .- G0.data
        #
        _matsubara_seed!(dG, F, R, β, sign; pcf = 5, order = 3)
        G.mat.data .= G.mat.data .- dG.data
    end

    return G
end

"""
    _dyson_mat_cg!(G::ℱ{C64}, Σ::ℱ{C64}, μ::F64, H::Cf{C64},
                   β::F64, I::Integrator; maxiter::I64 = 40, tol::F64 = 1e-12)

Solve the Matsubara Dyson equation with the Fletcher-Reeves conjugate
gradient method. See *Remarks* : *Matsubara Dyson Equation*. It is a
private function.

### Arguments
* G       -> Contour-ordered Green's function.
* Σ       -> Self-energy.
* μ       -> Chemical potential.
* H       -> Hamiltonian.
* β       -> Inverse temperature.
* I       -> An Integrator struct.
* maxiter -> Maximum number of iterations.
* tol     -> Tolerance of the residual.

### Returns
* niter -> Number of the performed iterations.
"""
function _dyson_mat_cg!(G::ℱ{C64}, Σ::ℱ{C64}, μ::F64, H::Cf{C64},
                        β::F64, I::Integrator; maxiter::I64 = 40, tol::F64 = 1e-12)
    sign = getsign(G)
    ntau = getntau(G)
    d = getdims(G)[1]

    # F = -G0 * Σ, Fcc = -Σ * G0, Q = G0
    G0, F = _dyson_mat_kernel(G, Σ, μ, H, β, I)
    Fcc = Gᵐᵃᵗ(ntau, d, d)
    conv_mat(Fcc, Σ.mat, G0, I, β, sign)
    smul!(Fcc, -one(C64))
    #
    _matsubara_seed!(G.mat, F, G0, β, sign; pcf = 3, order = 3)
    maxiter > 0 || return 0

    R = Gᵐᵃᵗ(ntau, d, d)
    P = Gᵐᵃᵗ(ntau, d, d)
    AP = Gᵐᵃᵗ(ntau, d, d)
    C₁ = Gᵐᵃᵗ(ntau, d, d)
    C₂ = Gᵐᵃᵗ(ntau, d, d)
    #
    # A(X) = X + [F * X + X * Fcc] / 2
    function apply_A!(AX, X)
        conv_mat(C₁, F, X, I, β, sign)
        conv_mat(C₂, X, Fcc, I, β, sign)
        AX.data .= 0.5 .* (C₁.data .+ C₂.data) .+ X.data
    end

    # Initial residual R = Q - A(G)
    apply_A!(R, G.mat)
    R.data .= G0.data .- R.data
    P.data .= R.data
    rsold = _mat_inner(R, R, I)
    isfinite(rsold) || throw(DomainError(rsold, "residual is not finite"))

    niter = 0
    if rsold > tol
        for _ = 1:maxiter
            niter += 1
            apply_A!(AP, P)
            c₂ = _mat_inner(P, AP, I)
            (isfinite(c₂) && !iszero(c₂)) ||
                throw(DomainError(c₂, "zero or non-finite denominator in CG"))
            α = rsold / c₂
            G.mat.data .= G.mat.data .+ α .* P.data
            R.data .= R.data .- α .* AP.data
            rsnew = _mat_inner(R, R, I)
            isfinite(rsnew) || throw(DomainError(rsnew, "residual is not finite"))
            P.data .= R.data .+ (rsnew / rsold) .* P.data
            rsold = rsnew
            sqrt(rsold) < tol && break
        end
    end

    return niter
end

"""
    _dyson_mat_fourier!(G::ℱ{C64}, Σ::ℱ{C64}, μ::F64, H::Cf{C64}, β::F64;
                        pcf::I64 = 10, order::I64 = 3)

Solve the Matsubara Dyson equation in the frequency domain. With ``K =
H - \\mu + \\Sigma(i\\omega_n)``, the non-zero modes contribute
``[(i\\omega_n - K)^{-1}/(i\\omega_n)] K / \\beta`` (the free tail
``1/(i\\omega_n)`` is treated analytically by the first order tail
correction), and the bosonic zero mode contributes ``-K^{-1}/\\beta``.
The frequency sum includes `2pcf + 1` copies of the basic window. The
number of imaginary time intervals must be even. It is a private
function.

### Arguments
* G     -> Contour-ordered Green's function.
* Σ     -> Self-energy.
* μ     -> Chemical potential.
* H     -> Hamiltonian, only `H[0]` is used.
* β     -> Inverse temperature.
* pcf   -> Number of the oversampling copies on each side.
* order -> Order of the boundary correction of the Fourier transform.

### Returns
* G -> Contour-ordered Green's function.

*References* :

Please see [`NESSi`] `cntr_dyson_impl.hpp`, `dyson_mat_fourier_dispatch`.
"""
function _dyson_mat_fourier!(G::ℱ{C64}, Σ::ℱ{C64}, μ::F64, H::Cf{C64}, β::F64;
                             pcf::I64 = 10, order::I64 = 3)
    sign = getsign(G)
    N = getntau(G) - 1
    d = getdims(G)[1]
    iseven(N) ||
        throw(ArgumentError("the :fourier method needs an even number of imaginary " *
                            "time intervals (an odd number of points), got $N intervals"))

    Id = Matrix{C64}(I, d, d)
    hj = H[0] .- μ .* Id
    m₂ = N ÷ 2
    one = (sign == FERMI) ? 1 : 0

    # Raw transform of Σ and the free tail
    sdft = _matsubara_dft(Σ.mat, sign)
    gmat = zeros(C64, d, d, N + 1)
    _first_order_tail!(gmat, Id, β, sign)

    for m = -m₂:m₂-1
        for p = -pcf:pcf
            mp = m + p * N
            iω = complex(0.0, π / β * (2 * mp + one))
            z₁ = _matsubara_ft_direct(mp, Σ.mat, sdft, sign, β, order) .+ hj
            if sign == BOSE && mp == 0
                # The bosonic zero mode, 1/(iωₙ) diverges
                z₂ = -inv(z₁)
            else
                # 1/(iωₙ - K) - 1/(iωₙ) = [iωₙ (iωₙ - K)]⁻¹ K
                zinv = inv(iω .* Id .- z₁) .* (1.0 / iω)
                z₂ = zinv * z₁
            end
            z₂ = z₂ .* (1.0 / β)
            for r = 0:N
                Z = view(gmat, :, :, r + 1)
                Z .= Z .+ z₂ .* exp(-(r * (β / N)) * iω)
            end
        end
    end

    copyto!(G.mat.data, gmat)
    all(isfinite, G.mat.data) || throw(DomainError(G.mat.data, "G is not finite"))

    return G
end

#=
### *Public API* : *Matsubara Dyson*
=#

"""
    dyson_mat!(G::ℱ{C64}, Σ::ℱ{C64}, μ::F64, H::Cf{C64}, β::F64;
               solve_order::I64 = 5, method::Symbol = :fixpoint,
               force_hermitian::Bool = true,
               maxiter::Union{Nothing,I64} = nothing, tol::F64 = 1e-12)

Solve the Matsubara Dyson equation
``[-\\partial_\\tau + \\mu - H] G^{M} - \\Sigma^{M} * G^{M} = \\delta(\\tau)``.
Only the Matsubara component of `G` is modified. See *Remarks* :
*Matsubara Dyson Equation*.

### Arguments
* G               -> Contour-ordered Green's function.
* Σ               -> Self-energy.
* μ               -> Chemical potential.
* H               -> Hamiltonian, only `H[0]` is used.
* β               -> Inverse temperature.
* solve_order     -> Order of the integration, 1 ≤ solve_order ≤ 5.
* method          -> `:fixpoint`, `:cg`, or `:fourier`.
* force_hermitian -> Whether to symmetrize the Matsubara component.
* maxiter         -> Number of iterations, 6 (`:fixpoint`) or 40 (`:cg`).
* tol             -> Tolerance of the residual (`:cg` only).

### Returns
* G -> Contour-ordered Green's function.

See also: [`dyson!`](@ref).
"""
function dyson_mat!(G::ℱ{C64}, Σ::ℱ{C64}, μ::F64, H::Cf{C64}, β::F64;
                    solve_order::I64 = 5, method::Symbol = :fixpoint,
                    force_hermitian::Bool = true,
                    maxiter::Union{Nothing,I64} = nothing, tol::F64 = 1e-12)
    # Sanity check
    _check_dyson(G, Σ, H, solve_order)
    _check_hamiltonian(H, 0, getdims(G)[1])
    all(isfinite, Σ.mat.data) || throw(DomainError(Σ.mat.data, "Σ is not finite"))
    β > 0.0 || throw(ArgumentError("β must be positive"))
    isnothing(maxiter) || maxiter ≥ 0 || throw(ArgumentError("maxiter must be nonnegative"))

    I = Integrator(solve_order)
    if method == :fixpoint
        _dyson_mat_fixpoint!(G, Σ, μ, H, β, I;
                             maxiter = something(maxiter, 6))
    elseif method == :cg
        _dyson_mat_cg!(G, Σ, μ, H, β, I;
                       maxiter = something(maxiter, 40), tol = tol)
    elseif method == :fourier
        _dyson_mat_fourier!(G, Σ, μ, H, β; pcf = 10, order = 3)
    else
        throw(ArgumentError("dyson_mat!: unknown method :$method, use :fixpoint, :cg, or :fourier"))
    end

    force_hermitian && force_matsubara_hermitian!(G)

    return G
end

#=
*Remarks* : *Real Time Dyson Equation*

In the following, the time indices `n`, `j`, `l`, `m` follow the
convention of [`NESSi`], i.e., ``t_n = nh`` with `n = 0, 1, ⋯`. The
corresponding index of `KadanoffBaym` is `n + 1`. Let ``D_{nl}`` be the
polynomial differentiation weights, ``I_{jnl}`` the polynomial
integration weights, ``\Omega_{nl}`` the Gregory weights, ``\omega_l``
the Gregory end weights, and ``b_p`` the backward differentiation weights
of order 𝑘 + 1.

**Start-up**: For the first 𝑘 + 1 time steps, the time derivatives and
integrals are evaluated by polynomial differentiation and integration.
For each `j = 0, ⋯, k - 1`, the unknowns ``G^{R}(n,j)`` (`n = j + 1, ⋯,
k`) satisfy

```math
\begin{equation}
\frac{i}{h}\sum^{k}_{l=0} D_{nl} G^{R}(l,j) + (\mu - H_n) G^{R}(n,j)
    - h\sum^{k}_{l=0} I_{jnl} \Sigma^{R}(n,l) G^{R}(l,j) = 0.
\end{equation}
```

The left-mixing and lesser components at `n = 1, ⋯, k` satisfy the
𝑘 × 𝑘 block system

```math
\begin{equation}
\sum^{k}_{q=1}\left[
    \frac{i}{h}D_{nq} + \delta_{nq}(\mu - H_n) - h\Omega_{nq}\Sigma^{R}(n,q)
\right] X(q) = Q_n,
\end{equation}
```

where ``X(q) = G^{\rceil}(q,\tau)`` or ``G^{<}(q,n')``, and ``Q_n``
contains the known source terms.

**Time stepping**: For `n > k` the time derivatives are evaluated by the
backward differentiation formula, and the integrals by the Gregory
quadrature. For example, the left-mixing component satisfies

```math
\begin{equation}
\left[\frac{i}{h}b_0 + \mu - H_n - h\omega_0 \Sigma^{R}(n,n)\right]
G^{\rceil}(n,\tau)
=
[\Sigma * G]^{\rceil}_{known}(n,\tau)
- \frac{i}{h}\sum^{k+1}_{p=1} b_p G^{\rceil}(n-p,\tau).
\end{equation}
```

The retarded component is obtained from the right action Dyson equation
by marching backward in ``t'``, and the lesser component ``G^{<}(j,n)``
by marching forward in ``j``. The source term of the lesser component is
``[\Sigma^{\rceil} * G^{\lceil} + \Sigma^{<} * G^{A}](j,n)``, while the
term ``\Sigma^{R} * G^{<}`` belongs to the linear system. The order of the
components is fixed: retarded, left-mixing, and then lesser.

*References* :

Please see [`NESSi`] `cntr_dyson_impl.hpp`.
=#

#=
### *Private Service Functions* : *Block Access*
=#

#
# Blocks of the contour functions, the time indices follow the NESSi
# convention (start from 0). Only the stored triangles are accessed.
#
@inline _ret_blk(G, n::I64, j::I64) = _block(G.ret, n + 1, j + 1)
@inline _tv_blk(G, n::I64, m::I64) = _block(G.lmix, n + 1, m + 1)
@inline _les_blk(G, j::I64, n::I64) = _block(G.less, j + 1, n + 1)
@inline _ham(H, n::I64) = H[n + 1]

"""
    _sigma_ret(Σ, n::I64, l::I64)

Return ``\\Sigma^{R}(n,l)`` (NESSi indices) and the sign of the weight.
If `n < l`, ``-\\Sigma^{R}(l,n)^{\\dagger}`` is returned as
``\\Sigma^{R}(l,n)^{\\dagger}`` together with the sign `-1`. It is a
private function.
"""
@inline function _sigma_ret(Σ, n::I64, l::I64)
    if n ≥ l
        return _ret_blk(Σ, n, l), 1.0
    else
        return _ret_blk(Σ, l, n)', -1.0
    end
end

"""
    _check_dyson_rt(G, Σ, H::Cf{C64}, k::I64, nmax::I64)

Check the arguments of the real time Dyson solvers. The time steps
`0, ⋯, nmax` (NESSi convention) of `G`, `Σ`, and `H` are needed. It is a
private function.
"""
function _check_dyson_rt(G, Σ, H::Cf{C64}, k::I64, nmax::I64)
    _check_dyson(G, Σ, H, k)
    getntime(G) ≥ nmax + 1 ||
        throw(ArgumentError("G must have at least $(nmax + 1) real time steps"))
    H.ntime ≥ nmax + 1 ||
        throw(ArgumentError("H must have at least $(nmax + 1) real time steps"))
    _check_hamiltonian(H, 1:nmax+1, getdims(G)[1])
end

#=
### *Private Service Functions* : *Retarded Component*
=#

"""
    _dyson_start_ret!(G::CnAbstractFunction{C64}, μ::F64,
                      H::CnAbstractFunction{C64}, Σ::CnAbstractFunction{C64},
                      h::F64, I::Integrator)

Start-up procedure of the retarded component, i.e., ``G^{R}(n,j)`` for
`0 ≤ j ≤ n ≤ k`. See *Remarks* : *Real Time Dyson Equation*. It is a
private function.

*References* :

Please see [`NESSi`] `cntr_dyson_impl.hpp`, `dyson_start_ret`.
"""
function _dyson_start_ret!(G::CnAbstractFunction{C64}, μ::F64,
                           H::CnAbstractFunction{C64}, Σ::CnAbstractFunction{C64},
                           h::F64, I::Integrator)
    k = I.k
    d = getdims(G)[1]
    Id = Matrix{C64}(LinearAlgebra.I, d, d)
    gtemp = zeros(C64, d, d)
    stemp = zeros(C64, d, d)

    # G^{R}(n,n) = -i
    for n = 0:k
        _ret_blk(G, n, n) .= -im .* Id
    end

    # Determine G^{R}(n,j), n = j+1, ⋯, k
    for j = 0:k-1
        dim = k - j
        ws = _BlockSolver(dim, d)
        fill!(ws.Mbig, zero(C64))
        fill!(ws.Qbig, zero(C64))
        #
        for n = j+1:k
            p = n - j
            Qₚ = _qblock(ws, p)
            for l = 0:k
                if l ≤ j
                    # Known G^{R}(l,j) = -G^{R}(j,l)†, it goes into Q
                    gtemp .= .-(_ret_blk(G, j, l))'
                    mul!(stemp, _ret_blk(Σ, n, l), gtemp)
                    @. Qₚ = Qₚ + (-im / h) * I.PDW[n,l] * gtemp +
                                 h * I.XIW[j,n,l] * stemp
                else
                    # Unknown G^{R}(l,j), it goes into M
                    Mₚₒ = _mblock(ws, p, l - j)
                    Mₚₒ .= (im / h * I.PDW[n,l]) .* Id
                    if l == n
                        Mₚₒ .= Mₚₒ .+ μ .* Id .- _ham(H, l)
                    end
                    S, s = _sigma_ret(Σ, n, l)
                    weight = s * h * I.XIW[j,n,l]
                    Mₚₒ .= Mₚₒ .- weight .* S
                end
            end
        end
        #
        _solve_block!(ws)
        for n = j+1:k
            _ret_blk(G, n, j) .= _xblock(ws, n - j)
        end
    end

    return G
end

"""
    _dyson_timestep_ret!(nkb::I64, G::CnAbstractFunction{C64}, μ::F64,
                         H::CnAbstractFunction{C64}, Σ::CnAbstractFunction{C64},
                         h::F64, I::Integrator)

Time stepping of the retarded component, i.e., ``G^{R}(n,j)`` for
`0 ≤ j ≤ n`, where `n = nkb - 1 > k`. See *Remarks* : *Real Time Dyson
Equation*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_dyson_impl.hpp`, `dyson_timestep_ret`.
"""
function _dyson_timestep_ret!(nkb::I64, G::CnAbstractFunction{C64}, μ::F64,
                              H::CnAbstractFunction{C64}, Σ::CnAbstractFunction{C64},
                              h::F64, I::Integrator)
    k = I.k
    n = nkb - 1
    @assert n > k
    d = getdims(G)[1]
    Id = Matrix{C64}(LinearAlgebra.I, d, d)

    # Clear the time step, G^{R}(n,n) = -i
    for j = 0:n
        fill!(_ret_blk(G, n, j), zero(C64))
    end
    _ret_blk(G, n, n) .= -im .* Id
    Gₙₙ = _ret_blk(G, n, n)

    # Start values G^{R}(n,n-j), j = 1, ⋯, k: solve a k × k problem
    # ∑_q X_q M_{pq} = Q_p with X_q = G^{R}(n,n-q)
    wsk = _BlockSolver(k, d)
    M = [zeros(C64, d, d) for _ = 1:k, _ = 1:k]
    Q = [zeros(C64, d, d) for _ = 1:k]
    X = [_ret_blk(G, n, n - q) for q = 1:k]
    for j = 1:k
        # Derivatives
        for l = 0:k
            cweight = im / h * I.PDW[j,l]
            if l == 0
                Q[j] .= Q[j] .- cweight .* Gₙₙ
            else
                M[j,l] .= M[j,l] .+ cweight .* Id
            end
        end
        # Hamiltonian
        M[j,j] .= M[j,j] .+ μ .* Id .- _ham(H, n - j)
        # Integral
        for l = 0:k
            weight = h * I.GIW[j,l]
            if l ≤ j
                S = _ret_blk(Σ, n - l, n - j)
            else
                S = _ret_blk(Σ, n - j, n - l)'
                weight = -weight
            end
            if l == 0
                mul!(Q[j], Gₙₙ, S, weight, one(C64))
            else
                M[j,l] .= M[j,l] .- weight .* S
            end
        end
    end
    _solve_block_left!(X, M, Q, wsk)

    # Contribution of G^{R}(n,m), m = n-k, ⋯, n, to the convolution
    # ∫ dm G^{R}(n,m) Σ^{R}(m,j), stored into qq(j) without the factor h
    qq = zeros(C64, d, d, n + 1)
    for m = n-k:n
        Gₙₘ = _ret_blk(G, n, m)
        for j = 0:n-k-1
            mul!(view(qq, :, :, j + 1), Gₙₘ, _ret_blk(Σ, m, j), I.GIW[n-j,n-m], one(C64))
        end
    end

    # The rest: t' = n-l, l = k+1, ⋯, n
    ws1 = _BlockSolver(1, d)
    diffw = [I.BDW[p] * im / h for p = 0:k+1]
    w₀ = h * I.GIW.ω[1]
    mm = zeros(C64, d, d)
    for l = k+1:n
        j = n - l
        qqⱼ = view(qq, :, :, j + 1)
        qqⱼ .= h .* qqⱼ
        for p = 1:k+1
            qqⱼ .= qqⱼ .- diffw[p+1] .* _ret_blk(G, n, j + p)
        end
        # M = i b₀ / h + μ - H(j)† - h ω₀ Σ^{R}(j,j)
        Hⱼ = _ham(H, j)
        mm .= (diffw[1] + μ) .* Id .- Hⱼ' .- w₀ .* _ret_blk(Σ, j, j)
        Gₙⱼ = _ret_blk(G, n, j)
        _solve_block_left!(Gₙⱼ, mm, qqⱼ, ws1)
        #
        # Contribution of G^{R}(n,j) to ∫ dm G^{R}(n,m) Σ^{R}(m,j₂), j₂ < j
        for j₂ = 0:j-1
            mul!(view(qq, :, :, j₂ + 1), Gₙⱼ, _ret_blk(Σ, j, j₂), I.GIW[n-j₂,n-j], one(C64))
        end
    end

    return G
end

#=
### *Private Service Functions* : *Left-mixing Component*
=#

"""
    _dyson_start_lmix!(G::CnAbstractFunction{C64}, μ::F64,
                       H::CnAbstractFunction{C64}, Σ::CnAbstractFunction{C64},
                       β::F64, h::F64, I::Integrator)

Start-up procedure of the left-mixing component, i.e., ``G^{\\rceil}(n,\\tau)``
for `0 ≤ n ≤ k`. The retarded component at the first 𝑘 + 1 time steps
must be known. See *Remarks* : *Real Time Dyson Equation*. It is a
private function.

*References* :

Please see [`NESSi`] `cntr_dyson_impl.hpp`, `dyson_start_tv`.
"""
function _dyson_start_lmix!(G::CnAbstractFunction{C64}, μ::F64,
                            H::CnAbstractFunction{C64}, Σ::CnAbstractFunction{C64},
                            β::F64, h::F64, I::Integrator)
    k = I.k
    d = getdims(G)[1]
    ntau = getntau(G)
    sign = getsign(G)
    δτ = β / (ntau - 1)
    Id = Matrix{C64}(LinearAlgebra.I, d, d)

    # Initial value: G^{⌉}(0,τ) = i ξ G^{M}(β - τ)
    for m = 1:ntau
        _block(G.lmix, 1, m) .= (im * sign) .* _block(G.mat, ntau - m + 1)
    end

    # Source term: δτ ∫ dτ' Σ^{⌉}(n,τ') G^{M}(τ' - τ), n = 1, ⋯, k. No
    # extra factor ±i is involved.
    cws = _ConvWorkspace(Σ.ret.data, G.ret.data)
    for n = 1:k
        for m = 1:ntau
            fill!(_tv_blk(G, n, m - 1), zero(C64))
        end
        _conv_lmix_mat_kernel!(_LmixRow(G.lmix, n + 1), n + 1, Σ.lmix, nothing,
                               G.mat, I, sign, δτ, cws)
    end

    # Block system ∑_q M_{pq} X_q = Q_p, X_q = G^{⌉}(q,τ) for all τ. The
    # matrix M does not depend on τ.
    ws = _BlockSolver(k, d, d * ntau)
    fill!(ws.Mbig, zero(C64))
    for n = 1:k
        Qₙ = _qblock(ws, n)
        for m = 1:ntau
            cols = (m - 1) * d + 1 : m * d
            G₀ₘ = _tv_blk(G, 0, m - 1)
            Qₙₘ = view(Qₙ, :, cols)
            # Integral Σ^{⌉} * G^{M}
            Qₙₘ .= _tv_blk(G, n, m - 1)
            # Derivative, j = 0
            Qₙₘ .= Qₙₘ .- (im / h * I.PDW[n,0]) .* G₀ₘ
            # Integral Σ^{R} * G^{⌉}, j = 0
            mul!(Qₙₘ, _ret_blk(Σ, n, 0), G₀ₘ, h * I.GIW[n,0], one(C64))
        end
        for j = 1:k
            Mₙⱼ = _mblock(ws, n, j)
            Mₙⱼ .= Mₙⱼ .+ (im / h * I.PDW[n,j]) .* Id
            S, s = _sigma_ret(Σ, n, j)
            Mₙⱼ .= Mₙⱼ .- (s * h * I.GIW[n,j]) .* S
        end
        Mₙₙ = _mblock(ws, n, n)
        Mₙₙ .= Mₙₙ .+ μ .* Id .- _ham(H, n)
    end
    _solve_block!(ws)
    for n = 1:k
        Xₙ = _xblock(ws, n)
        for m = 1:ntau
            _tv_blk(G, n, m - 1) .= view(Xₙ, :, (m - 1) * d + 1 : m * d)
        end
    end

    return G
end

"""
    _dyson_timestep_lmix!(nkb::I64, G::CnAbstractFunction{C64}, μ::F64,
                          H::CnAbstractFunction{C64}, Σ::CnAbstractFunction{C64},
                          β::F64, h::F64, I::Integrator)

Time stepping of the left-mixing component, i.e., ``G^{\\rceil}(n,\\tau)``,
where `n = nkb - 1 > k`. The retarded component at time step `n` must be
known. See *Remarks* : *Real Time Dyson Equation*. It is a private
function.

*References* :

Please see [`NESSi`] `cntr_dyson_impl.hpp`, `dyson_timestep_tv`.
"""
function _dyson_timestep_lmix!(nkb::I64, G::CnAbstractFunction{C64}, μ::F64,
                               H::CnAbstractFunction{C64}, Σ::CnAbstractFunction{C64},
                               β::F64, h::F64, I::Integrator; threaded::Bool = false)
    k = I.k
    n = nkb - 1
    @assert n > k
    d = getdims(G)[1]
    ntau = getntau(G)
    Id = Matrix{C64}(LinearAlgebra.I, d, d)

    # Clear the current row, so that the implicit term h ω₀ Σ^{R}(n,n)
    # G^{⌉}(n,τ) does not enter the source term
    for m = 1:ntau
        fill!(_block(G.lmix, nkb, m), zero(C64))
    end

    # Known part of the convolution [Σ * G]^{⌉}(n,τ)
    W = 𝒻(nkb, ntau, d, d, getsign(G))
    if _use_threads(threaded) && Σ isa ℱ && G isa ℱ
        _incr_tv_threaded!(nkb, W, Σ, Σ, G, G, I, β, h, one(C64))
    else
        incr_convolution_tv!(nkb, W, Σ, Σ, nothing, G, G, I, β, h, one(C64))
    end

    # Backward differentiation, G^{⌉}(m,τ), m = n-k-1, ⋯, n-1
    ws = _BlockSolver(1, d, d * ntau)
    for m = 1:ntau
        Qₘ = view(ws.Qbig, :, (m - 1) * d + 1 : m * d)
        Qₘ .= W.lmix.data[m]
        for l = n-k-1:n-1
            cweight = im / h * I.BDW[n-l]
            Qₘ .= Qₘ .- cweight .* _tv_blk(G, l, m - 1)
        end
    end

    # [i b₀ / h - h Ω_{n0} Σ^{R}(n,n) + μ - H(n)] G^{⌉}(n,τ) = Q(τ)
    ws.Mbig .= (im / h * I.BDW[0] + μ) .* Id .-
               (h * I.GIW[n,0]) .* _ret_blk(Σ, n, n) .- _ham(H, n)
    _solve_block!(ws)
    for m = 1:ntau
        _block(G.lmix, nkb, m) .= view(ws.Xbig, :, (m - 1) * d + 1 : m * d)
    end

    return G
end

#=
### *Private Service Functions* : *Lesser Component*
=#

"""
    _dyson_less_core!(n::I64, gles::AbstractVector, G, μ::F64, H, Σ,
                      h::F64, I::Integrator)

Solve the lesser component ``G^{<}(j,n)`` for `j = 0, ⋯, n` (NESSi
indices). On entry, `gles[j+1]` (`j = 0, ⋯, max(n,k)`) contains the source
term ``[\\Sigma^{\\rceil} * G^{\\lceil} + \\Sigma^{<} * G^{A}](j,n)``. On
exit, `gles[j+1]` (`j = 0, ⋯, n`) contains the solution, which is also
written into `G`. It is a private function.

*References* :

Please see [`NESSi`] `cntr_dyson_impl.hpp`, `dyson_timestep_les`.
"""
function _dyson_less_core!(n::I64, gles::AbstractVector, G, μ::F64, H, Σ,
                           h::F64, I::Integrator)
    k = I.k
    d = getdims(G)[1]
    Id = Matrix{C64}(LinearAlgebra.I, d, d)
    @assert length(gles) ≥ max(n, k) + 1

    # Initial value G^{<}(0,n) = -G^{⌉}(n,0)†
    g₀ = gles[1]
    g₀ .= .-(_tv_blk(G, n, 0))'

    # Start: j = 1, ⋯, k. Solve a k × k problem M X = Q,
    # X_q = G^{<}(q,n).
    ws = _BlockSolver(k, d)
    fill!(ws.Mbig, zero(C64))
    for j = 1:k
        Qⱼ = _qblock(ws, j)
        # Integral Σ^{⌉} * G^{⌈} + Σ^{<} * G^{A}
        Qⱼ .= gles[j+1]
        # Derivative
        Qⱼ .= Qⱼ .- (im / h * I.PDW[j,0]) .* g₀
        for m = 1:k
            Mⱼₘ = _mblock(ws, j, m)
            Mⱼₘ .= Mⱼₘ .+ (im / h * I.PDW[j,m]) .* Id
        end
        # Hamiltonian
        Mⱼⱼ = _mblock(ws, j, j)
        Mⱼⱼ .= Mⱼⱼ .+ μ .* Id .- _ham(H, j)
        # Integral Σ^{R} * G^{<}
        mul!(Qⱼ, _ret_blk(Σ, j, 0), g₀, h * I.GIW[j,0], one(C64))
        for m = 1:k
            S, s = _sigma_ret(Σ, j, m)
            Mⱼₘ = _mblock(ws, j, m)
            Mⱼₘ .= Mⱼₘ .- (s * h * I.GIW[j,m]) .* S
        end
    end
    _solve_block!(ws)
    for j = 1:k
        gles[j+1] .= _xblock(ws, j)
    end

    # Time stepping: j = k+1, ⋯, n
    ws1 = _BlockSolver(1, d)
    qq = zeros(C64, d, d)
    mm = zeros(C64, d, d)
    for j = k+1:n
        qq .= gles[j+1]
        for p = 1:k+1
            qq .= qq .+ (-im / h * I.BDW[p]) .* gles[j-p+1]
        end
        mm .= (im / h * I.BDW[0] + μ) .* Id .- _ham(H, j) .-
              (h * I.GIW[j,j]) .* _ret_blk(Σ, j, j)
        for m = 0:j-1
            mul!(qq, _ret_blk(Σ, j, m), gles[m+1], h * I.GIW[j,m], one(C64))
        end
        _solve_block_right!(gles[j+1], mm, qq, ws1)
    end

    # Write the stored triangle
    for j = 0:n
        _les_blk(G, j, n) .= gles[j+1]
    end

    return G
end

"""
    _dyson_start_less!(G::CnAbstractFunction{C64}, μ::F64,
                       H::CnAbstractFunction{C64}, Σ::CnAbstractFunction{C64},
                       β::F64, h::F64, I::Integrator)

Start-up procedure of the lesser component, i.e., ``G^{<}(j,n)`` for
`0 ≤ j ≤ n ≤ k`. The retarded and left-mixing components at the first
𝑘 + 1 time steps must be known. See *Remarks* : *Real Time Dyson
Equation*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_dyson_impl.hpp`, `dyson_start_les`.
"""
function _dyson_start_less!(G::CnAbstractFunction{C64}, μ::F64,
                            H::CnAbstractFunction{C64}, Σ::CnAbstractFunction{C64},
                            β::F64, h::F64, I::Integrator)
    k = I.k
    d = getdims(G)[1]
    ntau = getntau(G)
    cws = _ConvWorkspace(Σ.ret.data, G.ret.data)

    # NESSi n = 0, ⋯, k - 1: the private scratch-target convolutions
    # provide the source blocks j = 0, ⋯, k, including j > n.
    buf = [zeros(C64, d, d) for _ = 1:k+1]
    for n = 0:k-1
        tstp = n + 1
        foreach(b -> fill!(b, zero(C64)), buf)
        _incr_convolution_lmix_rmix!(buf, tstp, Σ, Σ, nothing, G, G, I, β, h, one(C64), cws)
        _incr_convolution_less_adv!(buf, tstp, Σ, Σ, nothing, G, G, I, β, h, one(C64), cws)
        _dyson_less_core!(n, buf, G, μ, H, Σ, h, I)
    end

    # NESSi n = k: the public slice-target convolutions, j = 0, ⋯, k
    W = 𝒻(k + 1, ntau, d, d, getsign(G))
    incr_convolution_lmix_rmix!(k + 1, W, Σ, Σ, nothing, G, G, I, β, h, one(C64))
    incr_convolution_less_adv!(k + 1, W, Σ, Σ, nothing, G, G, I, β, h, one(C64))
    _dyson_less_core!(k, W.less.data, G, μ, H, Σ, h, I)

    return G
end

"""
    _dyson_timestep_less!(nkb::I64, G::CnAbstractFunction{C64}, μ::F64,
                          H::CnAbstractFunction{C64}, Σ::CnAbstractFunction{C64},
                          β::F64, h::F64, I::Integrator)

Time stepping of the lesser component, i.e., ``G^{<}(j,n)`` for
`0 ≤ j ≤ n`, where `n = nkb - 1 > k`. The retarded and left-mixing
components at time step `n` must be known. See *Remarks* : *Real Time
Dyson Equation*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_dyson_impl.hpp`, `dyson_timestep_les`.
"""
function _dyson_timestep_less!(nkb::I64, G::CnAbstractFunction{C64}, μ::F64,
                               H::CnAbstractFunction{C64}, Σ::CnAbstractFunction{C64},
                               β::F64, h::F64, I::Integrator; threaded::Bool = false)
    n = nkb - 1
    @assert n > I.k
    d = getdims(G)[1]
    ntau = getntau(G)

    # Source term [Σ^{⌉} * G^{⌈} + Σ^{<} * G^{A}](j,n), j = 0, ⋯, n. The
    # term Σ^{R} * G^{<} is excluded.
    W = 𝒻(nkb, ntau, d, d, getsign(G))
    if _use_threads(threaded) && Σ isa ℱ && G isa ℱ
        _incr_lesser_src_threaded!(nkb, W, Σ, Σ, G, G, I, β, h, one(C64))
    else
        incr_convolution_lmix_rmix!(nkb, W, Σ, Σ, nothing, G, G, I, β, h, one(C64))
        incr_convolution_less_adv!(nkb, W, Σ, Σ, nothing, G, G, I, β, h, one(C64))
    end
    _dyson_less_core!(n, W.less.data, G, μ, H, Σ, h, I)

    return G
end

#=
### *Public API* : *Real Time Dyson*
=#

"""
    dyson_start!(G::ℱ{C64}, μ::F64, H::Cf{C64}, Σ::ℱ{C64}, β::F64, h::F64;
                 solve_order::I64 = 5)

Solve the Dyson equation for the first `solve_order + 1` real time
steps (`tstp = 1, ⋯, solve_order + 1`). The Matsubara component of `G`
must be known. The components are solved in the order: retarded,
left-mixing, and lesser.

### Arguments
* G           -> Contour-ordered Green's function.
* μ           -> Chemical potential.
* H           -> Hamiltonian.
* Σ           -> Self-energy.
* β           -> Inverse temperature.
* h           -> Time step.
* solve_order -> Order of the integration, 1 ≤ solve_order ≤ 5.

### Returns
* G -> Contour-ordered Green's function.

See also: [`dyson!`](@ref).
"""
function dyson_start!(G::ℱ{C64}, μ::F64, H::Cf{C64}, Σ::ℱ{C64}, β::F64, h::F64;
                      solve_order::I64 = 5)
    k = solve_order
    _check_dyson_rt(G, Σ, H, k, k)
    β > 0.0 || throw(ArgumentError("β must be positive"))
    h > 0.0 || throw(ArgumentError("h must be positive"))

    I = Integrator(k)
    _dyson_start_ret!(G, μ, H, Σ, h, I)
    _dyson_start_lmix!(G, μ, H, Σ, β, h, I)
    _dyson_start_less!(G, μ, H, Σ, β, h, I)

    return G
end

"""
    dyson_timestep!(nkb::I64, G::ℱ{C64}, μ::F64, H::Cf{C64}, Σ::ℱ{C64},
                    β::F64, h::F64; solve_order::I64 = 5)

Solve the Dyson equation at the real time step `nkb > solve_order + 1`.
The previous time steps of `G` must be known. The Matsubara component is
not recalculated. The components are solved in the order: retarded,
left-mixing, and lesser.

### Arguments
* nkb         -> Index of the time step.
* G           -> Contour-ordered Green's function.
* μ           -> Chemical potential.
* H           -> Hamiltonian.
* Σ           -> Self-energy.
* β           -> Inverse temperature.
* h           -> Time step.
* solve_order -> Order of the integration, 1 ≤ solve_order ≤ 5.

### Returns
* G -> Contour-ordered Green's function.

See also: [`dyson!`](@ref).
"""
function dyson_timestep!(nkb::I64, G::ℱ{C64}, μ::F64, H::Cf{C64}, Σ::ℱ{C64},
                         β::F64, h::F64; solve_order::I64 = 5,
                         threaded::Bool = false)
    k = solve_order
    nkb > k + 1 ||
        throw(ArgumentError("dyson_timestep!: time step $nkb must be larger than $(k + 1)"))
    _check_dyson_rt(G, Σ, H, k, nkb - 1)
    β > 0.0 || throw(ArgumentError("β must be positive"))
    h > 0.0 || throw(ArgumentError("h must be positive"))

    I = Integrator(k)
    _dyson_timestep_ret!(nkb, G, μ, H, Σ, h, I)
    _dyson_timestep_lmix!(nkb, G, μ, H, Σ, β, h, I; threaded = threaded)
    _dyson_timestep_less!(nkb, G, μ, H, Σ, β, h, I; threaded = threaded)

    return G
end

"""
    dyson!(G::ℱ{C64}, μ::F64, H::Cf{C64}, Σ::ℱ{C64}, β::F64, h::F64;
           solve_order::I64 = 5, matsubara_method::Symbol = :fixpoint,
           force_hermitian::Bool = true)

Solve the Dyson equation for all the components of `G`: Matsubara,
start-up, and then all the remaining real time steps.

### Arguments
* G                -> Contour-ordered Green's function.
* μ                -> Chemical potential.
* H                -> Hamiltonian.
* Σ                -> Self-energy.
* β                -> Inverse temperature.
* h                -> Time step.
* solve_order      -> Order of the integration, 1 ≤ solve_order ≤ 5.
* matsubara_method -> Method for the Matsubara component.
* force_hermitian  -> Whether to symmetrize the Matsubara component.

### Returns
* G -> Contour-ordered Green's function.

See also: [`dyson_mat!`](@ref), [`dyson_start!`](@ref),
[`dyson_timestep!`](@ref).
"""
function dyson!(G::ℱ{C64}, μ::F64, H::Cf{C64}, Σ::ℱ{C64}, β::F64, h::F64;
                solve_order::I64 = 5, matsubara_method::Symbol = :fixpoint,
                force_hermitian::Bool = true)
    k = solve_order
    _check_dyson_rt(G, Σ, H, k, getntime(G) - 1)

    dyson_mat!(G, Σ, μ, H, β; solve_order = k, method = matsubara_method,
               force_hermitian = force_hermitian)
    dyson_start!(G, μ, H, Σ, β, h; solve_order = k)
    for nkb = k+2:getntime(G)
        dyson_timestep!(nkb, G, μ, H, Σ, β, h; solve_order = k)
    end

    return G
end

#=
### *Moving Window* : *Truncated Dyson Equation*
=#

#=
*Remarks* : *Truncated Dyson Equation*

The truncated (moving) Dyson equation determines the leading row
``G(t, t - \delta_j)``, ``\delta_j = 0, \cdots, t_c``, of a moving Green's
function. The memory integrals are restricted to the window
``[t - t_c, t]``, and the Matsubara and mixing components do not exist.
In the following, `G(i,j)` denotes the block `(delti, deltj)` of the
window, i.e., ``G(t - i, t - i - j)``.

**Retarded component**: ``G^{R}(0,0) = -i``. The blocks `j = 1, ⋯, k`
are the solution of a 𝑘 × 𝑘 start-up system with polynomial
differentiation and Gregory weights. The blocks `j = k+1, ⋯, tc` are
obtained independently from

```math
\left[\frac{i}{h}b_0 + \mu - H(t) - h\omega_0\Sigma^{R}(t,t)\right]
G^{R}(t,t-j) = h\sum_{l=1}^{j} w_{jl}\Sigma^{R}(t,t-l)G^{R}(t-l,t-j)
- \frac{i}{h}\sum_{p=1}^{k+1} b_p G^{R}(t-p,t-j).
```

**Lesser component**: the blocks `j = tc, ⋯, k+1` are obtained
independently from the equation of motion in the first time argument
(the source contains ``\Sigma^{R} * G^{<}`` and ``\Sigma^{<} * G^{A}``
over the window), then the blocks `j = k, ⋯, 0` sequentially from the
equation of motion in the second time argument. The order of the second
part must not be changed.

The weights follow [`NESSi`] literally, e.g. ``\omega_0`` is used for the
diagonal term of every column. The mixed initial correlations
``\Sigma^{\rceil} \star G^{\lceil}`` are not represented. Hence the result
agrees with the full Dyson equation only up to the time propagation
error, even if the window covers the whole history.

*References* :

Please see [`NESSi`] `cntr_dyson_moving_impl.hpp`.
=#

"""
    _check_moving_solver(tstp, k, Xs...)

Check the moving objects of the truncated solvers: initialized, the same
cut-off, leading time step `tstp`, matrix size, and `tc ≥ 2k + 2`. It is a
private function.
"""
function _check_moving_solver(tstp::I64, k::I64, X₁, Xs...)
    1 ≤ k ≤ 5 || throw(ArgumentError("solve_order must be in 1:5, got $k"))
    for X in (X₁, Xs...)
        _check_initialized(X)
        X.tc == X₁.tc || throw(ArgumentError("the cut-offs differ"))
        X.tstp == tstp ||
            throw(ArgumentError("tstp = $tstp is not the leading time step $(X.tstp)"))
        getdims(X) == getdims(X₁) ||
            throw(DimensionMismatch("the matrix sizes differ"))
    end
    X₁.tc ≥ 2k + 2 ||
        throw(ArgumentError("the truncated solver of order $k requires tc ≥ $(2k + 2), got $(X₁.tc)"))
    return nothing
end

"""
    _dyson_moving_ret!(G::ℱᵐᵒᵛ{C64}, μ::F64, H::Cfᵐᵒᵛ{C64},
                       Σ::ℱᵐᵒᵛ{C64}, h::F64, I::Integrator)

Retarded component of the truncated Dyson equation at the leading row.
See *Remarks* : *Truncated Dyson Equation*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_dyson_moving_impl.hpp`,
`dyson_timestep_dispatch_ret`.
"""
function _dyson_moving_ret!(G::ℱᵐᵒᵛ{C64}, μ::F64, H::Cfᵐᵒᵛ{C64},
                            Σ::ℱᵐᵒᵛ{C64}, h::F64, I::Integrator;
                            threaded::Bool = false)
    k = I.k
    tc = G.tc
    d = getdims(G)[1]
    Id = Matrix{C64}(LinearAlgebra.I, d, d)
    Gr = G.ret
    Sr = Σ.ret

    # Clear the leading row, G^{R}(t,t) = -i
    fill!(_mrowview(Gr, 0), zero(C64))
    _mblock(Gr, 0, 0) .= -im .* Id
    G₀₀ = _mblock(Gr, 0, 0)

    # Start values G^{R}(t,t-j), j = 1, ⋯, k: ∑_q X_q M_{pq} = Q_p
    M = [zeros(C64, d, d) for _ = 1:k, _ = 1:k]
    Q = [zeros(C64, d, d) for _ = 1:k]
    X = [zeros(C64, d, d) for _ = 1:k]
    for j = 1:k
        # Derivatives
        for l = 0:k
            cweight = im / h * I.PDW[j,l]
            if l == 0
                Q[j] .= Q[j] .- cweight .* G₀₀
            else
                M[j,l] .= M[j,l] .+ cweight .* Id
            end
        end
        # Hamiltonian
        M[j,j] .= M[j,j] .- _fslot(H, j) .+ μ .* Id
        # Integral
        for l = 0:k
            weight = h * I.GIW[j,l]
            if j ≥ l
                S = _mblock(Sr, l, j - l)
            else
                S = _mblock(Sr, j, l - j)'
                weight = -weight
            end
            if l == 0
                mul!(Q[j], G₀₀, S, weight, one(C64))
            else
                M[j,l] .= M[j,l] .- weight .* S
            end
        end
    end
    _solve_block_left!(X, M, Q, _BlockSolver(k, d))
    for j = 1:k
        copyto!(_mblock(Gr, 0, j), X[j])
    end

    # The rest, j = k+1, ⋯, tc: independent 1 × 1 problems M X = qq
    diffw = [I.BDW[p] * im / h for p = 0:k+1]
    mm = (diffw[1] + μ) .* Id .- (h * I.GIW.ω[1]) .* _mblock(Sr, 0, 0) .- _fslot(H, 0)
    _moving_tforeach(k+1:tc, threaded) do js
        _dyson_moving_ret_far!(js, Gr, Sr, mm, diffw, h, I)
    end

    return G
end

"""
    _moving_tforeach(f, r::UnitRange{I64}, threaded::Bool)

Call `f(js)` for the chunks `js` of `r`, in parallel if `threaded` and
more than one thread is available. Every chunk owns its scratch. It is a
private function.
"""
function _moving_tforeach(f, r::UnitRange{I64}, threaded::Bool)
    if _use_threads(threaded) && length(r) > 1
        chunks = _tchunks(r)
        Threads.@threads :static for c in eachindex(chunks)
            f(chunks[c])
        end
    else
        f(r)
    end
    return nothing
end

"""
    _dyson_moving_ret_far!(js, Gr, Sr, mm, diffw, h, I)

Independent retarded blocks ``G^{R}(t,t-j)``, `j ∈ js`, `j > k`, of the
truncated Dyson equation. Only the history rows are read. It is a
private function.
"""
function _dyson_moving_ret_far!(js::UnitRange{I64}, Gr, Sr, mm, diffw, h::F64,
                                I::Integrator)
    k = I.k
    d = size(mm, 1)
    ws1 = _BlockSolver(1, d)
    stemp = zeros(C64, d, d)
    qq = zeros(C64, d, d)
    for j in js
        fill!(stemp, zero(C64))
        if j < 2k + 2
            for l = 1:j
                mul!(stemp, _mblock(Sr, 0, l), _mblock(Gr, l, j - l), I.GIW[j,l], one(C64))
            end
        else
            for l = 1:k
                mul!(stemp, _mblock(Sr, 0, l), _mblock(Gr, l, j - l), I.GIW.ω[l+1], one(C64))
            end
            for l = k+1:j-k-1
                mul!(stemp, _mblock(Sr, 0, l), _mblock(Gr, l, j - l), one(C64), one(C64))
            end
            for l = j-k:j
                mul!(stemp, _mblock(Sr, 0, l), _mblock(Gr, l, j - l), I.GIW.ω[j-l+1], one(C64))
            end
        end
        qq .= h .* stemp
        for p = 1:k+1
            qq .= qq .- diffw[p+1] .* _mblock(Gr, p, j - p)
        end
        _solve_block_right!(_mblock(Gr, 0, j), mm, qq, ws1)
    end

    return nothing
end

"""
    _dyson_moving_less!(G::ℱᵐᵒᵛ{C64}, μ::F64, H::Cfᵐᵒᵛ{C64},
                        Σ::ℱᵐᵒᵛ{C64}, h::F64, I::Integrator)

Lesser component of the truncated Dyson equation at the leading row. The
retarded component of the leading row must be known. See *Remarks* :
*Truncated Dyson Equation*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_dyson_moving_impl.hpp`,
`dyson_timestep_dispatch_les`.
"""
function _dyson_moving_less!(G::ℱᵐᵒᵛ{C64}, μ::F64, H::Cfᵐᵒᵛ{C64},
                             Σ::ℱᵐᵒᵛ{C64}, h::F64, I::Integrator;
                             threaded::Bool = false)
    k = I.k
    tc = G.tc
    d = getdims(G)[1]
    Id = Matrix{C64}(LinearAlgebra.I, d, d)
    Gr, Gl = G.ret, G.less
    Sr, Sl = Σ.ret, Σ.less

    diffw = [I.BDW[p] * im / h for p = 0:k+1]
    fill!(_mrowview(Gl, 0), zero(C64))

    mm = (diffw[1] + μ) .* Id .- (h * I.GIW.ω[1]) .* _mblock(Sr, 0, 0) .- _fslot(H, 0)

    # G^{<}(t,t-j), j = tc, ⋯, k+1, from the first time argument. These
    # blocks are independent, only the history rows are read.
    _moving_tforeach(k+1:tc, threaded) do js
        _dyson_moving_less_far!(js, Gr, Gl, Sr, Sl, mm, diffw, h, I)
    end

    _dyson_moving_less_near!(G, μ, H, Σ, h, I, diffw)

    return G
end

"""
    _dyson_moving_less_far!(js, Gr, Gl, Sr, Sl, mm, diffw, h, I)

Independent lesser blocks ``G^{<}(t,t-j)``, `j ∈ js`, `j > k`, of the
truncated Dyson equation (first time argument). It is a private function.
"""
function _dyson_moving_less_far!(js::UnitRange{I64}, Gr, Gl, Sr, Sl, mm, diffw,
                                 h::F64, I::Integrator)
    k = I.k
    tc = Gr.tc
    d = size(mm, 1)
    ws1 = _BlockSolver(1, d)
    qq = zeros(C64, d, d)
    stemp = zeros(C64, d, d)
    gtemp = zeros(C64, d, d)
    for j in reverse(js)
        # -∫ Σ^{R}(t,t-l) G^{<}(t-l,t-j), l = 1, ⋯, tc
        fill!(stemp, zero(C64))
        for l = 1:tc
            if j ≥ l
                copyto!(gtemp, _mblock(Gl, l, j - l))
            else
                adjoint!(gtemp, _mblock(Gl, j, l - j))
                gtemp .= .-gtemp
            end
            mul!(stemp, _mblock(Sr, 0, l), gtemp, I.GIW[tc,l], one(C64))
        end
        qq .= h .* stemp
        # ∫ Σ^{<}(t,s) G^{A}(s,t-j), s = t - tc + l
        fill!(stemp, zero(C64))
        dl1 = max(tc - j, k)
        for l = 0:dl1
            if tc - l ≥ j
                adjoint!(gtemp, _mblock(Gr, j, tc - l - j))
            else
                gtemp .= .-_mblock(Gr, tc - l, j - tc + l)
            end
            mul!(stemp, _mblock(Sl, 0, tc - l), gtemp, I.GIW[tc-j,l], one(C64))
        end
        qq .= qq .+ h .* stemp
        # Derivative
        for p = 1:k+1
            if j ≥ p
                copyto!(gtemp, _mblock(Gl, p, j - p))
            else
                adjoint!(gtemp, _mblock(Gl, j, p - j))
                gtemp .= .-gtemp
            end
            qq .= qq .- diffw[p+1] .* gtemp
        end
        _solve_block_right!(_mblock(Gl, 0, j), mm, qq, ws1)
    end

    return nothing
end

"""
    _dyson_moving_less_near!(G, μ, H, Σ, h, I, diffw)

Lesser blocks ``G^{<}(t,t-j)``, `j = k, ⋯, 0`, of the truncated Dyson
equation (second time argument). They are solved sequentially. It is a
private function.
"""
function _dyson_moving_less_near!(G::ℱᵐᵒᵛ{C64}, μ::F64, H::Cfᵐᵒᵛ{C64},
                                  Σ::ℱᵐᵒᵛ{C64}, h::F64, I::Integrator, diffw)
    k = I.k
    tc = G.tc
    d = getdims(G)[1]
    Id = Matrix{C64}(LinearAlgebra.I, d, d)
    Gr, Gl = G.ret, G.less
    Sr, Sl = Σ.ret, Σ.less
    ws1 = _BlockSolver(1, d)
    qq = zeros(C64, d, d)
    stemp = zeros(C64, d, d)
    gtemp = zeros(C64, d, d)

    # The sequence is important, the step j needs j+1, ⋯, tc.
    for j = k:-1:0
        # ∫ G^{R}(t,t-l) Σ^{<}(t-l,t-j), l = 0, ⋯, tc
        fill!(stemp, zero(C64))
        for l = 0:tc
            if j > l
                copyto!(gtemp, _mblock(Sl, l, j - l))
            else
                adjoint!(gtemp, _mblock(Sl, j, l - j))
                gtemp .= .-gtemp
            end
            mul!(stemp, _mblock(Gr, 0, l), gtemp, I.GIW[tc,l], one(C64))
        end
        qq .= h .* stemp
        # ∫ G^{<}(t,t-l) Σ^{A}(t-l,t-j), l = tc, ⋯, j+1
        fill!(stemp, zero(C64))
        for l = tc:-1:j+1
            adjoint!(gtemp, _mblock(Sr, j, l - j))
            mul!(stemp, _mblock(Gl, 0, l), gtemp, I.GIW[tc-j,l-j], one(C64))
        end
        qq .= qq .+ h .* stemp
        # Derivative -i d/dt'
        for p = 1:k+1
            qq .= qq .+ diffw[p+1] .* _mblock(Gl, 0, j + p)
        end
        mm₂ = (-diffw[1] + μ) .* Id .- (h * I.GIW.ω[1]) .* _mblock(Sr, j, 0)' .-
              _fslot(H, j)'
        _solve_block_left!(_mblock(Gl, 0, j), mm₂, qq, ws1)
    end

    return nothing
end

"""
    dyson_timestep!(tstp::I64, G::ℱᵐᵒᵛ{C64}, μ::F64, H::Cfᵐᵒᵛ{C64},
                    Σ::ℱᵐᵒᵛ{C64}, h::F64; solve_order::I64 = 5)

Solve the truncated Dyson equation for the leading row (KB time step
`tstp`) of the moving Green's function `G`. `G`, `Σ`, and `H` must be
initialized, with the same cut-off `tc ≥ 2 solve_order + 2` and leading
time step `tstp`. The history rows of `G` are not changed. See *Remarks* :
*Truncated Dyson Equation*.

### Arguments
* tstp        -> KB time step of the leading row.
* G           -> Moving Green's function.
* μ           -> Chemical potential.
* H           -> Moving Hamiltonian.
* Σ           -> Moving self-energy.
* h           -> Time step.
* solve_order -> Order of the integration, 1 ≤ solve_order ≤ 5.

### Returns
* G -> Moving Green's function.

*References* :

Please see [`NESSi`] `cntr_dyson_moving_impl.hpp`, `dyson_timestep`.
"""
function dyson_timestep!(tstp::I64, G::ℱᵐᵒᵛ{C64}, μ::F64, H::Cfᵐᵒᵛ{C64},
                         Σ::ℱᵐᵒᵛ{C64}, h::F64; solve_order::I64 = 5,
                         threaded::Bool = false)
    k = solve_order
    _check_moving_solver(tstp, k, G, Σ)
    _check_initialized(H)
    (H.tc ≥ G.tc && H.tstp == tstp && getdims(H) == getdims(G)) ||
        throw(ArgumentError("H does not fit to the moving Green's function"))
    G === Σ && throw(ArgumentError("G must not be an alias of Σ"))
    h > 0.0 || throw(ArgumentError("h must be positive"))

    I = Integrator(k)
    _dyson_moving_ret!(G, μ, H, Σ, h, I; threaded = threaded)
    _dyson_moving_less!(G, μ, H, Σ, h, I; threaded = threaded)

    return G
end

"""
    kinetic_energy_trunc(tstp::I64, G::ℱᵐᵒᵛ{C64}, D::ℱᵐᵒᵛ{C64},
                         I::Integrator, h::F64)

Truncated kinetic (or correlation) energy integral at the leading row,

```math
E = \\mathrm{Im}\\,\\mathrm{Tr}\\, h\\sum_{l=0}^{t_c} w_l
\\left[G^{R}(t,t-l)^{\\dagger} D^{<}(t,t-l)
- G^{<}(t,t-l)^{\\dagger} D^{R}(t,t-l)\\right],
```

with the Gregory end weights ``\\omega`` at both ends and the weight 1
in between. It requires `tc ≥ 2k + 2`.

*References* :

Please see [`NESSi`] `cntr_dyson_moving_impl.hpp`,
`get_kinetic_trunc_timestep` (without its debug output).
"""
function kinetic_energy_trunc(tstp::I64, G::ℱᵐᵒᵛ{C64}, D::ℱᵐᵒᵛ{C64},
                              I::Integrator, h::F64)
    k = I.k
    _check_moving_solver(tstp, k, G, D)
    tc = G.tc
    d = getdims(G)[1]
    s = zeros(C64, d, d)
    for l = 0:tc
        w = l ≤ k ? I.GIW.ω[l+1] : (l ≥ tc - k ? I.GIW.ω[tc-l+1] : 1.0)
        mul!(s, _mblock(G.ret, 0, l)', _mblock(D.less, 0, l), h * w, one(C64))
        mul!(s, _mblock(G.less, 0, l)', _mblock(D.ret, 0, l), -h * w, one(C64))
    end
    return imag(tr(s))
end
