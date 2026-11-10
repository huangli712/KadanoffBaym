#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

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


