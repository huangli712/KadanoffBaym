#
# Project : Lavender
# Source  : vie_mat.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Private Service Functions* : *Matsubara VIE2*
=#

"""
    _vie2_mat_fixpoint!(G::ℱ{C64}, F::ℱ{C64}, Q::ℱ{C64}, β::F64,
                        I::Integrator; nfixpoint::I64 = 6)

Solve the Matsubara VIE2 with the fixpoint method. See *Remarks* :
*Volterra Integral Equation*. It is a private function.

### Arguments
* G         -> Contour-ordered Green's function, the solution.
* F         -> Kernel.
* Q         -> Source term.
* β         -> Inverse temperature.
* I         -> An Integrator struct.
* nfixpoint -> Number of the residual corrections.

### Returns
* G -> Contour-ordered Green's function.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_mat_fixpoint_dispatch`.
"""
function _vie2_mat_fixpoint!(G::ℱ{C64}, F::ℱ{C64}, Q::ℱ{C64}, β::F64,
                             I::Integrator; nfixpoint::I64 = 6)
    sign = getsign(G)
    ntau = getntau(G)
    d = getdims(G)[1]

    _matsubara_seed!(G.mat, F.mat, Q.mat, β, sign; pcf = 5, order = 3)

    R = Gᵐᵃᵗ(ntau, d, d)
    dG = Gᵐᵃᵗ(ntau, d, d)
    for _ = 1:nfixpoint
        # R = F * G + G - Q
        conv_mat(R, F.mat, G.mat, I, β, sign)
        R.data .= R.data .+ G.mat.data .- Q.mat.data
        #
        _matsubara_seed!(dG, F.mat, R, β, sign; pcf = 5, order = 3)
        G.mat.data .= G.mat.data .- dG.data
    end

    return G
end


"""
    _vie2_mat_cg!(G::ℱ{C64}, F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64}, β::F64,
                  I::Integrator; maxiter::I64 = 40, tol::F64 = 1e-12,
                  trace = nothing)

Solve the Matsubara VIE2 with the Fletcher-Reeves conjugate gradient
method. See *Remarks* : *Volterra Integral Equation*. If `trace` is a
vector, the quantities `(rsold, c2, alpha, rsnew, beta, G)` of each
iteration are pushed into it, where `G` is a copy of the Matsubara data
after the update. It is a private function.

### Arguments
* G       -> Contour-ordered Green's function, the solution.
* F       -> Kernel.
* Fcc     -> Hermitian conjugate partner of the kernel.
* Q       -> Source term.
* β       -> Inverse temperature.
* I       -> An Integrator struct.
* maxiter -> Maximum number of iterations.
* tol     -> Tolerance of the residual.
* trace   -> `nothing` or a vector to record the iterations.

### Returns
* niter -> Number of the performed iterations.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_mat_steep_dispatch`.
"""
function _vie2_mat_cg!(G::ℱ{C64}, F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64}, β::F64,
                       I::Integrator; maxiter::I64 = 40, tol::F64 = 1e-12,
                       trace = nothing)
    sign = getsign(G)
    ntau = getntau(G)
    d = getdims(G)[1]

    _matsubara_seed!(G.mat, F.mat, Q.mat, β, sign; pcf = 3, order = 3)
    maxiter > 0 || return 0

    R = Gᵐᵃᵗ(ntau, d, d)
    P = Gᵐᵃᵗ(ntau, d, d)
    AP = Gᵐᵃᵗ(ntau, d, d)
    C₁ = Gᵐᵃᵗ(ntau, d, d)
    C₂ = Gᵐᵃᵗ(ntau, d, d)
    #
    # A(X) = X + [F * X + X * Fcc] / 2
    function apply_A!(AX, X)
        conv_mat(C₁, F.mat, X, I, β, sign)
        conv_mat(C₂, X, Fcc.mat, I, β, sign)
        AX.data .= 0.5 .* (C₁.data .+ C₂.data) .+ X.data
    end

    # Initial residual R = Q - A(G)
    apply_A!(R, G.mat)
    R.data .= Q.mat.data .- R.data
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
            βcg = rsnew / rsold
            P.data .= R.data .+ βcg .* P.data
            if !isnothing(trace)
                push!(trace, (rsold = rsold, c2 = c₂, alpha = α, rsnew = rsnew,
                              beta = βcg, G = copy(G.mat.data)))
            end
            rsold = rsnew
            sqrt(rsold) < tol && break
        end
    end

    return niter
end


"""
    _vie2_mat_fourier!(G::ℱ{C64}, F::ℱ{C64}, Q::ℱ{C64}, β::F64;
                       pcf::I64 = 20, order::I64 = 3)

Solve the Matsubara VIE2 in the frequency domain, see *Remarks* :
*Matsubara Seed* in `matsubara_fourier_core.jl`. The public `:fourier`
method uses `pcf = 20`, i.e. 41 copies of the basic frequency window.
The number of imaginary time intervals must be even. It is a private
function.

### Arguments
* G     -> Contour-ordered Green's function, the solution.
* F     -> Kernel.
* Q     -> Source term.
* β     -> Inverse temperature.
* pcf   -> Number of the oversampling copies on each side.
* order -> Order of the boundary correction of the Fourier transform.

### Returns
* G -> Contour-ordered Green's function.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_mat_fourier_dispatch`.
"""
function _vie2_mat_fourier!(G::ℱ{C64}, F::ℱ{C64}, Q::ℱ{C64}, β::F64;
                            pcf::I64 = 20, order::I64 = 3)
    N = getntau(G) - 1
    iseven(N) ||
        throw(ArgumentError("the :fourier method needs an even number of imaginary " *
                            "time intervals (an odd number of points), got $N intervals"))
    _matsubara_seed!(G.mat, F.mat, Q.mat, β, getsign(G); pcf = pcf, order = order)
    return G
end


#=
### *Public API* : *Matsubara VIE2*
=#

"""
    vie2_mat!(G::ℱ{C64}, F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64}, β::F64,
              I::Integrator; method::Symbol = :fixpoint, nfixpoint::I64 = 6,
              maxiter::I64 = 40, tol::F64 = 1e-12)

Solve the Matsubara component of the VIE2 ``[1 + F] * G = Q``. Only the
Matsubara component of `G` is modified. See *Remarks* : *Volterra
Integral Equation*.

### Arguments
* G         -> Contour-ordered Green's function, the solution.
* F         -> Kernel.
* Fcc       -> Hermitian conjugate partner of the kernel.
* Q         -> Source term.
* β         -> Inverse temperature.
* I         -> An Integrator struct.
* method    -> `:fixpoint`, `:cg`, or `:fourier` (`Fcc` is not used).
* nfixpoint -> Number of the residual corrections (`:fixpoint` only).
* maxiter   -> Maximum number of iterations (`:cg` only).
* tol       -> Tolerance of the residual (`:cg` only).

### Returns
* G -> Contour-ordered Green's function.

See also: [`vie2!`](@ref).
"""
function vie2_mat!(G::ℱ{C64}, F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64}, β::F64,
                   I::Integrator; method::Symbol = :fixpoint, nfixpoint::I64 = 6,
                   maxiter::I64 = 40, tol::F64 = 1e-12)
    # Sanity check
    _check_vie2(G, F, Fcc, Q)
    _check_vie2_order(I)
    β > 0.0 || throw(ArgumentError("β must be positive"))
    nfixpoint ≥ 0 || throw(ArgumentError("nfixpoint must be nonnegative"))
    maxiter ≥ 0 || throw(ArgumentError("maxiter must be nonnegative"))

    if method == :fixpoint
        _vie2_mat_fixpoint!(G, F, Q, β, I; nfixpoint = nfixpoint)
    elseif method == :cg
        _vie2_mat_cg!(G, F, Fcc, Q, β, I; maxiter = maxiter, tol = tol)
    elseif method == :fourier
        _vie2_mat_fourier!(G, F, Q, β; pcf = 20, order = 3)
    else
        throw(ArgumentError("vie2_mat!: unknown method :$method, use :fixpoint, :cg, or :fourier"))
    end

    return G
end


