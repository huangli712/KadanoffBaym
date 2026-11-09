#
# Project : Lavender
# Source  : vie.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/09
#

#=
*Remarks* : *Volterra Integral Equation*

The Volterra integral equation of the second kind (VIE2) reads

```math
\begin{equation}
[1 + F] * G = Q, \quad F^{cc}(t,t') = F(t',t)^{\dagger},
\end{equation}
```

where `*` denotes the convolution along the Kadanoff-Baym contour. The
functions `G` and `Q` are hermitian, so the equation is equivalent to
``G + G * F^{cc} = Q``. Note that ``F^{cc}`` is the hermitian conjugate
partner of ``F``, not its element-wise complex conjugate. The equation
is solved in three steps:

1. Matsubara component, `vie2_mat!()`.

2. Start-up of the real time components at the first 𝑘 + 1 time steps,
   `vie2_start!()`.

3. Time stepping of the real time components, `vie2_timestep!()`.

The full solution is obtained by `vie2!()`.

**Matsubara component**: Two methods are supported.

* `:fixpoint` -> Start from the frequency-domain solution (`pcf = 5`),
    then apply the preconditioned residual corrections ``R = F*G + G - Q``,
    ``G \to G - P^{-1}R`` for `nfixpoint` (default 6) times.

* `:cg` -> Start from the frequency-domain solution (`pcf = 3`), then
    apply the Fletcher-Reeves conjugate gradient method to the hermitian
    operator ``A(X) = X + [F*X + X*F^{cc}]/2``. The inner product is
    ``\langle X,Y \rangle = \sum_r w_r \sum_{ab} \mathrm{Re}[(X^{\dagger}_r
    Y_r)_{ab}]``, where ``w_r`` are the Gregory end weights.

The method `:fourier` is reserved for Phase 7.

**Real time components**: In the following, the time indices `n`, `j`,
`l`, `m` follow the convention of [`NESSi`], i.e., ``t_n = nh`` with
`n = 0, 1, ⋯`. The corresponding index of `KadanoffBaym` is `n + 1`. Let
``I_{jnl}`` be the polynomial integration weights, ``\Omega_{nl}`` the
Gregory weights, and ``\omega_l`` the Gregory end weights.

* Retarded component: In the start-up procedure, for each `j = 0, ⋯,
    k - 1`, the unknowns ``G^{R}(n,j)`` (`n = j + 1, ⋯, k`) satisfy

```math
\begin{equation}
G^{R}(n,j) + h\sum^{k}_{l=0} I_{jnl} F^{R}(n,l) G^{R}(l,j) = Q^{R}(n,j).
\end{equation}
```

    In the time stepping procedure, the right action equation
    ``G^{R} + G^{R} * F^{cc,R} = Q^{R}`` is solved by marching backward
    in ``t'``. Hence the kernel of the retarded component is ``F^{cc}``.

* Left-mixing component: For each ``\tau``,

```math
\begin{equation}
G^{\rceil}(n,\tau) + h\sum_{j} \Omega_{nj} F^{R}(n,j) G^{\rceil}(j,\tau)
    = Q^{\rceil}(n,\tau) - [F^{\rceil} \star G^{M}](n,\tau).
\end{equation}
```

* Lesser component: The source term is ``-[F^{\rceil} \star G^{\lceil}
    + F^{<} * G^{A}](j,n)``, while the term ``F^{R} * G^{<}`` belongs to
    the linear system, which is solved by marching forward in `j`.

The order of the components is fixed: retarded, left-mixing, and then
lesser.

*References* :

Please see [`NESSi`] Section `6` and `cntr_vie2_impl.hpp`.
=#

#=
### *Private Service Functions* : *Sanity Checks*
=#

"""
    _check_vie2(G::ℱ{C64}, F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64})

Check the compatibility of the arguments of the VIE2 solvers. It is a
private function.
"""
function _check_vie2(G::ℱ{C64}, F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64})
    getsign(G) in (FERMI, BOSE) || throw(ArgumentError("unknown statistics of G"))
    for (X, name) in ((F, "F"), (Fcc, "Fcc"), (Q, "Q"))
        getsign(X) == getsign(G) ||
            throw(ArgumentError("G and $name must have the same statistics"))
        getdims(X) == getdims(G) ||
            throw(DimensionMismatch("G and $name have different sizes"))
        getntau(X) == getntau(G) ||
            throw(DimensionMismatch("G and $name have different imaginary time grids"))
        getntime(X) == getntime(G) ||
            throw(DimensionMismatch("G and $name have different real time grids"))
    end
    d₁, d₂ = getdims(G)
    d₁ == d₂ || throw(DimensionMismatch("G must be square"))
    (G === F || G === Fcc || G === Q) &&
        throw(ArgumentError("G must not be an alias of F, Fcc, or Q"))
end

"""
    _check_vie2_order(I::Integrator)

Check the order of the integrator. It is a private function.
"""
function _check_vie2_order(I::Integrator)
    I.k in 1:5 || throw(ArgumentError("the order of the integrator must be in 1:5, got $(I.k)"))
end

"""
    _kernel_ret(F, Fcc, n::I64, l::I64)

Return ``F^{R}(n,l)`` (NESSi indices) and the sign `1` if `n ≥ l`.
Otherwise, ``F^{cc,R}(l,n)^{\\dagger}`` and the sign `-1` are returned,
i.e., ``F^{R}(n,l) = -F^{cc,R}(l,n)^{\\dagger}``. It is a private
function.
"""
@inline function _kernel_ret(F, Fcc, n::I64, l::I64)
    if n ≥ l
        return _ret_blk(F, n, l), 1.0
    else
        return _ret_blk(Fcc, l, n)', -1.0
    end
end

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
* method    -> `:fixpoint`, `:cg`, or `:fourier` (Phase 7).
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
        throw(ArgumentError("vie2_mat!: method :fourier is deferred to Phase 7"))
    else
        throw(ArgumentError("vie2_mat!: unknown method :$method, use :fixpoint, :cg, or :fourier"))
    end

    return G
end

#=
### *Private Service Functions* : *Retarded Component*
=#

"""
    _vie2_start_ret!(G, F, Fcc, Q, h::F64, I::Integrator)

Start-up procedure of the retarded component, i.e., ``G^{R}(n,j)`` for
`0 ≤ j ≤ n ≤ k`. See *Remarks* : *Volterra Integral Equation*. It is a
private function.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_start_ret`.
"""
function _vie2_start_ret!(G, F, Fcc, Q, h::F64, I::Integrator)
    k = I.k
    d = getdims(G)[1]
    Id = Matrix{C64}(LinearAlgebra.I, d, d)
    gtemp = zeros(C64, d, d)
    stemp = zeros(C64, d, d)

    # G^{R}(n,n) = Q^{R}(n,n)
    for n = 0:k
        _ret_blk(G, n, n) .= _ret_blk(Q, n, n)
    end

    # Determine G^{R}(n,j), n = j+1, ⋯, k
    for j = 0:k-1
        ws = _BlockSolver(k - j, d)
        fill!(ws.Mbig, zero(C64))
        fill!(ws.Qbig, zero(C64))
        #
        for n = j+1:k
            p = n - j
            Qₚ = _qblock(ws, p)
            Qₚ .= Qₚ .+ _ret_blk(Q, n, j)
            for l = 0:k
                if l ≤ j
                    # Known G^{R}(l,j) = -G^{R}(j,l)†, it goes into Q
                    gtemp .= .-(_ret_blk(G, j, l))'
                    mul!(stemp, _ret_blk(F, n, l), gtemp)
                    Qₚ .= Qₚ .- (h * I.XIW[j,n,l]) .* stemp
                else
                    # Unknown G^{R}(l,j), it goes into M
                    Mₚₒ = _mblock(ws, p, l - j)
                    fill!(Mₚₒ, zero(C64))
                    if l == n
                        Mₚₒ .= Mₚₒ .+ Id
                    end
                    S, s = _kernel_ret(F, Fcc, n, l)
                    Mₚₒ .= Mₚₒ .+ (s * h * I.XIW[j,n,l]) .* S
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
    _vie2_timestep_ret!(tstp::I64, G, K, Kcc, Q, h::F64, I::Integrator)

Time stepping of the retarded component, i.e., ``G^{R}(n,j)`` for
`0 ≤ j ≤ n`, where `n = tstp - 1 > k`. The right action equation
``G^{R} + G^{R} * K^{R} = Q^{R}`` is solved. Note that the caller passes
the kernel ``K = F^{cc}`` and its partner ``K^{cc} = F``. See *Remarks* :
*Volterra Integral Equation*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_timestep_ret`.
"""
function _vie2_timestep_ret!(tstp::I64, G, K, Kcc, Q, h::F64, I::Integrator)
    k = I.k
    n = tstp - 1
    @assert n > k
    d = getdims(G)[1]
    Id = Matrix{C64}(LinearAlgebra.I, d, d)

    # Clear the time step, G^{R}(n,n) = Q^{R}(n,n)
    for j = 0:n
        fill!(_ret_blk(G, n, j), zero(C64))
    end
    _ret_blk(G, n, n) .= _ret_blk(Q, n, n)
    Gₙₙ = _ret_blk(G, n, n)

    # Start values G^{R}(n,n-j), j = 1, ⋯, k: solve a k × k problem
    # ∑_q X_q M_{pq} = R_p with X_q = G^{R}(n,n-q)
    wsk = _BlockSolver(k, d)
    M = [zeros(C64, d, d) for _ = 1:k, _ = 1:k]
    R = [zeros(C64, d, d) for _ = 1:k]
    X = [_ret_blk(G, n, n - q) for q = 1:k]
    for j = 1:k
        R[j] .= _ret_blk(Q, n, n - j)
        M[j,j] .= M[j,j] .+ Id
        for l = 0:k
            weight = -h * I.GIW[j,l]
            if l ≤ j
                S = _ret_blk(K, n - l, n - j)
            else
                S = _ret_blk(Kcc, n - j, n - l)'
                weight = -weight
            end
            if l == 0
                mul!(R[j], Gₙₙ, S, weight, one(C64))
            else
                M[j,l] .= M[j,l] .- weight .* S
            end
        end
    end
    _solve_block_left!(X, M, R, wsk)

    # Contribution of G^{R}(n,m), m = n-k, ⋯, n, to the convolution
    # -∫ dm G^{R}(n,m) K^{R}(m,j), stored into qq(j) without the factor h
    qq = zeros(C64, d, d, n + 1)
    for m = n-k:n
        Gₙₘ = _ret_blk(G, n, m)
        for j = 0:n-k-1
            mul!(view(qq, :, :, j + 1), Gₙₘ, _ret_blk(K, m, j), -I.GIW[n-j,n-m], one(C64))
        end
    end

    # The rest: t' = n-l, l = k+1, ⋯, n
    ws1 = _BlockSolver(1, d)
    w₀ = h * I.GIW.ω[1]
    mm = zeros(C64, d, d)
    for l = k+1:n
        j = n - l
        qqⱼ = view(qq, :, :, j + 1)
        qqⱼ .= h .* qqⱼ .+ _ret_blk(Q, n, j)
        # G^{R}(n,j) [1 + h ω₀ K^{R}(j,j)] = qq(j)
        mm .= Id .+ w₀ .* _ret_blk(K, j, j)
        Gₙⱼ = _ret_blk(G, n, j)
        _solve_block_left!(Gₙⱼ, mm, qqⱼ, ws1)
        #
        # Contribution of G^{R}(n,j) to -∫ dm G^{R}(n,m) K^{R}(m,j₂), j₂ < j
        for j₂ = 0:j-1
            mul!(view(qq, :, :, j₂ + 1), Gₙⱼ, _ret_blk(K, j, j₂), -I.GIW[n-j₂,n-j], one(C64))
        end
    end

    return G
end

#=
### *Private Service Functions* : *Left-mixing Component*
=#

"""
    _vie2_start_tv!(G, F, Fcc, Q, β::F64, h::F64, I::Integrator)

Start-up procedure of the left-mixing component, i.e., ``G^{\\rceil}(n,\\tau)``
for `0 ≤ n ≤ k`. The Matsubara component of `G` and the retarded
component at the first 𝑘 + 1 time steps must be known. See *Remarks* :
*Volterra Integral Equation*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_start_tv`.
"""
function _vie2_start_tv!(G, F, Fcc, Q, β::F64, h::F64, I::Integrator)
    k = I.k
    d = getdims(G)[1]
    ntau = getntau(G)
    sign = getsign(G)
    δτ = β / (ntau - 1)
    Id = Matrix{C64}(LinearAlgebra.I, d, d)

    # Source term: -δτ ∫ dτ' F^{⌉}(n,τ') G^{M}(τ' - τ), n = 0, ⋯, k
    cws = _ConvWorkspace(F.ret.data, G.ret.data)
    for n = 0:k
        for m = 1:ntau
            fill!(_tv_blk(G, n, m - 1), zero(C64))
        end
        _conv_lmix_mat_kernel!(_LmixRow(G.lmix, n + 1), n + 1, F.lmix, nothing,
                               G.mat, I, sign, -δτ, cws)
    end

    # Block system ∑_q M_{pq} X_q = Q_p, X_q = G^{⌉}(q,τ) for all τ. The
    # matrix M does not depend on τ.
    ws = _BlockSolver(k + 1, d, d * ntau)
    fill!(ws.Mbig, zero(C64))
    for n = 0:k
        Qₙ = _qblock(ws, n + 1)
        for m = 1:ntau
            cols = (m - 1) * d + 1 : m * d
            view(Qₙ, :, cols) .= _tv_blk(G, n, m - 1) .+ _tv_blk(Q, n, m - 1)
        end
        Mₙₙ = _mblock(ws, n + 1, n + 1)
        Mₙₙ .= Mₙₙ .+ Id
        for j = 0:k
            # K = F^{R}(n,j) for n ≥ j, or -Fcc^{R}(j,n)† for n < j
            S, s = _kernel_ret(F, Fcc, n, j)
            Mₙⱼ = _mblock(ws, n + 1, j + 1)
            Mₙⱼ .= Mₙⱼ .+ (s * h * I.GIW[n,j]) .* S
        end
    end
    _solve_block!(ws)
    for n = 0:k
        Xₙ = _xblock(ws, n + 1)
        for m = 1:ntau
            _tv_blk(G, n, m - 1) .= view(Xₙ, :, (m - 1) * d + 1 : m * d)
        end
    end

    return G
end

"""
    _vie2_timestep_tv!(tstp::I64, G, F, Fcc, Q, β::F64, h::F64, I::Integrator)

Time stepping of the left-mixing component, i.e., ``G^{\\rceil}(n,\\tau)``,
where `n = tstp - 1 > k`. The retarded component at time step `n` must be
known. See *Remarks* : *Volterra Integral Equation*. It is a private
function.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_timestep_tv`.
"""
function _vie2_timestep_tv!(tstp::I64, G, F, Fcc, Q, β::F64, h::F64, I::Integrator)
    k = I.k
    n = tstp - 1
    @assert n > k
    d = getdims(G)[1]
    ntau = getntau(G)
    Id = Matrix{C64}(LinearAlgebra.I, d, d)

    # Clear the current row first, so that its old values do not enter
    # the source term
    for m = 1:ntau
        fill!(_block(G.lmix, tstp, m), zero(C64))
    end

    # Convolution [F * G]^{⌉}(n,τ) without the implicit term
    W = 𝒻(tstp, ntau, d, d, getsign(G))
    incr_convolution_tv!(tstp, W, F, Fcc, nothing, G, G, I, β, h, one(C64))

    # [1 + h Ω_{n0} F^{R}(n,n)] G^{⌉}(n,τ) = Q^{⌉}(n,τ) - [F * G]^{⌉}(n,τ)
    ws = _BlockSolver(1, d, d * ntau)
    for m = 1:ntau
        Qₘ = view(ws.Qbig, :, (m - 1) * d + 1 : m * d)
        Qₘ .= _tv_blk(Q, n, m - 1) .- W.lmix.data[m]
    end
    ws.Mbig .= Id .+ (h * I.GIW[n,0]) .* _ret_blk(F, n, n)
    _solve_block!(ws)
    for m = 1:ntau
        _block(G.lmix, tstp, m) .= view(ws.Xbig, :, (m - 1) * d + 1 : m * d)
    end

    return G
end

#=
### *Private Service Functions* : *Lesser Component*
=#

"""
    _vie2_less_core!(n::I64, gles::AbstractVector, G, F, Fcc, Q,
                     h::F64, I::Integrator)

Solve the lesser component ``G^{<}(j,n)`` for `j = 0, ⋯, n` (NESSi
indices). On entry, `gles[j+1]` (`j = 0, ⋯, max(n,k)`) contains the source
term ``-[F^{\\rceil} \\star G^{\\lceil} + F^{<} * G^{A}](j,n)``. On exit,
`gles[j+1]` (`j = 0, ⋯, n`) contains the solution, which is also written
into `G`. It is a private function.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_timestep_les`.
"""
function _vie2_less_core!(n::I64, gles::AbstractVector, G, F, Fcc, Q,
                          h::F64, I::Integrator)
    k = I.k
    d = getdims(G)[1]
    Id = Matrix{C64}(LinearAlgebra.I, d, d)
    @assert length(gles) ≥ max(n, k) + 1

    # Start: j = 0, ⋯, k. Solve a (k+1) × (k+1) problem M X = R,
    # X_q = G^{<}(q,n). For j > n, Q^{<}(j,n) = -Q^{<}(n,j)†.
    ws = _BlockSolver(k + 1, d)
    fill!(ws.Mbig, zero(C64))
    for j = 0:k
        Rⱼ = _qblock(ws, j + 1)
        Rⱼ .= gles[j+1]
        if j ≤ n
            Rⱼ .= Rⱼ .+ _les_blk(Q, j, n)
        else
            Rⱼ .= Rⱼ .- (_les_blk(Q, n, j))'
        end
        Mⱼⱼ = _mblock(ws, j + 1, j + 1)
        Mⱼⱼ .= Mⱼⱼ .+ Id
        # Integral F^{R} * G^{<}
        for m = 0:k
            S, s = _kernel_ret(F, Fcc, j, m)
            Mⱼₘ = _mblock(ws, j + 1, m + 1)
            Mⱼₘ .= Mⱼₘ .+ (s * h * I.GIW[j,m]) .* S
        end
    end
    _solve_block!(ws)
    for j = 0:k
        gles[j+1] .= _xblock(ws, j + 1)
    end

    # Time stepping: j = k+1, ⋯, n
    ws1 = _BlockSolver(1, d)
    qq = zeros(C64, d, d)
    mm = zeros(C64, d, d)
    for j = k+1:n
        qq .= gles[j+1] .+ _les_blk(Q, j, n)
        mm .= Id .+ (h * I.GIW[j,j]) .* _ret_blk(F, j, j)
        for m = 0:j-1
            mul!(qq, _ret_blk(F, j, m), gles[m+1], -h * I.GIW[j,m], one(C64))
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
    _vie2_start_less!(G, F, Fcc, Q, β::F64, h::F64, I::Integrator)

Start-up procedure of the lesser component, i.e., ``G^{<}(j,n)`` for
`0 ≤ j ≤ n ≤ k`. The retarded and left-mixing components at the first
𝑘 + 1 time steps must be known. See *Remarks* : *Volterra Integral
Equation*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_start_les`.
"""
function _vie2_start_less!(G, F, Fcc, Q, β::F64, h::F64, I::Integrator)
    k = I.k
    d = getdims(G)[1]
    ntau = getntau(G)
    cws = _ConvWorkspace(F.ret.data, G.ret.data)

    # NESSi n = 0, ⋯, k - 1: the private scratch-target convolutions
    # provide the source blocks j = 0, ⋯, k, including j > n.
    buf = [zeros(C64, d, d) for _ = 1:k+1]
    for n = 0:k-1
        tstp = n + 1
        foreach(b -> fill!(b, zero(C64)), buf)
        _incr_convolution_lmix_rmix!(buf, tstp, F, Fcc, nothing, G, G, I, β, h, -one(C64), cws)
        _incr_convolution_less_adv!(buf, tstp, F, Fcc, nothing, G, G, I, β, h, -one(C64), cws)
        _vie2_less_core!(n, buf, G, F, Fcc, Q, h, I)
    end

    # NESSi n = k: the public slice-target convolutions, j = 0, ⋯, k
    W = 𝒻(k + 1, ntau, d, d, getsign(G))
    incr_convolution_lmix_rmix!(k + 1, W, F, Fcc, nothing, G, G, I, β, h, -one(C64))
    incr_convolution_less_adv!(k + 1, W, F, Fcc, nothing, G, G, I, β, h, -one(C64))
    _vie2_less_core!(k, W.less.data, G, F, Fcc, Q, h, I)

    return G
end

"""
    _vie2_timestep_less!(tstp::I64, G, F, Fcc, Q, β::F64, h::F64,
                         I::Integrator)

Time stepping of the lesser component, i.e., ``G^{<}(j,n)`` for
`0 ≤ j ≤ n`, where `n = tstp - 1 > k`. The retarded and left-mixing
components at time step `n` must be known. The source term contains only
``F^{\\rceil} \\star G^{\\lceil}`` and ``F^{<} * G^{A}``. See *Remarks* :
*Volterra Integral Equation*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_timestep_les`.
"""
function _vie2_timestep_less!(tstp::I64, G, F, Fcc, Q, β::F64, h::F64,
                              I::Integrator)
    n = tstp - 1
    @assert n > I.k
    d = getdims(G)[1]
    ntau = getntau(G)

    W = 𝒻(tstp, ntau, d, d, getsign(G))
    incr_convolution_lmix_rmix!(tstp, W, F, Fcc, nothing, G, G, I, β, h, -one(C64))
    incr_convolution_less_adv!(tstp, W, F, Fcc, nothing, G, G, I, β, h, -one(C64))
    _vie2_less_core!(n, W.less.data, G, F, Fcc, Q, h, I)

    return G
end

#=
### *Private Service Functions* : *Zeroth Order Bootstrap*
=#

"""
    _vie2_bootstrap!(G, F, Q, β::F64)

Solve the VIE2 at the first real time step (`tstp = 1`, NESSi `n = 0`)
with the zeroth order integrator of [`NESSi`], i.e., the trapezoidal
rule along the imaginary time axis. Only the data at `t = 0` are used.
The Matsubara component of `G` must be known. It is a private function.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_timestep` with `n = 0`,
which calls `vie2_start` with `integration::I<T>(0)`.
"""
function _vie2_bootstrap!(G, F, Q, β::F64)
    d = getdims(G)[1]
    N = getntau(G) - 1
    sign = getsign(G)
    δτ = β / N
    c₁ = zeros(C64, d, d)
    c₂ = zeros(C64, d, d)

    # Retarded component, G^{R}(0,0) = Q^{R}(0,0)
    _ret_blk(G, 0, 0) .= _ret_blk(Q, 0, 0)

    # Left-mixing component, G^{⌉}(0,τ) = Q^{⌉}(0,τ) - δτ ∫ F^{⌉} G^{M}
    tw(l, a, b) = (l == a || l == b) ? 0.5 : 1.0
    for m = 0:N
        fill!(c₁, zero(C64))
        fill!(c₂, zero(C64))
        # 0 ⋯ τ
        if m > 0
            for l = 0:m
                mul!(c₁, _tv_blk(F, 0, m - l), _block(G.mat, N - l + 1), tw(l, 0, m), one(C64))
            end
        end
        # τ ⋯ β
        if m < N
            for l = m:N
                mul!(c₂, _tv_blk(F, 0, l), _block(G.mat, l - m + 1), tw(l, m, N), one(C64))
            end
        end
        _tv_blk(G, 0, m) .= _tv_blk(Q, 0, m) .- δτ .* (c₂ .+ sign .* c₁)
    end

    # Lesser component, G^{<}(0,0) = Q^{<}(0,0) - [F^{⌉} ⋆ G^{⌈}](0,0),
    # where G^{⌈}(τ,0) = -ξ G^{⌉}(0,β-τ)†
    fill!(c₁, zero(C64))
    for m = 0:N
        mul!(c₁, _tv_blk(F, 0, m), _tv_blk(G, 0, N - m)', tw(m, 0, N), one(C64))
    end
    _les_blk(G, 0, 0) .= _les_blk(Q, 0, 0) .- (im * δτ * sign) .* c₁

    return G
end

#=
### *Public API* : *Real Time VIE2*
=#

"""
    vie2_start!(G::ℱ{C64}, F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64},
                I::Integrator, β::F64, h::F64)

Solve the VIE2 ``[1 + F] * G = Q`` for the first `k + 1` real time steps
(`tstp = 1, ⋯, k + 1`), where `k = I.k`. The Matsubara component of `G`
must be known. The components are solved in the order: retarded,
left-mixing, and lesser.

### Arguments
* G   -> Contour-ordered Green's function, the solution.
* F   -> Kernel.
* Fcc -> Hermitian conjugate partner of the kernel.
* Q   -> Source term.
* I   -> An Integrator struct.
* β   -> Inverse temperature.
* h   -> Time step.

### Returns
* G -> Contour-ordered Green's function.

See also: [`vie2!`](@ref).
"""
function vie2_start!(G::ℱ{C64}, F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64},
                     I::Integrator, β::F64, h::F64)
    _check_vie2(G, F, Fcc, Q)
    _check_vie2_order(I)
    k = I.k
    getntime(G) ≥ k + 1 ||
        throw(ArgumentError("G must have at least $(k + 1) real time steps"))
    β > 0.0 || throw(ArgumentError("β must be positive"))
    h > 0.0 || throw(ArgumentError("h must be positive"))

    _vie2_start_ret!(G, F, Fcc, Q, h, I)
    _vie2_start_tv!(G, F, Fcc, Q, β, h, I)
    _vie2_start_less!(G, F, Fcc, Q, β, h, I)

    return G
end

"""
    vie2_timestep!(tstp::I64, G::ℱ{C64}, F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64},
                   I::Integrator, β::F64, h::F64; method::Symbol = :fixpoint)

Solve the VIE2 ``[1 + F] * G = Q`` at the time step `tstp`. Let
`k = I.k`. Following [`NESSi`], the time steps are treated as follows:

* `tstp = 0` -> The Matsubara component, see [`vie2_mat!`](@ref).
* `tstp = 1` -> The zeroth order start-up at `t = 0`.
* `2 ≤ tstp ≤ k + 1` -> The start-up of order `tstp - 1`, i.e., the
    time steps `1, ⋯, tstp` are (re)calculated.
* `tstp ≥ k + 2` -> The time stepping. The previous time steps must be
    known.

### Arguments
* tstp   -> Index of the time step.
* G      -> Contour-ordered Green's function, the solution.
* F      -> Kernel.
* Fcc    -> Hermitian conjugate partner of the kernel.
* Q      -> Source term.
* I      -> An Integrator struct.
* β      -> Inverse temperature.
* h      -> Time step.
* method -> Method for the Matsubara component.

### Returns
* G -> Contour-ordered Green's function.

See also: [`vie2!`](@ref).
"""
function vie2_timestep!(tstp::I64, G::ℱ{C64}, F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64},
                        I::Integrator, β::F64, h::F64; method::Symbol = :fixpoint)
    _check_vie2(G, F, Fcc, Q)
    _check_vie2_order(I)
    0 ≤ tstp ≤ getntime(G) ||
        throw(BoundsError(G, tstp))
    β > 0.0 || throw(ArgumentError("β must be positive"))
    h > 0.0 || throw(ArgumentError("h must be positive"))
    k = I.k

    if tstp == 0
        vie2_mat!(G, F, Fcc, Q, β, I; method = method)
    elseif tstp == 1
        _vie2_bootstrap!(G, F, Q, β)
    elseif tstp ≤ k + 1
        vie2_start!(G, F, Fcc, Q, Integrator(tstp - 1), β, h)
    else
        # The retarded component is solved from the right action
        # equation, so F and Fcc are exchanged.
        _vie2_timestep_ret!(tstp, G, Fcc, F, Q, h, I)
        _vie2_timestep_tv!(tstp, G, F, Fcc, Q, β, h, I)
        _vie2_timestep_less!(tstp, G, F, Fcc, Q, β, h, I)
    end

    return G
end

"""
    vie2!(G::ℱ{C64}, F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64},
          I::Integrator, β::F64, h::F64; method::Symbol = :fixpoint)

Solve the VIE2 ``[1 + F] * G = Q`` for all the components of `G`:
Matsubara, start-up, and then all the remaining real time steps.

### Arguments
* G      -> Contour-ordered Green's function, the solution.
* F      -> Kernel.
* Fcc    -> Hermitian conjugate partner of the kernel.
* Q      -> Source term.
* I      -> An Integrator struct.
* β      -> Inverse temperature.
* h      -> Time step.
* method -> Method for the Matsubara component.

### Returns
* G -> Contour-ordered Green's function.

See also: [`vie2_mat!`](@ref), [`vie2_start!`](@ref),
[`vie2_timestep!`](@ref).
"""
function vie2!(G::ℱ{C64}, F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64},
               I::Integrator, β::F64, h::F64; method::Symbol = :fixpoint)
    k = I.k
    vie2_mat!(G, F, Fcc, Q, β, I; method = method)
    vie2_start!(G, F, Fcc, Q, I, β, h)
    for tstp = k+2:getntime(G)
        vie2_timestep!(tstp, G, F, Fcc, Q, I, β, h)
    end

    return G
end

#=
*Remarks* : *VIE2 With Singular Kernel*

Let the kernel contain a singular term, ``F(t,t') + F_{\delta}(t)
\delta_{\mathcal{C}}(t,t')``, and the source term ``Q(t,t') +
Q_{\delta}(t)\delta_{\mathcal{C}}(t,t')``. Let ``D(t) = 1 + F_{\delta}(t)``.
The singular part of the solution is ``G_{\delta} = D^{-1} Q_{\delta}``,
and the regular part satisfies the ordinary VIE2 with

```math
\begin{equation}
F' = D^{-1} F, \quad F'^{cc} = F^{cc} D^{-1}, \quad
Q' = D^{-1} [Q - F G_{\delta}].
\end{equation}
```

Only the data at time step `i` of ``F'``, ``F'^{cc}``, and ``Q'`` are
needed to transform time step `i`, together with ``D^{-1}`` and
``G_{\delta}`` at the time steps ``\le i`` and at the Matsubara axis.
Hence the transformed functions are kept in a reusable workspace
(`VIE2SinWorkspace`), and each call transforms only the time steps which
may have changed:

* `tstp = 0` -> The Matsubara axis.

* `1 ≤ tstp ≤ k + 1` -> The Matsubara axis and the time steps
    `1, ⋯, k + 1` (all of them are involved in the start-up).

* `tstp ≥ k + 2` -> The time step `tstp`, and the earlier time steps
    which have not been transformed yet.

The time steps after the current one are invalidated. The arithmetic per
time step is the same as that of [`NESSi`], which transforms the time
steps `0, ⋯, n₂` of full copies of `F`, `Fcc`, and `Q` in every call.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_timestep_sin`.
=#

"""
    VIE2SinWorkspace

Reusable workspace of `vie2_timestep_sin!()`. It keeps the transformed
kernels ``F' = D^{-1}F``, ``F'^{cc} = F^{cc}D^{-1}``, and the transformed
source term ``Q' = D^{-1}[Q - FG_{\\delta}]`` at the time steps `0, ⋯,
ndone`, where `ndone = -1` means that nothing is kept. See *Remarks* :
*VIE2 With Singular Kernel*.

### Members
* F     -> Transformed kernel ``F'``.
* Fcc   -> Transformed kernel ``F'^{cc}``.
* Q     -> Transformed source term ``Q'``.
* FG    -> Work space for ``F' G_{\\delta}``.
* Finv  -> ``D^{-1}``.
* ndone -> Last time step kept (-1 for none).
* src   -> Objects `(Gsin, F, Fcc, Q, Fsin, Qsin)` of the last call.

See also: [`vie2_timestep_sin!`](@ref).
"""
mutable struct VIE2SinWorkspace
    F     :: ℱ{C64}
    Fcc   :: ℱ{C64}
    Q     :: ℱ{C64}
    FG    :: ℱ{C64}
    Finv  :: Cf{C64}
    ndone :: I64
    src   :: Any
end

"""
    VIE2SinWorkspace(G::ℱ{C64})

Create a workspace for `vie2_timestep_sin!()`, which is compatible with
the contour-ordered Green's function `G`.

See also: [`vie2_timestep_sin!`](@ref).
"""
function VIE2SinWorkspace(G::ℱ{C64})
    d₁, d₂ = getdims(G)
    C = Cn(getntime(G), getntau(G), d₁, d₂, 1.0, 1.0)
    s = getsign(G)
    VIE2SinWorkspace(ℱ(C, s), ℱ(C, s), ℱ(C, s), ℱ(C, s),
                     Cf(getntime(G), d₁, d₂), -1, nothing)
end

"""
    _vie2_sin_transform!(ws::VIE2SinWorkspace, i::I64, Gsin::Cf{C64},
                         F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64})

Transform the time step `i` of the kernels and the source term, i.e.,
``F' = D^{-1}F``, ``F'^{cc} = F^{cc}D^{-1}``, and ``Q' = D^{-1}[Q -
FG_{\\delta}]``. `ws.Finv` and `Gsin` must be known at the time steps
`0, ⋯, i`. It is a private function.
"""
function _vie2_sin_transform!(ws::VIE2SinWorkspace, i::I64, Gsin::Cf{C64},
                              F::ℱ{C64}, Fcc::ℱ{C64}, Q::ℱ{C64})
    memcpy!(F, ws.F, i)
    smul!(ws.Finv, ws.F, i)
    memcpy!(Fcc, ws.Fcc, i)
    smul!(ws.Fcc, ws.Finv, i)
    memcpy!(Q, ws.Q, i)
    smul!(ws.Finv, ws.Q, i)
    # Q' = D⁻¹ Q - F' Gsin
    memcpy!(ws.F, ws.FG, i)
    smul!(ws.FG, Gsin, i)
    incr!(ws.Q, ws.FG, i, -one(C64))
end

"""
    vie2_timestep_sin!(tstp::I64, G::ℱ{C64}, Gsin::Cf{C64},
                       F::ℱ{C64}, Fcc::ℱ{C64}, Fsin::Cf{C64},
                       Q::ℱ{C64}, Qsin::Cf{C64},
                       I::Integrator, β::F64, h::F64; method::Symbol = :fixpoint,
                       ws::Union{Nothing,VIE2SinWorkspace} = nothing)

Solve the VIE2 with singular kernel at the time step `tstp`. See
*Remarks* : *VIE2 With Singular Kernel*. The singular part of the solution
is written into `Gsin` at the time steps which are transformed in this
call (at most `0, ⋯, n₂`, where `n₂ = k + 1` for `1 ≤ tstp ≤ k + 1`,
and `n₂ = tstp` otherwise). `Fsin` must be hermitian.

For time stepping, pass the same workspace `ws` (see `VIE2SinWorkspace`)
to all the calls. Then the transformed functions are reused, and the
inputs `F`, `Fcc`, `Q`, `Fsin`, and `Qsin` at the time steps before
`tstp` must not be changed once `tstp ≥ k + 2` is reached (the time step
`tstp` itself may be changed between repeated calls). If other objects
are passed for `Gsin` or the inputs, the workspace is reset. Without `ws`, a temporary
workspace is created and all the time steps `0, ⋯, n₂` are transformed.

### Arguments
* tstp   -> Index of the time step.
* G      -> Contour-ordered Green's function, the regular part of the solution.
* Gsin   -> Singular part of the solution.
* F      -> Regular part of the kernel.
* Fcc    -> Hermitian conjugate partner of `F`.
* Fsin   -> Singular part of the kernel.
* Q      -> Regular part of the source term.
* Qsin   -> Singular part of the source term.
* I      -> An Integrator struct.
* β      -> Inverse temperature.
* h      -> Time step.
* method -> Method for the Matsubara component.
* ws     -> Reusable workspace, or `nothing`.

### Returns
* G -> Contour-ordered Green's function.

See also: [`vie2_timestep!`](@ref), [`VIE2SinWorkspace`](@ref).
"""
function vie2_timestep_sin!(tstp::I64, G::ℱ{C64}, Gsin::Cf{C64},
                            F::ℱ{C64}, Fcc::ℱ{C64}, Fsin::Cf{C64},
                            Q::ℱ{C64}, Qsin::Cf{C64},
                            I::Integrator, β::F64, h::F64; method::Symbol = :fixpoint,
                            ws::Union{Nothing,VIE2SinWorkspace} = nothing)
    _check_vie2(G, F, Fcc, Q)
    _check_vie2_order(I)
    0 ≤ tstp ≤ getntime(G) || throw(BoundsError(G, tstp))
    k = I.k
    d = getdims(G)[1]
    n₂ = (1 ≤ tstp ≤ k + 1) ? k + 1 : tstp
    n₂ ≤ getntime(G) ||
        throw(ArgumentError("G must have at least $n₂ real time steps"))
    for (X, name) in ((Gsin, "Gsin"), (Fsin, "Fsin"), (Qsin, "Qsin"))
        getdims(X) == (d, d) || throw(DimensionMismatch("$name must be $d × $d"))
        X.ntime ≥ n₂ || throw(ArgumentError("$name must have at least $n₂ real time steps"))
    end

    # Workspace
    if isnothing(ws)
        ws = VIE2SinWorkspace(G)
    else
        (getntime(ws.F) == getntime(G) && getntau(ws.F) == getntau(G) &&
         getdims(ws.F) == getdims(G) && getsign(ws.F) == getsign(G)) ||
            throw(DimensionMismatch("the workspace is not compatible with G"))
    end
    # The transformed source term depends on the earlier values of Gsin,
    # so Gsin is a part of the key as well
    src = (Gsin, F, Fcc, Q, Fsin, Qsin)
    if ws.src !== src
        ws.ndone = -1
        ws.src = src
    end

    # The time steps which may have changed
    lo = (tstp ≤ k + 1) ? 0 : min(ws.ndone + 1, tstp)

    # D⁻¹ = [1 + Fsin]⁻¹ and Gsin = D⁻¹ Qsin
    Id = Matrix{C64}(LinearAlgebra.I, d, d)
    for i = lo:n₂
        Fᵢ = Fsin[i]
        all(isfinite, Fᵢ) || throw(DomainError(i, "Fsin is not finite at time step $i"))
        norm(Fᵢ - Fᵢ') ≤ 1e-12 * max(1.0, norm(Fᵢ)) ||
            throw(ArgumentError("Fsin is not hermitian at time step $i"))
        Dinv = inv(Id + Fᵢ)
        ws.Finv[i] = Dinv
        Gsin[i] = Dinv * Qsin[i]
    end

    # F' = D⁻¹ F, Fcc' = Fcc D⁻¹, Q' = D⁻¹ [Q - F Gsin]
    for i = lo:n₂
        _vie2_sin_transform!(ws, i, Gsin, F, Fcc, Q)
    end
    ws.ndone = n₂

    vie2_timestep!(tstp, G, ws.F, ws.Fcc, ws.Q, I, β, h; method = method)

    return G
end
