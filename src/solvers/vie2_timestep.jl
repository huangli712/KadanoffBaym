#
# Project : Lavender
# Source  : vie2_timestep.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    _vie2_kernel_ret(F, Fcc, n::I64, l::I64)

Return ``F^{R}(n,l)`` (NESSi indices) and the sign `1` if `n ≥ l`.
Otherwise, ``F^{cc,R}(l,n)^{\\dagger}`` and the sign `-1` are returned,
i.e., ``F^{R}(n,l) = -F^{cc,R}(l,n)^{\\dagger}``. It is a private
function.
"""
@inline function _vie2_kernel_ret(F, Fcc, n::I64, l::I64)
    if n ≥ l
        return _ret_blk(F, n, l), 1.0
    else
        return _ret_blk(Fcc, l, n)', -1.0
    end
end

"""
    _vie2_timestep_ret!(tstp::I64, G::CnAbstractFunction{C64},
                        K::CnAbstractFunction{C64}, Kcc::CnAbstractFunction{C64},
                        Q::CnAbstractFunction{C64}, h::F64, I::Integrator)

Time stepping of the retarded component, i.e., ``G^{R}(n,j)`` for
`0 ≤ j ≤ n`, where `n = tstp - 1 > k`. The right action equation
``G^{R} + G^{R} * K^{R} = Q^{R}`` is solved. Note that the caller passes
the kernel ``K = F^{cc}`` and its partner ``K^{cc} = F``. See *Remarks* :
*Volterra Integral Equation*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_timestep_ret`.
"""
function _vie2_timestep_ret!(tstp::I64, G::CnAbstractFunction{C64},
                             K::CnAbstractFunction{C64}, Kcc::CnAbstractFunction{C64},
                             Q::CnAbstractFunction{C64}, h::F64, I::Integrator)
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

"""
    _vie2_timestep_lmix!(tstp::I64, G, F, Fcc, Q, β::F64, h::F64, I::Integrator)

Time stepping of the left-mixing component, i.e., ``G^{\\rceil}(n,\\tau)``,
where `n = tstp - 1 > k`. The retarded component at time step `n` must be
known. See *Remarks* : *Volterra Integral Equation*. It is a private
function.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_timestep_tv`.
"""
function _vie2_timestep_lmix!(tstp::I64, G, F, Fcc, Q, β::F64, h::F64, I::Integrator;
                            threaded::Bool = false)
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
    if _use_threads(threaded) && F isa ℱ && G isa ℱ
        _incr_tv_threaded!(tstp, W, F, Fcc, G, G, I, β, h, one(C64))
    else
        incr_convolution_tv!(tstp, W, F, Fcc, nothing, G, G, I, β, h, one(C64))
    end

    # [1 + h Ω_{n0} F^{R}(n,n)] G^{⌉}(n,τ) = Q^{⌉}(n,τ) - [F * G]^{⌉}(n,τ)
    ws = _BlockSolver(1, d, d * ntau)
    for m = 1:ntau
        Qₘ = view(ws.Qbig, :, (m - 1) * d + 1 : m * d)
        Qₘ .= _lmix_blk(Q, n, m - 1) .- W.lmix.data[m]
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
    _vie2_less_core!(n::I64, gles::AbstractVector, G::CnAbstractFunction{C64},
                     F::CnAbstractFunction{C64}, Fcc::CnAbstractFunction{C64},
                     Q::CnAbstractFunction{C64}, h::F64, I::Integrator)

Solve the lesser component ``G^{<}(j,n)`` for `j = 0, ⋯, n` (NESSi
indices). On entry, `gles[j+1]` (`j = 0, ⋯, max(n,k)`) contains the source
term ``-[F^{\\rceil} \\star G^{\\lceil} + F^{<} * G^{A}](j,n)``. On exit,
`gles[j+1]` (`j = 0, ⋯, n`) contains the solution, which is also written
into `G`. It is a private function.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_timestep_les`.
"""
function _vie2_less_core!(n::I64, gles::AbstractVector, G::CnAbstractFunction{C64},
                          F::CnAbstractFunction{C64}, Fcc::CnAbstractFunction{C64},
                          Q::CnAbstractFunction{C64}, h::F64, I::Integrator)
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
            Rⱼ .= Rⱼ .+ _less_blk(Q, j, n)
        else
            Rⱼ .= Rⱼ .- (_less_blk(Q, n, j))'
        end
        Mⱼⱼ = _mblock(ws, j + 1, j + 1)
        Mⱼⱼ .= Mⱼⱼ .+ Id
        # Integral F^{R} * G^{<}
        for m = 0:k
            S, s = _vie2_kernel_ret(F, Fcc, j, m)
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
        qq .= gles[j+1] .+ _less_blk(Q, j, n)
        mm .= Id .+ (h * I.GIW[j,j]) .* _ret_blk(F, j, j)
        for m = 0:j-1
            mul!(qq, _ret_blk(F, j, m), gles[m+1], -h * I.GIW[j,m], one(C64))
        end
        _solve_block_right!(gles[j+1], mm, qq, ws1)
    end

    # Write the stored triangle
    for j = 0:n
        _less_blk(G, j, n) .= gles[j+1]
    end

    return G
end

"""
    _vie2_timestep_less!(tstp::I64, G::CnAbstractFunction{C64},
                         F::CnAbstractFunction{C64}, Fcc::CnAbstractFunction{C64},
                         Q::CnAbstractFunction{C64}, β::F64, h::F64,
                         I::Integrator)

Time stepping of the lesser component, i.e., ``G^{<}(j,n)`` for
`0 ≤ j ≤ n`, where `n = tstp - 1 > k`. The retarded and left-mixing
components at time step `n` must be known. The source term contains only
``F^{\\rceil} \\star G^{\\lceil}`` and ``F^{<} * G^{A}``. See *Remarks* :
*Volterra Integral Equation*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_timestep_les`.
"""
function _vie2_timestep_less!(tstp::I64, G::CnAbstractFunction{C64},
                              F::CnAbstractFunction{C64}, Fcc::CnAbstractFunction{C64},
                              Q::CnAbstractFunction{C64}, β::F64, h::F64,
                              I::Integrator; threaded::Bool = false)
    n = tstp - 1
    @assert n > I.k
    d = getdims(G)[1]
    ntau = getntau(G)

    W = 𝒻(tstp, ntau, d, d, getsign(G))
    if _use_threads(threaded) && F isa ℱ && G isa ℱ
        _incr_lesser_src_threaded!(tstp, W, F, Fcc, G, G, I, β, h, -one(C64))
    else
        incr_convolution_lmix_rmix!(tstp, W, F, Fcc, nothing, G, G, I, β, h, -one(C64))
        incr_convolution_less_adv!(tstp, W, F, Fcc, nothing, G, G, I, β, h, -one(C64))
    end
    _vie2_less_core!(n, W.less.data, G, F, Fcc, Q, h, I)

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
                        I::Integrator, β::F64, h::F64; method::Symbol = :fixpoint,
                        threaded::Bool = false)
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
        _vie2_timestep_lmix!(tstp, G, F, Fcc, Q, β, h, I; threaded = threaded)
        _vie2_timestep_less!(tstp, G, F, Fcc, Q, β, h, I; threaded = threaded)
    end

    return G
end

"""
    _vie2_moving_ret!(G, F, Fcc, Q, h, I)

Retarded component of the truncated VIE2 at the leading row, the right
action equation with the kernel `K = Fcc` and its partner `Kcc = F`. See
*Remarks* : *Truncated VIE2*. It is a private function.
"""
function _vie2_moving_ret!(G::ℱᵐᵒᵛ{C64}, F::ℱᵐᵒᵛ{C64}, Fcc::ℱᵐᵒᵛ{C64},
                           Q::ℱᵐᵒᵛ{C64}, h::F64, I::Integrator)
    k = I.k
    n = G.tc
    d = getdims(G)[1]
    Id = Matrix{C64}(LinearAlgebra.I, d, d)
    K, Kcc = Fcc, F

    # Clear the leading row, G^{R}(n,n) = Q^{R}(n,n)
    fill!(_mrowview(G.ret, 0), zero(C64))
    copyto!(_wret(G, n, n), _wret(Q, n, n))
    Gₙₙ = _wret(G, n, n)

    # Start values G^{R}(n,n-j), j = 1, ⋯, k
    M = [zeros(C64, d, d) for _ = 1:k, _ = 1:k]
    R = [zeros(C64, d, d) for _ = 1:k]
    X = [zeros(C64, d, d) for _ = 1:k]
    for j = 1:k
        R[j] .= _wret(Q, n, n - j)
        M[j,j] .= M[j,j] .+ Id
        for l = 0:k
            weight = -h * I.GIW[j,l]
            if l ≤ j
                S = _wret(K, n - l, n - j)
            else
                S = _wret(Kcc, n - j, n - l)'
                weight = -weight
            end
            if l == 0
                mul!(R[j], Gₙₙ, S, weight, one(C64))
            else
                M[j,l] .= M[j,l] .- weight .* S
            end
        end
    end
    _solve_block_left!(X, M, R, _BlockSolver(k, d))
    for q = 1:k
        copyto!(_wret(G, n, n - q), X[q])
    end

    # Contribution of G^{R}(n,m), m = n-k, ⋯, n, to -∫ dm G^{R}(n,m) K^{R}(m,j)
    qq = zeros(C64, d, d, n + 1)
    for m = n-k:n
        Gₙₘ = _wret(G, n, m)
        for j = 0:n-k-1
            mul!(view(qq, :, :, j + 1), Gₙₘ, _wret(K, m, j), -I.GIW[n-j,n-m], one(C64))
        end
    end

    # The rest: j = n-k-1, ⋯, 0
    ws1 = _BlockSolver(1, d)
    w₀ = h * I.GIW.ω[1]
    mm = zeros(C64, d, d)
    for l = k+1:n
        j = n - l
        qqⱼ = view(qq, :, :, j + 1)
        qqⱼ .= h .* qqⱼ .+ _wret(Q, n, j)
        mm .= Id .+ w₀ .* _wret(K, j, j)
        Gₙⱼ = _wret(G, n, j)
        _solve_block_left!(Gₙⱼ, mm, qqⱼ, ws1)
        for j₂ = 0:j-1
            mul!(view(qq, :, :, j₂ + 1), Gₙⱼ, _wret(K, j, j₂), -I.GIW[n-j₂,n-j], one(C64))
        end
    end

    return G
end

"""
    _vie2_moving_less!(G, F, Fcc, Q, h, I)

Lesser component of the truncated VIE2 at the leading row. The retarded
component of the leading row must be known. See *Remarks* : *Truncated
VIE2*. It is a private function.
"""
function _vie2_moving_less!(G::ℱᵐᵒᵛ{C64}, F::ℱᵐᵒᵛ{C64}, Fcc::ℱᵐᵒᵛ{C64},
                            Q::ℱᵐᵒᵛ{C64}, h::F64, I::Integrator;
                            threaded::Bool = false)
    k = I.k
    n = G.tc
    d = getdims(G)[1]
    Id = Matrix{C64}(LinearAlgebra.I, d, d)

    # F^{<}(m,j) in the window, m ≥ j from F, otherwise from Fcc
    function fles!(dst, m, j)
        if m ≥ j
            copyto!(dst, _wles(F, m, j))
        else
            adjoint!(dst, _wles(Fcc, j, m))
            dst .= .-dst
        end
        return dst
    end

    # Source term -[F^{<} * G^{A}](m,n), m = 0, ⋯, n. The outputs m are
    # independent, the accumulation order over j is fixed.
    gles = [zeros(C64, d, d) for _ = 0:n]
    _moving_tforeach(0:n, threaded) do ms
        ftmp = zeros(C64, d, d)
        for j = 0:n
            # G^{A}(j,n) = G^{R}(n,j)†
            b = _wret(G, n, j)'
            w = -h * I.GIW[n,j]
            for m in ms
                mul!(gles[m+1], fles!(ftmp, m, j), b, w, one(C64))
            end
        end
    end

    # Q^{<}(j,n) = -Q^{<}(n,j)†
    qles(j) = -Matrix(_wles(Q, n, j)')

    # Start: j = 0, ⋯, k, M X = R
    ws = _BlockSolver(k + 1, d)
    fill!(ws.Mbig, zero(C64))
    for j = 0:k
        Rⱼ = _qblock(ws, j + 1)
        Rⱼ .= gles[j+1] .+ qles(j)
        Mⱼⱼ = _mblock(ws, j + 1, j + 1)
        Mⱼⱼ .= Mⱼⱼ .+ Id
        for m = 0:k
            if j ≥ m
                S, s = _wret(F, j, m), 1.0
            else
                S, s = _wret(Fcc, m, j)', -1.0
            end
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
        qq .= gles[j+1] .+ qles(j)
        mm .= Id .+ (h * I.GIW[j,j]) .* _wret(F, j, j)
        for m = 0:j-1
            mul!(qq, _wret(F, j, m), gles[m+1], -h * I.GIW[j,m], one(C64))
        end
        _solve_block_right!(gles[j+1], mm, qq, ws1)
    end

    # Write the leading row, G^{<}(n,j) = -G^{<}(j,n)† for j < n. The
    # diagonal block G^{<}(n,n) is stored as it is solved (it is
    # anti-hermitian only up to the discretization error), as in the full
    # storage.
    for j = 0:n-1
        adjoint!(_wles(G, n, j), gles[j+1])
        _wles(G, n, j) .= .-_wles(G, n, j)
    end
    copyto!(_wles(G, n, n), gles[n+1])

    return G
end

"""
    vie2_timestep!(tstp::I64, G::ℱᵐᵒᵛ{C64}, F::ℱᵐᵒᵛ{C64}, Fcc::ℱᵐᵒᵛ{C64},
                   Q::ℱᵐᵒᵛ{C64}, I::Integrator, h::F64)

Solve the truncated VIE2 ``G + F * G = Q`` for the leading row (KB time
step `tstp`) of the moving Green's function `G`. All moving objects must
be initialized with the same cut-off `tc ≥ 2k + 2` and leading time step.
The mixed initial correlations ``F^{\\rceil} \\star G^{\\lceil}`` are
omitted, see *Remarks* : *Truncated VIE2*; a large `tc` does not restore
them.

### Arguments
* tstp -> KB time step of the leading row.
* G    -> Moving Green's function, the solution.
* F    -> Moving kernel.
* Fcc  -> Hermitian conjugate partner of `F`.
* Q    -> Moving source term.
* I    -> An Integrator struct, 1 ≤ k ≤ 5.
* h    -> Time step.

### Returns
* G -> Moving Green's function.
"""
function vie2_timestep!(tstp::I64, G::ℱᵐᵒᵛ{C64}, F::ℱᵐᵒᵛ{C64}, Fcc::ℱᵐᵒᵛ{C64},
                        Q::ℱᵐᵒᵛ{C64}, I::Integrator, h::F64; threaded::Bool = false)
    _check_moving_solver(tstp, I.k, G, F, Fcc, Q)
    for X in (F, Fcc, Q)
        X === G && throw(ArgumentError("G must not be an alias of F, Fcc, or Q"))
    end
    getsign(F) == getsign(G) ||
        throw(ArgumentError("G and F must have the same statistics"))
    h > 0.0 || throw(ArgumentError("h must be positive"))

    _vie2_moving_ret!(G, F, Fcc, Q, h, I)
    _vie2_moving_less!(G, F, Fcc, Q, h, I; threaded = threaded)

    return G
end

