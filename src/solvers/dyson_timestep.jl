#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

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


