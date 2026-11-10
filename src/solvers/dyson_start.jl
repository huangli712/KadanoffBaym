#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

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


