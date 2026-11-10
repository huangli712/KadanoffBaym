#
# Project : Lavender
# Source  : vie2_start.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Private Service Functions* : *Retarded Component*
=#

"""
    _vie2_start_ret!(G::CnAbstractFunction{C64}, F::CnAbstractFunction{C64},
                     Fcc::CnAbstractFunction{C64}, Q::CnAbstractFunction{C64},
                     h::F64, I::Integrator)

Start-up procedure of the retarded component, i.e., ``G^{R}(n,j)`` for
`0 ≤ j ≤ n ≤ k`. See *Remarks* : *Volterra Integral Equation*. It is a
private function.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_start_ret`.
"""
function _vie2_start_ret!(G::CnAbstractFunction{C64}, F::CnAbstractFunction{C64},
                          Fcc::CnAbstractFunction{C64}, Q::CnAbstractFunction{C64},
                          h::F64, I::Integrator)
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
                    S, s = _vie2_kernel_ret(F, Fcc, n, l)
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

#=
### *Private Service Functions* : *Left-mixing Component*
=#

"""
    _vie2_start_lmix!(G, F, Fcc, Q, β::F64, h::F64, I::Integrator)

Start-up procedure of the left-mixing component, i.e., ``G^{\\rceil}(n,\\tau)``
for `0 ≤ n ≤ k`. The Matsubara component of `G` and the retarded
component at the first 𝑘 + 1 time steps must be known. See *Remarks* :
*Volterra Integral Equation*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_start_tv`.
"""
function _vie2_start_lmix!(G, F, Fcc, Q, β::F64, h::F64, I::Integrator)
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
            fill!(_lmix_blk(G, n, m - 1), zero(C64))
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
            view(Qₙ, :, cols) .= _lmix_blk(G, n, m - 1) .+ _lmix_blk(Q, n, m - 1)
        end
        Mₙₙ = _mblock(ws, n + 1, n + 1)
        Mₙₙ .= Mₙₙ .+ Id
        for j = 0:k
            # K = F^{R}(n,j) for n ≥ j, or -Fcc^{R}(j,n)† for n < j
            S, s = _vie2_kernel_ret(F, Fcc, n, j)
            Mₙⱼ = _mblock(ws, n + 1, j + 1)
            Mₙⱼ .= Mₙⱼ .+ (s * h * I.GIW[n,j]) .* S
        end
    end
    _solve_block!(ws)
    for n = 0:k
        Xₙ = _xblock(ws, n + 1)
        for m = 1:ntau
            _lmix_blk(G, n, m - 1) .= view(Xₙ, :, (m - 1) * d + 1 : m * d)
        end
    end

    return G
end

"""
    _vie2_start_less!(G::CnAbstractFunction{C64}, F::CnAbstractFunction{C64},
                      Fcc::CnAbstractFunction{C64}, Q::CnAbstractFunction{C64},
                      β::F64, h::F64, I::Integrator)

Start-up procedure of the lesser component, i.e., ``G^{<}(j,n)`` for
`0 ≤ j ≤ n ≤ k`. The retarded and left-mixing components at the first
𝑘 + 1 time steps must be known. See *Remarks* : *Volterra Integral
Equation*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_vie2_impl.hpp`, `vie2_start_les`.
"""
function _vie2_start_less!(G::CnAbstractFunction{C64}, F::CnAbstractFunction{C64},
                           Fcc::CnAbstractFunction{C64}, Q::CnAbstractFunction{C64},
                           β::F64, h::F64, I::Integrator)
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
                mul!(c₁, _lmix_blk(F, 0, m - l), _block(G.mat, N - l + 1), tw(l, 0, m), one(C64))
            end
        end
        # τ ⋯ β
        if m < N
            for l = m:N
                mul!(c₂, _lmix_blk(F, 0, l), _block(G.mat, l - m + 1), tw(l, m, N), one(C64))
            end
        end
        _lmix_blk(G, 0, m) .= _lmix_blk(Q, 0, m) .- δτ .* (c₂ .+ sign .* c₁)
    end

    # Lesser component, G^{<}(0,0) = Q^{<}(0,0) - [F^{⌉} ⋆ G^{⌈}](0,0),
    # where G^{⌈}(τ,0) = -ξ G^{⌉}(0,β-τ)†
    fill!(c₁, zero(C64))
    for m = 0:N
        mul!(c₁, _lmix_blk(F, 0, m), _lmix_blk(G, 0, N - m)', tw(m, 0, N), one(C64))
    end
    _less_blk(G, 0, 0) .= _less_blk(Q, 0, 0) .- (im * δτ * sign) .* c₁

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
    _vie2_start_lmix!(G, F, Fcc, Q, β, h, I)
    _vie2_start_less!(G, F, Fcc, Q, β, h, I)

    return G
end

