#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks* : *Pseudo-Particle Volterra Integral Equation*

The pseudo-particle VIE2 reads ``(1 + F) \ast G = Q``, where the
convolution is the pseudo-particle one (see `pseudo_convolution.jl`).

* Matsubara component. It is a causal Volterra equation,

```math
\begin{equation}
G^{M}(\tau) + \int^{\tau}_{0} d\bar{\tau} F^{M}(\tau - \bar{\tau})
G^{M}(\bar{\tau}) = Q^{M}(\tau), \quad G^{M}(0) = Q^{M}(0).
\end{equation}
```

* Left-mixing component in the start-up time steps. Let
``X(l) = G^{\rceil}(t,\beta - l)``, then

```math
\begin{equation}
X(l) = Q^{\rceil}(t,\beta - l)
     - \int^{l}_{0} d\bar{l} X(\bar{l}) F^{cc,M}(l - \bar{l})
     - \int^{t}_{0} d\bar{t} G^{>}(t,\bar{t}) F^{cc,\rceil}(\bar{t},\beta - l),
\end{equation}
```

with ``X(0) = \xi G^{>}(t,0)``.

* The greater (stored as `ret`) and lesser components in the start-up
time steps are solved by the standard VIE2 kernels, see `vie.jl`.

* For the later time steps (`tstp ≥ k + 2`), all the components are
obtained by the incremental convolutions (the unknown components at the
current time step are set to zero) and the end-point corrections, i.e.,
a 1 × 1 block problem for each matrix element. The left-mixing component
uses the pseudo-particle convolution.

The time indices of the private kernels follow the NESSi convention
(start from 0), see *Remarks* : *Real Time Dyson Equation* in `vide.jl`.

*References* :

Please see [`NESSi`] `cntr_pseudo_vie2_impl.hpp`.
=#

#=
### *Private Service Functions* : *Sanity Checks*
=#

"""
    _check_pseudo_vie2(G::PseudoGF{C64}, F::PseudoGF{C64}, Fcc::PseudoGF{C64},
                       Q::PseudoGF{C64}, I::Integrator, nmax::I64)

Check the arguments of the pseudo-particle VIE2 solvers. The time steps
`0, ⋯, nmax` (NESSi convention) are needed. If `nmax < 0`, only the
Matsubara axis is needed. It is a private function.
"""
function _check_pseudo_vie2(G::PseudoGF{C64}, F::PseudoGF{C64}, Fcc::PseudoGF{C64},
                            Q::PseudoGF{C64}, I::Integrator, nmax::I64)
    1 ≤ I.k ≤ 5 || throw(ArgumentError("the order must be in 1:5, got $(I.k)"))
    for X in (F, Fcc, Q)
        getsign(X) == getsign(G) ||
            throw(ArgumentError("all the Green's functions must have the same statistics"))
        getdims(X) == getdims(G) ||
            throw(DimensionMismatch("all the Green's functions must have the same size"))
        getntau(X) == getntau(G) ||
            throw(DimensionMismatch("all the Green's functions must have the same imaginary time grid"))
        getntime(X) == getntime(G) ||
            throw(DimensionMismatch("all the Green's functions must have the same real time grid"))
    end
    d₁, d₂ = getdims(G)
    d₁ == d₂ || throw(DimensionMismatch("G must be square"))
    getntau(G) - 1 > I.k ||
        throw(ArgumentError("ntau must be larger than solve_order + 1"))
    if nmax ≥ 0
        getntime(G) ≥ nmax + 1 ||
            throw(ArgumentError("G must have at least $(nmax + 1) real time steps"))
    end
end


#=
### *Private Service Functions* : *Matsubara Component*
=#

"""
    _pseudo_vie2_mat!(G::PseudoGF{C64}, F::PseudoGF{C64}, Q::PseudoGF{C64},
                      I::Integrator, β::F64)

Solve the Matsubara component of the pseudo-particle VIE2. See *Remarks* :
*Pseudo-Particle Volterra Integral Equation*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_pseudo_vie2_impl.hpp`, `pseudo_vie2_mat_dispatch`.
"""
function _pseudo_vie2_mat!(G::PseudoGF{C64}, F::PseudoGF{C64}, Q::PseudoGF{C64},
                           I::Integrator, β::F64)
    k = I.k
    N = getntau(G) - 1 # NESSi ntau
    d = getdims(G)[1]
    δτ = β / N
    Id = Matrix{C64}(LinearAlgebra.I, d, d)
    gm(m) = _block(G.mat, m + 1)
    fm(m) = _block(F.mat, m + 1)
    qm(m) = _block(Q.mat, m + 1)

    # Initial value: Gᴹ(0) = Qᴹ(0)
    gm(0) .= qm(0)

    # The first k steps, solve ∑_q M_{pq} X_q = R_p with X_q = Gᴹ(q)
    ws = _BlockSolver(k, d)
    fill!(ws.Mbig, zero(C64))
    for n = 1:k
        Rₚ = _qblock(ws, n)
        Rₚ .= qm(n)
        Mₚₚ = _mblock(ws, n, n)
        Mₚₚ .= Mₚₚ .+ Id
        # Integral ∫₀^τ dx Fᴹ(τ - x) Gᴹ(x)
        if n < k
            for m = 0:k
                for j = 0:k
                    cweight = δτ * I.BCW[n-1,m,j]
                    if j == 0
                        mul!(Rₚ, fm(m), gm(0), -cweight, true)
                    else
                        Mₚⱼ = _mblock(ws, n, j)
                        Mₚⱼ .= Mₚⱼ .+ cweight .* fm(m)
                    end
                end
            end
        else
            for m = 0:k
                cweight = δτ * I.GIW[k,m]
                if m == 0
                    mul!(Rₚ, fm(k), gm(0), -cweight, true)
                else
                    Mₚₘ = _mblock(ws, n, m)
                    Mₚₘ .= Mₚₘ .+ cweight .* fm(k - m)
                end
            end
        end
    end
    _solve_block!(ws)
    for n = 1:k
        gm(n) .= _xblock(ws, n)
    end

    # The remaining steps: Gregory integration
    ws1 = _BlockSolver(1, d)
    qq = zeros(C64, d, d)
    mm = zeros(C64, d, d)
    for n = k+1:N
        fill!(qq, zero(C64))
        for m = 0:n-1
            mul!(qq, fm(n - m), gm(m), -I.GIW[n,m], true)
        end
        qq .= δτ .* qq .+ qm(n)
        mm .= Id .+ (δτ * I.GIW[n,n]) .* fm(0)
        _solve_block_right!(gm(n), mm, qq, ws1)
    end

    return G
end


#=
### *Private Service Functions* : *Left-mixing Component*
=#

"""
    _pseudo_vie2_start_tv!(n::I64, G::PseudoGF{C64}, Fcc::PseudoGF{C64},
                           Q::PseudoGF{C64}, I::Integrator, β::F64, h::F64)

Solve the left-mixing component ``G^{\\rceil}(n,\\tau)`` (NESSi time index
`n ≤ k`) of the pseudo-particle VIE2 along the reversed imaginary time
axis. The greater component at the first `k + 1` time steps must be
known. It is used in the start-up procedure only. See *Remarks* :
*Pseudo-Particle Volterra Integral Equation*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_pseudo_vie2_impl.hpp`, `pseudo_vie2_timestep_tv`.
"""
function _pseudo_vie2_start_tv!(n::I64, G::PseudoGF{C64}, Fcc::PseudoGF{C64},
                                Q::PseudoGF{C64}, I::Integrator, β::F64, h::F64)
    k = I.k
    @assert 0 ≤ n ≤ k
    N = getntau(G) - 1 # NESSi ntau
    d = getdims(G)[1]
    sign = getsign(G)
    δτ = β / N
    n₁ = max(n, k)
    Id = Matrix{C64}(LinearAlgebra.I, d, d)
    fm(m) = _block(Fcc.mat, m + 1)
    gtemp = zeros(C64, d, d)

    # X(i) = G^⌉(n,β - τᵢ), i = 0, ⋯, N
    xx = [zeros(C64, d, d) for _ = 0:N]

    # Q^⌉(n,β - τᵢ) - ∫₀^t dj G^{>}(n,j) Fcc^⌉(j,β - τᵢ), i = 1, ⋯, N
    for i = 1:N
        xx[i+1] .= _tv_blk(Q, n, N - i)
    end
    for j = 0:n₁
        if j ≤ n
            gtemp .= _ret_blk(G, n, j)
        else
            gtemp .= .-(_ret_blk(G, j, n))'
        end
        gtemp .= (-I.GIW[n,j] * h) .* gtemp
        for i = 1:N
            mul!(xx[i+1], gtemp, _tv_blk(Fcc, j, N - i), true, true)
        end
    end

    # Initial value: X(0) = ξ G^{>}(n,0)
    xx[1] .= sign .* _ret_blk(G, n, 0)

    # The first k steps, solve ∑_q X_q M_{pq} = R_p with X_q = X(q)
    ws = _BlockSolver(k, d)
    M = [zeros(C64, d, d) for _ = 1:k, _ = 1:k]
    R = [copy(xx[i+1]) for i = 1:k]
    for i = 1:k
        Rₚ = R[i]
        M[i,i] .= M[i,i] .+ Id
        # Integral ∫₀^l dx X(x) Fccᴹ(l - x)
        if i < k
            for m = 0:k
                for j = 0:k
                    cweight = δτ * I.BCW[i-1,m,j]
                    if j == 0
                        mul!(Rₚ, xx[1], fm(m), -cweight, true)
                    else
                        M[i,j] .= M[i,j] .+ cweight .* fm(m)
                    end
                end
            end
        else
            for m = 0:k
                cweight = δτ * I.GIW[i,m]
                if m == 0
                    mul!(Rₚ, xx[1], fm(k), -cweight, true)
                else
                    M[i,m] .= M[i,m] .+ cweight .* fm(k - m)
                end
            end
        end
    end
    _solve_block_left!(view(xx, 2:k+1), M, R, ws)

    # The remaining steps: Gregory integration
    ws1 = _BlockSolver(1, d)
    qq = zeros(C64, d, d)
    mm = zeros(C64, d, d)
    for i = k+1:N
        fill!(qq, zero(C64))
        for m = 0:i-1
            mul!(qq, xx[m+1], fm(i - m), -I.GIW[i,m], true)
        end
        qq .= δτ .* qq .+ xx[i+1]
        mm .= Id .+ (δτ * I.GIW[i,i]) .* fm(0)
        _solve_block_left!(xx[i+1], mm, qq, ws1)
    end

    # G^⌉(n,τ_{N-l}) = X(l)
    for l = 0:N
        _tv_blk(G, n, N - l) .= xx[l+1]
    end

    return G
end


#=
### *Private Service Functions* : *Time Stepping*
=#

"""
    _pseudo_vie2_timestep!(tstp::I64, G::PseudoGF{C64}, F::PseudoGF{C64},
                           Fcc::PseudoGF{C64}, Q::PseudoGF{C64},
                           I::Integrator, β::F64, h::F64)

Solve the pseudo-particle VIE2 at time step `tstp ≥ k + 2` (NESSi
`n = tstp - 1 > k`). The source terms are evaluated by the incremental
convolutions with the components of `G` at the current time step set to
zero, then the end-point terms are solved. `Q` is not changed. See
*Remarks* : *Pseudo-Particle Volterra Integral Equation*. It is a private
function.

*References* :

Please see [`NESSi`] `cntr_pseudo_vie2_impl.hpp`,
`pseudo_vie2_timestep_new_dispatch`.
"""
function _pseudo_vie2_timestep!(tstp::I64, G::PseudoGF{C64}, F::PseudoGF{C64},
                                Fcc::PseudoGF{C64}, Q::PseudoGF{C64},
                                I::Integrator, β::F64, h::F64)
    k = I.k
    n = tstp - 1
    @assert n > k
    d = getdims(G)[1]
    ntau = getntau(G)
    Id = Matrix{C64}(LinearAlgebra.I, d, d)
    ws1 = _BlockSolver(1, d)
    mm = zeros(C64, d, d)

    # Source terms, a copy of Q at the current time step
    W = Q[tstp]
    zeros!(G, tstp)

    # h F^{R}(n,n), it enters the end-point terms
    mtemp = h .* _ret_blk(F, n, n)

    # Greater component: Q - F ∗ G, with G^{>}(n,j) = 0
    incr_convolution_ret!(tstp, W, F, Fcc, nothing, G, G, I, h, -1.0)
    for j = 0:n
        mm .= Id .+ I.GIW[n-j,0] .* mtemp
        _solve_block_right!(_ret_blk(G, n, j), mm, W.ret.data[j+1], ws1)
    end

    # Left-mixing component: Q - F ⋆ G, with G^⌉(n,τ) = 0
    incr_pseudo_convolution_tv!(tstp, W, F, Fcc, nothing, G, G, I, β, h, -1.0)
    mm .= Id .+ I.GIW[n,n] .* mtemp
    for m = 1:ntau
        _solve_block_right!(_block(G.lmix, tstp, m), mm, W.lmix.data[m], ws1)
    end

    # Lesser component: Q - G ∗ Fcc, with G^<(j,n) = 0. The blocks
    # j < n are solved first, then they enter the block j = n.
    cws = _ConvWorkspace(G, Fcc)
    lw = W.less.data
    mm .= (Id .+ I.GIW[n,n] .* mtemp)'
    for ms in (1:tstp-1, tstp:tstp)
        _incr_ret_less!(lw, ms, tstp, G, G, nothing, Fcc, F, I, β, h, -1.0, cws)
        _incr_less_adv!(lw, ms, tstp, G, G, nothing, Fcc, F, I, β, h, -1.0, cws;
                        diagcc = _incr_les_diagcc(tstp, I))
        _incr_lmix_rmix!(lw, ms, tstp, G, G, nothing, Fcc, F, I, β, h, -1.0, cws)
        for j in ms
            _solve_block_left!(_les_blk(G, j - 1, n), mm, lw[j], ws1)
        end
    end

    return G
end


#=
### *Public API* : *Pseudo-Particle VIE2*
=#

"""
    pseudo_vie2_mat!(G::PseudoGF{C64}, F::PseudoGF{C64}, Fcc::PseudoGF{C64},
                     Q::PseudoGF{C64}, β::F64, I::Integrator)

Solve the Matsubara component of the pseudo-particle VIE2
``(1 + F) \\ast G = Q``, i.e., a causal Volterra equation along the
imaginary time axis.

### Arguments
* G   -> Pseudo-particle Green's function, the solution.
* F   -> Kernel.
* Fcc -> Hermitian conjugate partner of F.
* Q   -> Source term.
* β   -> Inverse temperature.
* I   -> Struct for numerical integration, 1 ≤ I.k ≤ 5.

### Returns
* G -> Pseudo-particle Green's function.

*References* :

Please see [`NESSi`] `cntr_pseudo_vie2_impl.hpp`, `pseudo_vie2_mat`.
"""
function pseudo_vie2_mat!(G::PseudoGF{C64}, F::PseudoGF{C64}, Fcc::PseudoGF{C64},
                          Q::PseudoGF{C64}, β::F64, I::Integrator)
    _check_pseudo_vie2(G, F, Fcc, Q, I, -1)
    β > 0.0 || throw(ArgumentError("β must be positive"))
    _pseudo_vie2_mat!(G, F, Q, I, β)
end


"""
    pseudo_vie2_start!(G::PseudoGF{C64}, F::PseudoGF{C64}, Fcc::PseudoGF{C64},
                       Q::PseudoGF{C64}, I::Integrator, β::F64, h::F64)

Solve the pseudo-particle VIE2 at the first `k + 1` real time steps
(`tstp = 1, ⋯, k + 1`). The Matsubara component of `G` must be known.
The components are solved in the order: greater (stored as `ret`),
left-mixing, and lesser. `Q` must satisfy the boundary condition
``Q^{\\rceil}(t,\\beta) = \\xi G^{>}(t,0)``.

### Arguments
See [`pseudo_vie2_mat!`](@ref). `h` is the time step.

### Returns
* G -> Pseudo-particle Green's function.

*References* :

Please see [`NESSi`] `cntr_pseudo_vie2_impl.hpp`, `pseudo_vie2_start`.
"""
function pseudo_vie2_start!(G::PseudoGF{C64}, F::PseudoGF{C64}, Fcc::PseudoGF{C64},
                            Q::PseudoGF{C64}, I::Integrator, β::F64, h::F64)
    k = I.k
    _check_pseudo_vie2(G, F, Fcc, Q, I, k)
    β > 0.0 || throw(ArgumentError("β must be positive"))
    h > 0.0 || throw(ArgumentError("h must be positive"))

    _vie2_start_ret!(G, F, Fcc, Q, h, I)
    for n = 0:k
        _pseudo_vie2_start_tv!(n, G, Fcc, Q, I, β, h)
    end
    _vie2_start_less!(G, F, Fcc, Q, β, h, I)

    return G
end


"""
    pseudo_vie2_timestep!(tstp::I64, G::PseudoGF{C64}, F::PseudoGF{C64},
                          Fcc::PseudoGF{C64}, Q::PseudoGF{C64},
                          I::Integrator, β::F64, h::F64)

Solve the pseudo-particle VIE2 at time step `tstp ≥ k + 2`. The
components at the earlier time steps must be known. `Q` is not changed.
See [`pseudo_vie2_start!`](@ref) for the arguments.

*References* :

Please see [`NESSi`] `cntr_pseudo_vie2_impl.hpp`, `pseudo_vie2_timestep`.
"""
function pseudo_vie2_timestep!(tstp::I64, G::PseudoGF{C64}, F::PseudoGF{C64},
                               Fcc::PseudoGF{C64}, Q::PseudoGF{C64},
                               I::Integrator, β::F64, h::F64)
    k = I.k
    tstp ≥ k + 2 || throw(ArgumentError("tstp must be larger than solve_order + 1"))
    _check_pseudo_vie2(G, F, Fcc, Q, I, tstp - 1)
    β > 0.0 || throw(ArgumentError("β must be positive"))
    h > 0.0 || throw(ArgumentError("h must be positive"))

    _pseudo_vie2_timestep!(tstp, G, F, Fcc, Q, I, β, h)
end


"""
    pseudo_vie2!(G::PseudoGF{C64}, F::PseudoGF{C64}, Fcc::PseudoGF{C64},
                 Q::PseudoGF{C64}, I::Integrator, β::F64, h::F64)

Solve the pseudo-particle VIE2 at the Matsubara axis and at all the real
time steps, i.e., [`pseudo_vie2_mat!`](@ref), [`pseudo_vie2_start!`](@ref),
and [`pseudo_vie2_timestep!`](@ref) for `tstp = k + 2, ⋯, ntime`.

*References* :

Please see [`NESSi`] `cntr_pseudo_vie2_impl.hpp`, `pseudo_vie2`.
"""
function pseudo_vie2!(G::PseudoGF{C64}, F::PseudoGF{C64}, Fcc::PseudoGF{C64},
                      Q::PseudoGF{C64}, I::Integrator, β::F64, h::F64)
    k = I.k
    _check_pseudo_vie2(G, F, Fcc, Q, I, getntime(G) - 1)

    pseudo_vie2_mat!(G, F, Fcc, Q, β, I)
    pseudo_vie2_start!(G, F, Fcc, Q, I, β, h)
    for tstp = k+2:getntime(G)
        pseudo_vie2_timestep!(tstp, G, F, Fcc, Q, I, β, h)
    end

    return G
end

