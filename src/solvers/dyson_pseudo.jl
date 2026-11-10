#
# Project : Lavender
# Source  : dyson_pseudo.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks* : *Pseudo-Particle Dyson Equation*

The pseudo-particle Dyson equation reads

```math
\begin{equation}
[i\partial_t + \lambda_0 - H(t) - \Sigma] \ast G = 1,
\end{equation}
```

where ``\lambda_0`` is not included in ``H``. The retarded (greater) and
lesser components are solved by the standard real-time Dyson kernels,
see `vide.jl`. Only the Matsubara and the left-mixing components are
special.

* Matsubara component. It is a causal initial value problem,

```math
\begin{equation}
[-\partial_\tau + \lambda_0 - H_0] G^{M}(\tau)
- \int^{\tau}_{0} d\bar{\tau} \Sigma^{M}(\tau - \bar{\tau}) G^{M}(\bar{\tau})
= 0, \quad G^{M}(0) = -1.
\end{equation}
```

The first ``k`` points are solved together (polynomial differentiation
and boundary corrections), the others by the BDF(k+1) and Gregory rules.

* Left-mixing component. Let ``X(l) = G^{\rceil}(t,\beta - l)``, then

```math
\begin{equation}
[-\partial_l + \lambda_0 - H_0^{\dagger}] X(l)
- \int^{l}_{0} d\bar{l} X(\bar{l}) \Sigma^{M}(l - \bar{l})
= \int^{t}_{0} d\bar{t} G^{>}(t,\bar{t}) \Sigma^{\rceil}(\bar{t},\beta - l),
\end{equation}
```

with ``X(0) = \xi G^{>}(t,0)``. The equation is solved from ``l = 0`` to
``l = \beta`` for fixed ``t``, using the same rules as the Matsubara
component. Note that ``H_0`` (the Hamiltonian at the Matsubara axis) is
used.

The time indices of the private kernels follow the NESSi convention
(start from 0), see *Remarks* : *Real Time Dyson Equation* in `vide.jl`.

*References* :

Please see [`NESSi`] `cntr_pseudodyson_impl.hpp`.
=#

#=
### *Private Service Functions* : *Sanity Checks*
=#

"""
    _check_pseudodyson(G::PseudoGF{C64}, Σ::PseudoGF{C64}, H::Cf{C64},
                       I::Integrator, nmax::I64)

Check the arguments of the pseudo-particle Dyson solvers. The time steps
`0, ⋯, nmax` (NESSi convention) of `H` are checked. If `nmax < 0`, only
the Matsubara axis is checked. It is a private function.
"""
function _check_pseudodyson(G::PseudoGF{C64}, Σ::PseudoGF{C64}, H::Cf{C64},
                            I::Integrator, nmax::I64)
    1 ≤ I.k ≤ 5 || throw(ArgumentError("the order must be in 1:5, got $(I.k)"))
    getsign(G) == getsign(Σ) ||
        throw(ArgumentError("G and Σ must have the same statistics"))
    d₁, d₂ = getdims(G)
    d₁ == d₂ || throw(DimensionMismatch("G must be square"))
    getdims(Σ) == (d₁, d₂) || throw(DimensionMismatch("G and Σ have different sizes"))
    getntau(G) == getntau(Σ) ||
        throw(DimensionMismatch("G and Σ have different imaginary time grids"))
    getntime(G) == getntime(Σ) ||
        throw(DimensionMismatch("G and Σ have different real time grids"))
    getntau(G) - 1 > I.k ||
        throw(ArgumentError("ntau must be larger than solve_order + 1"))
    if nmax ≥ 0
        getntime(G) ≥ nmax + 1 ||
            throw(ArgumentError("G must have at least $(nmax + 1) real time steps"))
        H.ntime ≥ nmax + 1 ||
            throw(ArgumentError("H must have at least $(nmax + 1) real time steps"))
    end
    _check_hamiltonian(H, nmax ≥ 0 ? (0:nmax+1) : 0, d₁)
end


#=
### *Private Service Functions* : *Hamiltonian Adapter*
=#

"""
    _assemble_pseudo_hamiltonian(G::PseudoGF, H0, H)

Assemble the Hamiltonian at the Matsubara axis `H0` and the real-time
Hamiltonian `H` into a `Cf{C64}` object `Hcf`, where `Hcf[0] = H0` and
`Hcf[n] = H[n]` (`n ≥ 1`, i.e., `t = (n-1)h`). `H0` is a matrix and `H`
is a vector of matrices. For 1 × 1 Green's functions, `H0` and `H` can
also be a number and a vector of numbers. It is a private function.
"""
function _assemble_pseudo_hamiltonian(G::PseudoGF, H0::AbstractMatrix,
                                      H::AbstractVector{<:AbstractMatrix})
    d = getdims(G)[1]
    size(H0) == (d, d) || throw(DimensionMismatch("H0 must be $d × $d"))
    Hcf = Cf(length(H), d, d)
    Hcf[0] = Matrix{C64}(H0)
    for n in eachindex(H)
        size(H[n]) == (d, d) || throw(DimensionMismatch("H[$n] must be $d × $d"))
        Hcf[n - firstindex(H) + 1] = Matrix{C64}(H[n])
    end
    return Hcf
end
#

function _assemble_pseudo_hamiltonian(G::PseudoGF, H0::Number,
                                      H::AbstractVector{<:Number})
    getdims(G) == (1, 1) ||
        throw(DimensionMismatch("scalar Hamiltonians need 1 × 1 Green's functions"))
    _assemble_pseudo_hamiltonian(G, fill(C64(H0), 1, 1),
                                 [fill(C64(x), 1, 1) for x in H])
end


#=
### *Private Service Functions* : *Matsubara Component*
=#

"""
    _pseudodyson_mat!(G::PseudoGF{C64}, λ0::F64, H0::AbstractMatrix,
                      Σ::PseudoGF{C64}, I::Integrator, β::F64)

Solve the Matsubara component of the pseudo-particle Dyson equation. See
*Remarks* : *Pseudo-Particle Dyson Equation*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_pseudodyson_impl.hpp`, `pseudodyson_mat_dispatch`.
"""
function _pseudodyson_mat!(G::PseudoGF{C64}, λ0::F64, H0::AbstractMatrix,
                           Σ::PseudoGF{C64}, I::Integrator, β::F64)
    k = I.k
    N = getntau(G) - 1 # NESSi ntau
    d = getdims(G)[1]
    δτ = β / N
    Id = Matrix{C64}(LinearAlgebra.I, d, d)
    gm(m) = _block(G.mat, m + 1)
    sm(m) = _block(Σ.mat, m + 1)

    # Initial value: Gᴹ(0) = -1
    gm(0) .= .-Id

    # The first k steps, solve ∑_q M_{pq} X_q = Q_p with X_q = Gᴹ(q)
    ws = _BlockSolver(k, d)
    fill!(ws.Mbig, zero(C64))
    fill!(ws.Qbig, zero(C64))
    for n = 1:k
        Qₚ = _qblock(ws, n)
        # Derivative -d/dτ Gᴹ(n)
        for m = 0:k
            cweight = -1.0 / δτ * I.PDW[n,m]
            if m == 0
                Qₚ .= Qₚ .- cweight .* gm(0)
            else
                Mₚₘ = _mblock(ws, n, m)
                Mₚₘ .= Mₚₘ .+ cweight .* Id
            end
        end
        # Hamiltonian
        Mₚₚ = _mblock(ws, n, n)
        Mₚₚ .= Mₚₚ .+ λ0 .* Id .- H0
        # Integral ∫₀^τ dx Σᴹ(τ - x) Gᴹ(x)
        if n < k
            for m = 0:k
                for j = 0:k
                    cweight = δτ * I.BCW[n-1,m,j]
                    if j == 0
                        mul!(Qₚ, sm(m), gm(0), cweight, true)
                    else
                        Mₚⱼ = _mblock(ws, n, j)
                        Mₚⱼ .= Mₚⱼ .- cweight .* sm(m)
                    end
                end
            end
        else
            for m = 0:k
                cweight = δτ * I.GIW[k,m]
                if m == 0
                    mul!(Qₚ, sm(k), gm(0), cweight, true)
                else
                    Mₚₘ = _mblock(ws, n, m)
                    Mₚₘ .= Mₚₘ .- cweight .* sm(k - m)
                end
            end
        end
    end
    _solve_block!(ws)
    for n = 1:k
        gm(n) .= _xblock(ws, n)
    end

    # The remaining steps: BDF(k+1) and Gregory integration
    ws1 = _BlockSolver(1, d)
    qq = zeros(C64, d, d)
    mm = zeros(C64, d, d)
    for n = k+1:N
        fill!(qq, zero(C64))
        for m = 0:n-1
            mul!(qq, sm(n - m), gm(m), I.GIW[n,m], true)
        end
        qq .= δτ .* qq
        for m = 1:k+1
            qq .= qq .+ (I.BDW[m] / δτ) .* gm(n - m)
        end
        mm .= (-I.BDW[0] / δτ + λ0) .* Id .- H0 .- (δτ * I.GIW[n,n]) .* sm(0)
        _solve_block_right!(gm(n), mm, qq, ws1)
    end

    return G
end


#=
### *Private Service Functions* : *Left-mixing Component*
=#

"""
    _pseudodyson_tv!(n::I64, G::PseudoGF{C64}, λ0::F64, H0::AbstractMatrix,
                     Σ::PseudoGF{C64}, I::Integrator, β::F64, h::F64)

Solve the left-mixing component ``G^{\\rceil}(n,\\tau)`` (NESSi time index
`n`) of the pseudo-particle Dyson equation. The greater component at time
step `n` (and at the first `k + 1` steps) must be known. See *Remarks* :
*Pseudo-Particle Dyson Equation*. It is a private function.

*References* :

Please see [`NESSi`] `cntr_pseudodyson_impl.hpp`, `pseudodyson_timestep_tv`.
"""
function _pseudodyson_tv!(n::I64, G::PseudoGF{C64}, λ0::F64, H0::AbstractMatrix,
                          Σ::PseudoGF{C64}, I::Integrator, β::F64, h::F64)
    k = I.k
    N = getntau(G) - 1 # NESSi ntau
    d = getdims(G)[1]
    sign = getsign(G)
    δτ = β / N
    n₁ = max(n, k)
    Id = Matrix{C64}(LinearAlgebra.I, d, d)
    H0d = Matrix{C64}(H0')
    sm(m) = _block(Σ.mat, m + 1)
    gtemp = zeros(C64, d, d)

    # X(i) = G^⌉(n,β - τᵢ), i = 0, ⋯, N
    xx = [zeros(C64, d, d) for _ = 0:N]

    # Source term: ∫₀^t dj G^{>}(n,j) Σ^⌉(j,β - τᵢ), i = 1, ⋯, N
    for j = 0:n₁
        if j ≤ n
            gtemp .= _ret_blk(G, n, j)
        else
            gtemp .= .-(_ret_blk(G, j, n))'
        end
        gtemp .= (I.GIW[n,j] * h) .* gtemp
        for i = 1:N
            mul!(xx[i+1], gtemp, _tv_blk(Σ, j, N - i), true, true)
        end
    end

    # Initial value: X(0) = ξ G^{>}(n,0)
    xx[1] .= sign .* _ret_blk(G, n, 0)

    # The first k steps, solve ∑_q X_q M_{pq} = Q_p with X_q = X(q)
    ws = _BlockSolver(k, d)
    M = [zeros(C64, d, d) for _ = 1:k, _ = 1:k]
    Q = [copy(xx[i+1]) for i = 1:k]
    for i = 1:k
        Qₚ = Q[i]
        # Derivative -d/dl X(i)
        for m = 0:k
            cweight = -1.0 / δτ * I.PDW[i,m]
            if m == 0
                Qₚ .= Qₚ .- cweight .* xx[1]
            else
                M[i,m] .= M[i,m] .+ cweight .* Id
            end
        end
        # Hamiltonian
        M[i,i] .= M[i,i] .+ λ0 .* Id .- H0d
        # Integral ∫₀^l dx X(x) Σᴹ(l - x)
        if i < k
            for m = 0:k
                for j = 0:k
                    cweight = δτ * I.BCW[i-1,m,j]
                    if j == 0
                        mul!(Qₚ, xx[1], sm(m), cweight, true)
                    else
                        M[i,j] .= M[i,j] .- cweight .* sm(m)
                    end
                end
            end
        else
            for m = 0:k
                cweight = δτ * I.GIW[i,m]
                if m == 0
                    mul!(Qₚ, xx[1], sm(k), cweight, true)
                else
                    M[i,m] .= M[i,m] .- cweight .* sm(k - m)
                end
            end
        end
    end
    _solve_block_left!(view(xx, 2:k+1), M, Q, ws)

    # The remaining steps: BDF(k+1) and Gregory integration
    ws1 = _BlockSolver(1, d)
    qq = zeros(C64, d, d)
    mm = zeros(C64, d, d)
    for i = k+1:N
        fill!(qq, zero(C64))
        for m = 0:i-1
            mul!(qq, xx[m+1], sm(i - m), I.GIW[i,m], true)
        end
        qq .= δτ .* qq .+ xx[i+1]
        for m = 1:k+1
            qq .= qq .+ (I.BDW[m] / δτ) .* xx[i-m+1]
        end
        mm .= (-I.BDW[0] / δτ + λ0) .* Id .- H0d .- (δτ * I.GIW[i,i]) .* sm(0)
        _solve_block_left!(xx[i+1], mm, qq, ws1)
    end

    # G^⌉(n,τ_{N-l}) = X(l)
    for l = 0:N
        _tv_blk(G, n, N - l) .= xx[l+1]
    end

    return G
end


#=
### *Public API* : *Pseudo-Particle Dyson*
=#

"""
    pseudodyson_mat!(G::PseudoGF{C64}, λ0::F64, H0, Σ::PseudoGF{C64},
                     I::Integrator, β::F64)

Solve the Matsubara component of the pseudo-particle Dyson equation
``[-\\partial_\\tau + \\lambda_0 - H_0 - \\Sigma] \\ast G = 0`` with
``G^{M}(0) = -1``. Here `H0` is a matrix, a number (1 × 1 only), or a
`Cf{C64}` object (`H0[0]` is used).

### Arguments
* G  -> Pseudo-particle Green's function.
* λ0 -> Pseudo-particle chemical potential, not included in `H0`.
* H0 -> Hamiltonian at the Matsubara axis.
* Σ  -> Pseudo-particle self-energy.
* I  -> Struct for numerical integration, 1 ≤ I.k ≤ 5.
* β  -> Inverse temperature.

### Returns
* G -> Pseudo-particle Green's function.

*References* :

Please see [`NESSi`] `cntr_pseudodyson_impl.hpp`, `pseudodyson_mat`.
"""
function pseudodyson_mat!(G::PseudoGF{C64}, λ0::F64, H0::Cf{C64},
                          Σ::PseudoGF{C64}, I::Integrator, β::F64)
    _check_pseudodyson(G, Σ, H0, I, -1)
    β > 0.0 || throw(ArgumentError("β must be positive"))
    _pseudodyson_mat!(G, λ0, H0[0], Σ, I, β)
end
#

function pseudodyson_mat!(G::PseudoGF{C64}, λ0::F64, H0::Union{AbstractMatrix,Number},
                          Σ::PseudoGF{C64}, I::Integrator, β::F64)
    Hcf = _assemble_pseudo_hamiltonian(G, H0, H0 isa Number ? C64[] : Matrix{C64}[])
    pseudodyson_mat!(G, λ0, Hcf, Σ, I, β)
end


"""
    pseudodyson_start!(G::PseudoGF{C64}, λ0::F64, H::Cf{C64},
                       Σ::PseudoGF{C64}, I::Integrator, β::F64, h::F64)
    pseudodyson_start!(G::PseudoGF{C64}, λ0::F64, H0, H,
                       Σ::PseudoGF{C64}, I::Integrator, β::F64, h::F64)

Solve the pseudo-particle Dyson equation at the first `k + 1` real time
steps (`tstp = 1, ⋯, k + 1`). The Matsubara component of `G` must be
known. The components are solved in the order: greater (stored as
`ret`), left-mixing, and lesser. `H[0]` (or `H0`) is the Hamiltonian at
the Matsubara axis, which is used in the left-mixing component, while
`H[n]` (`n ≥ 1`) is the Hamiltonian at `t = (n-1)h`. For 1 × 1 Green's
functions, `H0` and `H` can be a number and a vector of numbers.

### Arguments
* G  -> Pseudo-particle Green's function.
* λ0 -> Pseudo-particle chemical potential, not included in `H`.
* H  -> Hamiltonian.
* Σ  -> Pseudo-particle self-energy.
* I  -> Struct for numerical integration, 1 ≤ I.k ≤ 5.
* β  -> Inverse temperature.
* h  -> Time step.

### Returns
* G -> Pseudo-particle Green's function.

*References* :

Please see [`NESSi`] `cntr_pseudodyson_impl.hpp`, `pseudodyson_start`.
"""
function pseudodyson_start!(G::PseudoGF{C64}, λ0::F64, H::Cf{C64},
                            Σ::PseudoGF{C64}, I::Integrator, β::F64, h::F64)
    k = I.k
    _check_pseudodyson(G, Σ, H, I, k)
    β > 0.0 || throw(ArgumentError("β must be positive"))
    h > 0.0 || throw(ArgumentError("h must be positive"))

    _dyson_start_ret!(G, λ0, H, Σ, h, I)
    for n = 0:k
        _pseudodyson_tv!(n, G, λ0, H[0], Σ, I, β, h)
    end
    _dyson_start_less!(G, λ0, H, Σ, β, h, I)

    return G
end
#

function pseudodyson_start!(G::PseudoGF{C64}, λ0::F64, H0, H::AbstractVector,
                            Σ::PseudoGF{C64}, I::Integrator, β::F64, h::F64)
    Hcf = _assemble_pseudo_hamiltonian(G, H0, H)
    pseudodyson_start!(G, λ0, Hcf, Σ, I, β, h)
end


"""
    pseudodyson_timestep!(tstp::I64, G::PseudoGF{C64}, λ0::F64, H::Cf{C64},
                          Σ::PseudoGF{C64}, I::Integrator, β::F64, h::F64)
    pseudodyson_timestep!(tstp::I64, G::PseudoGF{C64}, λ0::F64, H0, H,
                          Σ::PseudoGF{C64}, I::Integrator, β::F64, h::F64)

Solve the pseudo-particle Dyson equation at time step `tstp ≥ k + 2`.
The components at the earlier time steps must be known. See
[`pseudodyson_start!`](@ref) for the arguments. Note that the left-mixing
component always uses the Hamiltonian at the Matsubara axis.

*References* :

Please see [`NESSi`] `cntr_pseudodyson_impl.hpp`, `pseudodyson_timestep`.
"""
function pseudodyson_timestep!(tstp::I64, G::PseudoGF{C64}, λ0::F64, H::Cf{C64},
                               Σ::PseudoGF{C64}, I::Integrator, β::F64, h::F64)
    k = I.k
    tstp ≥ k + 2 || throw(ArgumentError("tstp must be larger than solve_order + 1"))
    _check_pseudodyson(G, Σ, H, I, tstp - 1)
    β > 0.0 || throw(ArgumentError("β must be positive"))
    h > 0.0 || throw(ArgumentError("h must be positive"))

    n = tstp - 1
    _dyson_timestep_ret!(tstp, G, λ0, H, Σ, h, I)
    _pseudodyson_tv!(n, G, λ0, H[0], Σ, I, β, h)
    _dyson_timestep_less!(tstp, G, λ0, H, Σ, β, h, I)

    return G
end
#

function pseudodyson_timestep!(tstp::I64, G::PseudoGF{C64}, λ0::F64, H0, H::AbstractVector,
                               Σ::PseudoGF{C64}, I::Integrator, β::F64, h::F64)
    Hcf = _assemble_pseudo_hamiltonian(G, H0, H)
    pseudodyson_timestep!(tstp, G, λ0, Hcf, Σ, I, β, h)
end


"""
    pseudodyson!(G::PseudoGF{C64}, λ0::F64, H::Cf{C64},
                 Σ::PseudoGF{C64}, I::Integrator, β::F64, h::F64)
    pseudodyson!(G::PseudoGF{C64}, λ0::F64, H0, H,
                 Σ::PseudoGF{C64}, I::Integrator, β::F64, h::F64)

Solve the pseudo-particle Dyson equation at the Matsubara axis and at all
the real time steps, i.e., [`pseudodyson_mat!`](@ref),
[`pseudodyson_start!`](@ref), and [`pseudodyson_timestep!`](@ref) for
`tstp = k + 2, ⋯, ntime`. See [`pseudodyson_start!`](@ref) for the
arguments.
"""
function pseudodyson!(G::PseudoGF{C64}, λ0::F64, H::Cf{C64},
                      Σ::PseudoGF{C64}, I::Integrator, β::F64, h::F64)
    k = I.k
    _check_pseudodyson(G, Σ, H, I, getntime(G) - 1)

    pseudodyson_mat!(G, λ0, H, Σ, I, β)
    pseudodyson_start!(G, λ0, H, Σ, I, β, h)
    for tstp = k+2:getntime(G)
        pseudodyson_timestep!(tstp, G, λ0, H, Σ, I, β, h)
    end

    return G
end
#

function pseudodyson!(G::PseudoGF{C64}, λ0::F64, H0, H::AbstractVector,
                      Σ::PseudoGF{C64}, I::Integrator, β::F64, h::F64)
    Hcf = _assemble_pseudo_hamiltonian(G, H0, H)
    pseudodyson!(G, λ0, Hcf, Σ, I, β, h)
end

