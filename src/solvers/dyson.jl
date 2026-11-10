#
# Project : Lavender
# Source  : dyson.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
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


