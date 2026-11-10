#
# Project : Lavender
# Source  : green_from_H.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks* : *Free Green's Functions From Hamiltonian*

Let ``K = \mu - H(-i\beta) = R\ \mathrm{diag}(\lambda) R^{\dagger}``, and
``U_n`` be the propagator from 0 to ``t_n``, including the phase
``e^{i\mu t_n}``. For fermions,

```math
\begin{equation}
G^{M}(\tau) = -R\ \mathrm{diag}[e^{\tau\lambda}n_F(\lambda)] R^{\dagger},
\quad
G^{\rceil}(t_n,\tau) = iU_n R\ \mathrm{diag}[e^{-\tau\lambda}n_F(-\lambda)]
R^{\dagger},
\end{equation}
```

```math
\begin{equation}
G^{R}(t_n,t_j) = -iU_n U^{\dagger}_j,
\quad
G^{<}(t_j,t_n) = iU_j N U^{\dagger}_n,
\quad
N = R\ \mathrm{diag}[n_F(-\lambda)] R^{\dagger}.
\end{equation}
```

For bosons, ``G^{M}`` and ``N`` change sign, ``n_F`` is replaced by
``n_B``, and ``G^{\rceil}`` is multiplied by -1.

For a constant Hamiltonian, ``U_n = e^{ihnK}``. For a time-dependent
Hamiltonian, the commutator-free expansions of order 2 and 4 are used,

```math
\begin{equation}
U_n = e^{-ih H(t_{n-1/2})} U_{n-1}, \quad
U_n = e^{-ih(a_1 H_1 + a_2 H_2)} e^{-ih(a_2 H_1 + a_1 H_2)} U_{n-1},
\end{equation}
```

where ``H_{1,2} = H(t_{n-1} + c_{1,2} h)``, ``c_{1,2} = 1/2 \mp
\sqrt{3}/6``, ``a_{1,2} = (3 \mp 2\sqrt{3})/12``. The Hamiltonian at the
fractional time points is obtained by polynomial interpolation of order
`solve_order`. Finally ``U_n \to U_n e^{i\mu t_n}``.

*References* :

Please see [`NESSi`] `cntr_equilibrium_impl.hpp`, `green_from_H`,
`propagator_exp`.
=#

#=
### *Private Service Functions* : *Free Green's Functions*
=#

"""
    _hamiltonian_cf(nkb::I64, H::Cf{C64}, kt::I64, fix::Bool)

Prepare the Hamiltonian for the propagation from the KB time step
`nkb - 1` to `nkb`. If `fix = false`, `H[nkb]` is extrapolated from the
previous time steps when `nkb - 1 > kt`. It returns the KB index of the
last grid point of the interpolation interval. It is a private function.
"""
function _hamiltonian_cf(nkb::I64, H::Cf{C64}, kt::I64, fix::Bool)
    tstp = nkb - 1 # NESSi index
    if !fix && tstp > kt
        extrapolate_timestep!(nkb - 1, H, Integrator(kt))
    end
    n₁ = tstp ≤ kt ? kt : tstp
    return n₁ + 1
end

"""
    _propagator_cf2!(U::Vector{Matrix{C64}}, nkb::I64, H::Cf{C64},
                     h::F64, kt::I64, fix::Bool)

Second order commutator-free propagation, ``U_n = e^{-ihH(t_{n-1/2})}
U_{n-1}``, where `n = nkb - 1`. It is a private function.

*References* :

Please see [`NESSi`] `cntr_equilibrium_impl.hpp`, `propagator_exp`.
"""
function _propagator_cf2!(U::Vector{Matrix{C64}}, nkb::I64, H::Cf{C64},
                          h::F64, kt::I64, fix::Bool)
    n₁ = _hamiltonian_cf(nkb, H, kt, fix)
    Hm = interpolation(n₁, nkb - 0.5, H, Integrator(kt))
    U[nkb] = exp((-im * h) .* Hm) * U[nkb-1]
    return U
end

"""
    _propagator_cf4!(U::Vector{Matrix{C64}}, nkb::I64, H::Cf{C64},
                     h::F64, kt::I64, fix::Bool)

Fourth order commutator-free propagation, ``U_n = e^{-ih(a_1 H_1 + a_2
H_2)} e^{-ih(a_2 H_1 + a_1 H_2)} U_{n-1}``, where `n = nkb - 1`. It is a
private function.

*References* :

Please see [`NESSi`] `cntr_equilibrium_impl.hpp`, `propagator_exp`.
"""
function _propagator_cf4!(U::Vector{Matrix{C64}}, nkb::I64, H::Cf{C64},
                          h::F64, kt::I64, fix::Bool)
    n₁ = _hamiltonian_cf(nkb, H, kt, fix)
    H₁ = interpolation(n₁, nkb - 0.5 - sqrt(3) / 6.0, H, Integrator(kt))
    H₂ = interpolation(n₁, nkb - 0.5 + sqrt(3) / 6.0, H, Integrator(kt))
    a₁ = (3.0 - 2.0 * sqrt(3.0)) / 12.0
    a₂ = (3.0 + 2.0 * sqrt(3.0)) / 12.0
    arg₁ = (-im * h) .* (a₁ .* H₁ .+ a₂ .* H₂)
    arg₂ = (-im * h) .* (a₂ .* H₁ .+ a₁ .* H₂)
    U[nkb] = exp(arg₁) * exp(arg₂) * U[nkb-1]
    return U
end

"""
    _fill_green_from_propagator!(G::ℱ{C64}, μ::F64, H0::AbstractMatrix{C64},
                                 β::F64, U::Vector{Matrix{C64}})

Fill all the components of the free Green's function `G` from the
Hamiltonian `H0` at the Matsubara axis and the propagators `U[n]`
(including the phase ``e^{i\\mu t}``). See *Remarks* : *Free Green's
Functions From Hamiltonian*. It is a private function.
"""
function _fill_green_from_propagator!(G::ℱ{C64}, μ::F64, H0::AbstractMatrix{C64},
                                      β::F64, U::Vector{Matrix{C64}})
    sign = getsign(G)
    ntime = getntime(G)
    ntau = getntau(G)
    d = getdims(G)[1]
    δτ = β / (ntau - 1)

    K = Hermitian(μ * Matrix{C64}(I, d, d) - H0)
    λ, R = eigen(K)

    # Matsubara and left-mixing components
    for m = 1:ntau
        τ = (m - 1) * δτ
        if sign == FERMI
            _block(G.mat, m) .= -R * Diagonal(fermi_exp(β, τ, λ)) * R'
            X = im * R * Diagonal(fermi_exp(β, τ, -λ)) * R'
        else
            _block(G.mat, m) .= R * Diagonal(bose_exp(β, τ, λ)) * R'
            X = -im * R * Diagonal(bose_exp(β, τ, -λ)) * R'
        end
        for n = 1:ntime
            mul!(_block(G.lmix, n, m), U[n], X)
        end
    end

    # Retarded and lesser components
    if sign == FERMI
        N = R * Diagonal(fermi(β, -λ)) * R'
    else
        N = -R * Diagonal(bose(β, -λ)) * R'
    end
    for m = 1:ntime, n = 1:m
        _block(G.ret, m, n) .= -im .* (U[m] * U[n]')
        _block(G.less, n, m) .= im .* (U[n] * N * U[m]')
    end

    return G
end

"""
    _check_free_hamiltonian(G::ℱ{C64}, H0::AbstractMatrix{C64})

Check the statistics and the shapes for the free Green's functions. It
is a private function.
"""
function _check_free_hamiltonian(G::ℱ{C64}, H0::AbstractMatrix{C64})
    getsign(G) in (FERMI, BOSE) || throw(ArgumentError("unknown statistics of G"))
    d₁, d₂ = getdims(G)
    d₁ == d₂ || throw(DimensionMismatch("G must be square"))
    size(H0) == (d₁, d₂) || throw(DimensionMismatch("H must be $d₁ × $d₂"))
    all(isfinite, H0) || throw(DomainError(H0, "H is not finite"))
    if norm(H0 - H0') > 1e-12 * max(1.0, norm(H0))
        throw(ArgumentError("H is not hermitian"))
    end
end

#=
### *Public API* : *Free Green's Functions From Hamiltonian*
=#

"""
    green_from_H!(G::ℱ{C64}, μ::F64, H0::Matrix{C64}, β::F64, h::F64)

Calculate the free Green's function of the constant hermitian
Hamiltonian `H0`. See *Remarks* : *Free Green's Functions From
Hamiltonian*.

### Arguments
* G  -> Contour-ordered Green's function.
* μ  -> Chemical potential.
* H0 -> Hamiltonian.
* β  -> Inverse temperature.
* h  -> Time step.

### Returns
* G -> Contour-ordered Green's function.
"""
function green_from_H!(G::ℱ{C64}, μ::F64, H0::Matrix{C64}, β::F64, h::F64)
    _check_free_hamiltonian(G, H0)
    β > 0.0 || throw(ArgumentError("β must be positive"))
    d = getdims(G)[1]
    ntime = getntime(G)

    # U_n = exp(i h (μ - H0))^n
    Udt = exp((im * h) .* (μ * Matrix{C64}(I, d, d) - H0))
    U = Vector{Matrix{C64}}(undef, ntime)
    U[1] = Matrix{C64}(I, d, d)
    for n = 2:ntime
        U[n] = U[n-1] * Udt
    end

    _fill_green_from_propagator!(G, μ, H0, β, U)
end

"""
    green_from_H!(G::ℱ{C64}, μ::F64, H::Cf{C64}, β::F64, h::F64;
                  fix_hamiltonian::Bool = true, solve_order::I64 = 5,
                  cf_order::I64 = 4)

Calculate the free Green's function of the time-dependent hermitian
Hamiltonian `H`. The Matsubara component is determined by `H[0]`. The
propagators are obtained by the commutator-free expansion of order
`cf_order` (2 or 4), and the Hamiltonian at the fractional time points
is interpolated with the order `solve_order`. If `fix_hamiltonian =
false`, the values of `H` at `t > solve_order` are extrapolated from the
previous time steps and written into `H`.

### Arguments
* G               -> Contour-ordered Green's function.
* μ               -> Chemical potential.
* H               -> Hamiltonian.
* β               -> Inverse temperature.
* h               -> Time step.
* fix_hamiltonian -> Whether `H` is given at all the time steps.
* solve_order     -> Order of the interpolation, 1 ≤ solve_order ≤ 5.
* cf_order        -> Order of the commutator-free expansion, 2 or 4.

### Returns
* G -> Contour-ordered Green's function.
"""
function green_from_H!(G::ℱ{C64}, μ::F64, H::Cf{C64}, β::F64, h::F64;
                       fix_hamiltonian::Bool = true, solve_order::I64 = 5,
                       cf_order::I64 = 4)
    # Sanity check
    solve_order in 1:5 ||
        throw(ArgumentError("solve_order must be in 1:5, got $solve_order"))
    cf_order in (2, 4) || throw(ArgumentError("cf_order must be 2 or 4, got $cf_order"))
    β > 0.0 || throw(ArgumentError("β must be positive"))
    h > 0.0 || throw(ArgumentError("h must be positive"))
    _check_free_hamiltonian(G, H[0])
    d = getdims(G)[1]
    ntime = getntime(G)
    kt = solve_order
    #
    # The Hamiltonian is needed at the KB time steps 1, ⋯, max(kt, nt) + 1
    # (fix_hamiltonian = true), or 1, ⋯, kt + 1 (fix_hamiltonian = false).
    nmax = fix_hamiltonian ? max(kt + 1, ntime) : kt + 1
    H.ntime ≥ max(nmax, ntime) ||
        throw(ArgumentError("H must have at least $(max(nmax, ntime)) real time steps"))
    _check_hamiltonian(H, 1:nmax, d)

    # Propagators without the phase
    U = Vector{Matrix{C64}}(undef, ntime)
    U[1] = Matrix{C64}(I, d, d)
    for nkb = 2:ntime
        if cf_order == 2
            _propagator_cf2!(U, nkb, H, h, kt, fix_hamiltonian)
        else
            _propagator_cf4!(U, nkb, H, h, kt, fix_hamiltonian)
        end
    end

    # Phase exp(i μ t)
    for nkb = 1:ntime
        U[nkb] = U[nkb] * cis(μ * h * (nkb - 1))
    end

    _fill_green_from_propagator!(G, μ, H[0], β, U)
end

#=
*Remarks* : *How To Initialize Keldysh Green's Functions*

**Free Green's Functions**

Free Green's functions ``G_0(t,t')`` are determined from the following
equation of motion:

```math
\begin{equation}
[ i\partial_t - \epsilon(t) ] G_0(t,t') = \delta_{\mathcal{C}}(t,t').
\end{equation}
```

Let us denote the eigenvalues of the Hamiltonian matrix ``\epsilon(0^-)``
by ``\varepsilon_{\alpha}`` and the corresponding basis transformation
matrix by ``\mathbb{R}``, such that

```math
\begin{equation}
\epsilon(0^-) = \mathbb{R}~
                \text{diag}\{\varepsilon_{\alpha}\}~
                \mathbb{R}^{\dagger}.
\end{equation}
```

**Matsubara Component**

The Matsubara component is then given by

```math
\begin{equation}
G^M_0(\tau) = \mathbb{R}~
              \text{diag}
              \left\{
                  f_{\xi}(\mu - \varepsilon_{\alpha})
                  e^{(\mu-\varepsilon_{\alpha})\tau}
              \right\}~
              \mathbb{R}^{\dagger}.
\end{equation}
```

for ``\tau \in (0,\beta)``.

**Unitary Evolution Operator**

All other Keldysh components of ``G_0(t,t')`` are governed by the unitary
evolution with respect to the single-particle Hamiltonian ``\epsilon(t)``.
The time evolution operator is defined as follows:

```math
\begin{equation}
U(t_1,t_2) = T \exp
             \left[
             -i \int^{t_1}_{t_2} dt~H_{\mathcal{C}}(t)
             \right]
\end{equation}
```

for ``t_1 > t_2`` and

```math
\begin{equation}
U(t_1,t_2) = \bar{T} \exp
             \left[
             i \int^{t_2}_{t_1} dt~H_{\mathcal{C}}(t)
             \right]
\end{equation}
```

for ``t_2 > t_1``. Here, ``T (\bar{T})`` denotes the chronological
(or anti-chronological) time ordering symbol. On the equidistant grid
``t_n = nh``, we approximate the propagator ``U_{n,j} \equiv U(nh,jh)``
by the commutator-free matrix exponential approximation. In particular,
the semi-group property:

```math
\begin{equation}
U_{n,j} = U_{n,n-1} U_{n-1,n-2} \cdots U_{j+1,j},
\end{equation}
```

is applied.

**Other Keldysh Components**

Based on the commutator-free matrix exponential approximation, the other
Keldysh components are determined by:

```math
\begin{equation}
G^{⌉}_0(nh,\tau) = -i \xi U_{n,0}(nh,0)
                   \mathbb{R}~
                   \text{diag}
                   \left\{
                   f_{\xi}(\varepsilon_{\alpha} - \mu)
                   e^{(\varepsilon_{\alpha}-\mu)\tau}
                   \right\}~
                   \mathbb{R}^{\dagger},
\end{equation}
```

```math
\begin{equation}
G^{R}_0(nh,jh) = -i U_{n,j} = U_{n,0}[U_{j,0}]^{\dagger},
\end{equation}
```

```math
\begin{equation}
G^{<}_0(jh,nh) = i U_{j,0}
                 \mathbb{R}~
                 \text{diag}
                 \left\{
                     f_{\xi}(\varepsilon_{\alpha}-\mu)
                 \right\}
                 \mathbb{R}^{\dagger}
                 [U_{n,0}]^{\dagger}.
\end{equation}
```

*References* :

Please see [`NESSi`] Section `14` for more details.
=#

"""
    init_green!(G::ℱ{T}, H₀::Matrix{T}, μ::F64, β::F64, h::F64)

Try to generate initial contour-ordered Green's function `G`. Here, `H₀`
is the band dispersion, `μ` is the chemical potential, `β` (≡ 1/𝑇) is the
inverse temperature, and `h` (≡ δ𝑡) is the length of time step at real
time axis. It is a wrapper of `green_from_H!()`.

See also: [`ℱ`](@ref), [`green_from_H!`](@ref).
"""
function init_green!(G::ℱ{T}, H₀::Matrix{T}, μ::F64, β::F64, h::F64) where {T}
    green_from_H!(G, μ, H₀, β, h)
end

