#
# Project : Lavender
# Source  : observables.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/09
#

#=
*Remarks* : *Observables*

The density matrix is ``\rho = -G^{M}(\beta)`` in equilibrium, and
``\rho(t) = i\xi G^{<}(t,t)`` at real time ``t``. The kinetic energy is
``E_{kin} = \mathrm{Re}\ \mathrm{Tr}[H \rho]``. The correlation energy is

```math
\begin{equation}
E_{corr} = \frac{1}{2} \mathrm{Re}\ \mathrm{Tr}[\rho_{\Sigma G}],
\end{equation}
```

where ``\rho_{\Sigma G} = -[\Sigma * G]^{M}(\beta)`` in equilibrium, and
``\rho_{\Sigma G} = -i[\Sigma * G]^{<}(t,t)`` at real time ``t``. Note
that the order of the convolution is ``\Sigma * G``.

*References* :

Please see [`NESSi`] `cntr_herm_matrix_impl.hpp`, `density_matrix`, and
`cntr_utilities_impl.hpp`, `correlation_energy`.
=#

"""
    density_matrix(cfm::ℱ{T}, tstp::I64)
    density_matrix(cfm::PseudoGF{T}, tstp::I64)

Returns the density matrix at given time step `tstp`. If `tstp = 0`, it
is the equilibrium one, ``-G^{M}(\\beta)``. If `tstp > 0`, it is
``i\\xi G^{<}(t,t)``. The same formulas are used for the pseudo-particle
Green's function, see [`NESSi`] `cntr_herm_pseudo_impl.hpp`,
`density_matrix`.

### Arguments
* cfm  -> Contour-ordered (or pseudo-particle) Green's function.
* tstp -> Index of time step.

### Returns
* ρ -> Density matrix.

See also: [`density`](@ref).
"""
function density_matrix(cfm::Union{ℱ{T},PseudoGF{T}}, tstp::I64) where {T}
    # Sanity check
    0 ≤ tstp ≤ getntime(cfm) || throw(BoundsError(cfm, tstp))

    if tstp == 0
        return -cfm.mat[getntau(cfm)]
    else
        return cfm.less[tstp, tstp] * getsign(cfm) * im
    end
end

"""
    density_matrix(cfv::𝒻{S}, tstp::I64)
    density_matrix(cfv::PseudoGFStep{S}, tstp::I64)

Returns the density matrix at given time step `tstp`, which must be the
time step of `cfv`. See [`density_matrix`](@ref).

See also: [`density`](@ref).
"""
function density_matrix(cfv::Union{𝒻{S},PseudoGFStep{S}}, tstp::I64) where {S}
    # Sanity check
    tstp == gettstp(cfv) || throw(ArgumentError("tstp must be the time step of cfv"))

    if tstp == 0
        return -cfv.mat[getntau(cfv)]
    else
        return cfv.less[tstp] * getsign(cfv) * im
    end
end

"""
    density(cfm::ℱ{T}, tstp::I64)

Returns the density matrix at given time step `tstp`. If `tstp = 0`,
it denotes the equilibrium state. However, when `tstp > 0`, it means
the nonequilibrium state. It is a wrapper of `density_matrix()`.

See also: [`density_matrix`](@ref).
"""
function density(cfm::Union{ℱ{T},PseudoGF{T}}, tstp::I64) where {T}
    density_matrix(cfm, tstp)
end

"""
    density(cfv::𝒻{S}, tstp::I64)

Returns the density matrix at given time step `tstp`. If `tstp = 0`,
it denotes the equilibrium state. However, when `tstp > 0`, it means
the nonequilibrium state. It is a wrapper of `density_matrix()`.

See also: [`density_matrix`](@ref).
"""
function density(cfv::Union{𝒻{S},PseudoGFStep{S}}, tstp::I64) where {S}
    density_matrix(cfv, tstp)
end

"""
    kinetic_energy(H::AbstractMatrix, ρ::AbstractMatrix)

Returns the kinetic energy ``\\mathrm{Re}\\ \\mathrm{Tr}[H \\rho]``.

### Arguments
* H -> Hamiltonian.
* ρ -> Density matrix.

### Returns
* E -> Kinetic energy.
"""
function kinetic_energy(H::AbstractMatrix, ρ::AbstractMatrix)
    size(H, 2) == size(ρ, 1) && size(H, 1) == size(ρ, 2) ||
        throw(DimensionMismatch("H and ρ are not compatible"))
    return real(tr(H * ρ))
end

"""
    kinetic_energy(H::Cf{T}, cfm::Union{ℱ{T},𝒻{T}}, tstp::I64)

Returns the kinetic energy at given time step `tstp`, i.e.,
``\\mathrm{Re}\\ \\mathrm{Tr}[H(t) \\rho(t)]``. Here `H[tstp]` is used.

See also: [`density_matrix`](@ref).
"""
function kinetic_energy(H::Cf{T}, cfm::Union{ℱ{T},𝒻{T}}, tstp::I64) where {T}
    0 ≤ tstp ≤ H.ntime || throw(BoundsError(H, tstp))
    return kinetic_energy(H[tstp], density_matrix(cfm, tstp))
end

"""
    correlation_energy(tstp::I64, G::ℱ{T}, Σ::ℱ{T}, I::Integrator,
                       β::F64, h::F64)

Returns the correlation energy at given time step `tstp`, see *Remarks* :
*Observables*.

### Arguments
* tstp -> Index of time step.
* G    -> Contour-ordered Green's function.
* Σ    -> Self-energy.
* I    -> An Integrator struct.
* β    -> Inverse temperature.
* h    -> Time step.

### Returns
* E -> Correlation energy.

See also: [`convolution_density_matrix`](@ref).
"""
function correlation_energy(tstp::I64, G::ℱ{T}, Σ::ℱ{T}, I::Integrator,
                            β::F64, h::F64) where {T}
    0 ≤ tstp ≤ getntime(G) || throw(BoundsError(G, tstp))
    β > 0.0 || throw(ArgumentError("β must be positive"))
    h > 0.0 || throw(ArgumentError("h must be positive"))
    ρ = convolution_density_matrix(tstp, Σ, G, I, β, h)
    return 0.5 * real(tr(ρ))
end

"""
    correlation_energy(tstp::I64, G::ℱ{T}, Σ::ℱ{T}, β::F64, h::F64,
                       solve_order::I64)

Returns the correlation energy at given time step `tstp`. The order of
the integration is `solve_order`.

See also: [`correlation_energy`](@ref).
"""
function correlation_energy(tstp::I64, G::ℱ{T}, Σ::ℱ{T}, β::F64, h::F64,
                            solve_order::I64) where {T}
    solve_order in 1:5 ||
        throw(ArgumentError("solve_order must be in 1:5, got $solve_order"))
    correlation_energy(tstp, G, Σ, Integrator(solve_order), β, h)
end

"""
    current(v::AbstractVector{<:AbstractMatrix},
            ρ::AbstractVector{<:AbstractMatrix},
            w::AbstractVector{<:Real})

Discrete k sum of the current, ``J = \\sum_k w_k \\mathrm{Re\\,Tr}[v_k \\rho_k]``,
where `v[k]` is the velocity matrix, `ρ[k]` the density matrix, and `w[k]`
the weight of the k point. The sum runs over the k points in ascending
order. The model dependent velocities are not part of this function.

### Arguments
* v -> Velocity matrices.
* ρ -> Density matrices.
* w -> Weights of the k points.

### Returns
* J -> Current.

*References* :

Please see [`NESSi`] `examples/programs/gw_selfene_impl.cpp`, `current`.
"""
function current(v::AbstractVector{<:AbstractMatrix},
                 ρ::AbstractVector{<:AbstractMatrix},
                 w::AbstractVector{<:Real})
    (length(v) == length(ρ) == length(w)) ||
        throw(DimensionMismatch("v, ρ, and w must have the same length"))
    J = 0.0
    for k in eachindex(v, ρ, w)
        (size(v[k], 2) == size(ρ[k], 1) && size(v[k], 1) == size(ρ[k], 2)) ||
            throw(DimensionMismatch("v[$k] and ρ[$k] are not compatible"))
        J += w[k] * real(tr(v[k] * ρ[k]))
    end
    return J
end
