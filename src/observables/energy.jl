#
# Project : Lavender
# Source  : energy.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

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
    kinetic_energy_trunc(tstp::I64, G::ℱᵐᵒᵛ{C64}, D::ℱᵐᵒᵛ{C64},
                         I::Integrator, h::F64)

Truncated kinetic (or correlation) energy integral at the leading row,

```math
E = \\mathrm{Im}\\,\\mathrm{Tr}\\, h\\sum_{l=0}^{t_c} w_l
\\left[G^{R}(t,t-l)^{\\dagger} D^{<}(t,t-l)
- G^{<}(t,t-l)^{\\dagger} D^{R}(t,t-l)\\right],
```

with the Gregory end weights ``\\omega`` at both ends and the weight 1
in between. It requires `tc ≥ 2k + 2`.

*References* :

Please see [`NESSi`] `cntr_dyson_moving_impl.hpp`,
`get_kinetic_trunc_timestep` (without its debug output).
"""
function kinetic_energy_trunc(tstp::I64, G::ℱᵐᵒᵛ{C64}, D::ℱᵐᵒᵛ{C64},
                              I::Integrator, h::F64)
    k = I.k
    _check_moving_solver(tstp, k, G, D)
    tc = G.tc
    d = getdims(G)[1]
    s = zeros(C64, d, d)
    for l = 0:tc
        w = l ≤ k ? I.GIW.ω[l+1] : (l ≥ tc - k ? I.GIW.ω[tc-l+1] : 1.0)
        mul!(s, _mblock(G.ret, 0, l)', _mblock(D.less, 0, l), h * w, one(C64))
        mul!(s, _mblock(G.less, 0, l)', _mblock(D.ret, 0, l), -h * w, one(C64))
    end
    return imag(tr(s))
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

