#
# Project : Lavender
# Source  : density.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
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
    density(cfm::ℱ{T}, tstp::I64)

Returns the density matrix at given time step `tstp`. If `tstp = 0`,
it denotes the equilibrium state. However, when `tstp > 0`, it means
the nonequilibrium state. It is a wrapper of `density_matrix()`.

See also: [`density_matrix`](@ref).
"""
function density(cfm::Union{ℱ{T},ℱᵖˢ{T}}, tstp::I64) where {T}
    density_matrix(cfm, tstp)
end

"""
    density(cfv::𝒻{S}, tstp::I64)

Returns the density matrix at given time step `tstp`. If `tstp = 0`,
it denotes the equilibrium state. However, when `tstp > 0`, it means
the nonequilibrium state. It is a wrapper of `density_matrix()`.

See also: [`density_matrix`](@ref).
"""
function density(cfv::Union{𝒻{S},𝒻ᵖˢ{S}}, tstp::I64) where {S}
    density_matrix(cfv, tstp)
end

"""
    density_matrix(cfm::ℱ{T}, tstp::I64)
    density_matrix(cfm::ℱᵖˢ{T}, tstp::I64)

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
function density_matrix(cfm::Union{ℱ{T},ℱᵖˢ{T}}, tstp::I64) where {T}
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
    density_matrix(cfv::𝒻ᵖˢ{S}, tstp::I64)

Returns the density matrix at given time step `tstp`, which must be the
time step of `cfv`. See [`density_matrix`](@ref).

See also: [`density`](@ref).
"""
function density_matrix(cfv::Union{𝒻{S},𝒻ᵖˢ{S}}, tstp::I64) where {S}
    # Sanity check
    tstp == gettstp(cfv) || throw(ArgumentError("tstp must be the time step of cfv"))

    if tstp == 0
        return -cfv.mat[getntau(cfv)]
    else
        return cfv.less[tstp] * getsign(cfv) * im
    end
end

"""
    density_matrix(G::NessHermMat{C64}, sign::Integer)

Density matrix ``\\rho = \\xi i G^<(0)`` of a steady-state Green's
function, where ``\\xi`` is `sign` (FERMI or BOSE).

*References* :

Please see [`NESSi`] `ness2_utils_impl.hpp`, `density_matrix`.
"""
function density_matrix(G::NessHermMat{C64}, sign::Integer)
    sign in (FERMI, BOSE) || throw(ArgumentError("sign must be FERMI or BOSE"))
    return G.les.time[:, :, 1] .* complex(0.0, F64(sign))
end

