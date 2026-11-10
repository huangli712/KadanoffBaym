#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

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


