#
# Project : Lavender
# Source  : utilities.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/09
#

#=
### *Public API* : *Initial State*
=#

#=
*Remarks* : *Boundary Values From Matsubara Component*

At ``t = 0``, the real-time components are determined by the Matsubara
component:

```math
\begin{equation}
G^{\rceil}(0,\tau) = i \xi G^{M}(\beta - \tau),
\end{equation}
```

```math
\begin{equation}
G^{<}(0,0) = G^{\rceil}(0,0),
\end{equation}
```

```math
\begin{equation}
G^{R}(0,0) = i G^{M}(0) - G^{<}(0,0).
\end{equation}
```

Here ``\xi = -1`` (``+1``) for fermions (bosons).

*References* :

Please see [`NESSi`] `cntr_utilities_impl.hpp`, `set_t0_from_mat`.
=#

"""
    _set_tn_from_mat!(n::I64, G::ℱ{T})

Setup the `lmix` component at time step `n`, and the diagonal elements
of the `ret` and `less` components at `(n, n)`, by using the Matsubara
component. It is a private function.

See also: [`set_t0_from_mat!`](@ref), [`set_tk_from_mat!`](@ref).
"""
function _set_tn_from_mat!(n::I64, G::ℱ{T}) where {T}
    ntau = getntau(G)
    sign = getsign(G)

    # G^{⌉}(tₙ,τₘ) = i ξ G^{M}(β - τₘ)
    for m = 1:ntau
        _block(G.lmix, n, m) .= (im * sign) .* _block(G.mat, ntau - m + 1)
    end

    # G^{<}(tₙ,tₙ) = G^{⌉}(0,0)
    copyto!(_block(G.less, n, n), _block(G.lmix, 1, 1))

    # G^{R}(tₙ,tₙ) = i G^{M}(0) - G^{<}(0,0)
    _block(G.ret, n, n) .= im .* _block(G.mat, 1) .- _block(G.less, 1, 1)
end

"""
    set_t0_from_mat!(G::ℱ{T})

Setup the first real time step (`t = 0`, i.e., `tstp = 1`) of the `lmix`,
`ret`, and `less` components of `G` by using its Matsubara component.

### Arguments
* G -> A contour-ordered Green's function.

### Returns
N/A
"""
function set_t0_from_mat!(G::ℱ{T}) where {T}
    _set_tn_from_mat!(1, G)
end

"""
    set_tk_from_mat!(G::ℱ{T}, k::I64)

Setup the time steps `1, 2, ..., k + 1` of the `lmix` component, and the
diagonal elements of the `ret` and `less` components, by using the
Matsubara component of `G`. The values are the same as those at `t = 0`.
The off-diagonal elements of `ret` and `less` are not touched.

### Arguments
* G -> A contour-ordered Green's function.
* k -> The last time step to be set is `k + 1`.

### Returns
N/A
"""
function set_tk_from_mat!(G::ℱ{T}, k::I64) where {T}
    # Sanity check
    @assert k ≥ 0
    @assert getntime(G) ≥ k + 1

    for n = 1:k+1
        _set_tn_from_mat!(n, G)
    end
end

