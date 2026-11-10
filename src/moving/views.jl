#
# Project : Lavender
# Source  : views.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Moving Window* : *Traits*
=#

#=
*Remarks* : *Moving Time Steps*

The traits of the moving types act on one time step (a row of the
window): an owning `𝒻ᵐᵒᵛ` or a view `𝒻ᵛⁱᵉʷᵐᵒᵛ` of a row of `ℱᵐᵒᵛ`. They
work on the contiguous row slices directly. A view writes through to its
parent and is checked against the generation of the parent first.

Please see [`NESSi`] `cntr_herm_matrix_timestep_moving_view_impl.hpp`.
=#

"""
    _mrows(x::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ}, c::Symbol)

Zero-copy view `(ndim1, ndim2, tc + 1)` of the component `c` (`:ret` or
`:less`) of a moving time step. It is a private function.
"""
@inline function _mrows(x::𝒻ᵐᵒᵛ, c::Symbol)
    return c == :ret ? x.ret.data : x.less.data
end
#

@inline function _mrows(v::𝒻ᵛⁱᵉʷᵐᵒᵛ, c::Symbol)
    _check_view(v)
    A = c == :ret ? v.parent.ret : v.parent.less
    return _mrowview(A, v.delti)
end

"""
    _check_mrow(x, y)

Check that two moving time steps have the same cut-off and matrix size.
It is a private function.
"""
function _check_mrow(x, y)
    getsize(x) == getsize(y) ||
        throw(ArgumentError("the cut-offs differ: $(getsize(x)) and $(getsize(y))"))
    getdims(x) == getdims(y) ||
        throw(DimensionMismatch("the matrix sizes differ"))
    return nothing
end

#=
### *Moving Window* : *Truncated VIE2*
=#

#=
*Remarks* : *Truncated VIE2*

The truncated VIE2 ``G + F * G = Q`` determines the leading row
``G(t, t - \delta_j)`` of a moving Green's function with all memory
integrals restricted to the window ``[t_o, t]``, ``t_o = t - t_c``. Let
the window time `a = 0, ⋯, tc` denote ``t_o + a h``. The equation is
solved exactly as the full VIE2 at the time step `n = tc` of the window,
i.e. with the time origin shifted to ``t_o``:

* retarded: the right action equation ``G^{R} + G^{R} * F^{cc,R} = Q^{R}``
  for ``G^{R}(n, j)``, `j = n, ⋯, 0`, with a 𝑘 × 𝑘 start-up system;
* lesser: ``G^{<}(j, n)`` for `j = 0, ⋯, n` with a (𝑘+1) × (𝑘+1) start-up
  system, the source term being ``-[F^{<} * G^{A}](j, n)``.

The full lesser equation additionally contains the mixed initial
correlation ``F^{\rceil} \star G^{\lceil}``, which is not represented in
the moving types and omitted here. Therefore the truncated VIE2 agrees
with the full VIE2 only if the window covers the whole real time history
(``t_o`` is the KB time step 1, i.e. `tc = tstp - 1`) and the mixing
components of the full input vanish; otherwise there is no full
equivalent. A large `tc` does not restore the initial correlations.

The lesser blocks with the first time earlier than the second one are
reconstructed from the hermitian conjugate partner, e.g.
``F^{<}(a, b) = -F^{cc,<}(b, a)^{\dagger}`` for `a < b`, and
``Q^{<}(a, b) = -Q^{<}(b, a)^{\dagger}``.

[`NESSi`] has no moving VIE2. The formulas are those of the full kernels
`vie2_timestep_ret` and `vie2_timestep_les` without the `tv*vt` term,
see `cntr_vie2_impl.hpp`.
=#

#
# Window accessors of the truncated VIE2, `a ≥ b` are window times. They
# are private functions.
#
@inline _wret(X::ℱᵐᵒᵛ, a::I64, b::I64) = _mblock(X.ret, X.tc - a, a - b)

@inline _wles(X::ℱᵐᵒᵛ, a::I64, b::I64) = _mblock(X.less, X.tc - a, a - b)

"""
    _moving_tforeach(f, r::UnitRange{I64}, threaded::Bool)

Call `f(js)` for the chunks `js` of `r`, in parallel if `threaded` and
more than one thread is available. Every chunk owns its scratch. It is a
private function.
"""
function _moving_tforeach(f, r::UnitRange{I64}, threaded::Bool)
    if _use_threads(threaded) && length(r) > 1
        chunks = _tchunks(r)
        Threads.@threads :static for c in eachindex(chunks)
            f(chunks[c])
        end
    else
        f(r)
    end
    return nothing
end

#=
*Remarks* : *Moving (Truncated) Window*

The moving types (`ℱᵐᵒᵛ`, `𝒻ᵐᵒᵛ`, `𝒻ᵛⁱᵉʷᵐᵒᵛ`, `Cfᵐᵒᵛ`) store a window
of `tc + 1` real time steps. The block `(delti, deltj)` of `ℱᵐᵒᵛ` is

```math
G^{R/<}(t_0 - \delta_i, t_0 - \delta_i - \delta_j),
```

where ``t_0`` is the KB time step `tstp` of the leading row. For an
initialized object the absolute KB time steps `(n, m)` with `n ≥ m` are
mapped to `delti = tstp - n` and `deltj = n - m`. The blocks with `n < m`
are reconstructed by the hermitian relations

```math
G^{R}(t_n, t_m) = -G^{R}(t_m, t_n)^{\dagger},\quad
G^{<}(t_n, t_m) = -G^{<}(t_m, t_n)^{\dagger}.
```

Relative accesses (on the components and the views) use the NESSi
indices `(delti, deltj)` directly. Only the boundaries to the full
contour (`set_from_G_backward!`) and the steady state
(`initialize_from_ness!`) translate between the NESSi 0-based and the KB
1-based time steps.

Please see [`NESSi`] `cntr_herm_matrix_moving_impl.hpp`,
`cntr_herm_matrix_timestep_moving_view_impl.hpp`, and
`cntr_function_moving_impl.hpp`.
=#

#=
### *Moving Window* : *Private Helpers*
=#

"""
    _mrow(head::I64, tc::I64, delti::I64)

Physical row of the logical row `delti`. It is a private function.
"""
@inline _mrow(head::I64, tc::I64, delti::I64) = mod1(head + delti, tc + 1)

"""
    _mblock(A::Union{Gʳᵉᵗᵐᵒᵛ{T},Gˡᵉˢˢᵐᵒᵛ{T}}, delti::I64, deltj::I64)

Zero-copy (write-through) view of the block `(delti, deltj)` of a moving
component. The view must not escape the calling kernel. It is a private
function.
"""
@inline function _mblock(A::Union{Gʳᵉᵗᵐᵒᵛ{T},Gˡᵉˢˢᵐᵒᵛ{T}},
                         delti::I64, deltj::I64) where {T}
    @assert 0 ≤ delti ≤ A.tc && 0 ≤ deltj ≤ A.tc
    return view(A.data, :, :, deltj + 1, _mrow(A.head, A.tc, delti))
end
#

@inline function _mblock(A::Union{gʳᵉᵗᵐᵒᵛ{S},gˡᵉˢˢᵐᵒᵛ{S}},
                         deltj::I64) where {S}
    @assert 0 ≤ deltj ≤ A.tc
    return view(A.data, :, :, deltj + 1)
end

"""
    _mrowview(A::Union{Gʳᵉᵗᵐᵒᵛ{T},Gˡᵉˢˢᵐᵒᵛ{T}}, delti::I64)

Zero-copy view `(ndim1, ndim2, tc + 1)` of the logical row `delti`. It is
a private function.
"""
@inline function _mrowview(A::Union{Gʳᵉᵗᵐᵒᵛ{T},Gˡᵉˢˢᵐᵒᵛ{T}},
                           delti::I64) where {T}
    @assert 0 ≤ delti ≤ A.tc
    return view(A.data, :, :, :, _mrow(A.head, A.tc, delti))
end

#=
### *ℱᵐᵒᵛ* : *Absolute Indexing*
=#

"""
    _mabs(G::ℱᵐᵒᵛ, n::I64, m::I64)

Translate the absolute KB time steps `(n, m)`, `n ≥ m`, to the relative
indices `(delti, deltj)`. Throw a `BoundsError` outside the window. It is
a private function.
"""
function _mabs(G::ℱᵐᵒᵛ, n::I64, m::I64)
    delti = G.tstp - n
    deltj = n - m
    (m ≥ 1 && 0 ≤ delti ≤ G.tc && 0 ≤ deltj ≤ G.tc) ||
        throw(BoundsError(G, (n, m)))
    return delti, deltj
end

#=
### *ℱᵐᵒᵛ* : *Views And Time Steps*
=#

"""
    timestep_view(G::ℱᵐᵒᵛ{T}, delti::I64 = 0)

Return a writable view of the logical row `delti` of `G`. The view becomes
stale after `forward!`, `set_from_G_backward!`, `initialize_from_ness!`,
or a successful `read!`.
"""
function timestep_view(G::ℱᵐᵒᵛ{T}, delti::I64 = 0) where {T}
    _check_initialized(G)
    0 ≤ delti ≤ G.tc || throw(BoundsError(G, delti))
    return 𝒻ᵛⁱᵉʷᵐᵒᵛ{T}(G, delti, G.generation)
end

"""
    _vblock(v::𝒻ᵛⁱᵉʷᵐᵒᵛ, c::Symbol, deltj::I64)

Zero-copy view of the block `deltj` of the component `c` of a moving
view. It is a private function.
"""
@inline function _vblock(v::𝒻ᵛⁱᵉʷᵐᵒᵛ, c::Symbol, deltj::I64)
    _check_view(v)
    A = c == :ret ? v.parent.ret : v.parent.less
    return _mblock(A, v.delti, deltj)
end
#

@inline function _vblock(v::𝒻ᵐᵒᵛ, c::Symbol, deltj::I64)
    A = c == :ret ? v.ret : v.less
    return _mblock(A, deltj)
end

"""
    get_timestep(G::ℱᵐᵒᵛ{T}, delti::I64 = 0)

Return an owning copy (`𝒻ᵐᵒᵛ`) of the logical row `delti`.
"""
function get_timestep(G::ℱᵐᵒᵛ{T}, delti::I64 = 0) where {T}
    _check_initialized(G)
    0 ≤ delti ≤ G.tc || throw(BoundsError(G, delti))
    x = 𝒻ᵐᵒᵛ(G.tc, G.ret.ndim1, G.tstp - delti, G.sign)
    copyto!(x.ret.data, _mrowview(G.ret, delti))
    copyto!(x.less.data, _mrowview(G.less, delti))
    return x
end

