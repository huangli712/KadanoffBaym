#
# Project : Lavender
# Source  : layout.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

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
### *Private Service Functions* : *Sanity Checks*
=#

"""
    _check_initialized(G)

Throw an `ArgumentError` if the moving object `G` is not initialized. It
is a private function.
"""
function _check_initialized(G::Union{ℱᵐᵒᵛ,Cfᵐᵒᵛ})
    G.initialized ||
        throw(ArgumentError("the moving object is not initialized"))
    return nothing
end

"""
    _check_view(v::𝒻ᵛⁱᵉʷᵐᵒᵛ)

Throw an `ArgumentError` if the view is stale. It is a private function.
"""
function _check_view(v::𝒻ᵛⁱᵉʷᵐᵒᵛ)
    v.generation == v.parent.generation ||
        throw(ArgumentError("stale view: the parent moving object has changed"))
    return nothing
end

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

"""
    _fslot(f::Cfᵐᵒᵛ, delti::I64)

Zero-copy view of the slot `delti`. It is a private function.
"""
@inline _fslot(f::Cfᵐᵒᵛ, delti::I64) = view(f.data, :, :, _mrow(f.head, f.tc, delti))

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
