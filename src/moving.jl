#
# Project : Lavender
# Source  : moving.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks* : *Moving (Truncated) Window*

The moving types (`ℱᵐᵒᵛ`, `𝒻ᵐᵒᵛ`, `𝒻ᵐᵒᵛView`, `Cfᵐᵒᵛ`) store a window
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
    _check_view(v::𝒻ᵐᵒᵛView)

Throw an `ArgumentError` if the view is stale. It is a private function.
"""
function _check_view(v::𝒻ᵐᵒᵛView)
    v.generation == v.parent.generation ||
        throw(ArgumentError("stale view: the parent moving object has changed"))
    return nothing
end

#=
### *ℱᵐᵒᵛ* : *Relative Indexing*
=#

"""
    Base.getindex(A::Union{Gʳᵉᵗᵐᵒᵛ{T},Gˡᵉˢˢᵐᵒᵛ{T}}, delti::I64, deltj::I64)

Return a copy of the block `(delti, deltj)` (relative NESSi indices).
"""
function Base.getindex(A::Union{Gʳᵉᵗᵐᵒᵛ{T},Gˡᵉˢˢᵐᵒᵛ{T}},
                       delti::I64, deltj::I64) where {T}
    (0 ≤ delti ≤ A.tc && 0 ≤ deltj ≤ A.tc) || throw(BoundsError(A, (delti, deltj)))
    return Matrix(_mblock(A, delti, deltj))
end

"""
    Base.setindex!(A::Union{Gʳᵉᵗᵐᵒᵛ{T},Gˡᵉˢˢᵐᵒᵛ{T}}, x::AbstractMatrix,
                   delti::I64, deltj::I64)

Set the block `(delti, deltj)` (relative NESSi indices).
"""
function Base.setindex!(A::Union{Gʳᵉᵗᵐᵒᵛ{T},Gˡᵉˢˢᵐᵒᵛ{T}}, x::AbstractMatrix,
                        delti::I64, deltj::I64) where {T}
    (0 ≤ delti ≤ A.tc && 0 ≤ deltj ≤ A.tc) || throw(BoundsError(A, (delti, deltj)))
    size(x) == (A.ndim1, A.ndim2) ||
        throw(DimensionMismatch("expected a $(A.ndim1) × $(A.ndim2) matrix"))
    copyto!(_mblock(A, delti, deltj), x)
    return A
end

"""
    Base.getindex(A::Union{gʳᵉᵗᵐᵒᵛ{S},gˡᵉˢˢᵐᵒᵛ{S}}, deltj::I64)

Return a copy of the block `deltj` of a moving time step.
"""
function Base.getindex(A::Union{gʳᵉᵗᵐᵒᵛ{S},gˡᵉˢˢᵐᵒᵛ{S}}, deltj::I64) where {S}
    0 ≤ deltj ≤ A.tc || throw(BoundsError(A, deltj))
    return Matrix(_mblock(A, deltj))
end

"""
    Base.setindex!(A::Union{gʳᵉᵗᵐᵒᵛ{S},gˡᵉˢˢᵐᵒᵛ{S}}, x::AbstractMatrix,
                   deltj::I64)

Set the block `deltj` of a moving time step.
"""
function Base.setindex!(A::Union{gʳᵉᵗᵐᵒᵛ{S},gˡᵉˢˢᵐᵒᵛ{S}}, x::AbstractMatrix,
                        deltj::I64) where {S}
    0 ≤ deltj ≤ A.tc || throw(BoundsError(A, deltj))
    size(x) == (A.ndim1, A.ndim2) ||
        throw(DimensionMismatch("expected a $(A.ndim1) × $(A.ndim2) matrix"))
    copyto!(_mblock(A, deltj), x)
    return A
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
    Base.getindex(G::ℱᵐᵒᵛ{T}, c::Symbol, n::I64, m::I64)

Return a copy of the component `c` (`:ret`, `:less`, or `:gtr`) at the
absolute KB time steps `(n, m)`. The blocks with `n < m` are
reconstructed by the hermitian relations. The greater component is
``G^{>} = G^{R} + G^{<}`` for `n ≥ m` and ``-G^{>}(t_m, t_n)^{\\dagger}``
otherwise.
"""
function Base.getindex(G::ℱᵐᵒᵛ{T}, c::Symbol, n::I64, m::I64) where {T}
    _check_initialized(G)
    c in (:ret, :less, :gtr) ||
        throw(ArgumentError("component must be :ret, :less, or :gtr, got $c"))
    if n ≥ m
        delti, deltj = _mabs(G, n, m)
        R = Matrix(_mblock(G.ret, delti, deltj))
        L = Matrix(_mblock(G.less, delti, deltj))
        c == :ret && return R
        c == :less && return L
        return R + L
    else
        X = G[c, m, n]
        return -Matrix(X')
    end
end

"""
    Base.setindex!(G::ℱᵐᵒᵛ{T}, x::AbstractMatrix, c::Symbol, n::I64, m::I64)

Set the component `c` (`:ret` or `:less`) at the absolute KB time steps
`(n, m)`. Only the stored domain `n ≥ m` is accepted.
"""
function Base.setindex!(G::ℱᵐᵒᵛ{T}, x::AbstractMatrix, c::Symbol,
                        n::I64, m::I64) where {T}
    _check_initialized(G)
    c in (:ret, :less) ||
        throw(ArgumentError("component must be :ret or :less, got $c"))
    n ≥ m || throw(ArgumentError("only the stored domain n ≥ m can be set"))
    delti, deltj = _mabs(G, n, m)
    A = c == :ret ? G.ret : G.less
    A[delti, deltj] = x
    return G
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
    return 𝒻ᵐᵒᵛView{T}(G, delti, G.generation)
end

"""
    _vblock(v::𝒻ᵐᵒᵛView, c::Symbol, deltj::I64)

Zero-copy view of the block `deltj` of the component `c` of a moving
view. It is a private function.
"""
@inline function _vblock(v::𝒻ᵐᵒᵛView, c::Symbol, deltj::I64)
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
    Base.getindex(v::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView}, c::Symbol, deltj::I64)

Return a copy of the block `deltj` of the component `c` (`:ret` or
`:less`) of a moving time step or view.
"""
function Base.getindex(v::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView}, c::Symbol, deltj::I64)
    c in (:ret, :less) ||
        throw(ArgumentError("component must be :ret or :less, got $c"))
    0 ≤ deltj ≤ getsize(v) || throw(BoundsError(v, deltj))
    return Matrix(_vblock(v, c, deltj))
end

"""
    Base.setindex!(v::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView}, x::AbstractMatrix, c::Symbol,
                   deltj::I64)

Set the block `deltj` of the component `c` of a moving time step or view.
A view writes through to its parent.
"""
function Base.setindex!(v::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView}, x::AbstractMatrix,
                        c::Symbol, deltj::I64)
    c in (:ret, :less) ||
        throw(ArgumentError("component must be :ret or :less, got $c"))
    0 ≤ deltj ≤ getsize(v) || throw(BoundsError(v, deltj))
    B = _vblock(v, c, deltj)
    size(x) == size(B) || throw(DimensionMismatch("expected a $(size(B)) matrix"))
    copyto!(B, x)
    return v
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

#=
### *ℱᵐᵒᵛ* : *Window Operations*
=#

"""
    forward!(G::ℱᵐᵒᵛ{T})

Move the window forward by one time step. The physical rows are rotated,
the new leading row (`delti = 0`) is set to zero, `tstp` is incremented
and every existing view becomes stale. No data are copied.

*References* :

Please see [`NESSi`] `herm_matrix_moving::forward`.
"""
function forward!(G::ℱᵐᵒᵛ{T}) where {T}
    _check_initialized(G)
    tc = G.tc
    head = mod1(G.ret.head - 1, tc + 1)
    G.ret.head = head
    G.less.head = head
    fill!(_mrowview(G.ret, 0), zero(T))
    fill!(_mrowview(G.less, 0), zero(T))
    G.tstp += 1
    G.generation += 1
    return G
end

"""
    set_from_G_backward!(Gm::ℱᵐᵒᵛ{C64}, G::ℱ{C64}, Gcc::ℱ{C64}, n₀::I64)
    set_from_G_backward!(Gm::ℱᵐᵒᵛ{C64}, G::ℱ{C64}, n₀::I64)

Initialize the window from the full contour Green's function `G` at the
KB time step `n₀`: the row `delti` is the time step `n₀ - delti` of `G`,

```math
G_m^{R}(\\delta_i, \\delta_j) = G^{R}(t_{n}, t_{n-\\delta_j}),\\quad
G_m^{<}(\\delta_i, \\delta_j) = -G_{cc}^{<}(t_{n-\\delta_j}, t_n)^{\\dagger},
```

with `n = n₀ - delti` and `deltj ≤ n - 1` (the remaining blocks are
zero). `Gcc` is the hermitian conjugate of `G` (`Gcc = G` for hermitian
`G`). It requires `n₀ - tc ≥ 1`. Afterwards `Gm.tstp = n₀`.

*References* :

Please see [`NESSi`] `herm_matrix_moving::set_from_G_backward` and
`herm_matrix_timestep_moving_view::set_timestep`.
"""
function set_from_G_backward!(Gm::ℱᵐᵒᵛ{C64}, G::ℱ{C64}, Gcc::ℱ{C64}, n₀::I64)
    tc = Gm.tc
    d = Gm.ret.ndim1
    getdims(G) == (d, d) && getdims(Gcc) == (d, d) ||
        throw(DimensionMismatch("the matrix sizes do not match"))
    (n₀ - tc ≥ 1 && n₀ ≤ getntime(G) && n₀ ≤ getntime(Gcc)) ||
        throw(ArgumentError("set_from_G_backward! requires 1 ≤ n₀ - tc and n₀ ≤ ntime"))
    getsign(G) == Gm.sign ||
        throw(ArgumentError("the statistics of G and Gm differ"))
    fill!(Gm.ret.data, zero(C64))
    fill!(Gm.less.data, zero(C64))
    for delti = 0:tc
        n = n₀ - delti
        for deltj = 0:min(tc, n - 1)
            m = n - deltj
            copyto!(_mblock(Gm.ret, delti, deltj), _block(G.ret, n, m))
            B = _mblock(Gm.less, delti, deltj)
            S = _block(Gcc.less, m, n)
            @inbounds for b = 1:d, a = 1:d
                B[a, b] = -conj(S[b, a])
            end
        end
    end
    Gm.tstp = n₀
    Gm.initialized = true
    Gm.generation += 1
    return Gm
end
#
set_from_G_backward!(Gm::ℱᵐᵒᵛ{C64}, G::ℱ{C64}, n₀::I64) =
    set_from_G_backward!(Gm, G, G, n₀)

"""
    initialize_from_ness!(Gm::ℱᵐᵒᵛ{C64}, Gness::HermMatrixNESS{C64})

Initialize every row of the window with the time-translation invariant
steady state `Gness`,
``G^{R/<}_m(\\delta_i, \\delta_j) = G^{R/<}_{ness}(\\delta_j h)`` for
``\\delta_j \\le \\min(t_c, N_{ft}/2 - 1)`` (the remaining blocks are
zero). The relative NESS step 0 is the KB time step 1, i.e. afterwards
`Gm.tstp = 1`, and the first `forward!` leads to the KB time step 2. The
negative relative history is accessible only through the relative
indices.

*References* :

Please see [`NESSi`] `ness2_utils_decl.hpp`, `ness2cntr` for
`herm_matrix_moving`.
"""
function initialize_from_ness!(Gm::ℱᵐᵒᵛ{C64}, Gness::HermMatrixNESS{C64})
    d = Gm.ret.ndim1
    size(Gness.ret.time, 1) == d ||
        throw(DimensionMismatch("ness2cntr: matrix size mismatch"))
    tc = Gm.tc
    dtmax = min(tc, getnft(Gness) ÷ 2 - 1)
    fill!(Gm.ret.data, zero(C64))
    fill!(Gm.less.data, zero(C64))
    for delti = 0:tc, dt = 0:dtmax
        copyto!(_mblock(Gm.ret, delti, dt), view(Gness.ret.time, :, :, dt + 1))
        copyto!(_mblock(Gm.less, delti, dt), view(Gness.les.time, :, :, dt + 1))
    end
    Gm.tstp = 1
    Gm.initialized = true
    Gm.generation += 1
    return Gm
end

#=
### *Cfᵐᵒᵛ* : *Indexing And Window Operations*
=#

"""
    Base.getindex(f::Cfᵐᵒᵛ{T}, delti::I64)

Return a copy of the slot `delti`, i.e. ``f(t_0 - \\delta_i)``.
"""
function Base.getindex(f::Cfᵐᵒᵛ{T}, delti::I64) where {T}
    0 ≤ delti ≤ f.tc || throw(BoundsError(f, delti))
    return Matrix(view(f.data, :, :, _mrow(f.head, f.tc, delti)))
end

"""
    Base.setindex!(f::Cfᵐᵒᵛ{T}, x::AbstractMatrix, delti::I64)

Set the slot `delti`.
"""
function Base.setindex!(f::Cfᵐᵒᵛ{T}, x::AbstractMatrix, delti::I64) where {T}
    0 ≤ delti ≤ f.tc || throw(BoundsError(f, delti))
    size(x) == (f.ndim1, f.ndim2) ||
        throw(DimensionMismatch("expected a $(f.ndim1) × $(f.ndim2) matrix"))
    copyto!(view(f.data, :, :, _mrow(f.head, f.tc, delti)), x)
    return f
end

"""
    _fslot(f::Cfᵐᵒᵛ, delti::I64)

Zero-copy view of the slot `delti`. It is a private function.
"""
@inline _fslot(f::Cfᵐᵒᵛ, delti::I64) = view(f.data, :, :, _mrow(f.head, f.tc, delti))

"""
    forward!(f::Cfᵐᵒᵛ{T})

Move the window of the single-time function forward by one time step.
The new leading slot is set to zero.

*References* :

Please see [`NESSi`] `function_moving::forward`.
"""
function forward!(f::Cfᵐᵒᵛ{T}) where {T}
    _check_initialized(f)
    f.head = mod1(f.head - 1, f.tc + 1)
    fill!(_fslot(f, 0), zero(T))
    f.tstp += 1
    return f
end

"""
    set_from_function_backward!(f::Cfᵐᵒᵛ{C64}, cf::Cf{C64}, n₀::I64)

Initialize the slots from the single-time function `cf`: the slot `delti`
is `cf[n₀ - delti]`. It requires `n₀ - tc ≥ 1` (the Matsubara value
`cf[0]` never enters the window). Afterwards `f.tstp = n₀`.

*References* :

Please see [`NESSi`] `function_moving::set_from_function_forward`. The
source and target indices of NESSi `set_from_function_backward` appear to
be interchanged, the semantics of `set_from_function_forward` is used.
"""
function set_from_function_backward!(f::Cfᵐᵒᵛ{C64}, cf::Cf{C64}, n₀::I64)
    getdims(cf) == (f.ndim1, f.ndim2) ||
        throw(DimensionMismatch("the matrix sizes do not match"))
    (n₀ - f.tc ≥ 1 && n₀ ≤ getntime(cf)) ||
        throw(ArgumentError("set_from_function_backward! requires 1 ≤ n₀ - tc and n₀ ≤ ntime"))
    for delti = 0:f.tc
        copyto!(_fslot(f, delti), cf[n₀ - delti])
    end
    f.tstp = n₀
    f.initialized = true
    return f
end

#=
### *Moving Window* : *Plain Text I/O*
=#

#=
*Remarks* : *Format*

The first line is the header

    # KB_MOVING_V1 kind tc initialized tstp sign ndim

with `kind = F` (`ℱᵐᵒᵛ`) or `kind = Cf` (`Cfᵐᵒᵛ`, `sign = 0`). For `F`
the retarded blocks follow, then the lesser blocks, every line contains
`delti deltj a b re im` in the logical order (`delti`, `deltj`, `b`, `a`
with `a` running fastest). For `Cf` every line contains
`delti a b re im`. The physical ring order is never written.
=#

"""
    Base.show(io::IO, G::ℱᵐᵒᵛ{T})

Write the moving Green's function as plain text.
"""
function Base.show(io::IO, G::ℱᵐᵒᵛ{T}) where {T}
    tc = G.tc
    d = G.ret.ndim1
    println(io, "# KB_MOVING_V1 F $tc $(Int(G.initialized)) $(G.tstp) $(G.sign) $d")
    for A in (G.ret, G.less), delti = 0:tc, deltj = 0:tc
        B = _mblock(A, delti, deltj)
        for b = 1:d, a = 1:d
            z = B[a, b]
            @printf(io, "%d %d %d %d %.16e %.16e\n", delti, deltj, a, b, real(z), imag(z))
        end
    end
end

"""
    Base.show(io::IO, f::Cfᵐᵒᵛ{T})

Write the moving single-time function as plain text.
"""
function Base.show(io::IO, f::Cfᵐᵒᵛ{T}) where {T}
    d = f.ndim1
    println(io, "# KB_MOVING_V1 Cf $(f.tc) $(Int(f.initialized)) $(f.tstp) 0 $d")
    for delti = 0:f.tc
        B = _fslot(f, delti)
        for b = 1:d, a = 1:d
            z = B[a, b]
            @printf(io, "%d %d %d %.16e %.16e\n", delti, a, b, real(z), imag(z))
        end
    end
end

"""
    Base.write(fn::AbstractString, G::Union{ℱᵐᵒᵛ,Cfᵐᵒᵛ})

Write the moving object to the file `fn` as plain text.
"""
function Base.write(fn::AbstractString, G::Union{ℱᵐᵒᵛ,Cfᵐᵒᵛ})
    open(fn, "w") do io
        show(io, G)
    end
end

"""
    _read_moving_header(lines, kind, tc, ndim)

Parse and validate the header of a moving text file. It is a private
function.
"""
function _read_moving_header(line::AbstractString, kind::String, tc::I64, ndim::I64)
    f = split(line)
    (length(f) == 8 && f[1] == "#" && f[2] == "KB_MOVING_V1") ||
        throw(ArgumentError("not a KB_MOVING_V1 file"))
    f[3] == kind || throw(ArgumentError("wrong kind $(f[3]), expected $kind"))
    ftc = parse(I64, f[4])
    init = parse(I64, f[5])
    tstp = parse(I64, f[6])
    sign = parse(I64, f[7])
    fd = parse(I64, f[8])
    ftc == tc || throw(ArgumentError("wrong tc $ftc, expected $tc"))
    fd == ndim || throw(ArgumentError("wrong ndim $fd, expected $ndim"))
    init in (0, 1) || throw(ArgumentError("bad initialized flag $init"))
    # An uninitialized object carries the placeholder tstp = 0, an
    # initialized one a valid KB time step
    (init == 0 ? tstp == 0 : tstp ≥ 1) ||
        throw(ArgumentError("inconsistent initialized/tstp: $init/$tstp"))
    return init == 1, tstp, sign
end

"""
    Base.read!(fn::AbstractString, G::ℱᵐᵒᵛ{C64})

Read the moving Green's function from the file `fn`. The file is
validated completely before `G` is modified; on success the state is
replaced and every existing view becomes stale.
"""
function Base.read!(fn::AbstractString, G::ℱᵐᵒᵛ{C64})
    tc = G.tc
    d = G.ret.ndim1
    lines = readlines(fn)
    isempty(lines) && throw(ArgumentError("empty file"))
    init, tstp, sign = _read_moving_header(lines[1], "F", tc, d)
    sign == G.sign || throw(ArgumentError("wrong sign $sign, expected $(G.sign)"))
    nrec = 2 * (tc + 1)^2 * d^2
    length(lines) == nrec + 1 ||
        throw(ArgumentError("wrong number of records $(length(lines) - 1), expected $nrec"))
    R = zeros(C64, d, d, tc + 1, tc + 1)
    L = zeros(C64, d, d, tc + 1, tc + 1)
    k = 1
    for X in (R, L), delti = 0:tc, deltj = 0:tc, b = 1:d, a = 1:d
        k += 1
        f = split(lines[k])
        length(f) == 6 || throw(ArgumentError("bad record at line $k"))
        idx = parse.(I64, f[1:4])
        idx == [delti, deltj, a, b] ||
            throw(ArgumentError("unexpected index at line $k"))
        re = parse(F64, f[5])
        im = parse(F64, f[6])
        (isfinite(re) && isfinite(im)) ||
            throw(ArgumentError("non-finite value at line $k"))
        X[a, b, deltj + 1, delti + 1] = complex(re, im)
    end
    # Store in the logical order (head = 1)
    G.ret.head = 1
    G.less.head = 1
    copyto!(G.ret.data, R)
    copyto!(G.less.data, L)
    G.tstp = tstp
    G.initialized = init
    G.generation += 1
    return G
end

"""
    Base.read!(fn::AbstractString, f::Cfᵐᵒᵛ{C64})

Read the moving single-time function from the file `fn`.
"""
function Base.read!(fn::AbstractString, f::Cfᵐᵒᵛ{C64})
    tc = f.tc
    d = f.ndim1
    lines = readlines(fn)
    isempty(lines) && throw(ArgumentError("empty file"))
    init, tstp, _ = _read_moving_header(lines[1], "Cf", tc, d)
    nrec = (tc + 1) * d^2
    length(lines) == nrec + 1 ||
        throw(ArgumentError("wrong number of records $(length(lines) - 1), expected $nrec"))
    X = zeros(C64, d, d, tc + 1)
    k = 1
    for delti = 0:tc, b = 1:d, a = 1:d
        k += 1
        fs = split(lines[k])
        length(fs) == 5 || throw(ArgumentError("bad record at line $k"))
        parse.(I64, fs[1:3]) == [delti, a, b] ||
            throw(ArgumentError("unexpected index at line $k"))
        re = parse(F64, fs[4])
        im = parse(F64, fs[5])
        (isfinite(re) && isfinite(im)) ||
            throw(ArgumentError("non-finite value at line $k"))
        X[a, b, delti + 1] = complex(re, im)
    end
    f.head = 1
    copyto!(f.data, X)
    f.tstp = tstp
    f.initialized = init
    return f
end
