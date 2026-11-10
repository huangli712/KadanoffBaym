#
# Project : Lavender
# Source  : window.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *ℱᵐᵒᵛ* : *Window Operations*
=#

"""
    init_from_G!(Gm::ℱᵐᵒᵛ{C64}, G::ℱ{C64}, Gcc::ℱ{C64}, n₀::I64)
    init_from_G!(Gm::ℱᵐᵒᵛ{C64}, G::ℱ{C64}, n₀::I64)

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
function init_from_G!(Gm::ℱᵐᵒᵛ{C64}, G::ℱ{C64}, Gcc::ℱ{C64}, n₀::I64)
    tc = Gm.tc
    d = Gm.ret.ndim1
    getdims(G) == (d, d) && getdims(Gcc) == (d, d) ||
        throw(DimensionMismatch("the matrix sizes do not match"))
    (n₀ - tc ≥ 1 && n₀ ≤ getntime(G) && n₀ ≤ getntime(Gcc)) ||
        throw(ArgumentError("init_from_G! requires 1 ≤ n₀ - tc and n₀ ≤ ntime"))
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

init_from_G!(Gm::ℱᵐᵒᵛ{C64}, G::ℱ{C64}, n₀::I64) =
    init_from_G!(Gm, G, G, n₀)

"""
    init_from_ness!(Gm::ℱᵐᵒᵛ{C64}, Gness::HermMatrixNESS{C64})

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
function init_from_ness!(Gm::ℱᵐᵒᵛ{C64}, Gness::HermMatrixNESS{C64})
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

"""
    init_from_function!(f::Cfᵐᵒᵛ{C64}, cf::Cf{C64}, n₀::I64)

Initialize the slots from the single-time function `cf`: the slot `delti`
is `cf[n₀ - delti]`. It requires `n₀ - tc ≥ 1` (the Matsubara value
`cf[0]` never enters the window). Afterwards `f.tstp = n₀`.

*References* :

Please see [`NESSi`] `function_moving::set_from_function_forward`. The
source and target indices of NESSi `set_from_function_backward` appear to
be interchanged, the semantics of `set_from_function_forward` is used.
"""
function init_from_function!(f::Cfᵐᵒᵛ{C64}, cf::Cf{C64}, n₀::I64)
    getdims(cf) == (f.ndim1, f.ndim2) ||
        throw(DimensionMismatch("the matrix sizes do not match"))
    (n₀ - f.tc ≥ 1 && n₀ ≤ getntime(cf)) ||
        throw(ArgumentError("init_from_function! requires 1 ≤ n₀ - tc and n₀ ≤ ntime"))
    for delti = 0:f.tc
        copyto!(_fslot(f, delti), cf[n₀ - delti])
    end
    f.tstp = n₀
    f.initialized = true
    return f
end

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

#=
### *ℱᵐᵒᵛ* : *Views And Time Steps*
=#

"""
    timestep_view(G::ℱᵐᵒᵛ{T}, delti::I64 = 0)

Return a writable view of the logical row `delti` of `G`. The view becomes
stale after `forward!`, `init_from_G!`, `init_from_ness!`,
or a successful `read!`.
"""
function timestep_view(G::ℱᵐᵒᵛ{T}, delti::I64 = 0) where {T}
    _check_initialized(G)
    0 ≤ delti ≤ G.tc || throw(BoundsError(G, delti))
    return 𝒻ᵛⁱᵉʷᵐᵒᵛ{T}(G, delti, G.generation)
end

"""
    copy_timestep(G::ℱᵐᵒᵛ{T}, delti::I64 = 0)

Return an owning copy (`𝒻ᵐᵒᵛ`) of the logical row `delti`.
"""
function copy_timestep(G::ℱᵐᵒᵛ{T}, delti::I64 = 0) where {T}
    _check_initialized(G)
    0 ≤ delti ≤ G.tc || throw(BoundsError(G, delti))
    x = 𝒻ᵐᵒᵛ(G.tc, G.ret.ndim1, G.tstp - delti, G.sign)
    copyto!(x.ret.data, _mrowview(G.ret, delti))
    copyto!(x.less.data, _mrowview(G.less, delti))
    return x
end
