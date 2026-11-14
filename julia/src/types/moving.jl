#
# Project : Lavender
# Source  : moving.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Customized Structs* : *Moving Window*
=#

#=
*Remarks* : *Moving (Truncated) Window*

A moving function stores the retarded and lesser components on a window
of `tc + 1` time steps. The leading time step is `t₀`. The block
`(delti, deltj)`, `0 ≤ delti, deltj ≤ tc`, contains

```math
G^{R}(t_0 - \delta_i, t_0 - \delta_i - \delta_j),\quad
G^{<}(t_0 - \delta_i, t_0 - \delta_i - \delta_j).
```

Every component owns one contiguous array of the size
`(ndim1, ndim2, tc + 1, tc + 1)`, the last two indices are `deltj + 1`
and the physical row. The logical row `delti` is stored in the physical
row `mod1(head + delti, tc + 1)`, so that moving the window forward only
rotates `head`. The Matsubara and mixing components are not stored.

Please see [`NESSi`] `cntr_herm_matrix_moving_decl.hpp` and
`cntr_herm_matrix_moving_impl.hpp`.
=#

"""
    Gʳᵉᵗᵐᵒᵛ{T}

Retarded component of a moving contour function. `data[:, :, deltj + 1, p]`
is the block `(delti, deltj)` with the physical row
`p = mod1(head + delti, tc + 1)`.
"""
mutable struct Gʳᵉᵗᵐᵒᵛ{T} <: CnAbstractMovingMatrix{T}
    tc    :: I64
    ndim1 :: I64
    ndim2 :: I64
    head  :: I64
    data  :: Array{T,4}
end

"""
    gʳᵉᵗᵐᵒᵛ{S}

Retarded component of a moving function at one time step,
`data[:, :, deltj + 1]` is ``G^{R}(t, t - \\delta_j)``.
"""
mutable struct gʳᵉᵗᵐᵒᵛ{S} <: CnAbstractMovingVector{S}
    tc    :: I64
    ndim1 :: I64
    ndim2 :: I64
    data  :: Array{S,3}
end

"""
    Gˡᵉˢˢᵐᵒᵛ{T}

Lesser component of a moving contour function, the layout is the same as
the one of `Gʳᵉᵗᵐᵒᵛ`. The block `(delti, deltj)` contains
``G^{<}(t_0 - \\delta_i, t_0 - \\delta_i - \\delta_j)``.
"""
mutable struct Gˡᵉˢˢᵐᵒᵛ{T} <: CnAbstractMovingMatrix{T}
    tc    :: I64
    ndim1 :: I64
    ndim2 :: I64
    head  :: I64
    data  :: Array{T,4}
end

"""
    gˡᵉˢˢᵐᵒᵛ{S}

Lesser component of a moving function at one time step,
`data[:, :, deltj + 1]` is ``G^{<}(t, t - \\delta_j)``.
"""
mutable struct gˡᵉˢˢᵐᵒᵛ{S} <: CnAbstractMovingVector{S}
    tc    :: I64
    ndim1 :: I64
    ndim2 :: I64
    data  :: Array{S,3}
end

"""
    ℱᵐᵒᵛ{T}

Moving (truncated) contour-ordered Green's function. `tstp` is the KB
real time step of the leading row (`delti = 0`); it is meaningful only if
`initialized` is true. `generation` is incremented by every operation
that changes the mapping between the logical time and the storage, so
that stale views can be detected.
"""
mutable struct ℱᵐᵒᵛ{T} <: CnAbstractMovingFunction{T}
    tc          :: I64
    tstp        :: I64
    initialized :: Bool
    sign        :: I64
    generation  :: UInt
    ret         :: Gʳᵉᵗᵐᵒᵛ{T}
    less        :: Gˡᵉˢˢᵐᵒᵛ{T}
end

"""
    ℱᵐᵒᵛ(tc::I64, ndim::I64, sign::I64 = FERMI)

Constructor. Create an uninitialized moving Green's function with the
cut-off `tc` and square `ndim × ndim` matrices. Only `C64` is supported.
"""
function ℱᵐᵒᵛ(tc::I64, ndim::I64, sign::I64 = FERMI)
    tc ≥ 0 || throw(ArgumentError("tc must be ≥ 0, got $tc"))
    ndim ≥ 1 || throw(ArgumentError("ndim must be ≥ 1, got $ndim"))
    sign in (FERMI, BOSE) || throw(ArgumentError("sign must be FERMI or BOSE, got $sign"))
    _check_payload(tc, ndim, ndim, tc + 1)
    ret = Gʳᵉᵗᵐᵒᵛ{C64}(tc, ndim, ndim, 1, zeros(C64, ndim, ndim, tc + 1, tc + 1))
    less = Gˡᵉˢˢᵐᵒᵛ{C64}(tc, ndim, ndim, 1, zeros(C64, ndim, ndim, tc + 1, tc + 1))
    return ℱᵐᵒᵛ{C64}(tc, 0, false, sign, UInt(0), ret, less)
end

"""
    𝒻ᵐᵒᵛ{S}

Owning time step of a moving Green's function. It contains
``G^{R}(t, t - \\delta_j)`` and ``G^{<}(t, t - \\delta_j)`` for
`deltj = 0, …, tc`, where `t` is the KB time step `tstp`.
"""
mutable struct 𝒻ᵐᵒᵛ{S} <: CnAbstractMovingFunction{S}
    tc   :: I64
    tstp :: I64
    sign :: I64
    ret  :: gʳᵉᵗᵐᵒᵛ{S}
    less :: gˡᵉˢˢᵐᵒᵛ{S}
end

"""
    𝒻ᵐᵒᵛ(tc::I64, ndim::I64, tstp::I64, sign::I64 = FERMI)

Constructor. Create an owning moving time step with zero values.
"""
function 𝒻ᵐᵒᵛ(tc::I64, ndim::I64, tstp::I64, sign::I64 = FERMI)
    tc ≥ 0 || throw(ArgumentError("tc must be ≥ 0, got $tc"))
    ndim ≥ 1 || throw(ArgumentError("ndim must be ≥ 1, got $ndim"))
    sign in (FERMI, BOSE) || throw(ArgumentError("sign must be FERMI or BOSE, got $sign"))
    _check_payload(tc, ndim, ndim, 1)
    ret = gʳᵉᵗᵐᵒᵛ{C64}(tc, ndim, ndim, zeros(C64, ndim, ndim, tc + 1))
    less = gˡᵉˢˢᵐᵒᵛ{C64}(tc, ndim, ndim, zeros(C64, ndim, ndim, tc + 1))
    return 𝒻ᵐᵒᵛ{C64}(tc, tstp, sign, ret, less)
end

"""
    𝒻ᵛⁱᵉʷᵐᵒᵛ{S}

Non-owning view of the row `delti` of a moving Green's function. The
view is valid as long as `generation == parent.generation`; afterwards
every access throws an `ArgumentError`.
"""
struct 𝒻ᵛⁱᵉʷᵐᵒᵛ{S} <: CnAbstractMovingFunction{S}
    parent     :: ℱᵐᵒᵛ{S}
    delti      :: I64
    generation :: UInt
end

"""
    Cfᵐᵒᵛ{T}

Moving single-time function. The slot `delti` (`0 ≤ delti ≤ tc`)
contains ``f(t_0 - \\delta_i)`` and is stored in
`data[:, :, mod1(head + delti, tc + 1)]`.
"""
mutable struct Cfᵐᵒᵛ{T} <: CnAbstractMovingFunction{T}
    tc          :: I64
    tstp        :: I64
    initialized :: Bool
    head        :: I64
    ndim1       :: I64
    ndim2       :: I64
    data        :: Array{T,3}
end

"""
    Cfᵐᵒᵛ(tc::I64, ndim::I64)

Constructor. Create an uninitialized moving single-time function.
"""
function Cfᵐᵒᵛ(tc::I64, ndim::I64)
    tc ≥ 0 || throw(ArgumentError("tc must be ≥ 0, got $tc"))
    ndim ≥ 1 || throw(ArgumentError("ndim must be ≥ 1, got $ndim"))
    _check_payload(tc, ndim, ndim, 1)
    return Cfᵐᵒᵛ{C64}(tc, 0, false, 1, ndim, ndim, zeros(C64, ndim, ndim, tc + 1))
end

"""
    _check_payload(tc::I64, ndim1::I64, ndim2::I64, nrow::I64)

Number of elements of a moving payload with checked arithmetic. It is a
private function.
"""
function _check_payload(tc::I64, ndim1::I64, ndim2::I64, nrow::I64)
    n = Base.checked_mul(ndim1, ndim2)
    n = Base.checked_mul(n, Base.checked_add(tc, 1))
    return Base.checked_mul(n, nrow)
end
