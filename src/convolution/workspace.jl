#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Private Convolution Kernels*
=#

#=
*Remarks* : *Block-Target Kernels*

The following private kernels implement the integration loops of the
retarded, left-mixing, and lesser convolutions. They accumulate the
results directly into the target blocks, i.e., `tgt[m] += s * (...)`,
where `tgt` is any object whose `getindex` returns a mutable matrix
block, such as a time slice of a contour-ordered Green's function, a
time slice of a `𝒻` object, or a scratch vector of matrices. All matrix
products are evaluated by `mul!`, and all intermediate results are kept
in a preallocated workspace `ws`. Thus, the kernels do not allocate
memory which grows with the number of time steps.

The target blocks must not be aliased with the input blocks.

The formulas, weights, and summation domains are the same as those in
the public `conv_*` functions. Please see [`NESSi`] Eqs. (110) - (122).
=#

#
# Contour-ordered Green's functions accepted by the real-time convolution
# kernels. The pseudo-particle Green's function shares the storage layout
# of ℱ, and its real-time components (ret and less) are evaluated with the
# same kernels. The Matsubara and left-mixing convolutions of PseudoGF are
# different, see pseudo_convolution.jl. See [NESSi]
# cntr_pseudo_convolution_impl.hpp, incr_pseudo_convolution.
#
const _ConvGF{T} = Union{ℱ{T},PseudoGF{T}}

const _ConvTarget{T} = Union{ℱ{T},𝒻{T},PseudoGF{T},PseudoGFStep{T}}


"""
    _ConvWorkspace{T}

Workspace for the private convolution kernels. `af` stores `A f`, `fb`
stores `f B`, and `fbm` stores `f₀ Bᴹ(τ)` for all imaginary time points.
It is a private struct.
"""
struct _ConvWorkspace{T}
    af  :: Matrix{T}
    fb  :: Matrix{T}
    fbm :: Vector{Matrix{T}}
end


"""
    _ConvWorkspace(A::AbstractArray{T}, B::AbstractArray{T})

Constructor. Here `A` and `B` are the data arrays of the components of
the input Green's functions. Only their first two dimensions are used.
"""
function _ConvWorkspace(A::AbstractArray{T}, B::AbstractArray{T}) where {T}
    af = zeros(T, size(A, 1), size(B, 1))
    fb = zeros(T, size(A, 2), size(B, 2))
    return _ConvWorkspace{T}(af, fb, Matrix{T}[])
end


"""
    _ConvWorkspace(A::_ConvGF{T}, B::_ConvGF{T})

Constructor for contour-ordered Green's functions `A` and `B`.
"""
function _ConvWorkspace(A::_ConvGF{T}, B::_ConvGF{T}) where {T}
    return _ConvWorkspace(A.ret.data, B.ret.data)
end

#
# Targets: write-through views of the stored blocks at given time step.
#
# _RetRow  : Cᴿ(tₙ,tₘ), m = 1, ⋯, n
# _LmixRow : C^⌉(tₙ,τₘ), m = 1, ⋯, ntau
# _LessCol : C^<(tⱼ,tₙ), j = 1, ⋯, n
# _MatBlocks : Bᴹ(τₘ), m = 1, ⋯, ntau
#

struct _RetRow{T}
    ret :: Gʳᵉᵗ{T}
    n   :: I64
end

@inline Base.getindex(r::_RetRow, m::I64) = _block(r.ret, r.n, m)
#

struct _LmixRow{T}
    lmix :: Gˡᵐⁱˣ{T}
    n    :: I64
end

@inline Base.getindex(r::_LmixRow, m::I64) = _block(r.lmix, r.n, m)
#

struct _LessCol{T}
    less :: Gˡᵉˢˢ{T}
    n    :: I64
end

@inline Base.getindex(r::_LessCol, j::I64) = _block(r.less, j, r.n)
#

struct _MatBlocks{T}
    mat :: Gᵐᵃᵗ{T}
end

@inline Base.getindex(r::_MatBlocks, m::I64) = _block(r.mat, m)

#
# Helpers for the optional contour function f. If `f` is `nothing`, the
# operand is returned as it is.
#
# _af! : a → a fⱼ
# _fb! : b → fⱼ b
# _f0b! : b → f₀ b
#

@inline _af!(ws::_ConvWorkspace, a, ::Nothing, j::I64) = a

@inline function _af!(ws::_ConvWorkspace, a, f, j::I64)
    mul!(ws.af, a, f[j])
    return ws.af
end
#

@inline _fb!(ws::_ConvWorkspace, ::Nothing, j::I64, b) = b

@inline function _fb!(ws::_ConvWorkspace, f, j::I64, b)
    mul!(ws.fb, f[j], b)
    return ws.fb
end
#

@inline _f0b!(ws::_ConvWorkspace, ::Nothing, b) = b

@inline function _f0b!(ws::_ConvWorkspace, f₀, b)
    mul!(ws.fb, f₀, b)
    return ws.fb
end


"""
    _fbmat!(ws::_ConvWorkspace, f₀, B::Gᵐᵃᵗ{T})

Return the Matsubara blocks `f₀ Bᴹ(τₘ)`. If `f₀` is `nothing`, the blocks
of `B` are returned directly. Otherwise, they are evaluated in the
workspace. It is a private function.
"""
_fbmat!(ws::_ConvWorkspace, ::Nothing, B::Gᵐᵃᵗ) = _MatBlocks(B)
#

function _fbmat!(ws::_ConvWorkspace{T}, f₀, B::Gᵐᵃᵗ{T}) where {T}
    ntau = B.ntau
    while length(ws.fbm) < ntau
        push!(ws.fbm, similar(ws.fb))
    end
    for m = 1:ntau
        mul!(ws.fbm[m], f₀, _block(B, m))
    end
    return ws.fbm
end


"""
    _conv_lmix_row!(tgt, a, α, B, j)

Accumulate `α * a * B^⌉(tⱼ,τₘ)` into `tgt[m]` for all `m`. It is a private
function.
"""
function _conv_lmix_row!(tgt, a, α::Number, B::Gˡᵐⁱˣ, j::I64,
                         ms::UnitRange{I64} = 1:B.ntau)
    for m in ms
        mul!(tgt[m], a, _block(B, j, m), α, true)
    end
end


#=
### *Incremental Convolution* : *Targets*
=#

#=
*Remarks* : *Incremental Convolution*

The incremental convolution implements

```math
\begin{equation}
C^{x} \leftarrow C^{x} + \alpha [A \ast f \ast B]^{x},
\end{equation}
```

where ``x \in \{M, R, \rceil, <\}``. The target `C` can be a `ℱ` object
or a `𝒻` object (a time slice). The inputs `A`, `Acc`, `B`, and `Bcc`
must be `ℱ` objects, because the integrals involve all the earlier time
steps. If the contour function `f` is `nothing`, then `f = 1`.

The real-time components are calculated by the same private kernels as
those used in [`convolution_time_step`](@ref). The `mask` argument of
`NESSi` is not supported.

*References* :

Please see [`NESSi`] `cntr_convolution_impl.hpp`, `incr_convolution`.
=#

#
# Private accessors to the target blocks at given time step `tstp`.
#
_mat_target(C::ℱ, tstp::I64) = _MatBlocks(C.mat)

_mat_target(C::𝒻, tstp::I64) = (_check_tstp(C, tstp); C.mat.data)
#

_ret_target(C::ℱ, tstp::I64) = _RetRow(C.ret, tstp)

_ret_target(C::𝒻, tstp::I64) = (_check_tstp(C, tstp); C.ret.data)
#

_lmix_target(C::ℱ, tstp::I64) = _LmixRow(C.lmix, tstp)

_lmix_target(C::𝒻, tstp::I64) = (_check_tstp(C, tstp); C.lmix.data)
#

_less_target(C::ℱ, tstp::I64) = _LessCol(C.less, tstp)

_less_target(C::𝒻, tstp::I64) = (_check_tstp(C, tstp); C.less.data)
#

_mat_target(C::PseudoGF, tstp::I64) = _MatBlocks(C.mat)

_mat_target(C::PseudoGFStep, tstp::I64) = (_check_tstp(C, tstp); C.mat.data)

_ret_target(C::PseudoGF, tstp::I64) = _RetRow(C.ret, tstp)

_ret_target(C::PseudoGFStep, tstp::I64) = (_check_tstp(C, tstp); C.ret.data)

_lmix_target(C::PseudoGF, tstp::I64) = _LmixRow(C.lmix, tstp)

_lmix_target(C::PseudoGFStep, tstp::I64) = (_check_tstp(C, tstp); C.lmix.data)

_less_target(C::PseudoGF, tstp::I64) = _LessCol(C.less, tstp)

_less_target(C::PseudoGFStep, tstp::I64) = (_check_tstp(C, tstp); C.less.data)

#
# A single block target. It is used to evaluate C^<(tₙ,tₙ) only.
#

struct _OneBlock{T}
    blk :: Matrix{T}
    n   :: I64
end

@inline function Base.getindex(r::_OneBlock, m::I64)
    @assert m == r.n
    return r.blk
end

#
# Value of the contour function at the Matsubara branch.
#

_f0(::Nothing) = nothing

_f0(f::Cf) = f[0]


"""
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target)

Sanity check for the incremental convolution. `target` lists the
components of `C` which are written in place, see [`_check_noalias`](@ref).
It is a private function.
"""
function _check_incr(tstp::I64, C, A::_ConvGF, Acc::_ConvGF, f, B::_ConvGF, Bcc::_ConvGF;
                     target::Tuple = _ALL_COMPONENTS)
    @assert iscompatible(A, B)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert getntau(C) == getntau(A)
    @assert getdims(C) == getdims(A)
    @assert 0 ≤ tstp ≤ getntime(A)
    if f isa Cf
        @assert getdims(f) == (getdims(A)[2], getdims(B)[1])
        @assert f.ntime ≥ tstp
    end
    _check_noalias(C, A, Acc, B, Bcc, target)
end


const _ALL_COMPONENTS = (:mat, :ret, :lmix, :less)


"""
    _check_noalias(C, A, Acc, B, Bcc, target = _ALL_COMPONENTS)

The components `target` of `C` are updated in place by the private
convolution kernels while the inputs are still being read. Thus, they
must not share storage with any component of the inputs. It is a private
function.
"""
function _check_noalias(C::_ConvGF, A::_ConvGF, Acc::_ConvGF, B::_ConvGF, Bcc::_ConvGF,
                        target::Tuple = _ALL_COMPONENTS)
    for X in (A, Acc, B, Bcc), s in target, t in _ALL_COMPONENTS
        c = getfield(C, s).data
        x = getfield(X, t).data
        @assert !Base.mightalias(c, x)
    end
end
#
# The blocks of a 𝒻 object are separate matrices, which can not share
# storage with the inputs of type ℱ.
_check_noalias(C::Union{𝒻,PseudoGFStep}, A::_ConvGF, Acc::_ConvGF, B::_ConvGF, Bcc::_ConvGF,
               target::Tuple = ()) = nothing


