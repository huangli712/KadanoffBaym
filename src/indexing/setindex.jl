#
# Project : Lavender
# Source  : setindex.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    Base.setindex!(cf::Cf{T}, x::AbstractMatrix{T}, i::I64)

Setup the element stored in `Cf` object. If `i = 0`, it will setup the
element at Matsubara axis to `x`. On the other hand, if `i > 0`, it
will setup elements at real time axis.

See also: [`Cf`](@ref).
"""
function Base.setindex!(cf::Cf{T}, x::AbstractMatrix{T}, i::I64) where {T}
    # Sanity check
    @assert size(x) == getdims(cf)
    @assert 0 ≤ i ≤ cf.ntime

    # 𝑓(𝑡ᵢ) = x
    if i == 0 # Matsubara axis
        cf.data[end] = copy(x)
    else      # Real time axis
        cf.data[i] = copy(x)
    end
end


"""
    Base.setindex!(cf::Cf{T}, v::T, i::I64)

Setup the element stored in `Cf` object. If `i = 0`, it will setup the
element at Matsubara axis to `v`. On the other hand, if `i > 0`, it
will setup elements at real time axis.

Here, `v` should be a scalar number.

### Examples
```julia
using KadanoffBaym
cf = Cf(11,2)
@show cf[1]
cf[1] = 0.2 + 0.3im
@show cf[1]
```

See also: [`Cf`](@ref).
"""
function Base.setindex!(cf::Cf{T}, v::T, i::I64) where {T}
    # Sanity check
    @assert 0 ≤ i ≤ cf.ntime

    # 𝑓(𝑡ᵢ) .= v
    if i == 0 # Matsubara axis
        fill!(cf.data[end], v)
    else      # Real time axis
        fill!(cf.data[i], v)
    end
end


"""
    Base.setindex!(mat::Gᵐᵃᵗ{T}, x::AbstractMatrix{T}, i::I64)

Setup the element stored in `Gᵐᵃᵗ` object. `x` should be a 2D array, and
`i` is index for imaginary time.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function Base.setindex!(mat::Gᵐᵃᵗ{T}, x::AbstractMatrix{T}, i::I64) where {T}
    # Sanity check
    @assert size(x) == getdims(mat)
    @assert 1 ≤ i ≤ mat.ntau

    # G^{M}(τᵢ) = x
    copyto!(_block(mat, i), x)
end


"""
    Base.setindex!(mat::Gᵐᵃᵗ{T}, v::T, i::I64)

Setup the element stored in `Gᵐᵃᵗ` object. `v` should be a scalar number,
and `i` is index for imaginary time.

### Examples
```julia
using KadanoffBaym
mat = Gᵐᵃᵗ(5,2)
@show mat[2]
mat[2] = 0.1 + 0.2im
@show mat[2]
```

See also: [`Gᵐᵃᵗ`](@ref).
"""
function Base.setindex!(mat::Gᵐᵃᵗ{T}, v::T, i::I64) where {T}
    # Sanity check
    @assert 1 ≤ i ≤ mat.ntau

    # G^{M}(τᵢ) .= v
    fill!(_block(mat, i), v)
end


"""
    Base.setindex!(ret::Gʳᵉᵗ{T}, x::AbstractMatrix{T}, i::I64, j::I64)

Setup the element stored in `Gʳᵉᵗ` object. `x` should be a 2D array, `i`
and `j` are indices for real time.

See also: [`Gʳᵉᵗ`](@ref).
"""
function Base.setindex!(ret::Gʳᵉᵗ{T}, x::AbstractMatrix{T}, i::I64, j::I64) where {T}
    # Sanity check
    @assert size(x) == getdims(ret)
    @assert 1 ≤ i ≤ ret.ntime
    @assert 1 ≤ j ≤ ret.ntime
    @assert i ≥ j

    # G^{R}(tᵢ, tⱼ) = x
    copyto!(_block(ret, i, j), x)
end


"""
    Base.setindex!(ret::Gʳᵉᵗ{T}, v::T, i::I64, j::I64)

Setup the element stored in `Gʳᵉᵗ` object. `v` should be a scalar number,
`i` and `j` are indices for real time.

### Examples
```julia
using KadanoffBaym
ret = Gʳᵉᵗ(5,2,2)
@show ret[2,2]
ret[2,2] = 1.0 - 0.3im
@show ret[2,2]
```

See also: [`Gʳᵉᵗ`](@ref).
"""
function Base.setindex!(ret::Gʳᵉᵗ{T}, v::T, i::I64, j::I64) where {T}
    # Sanity check
    @assert 1 ≤ i ≤ ret.ntime
    @assert 1 ≤ j ≤ ret.ntime
    @assert i ≥ j

    # G^{R}(tᵢ, tⱼ) .= v
    fill!(_block(ret, i, j), v)
end


"""
    Base.setindex!(lmix::Gˡᵐⁱˣ{T}, x::AbstractMatrix{T}, i::I64, j::I64)

Setup the element stored in `Gˡᵐⁱˣ` object. `x` should be a 2D array,
`i` is index for real time, and `j` is index for imaginary time.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function Base.setindex!(lmix::Gˡᵐⁱˣ{T}, x::AbstractMatrix{T}, i::I64, j::I64) where {T}
    # Sanity check
    @assert size(x) == getdims(lmix)
    @assert 1 ≤ i ≤ lmix.ntime
    @assert 1 ≤ j ≤ lmix.ntau

    # G^{⌉}(tᵢ, τⱼ) = x
    copyto!(_block(lmix, i, j), x)
end


"""
    Base.setindex!(lmix::Gˡᵐⁱˣ{T}, v::T, i::I64, j::I64)

Setup the element stored in `Gˡᵐⁱˣ` object. `v` should be a scalar
number, `i` is index for real time, `j` is index for imaginary time.

### Examples
```julia
using KadanoffBaym
lmix = Gˡᵐⁱˣ(8,10,2,2,0.7im)
@show lmix[4,5]
lmix[4,5] = -0.12 - 0.45im
@show lmix[4,5]
```

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function Base.setindex!(lmix::Gˡᵐⁱˣ{T}, v::T, i::I64, j::I64) where {T}
    # Sanity check
    @assert 1 ≤ i ≤ lmix.ntime
    @assert 1 ≤ j ≤ lmix.ntau

    # G^{⌉}(tᵢ, τⱼ) .= v
    fill!(_block(lmix, i, j), v)
end


"""
    Base.setindex!(less::Gˡᵉˢˢ{T}, x::AbstractMatrix{T}, i::I64, j::I64)

Setup the element stored in `Gˡᵉˢˢ` object. `x` should be a 2D array,
`i` and `j` are indices for real time.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function Base.setindex!(less::Gˡᵉˢˢ{T}, x::AbstractMatrix{T}, i::I64, j::I64) where {T}
    # Sanity check
    @assert size(x) == getdims(less)
    @assert 1 ≤ i ≤ less.ntime
    @assert 1 ≤ j ≤ less.ntime
    @assert i ≤ j

    # G^{<}(tᵢ, tⱼ) = x
    copyto!(_block(less, i, j), x)
end


"""
    Base.setindex!(less::Gˡᵉˢˢ{T}, v::T, i::I64, j::I64)

Setup the element stored in `Gˡᵉˢˢ` object. `v` should be a scalar
number, `i` and `j` are indices for real time.

### Examples
``` julia
using KadanoffBaym
less = Gˡᵉˢˢ(5,2,2,0.3+0.4im)
@show less[1,2]
less[1,2] = 0.2im
@show less[1,2]
```

See also: [`Gˡᵉˢˢ`](@ref).
"""
function Base.setindex!(less::Gˡᵉˢˢ{T}, v::T, i::I64, j::I64) where {T}
    # Sanity check
    @assert 1 ≤ i ≤ less.ntime
    @assert 1 ≤ j ≤ less.ntime
    @assert i ≤ j

    # G^{<}(tᵢ, tⱼ) .= v
    fill!(_block(less, i, j), v)
end


"""
    Base.setindex!(mat::gᵐᵃᵗ{S}, x::AbstractMatrix{S}, i::I64)

Setup the element stored in `gᵐᵃᵗ` object. `x` should be a 2D array, and
`i` is index for imaginary time.

See also: [`gᵐᵃᵗ`](@ref).
"""
function Base.setindex!(mat::gᵐᵃᵗ{S}, x::AbstractMatrix{S}, i::I64) where {S}
    # Sanity check
    @assert size(x) == getdims(mat)
    @assert 1 ≤ i ≤ mat.ntau

    # G^{M}(τᵢ) = x
    mat.data[i] = copy(x)
end


"""
    Base.setindex!(mat::gᵐᵃᵗ{S}, v::S, i::I64)

Setup the element stored in `gᵐᵃᵗ` object. `v` should be a scalar number,
and `i` is index for imaginary time.

### Examples
```julia
using KadanoffBaym
mat = gᵐᵃᵗ(5,2,2,0.4im)
@show mat[2]
mat[2] = 1.0 - 0.3im
@show mat[2]
```

See also: [`gᵐᵃᵗ`](@ref).
"""
function Base.setindex!(mat::gᵐᵃᵗ{S}, v::S, i::I64) where {S}
    # Sanity check
    @assert 1 ≤ i ≤ mat.ntau

    # G^{M}(τᵢ) .= v
    fill!(mat.data[i], v)
end


"""
    Base.setindex!(ret::gʳᵉᵗ{S}, x::AbstractMatrix{S}, j::I64)

Setup the element stored in `gʳᵉᵗ` object. `x` should be a 2D array, `j`
is index for real time.

See also: [`gʳᵉᵗ`](@ref).
"""
function Base.setindex!(ret::gʳᵉᵗ{S}, x::AbstractMatrix{S}, j::I64) where {S}
    # Sanity check
    @assert size(x) == getdims(ret)
    @assert 1 ≤ j ≤ ret.tstp

    # G^{R}(tᵢ ≡ tstp, tⱼ) = x
    ret.data[j] = copy(x)
end


"""
    Base.setindex!(ret::gʳᵉᵗ{S}, v::S, j::I64)

Setup the element stored in `gʳᵉᵗ` object. `v` should be a scalar number,
`j` is index for real time.

### Examples
```julia
using KadanoffBaym
ret = gʳᵉᵗ(5,2,2,0.3im)
@show ret[2]
ret[2] = 0.3im - 0.2
@show ret[2]
```

See also: [`gʳᵉᵗ`](@ref).
"""
function Base.setindex!(ret::gʳᵉᵗ{S}, v::S, j::I64) where {S}
    # Sanity check
    @assert 1 ≤ j ≤ ret.tstp

    # G^{R}(tᵢ ≡ tstp, tⱼ) .= v
    fill!(ret.data[j], v)
end


"""
    Base.setindex!(lmix::gˡᵐⁱˣ{S}, x::AbstractMatrix{S}, j::I64)

Setup the element stored in `gˡᵐⁱˣ` object. `x` should be a 2D array, `j`
is index for imaginary time.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function Base.setindex!(lmix::gˡᵐⁱˣ{S}, x::AbstractMatrix{S}, j::I64) where {S}
    # Sanity check
    @assert size(x) == getdims(lmix)
    @assert 1 ≤ j ≤ lmix.ntau

    # G^{⌉}(tᵢ ≡ tstp, τⱼ) = x
    lmix.data[j] = copy(x)
end


"""
    Base.setindex!(lmix::gˡᵐⁱˣ{S}, v::S, j::I64)

Setup the element stored in `gˡᵐⁱˣ` object. `v` should be a scalar number,
`j` is index for imaginary time.

### Examples
```julia
using KadanoffBaym
lmix = gˡᵐⁱˣ(5,2,2,0.3im)
@show lmix[2]
lmix[2] = 0.2im + 0.3
@show lmix[2]
```

See also: [`gˡᵐⁱˣ`](@ref).
"""
function Base.setindex!(lmix::gˡᵐⁱˣ{S}, v::S, j::I64) where {S}
    # Sanity check
    @assert 1 ≤ j ≤ lmix.ntau

    # G^{⌉}(tᵢ ≡ tstp, τⱼ) .= v
    fill!(lmix.data[j], v)
end


"""
    Base.setindex!(less::gˡᵉˢˢ{S}, x::AbstractMatrix{S}, i::I64)

Setup the element stored in `gˡᵉˢˢ` object. `x` should be a 2D array, `i`
is index for real time.

See also: [`gˡᵉˢˢ`](@ref).
"""
function Base.setindex!(less::gˡᵉˢˢ{S}, x::AbstractMatrix{S}, i::I64) where {S}
    # Sanity check
    @assert size(x) == getdims(less)
    @assert 1 ≤ i ≤ less.tstp

    # G^{<}(tᵢ, tⱼ ≡ tstp) = x
    less.data[i] = copy(x)
end


"""
    Base.setindex!(less::gˡᵉˢˢ{S}, v::S, i::I64)

Setup the element stored in `gˡᵉˢˢ` object. `v` should be a scalar
number, `i` is index for real time.

### Examples
```julia
using KadanoffBaym
less = gˡᵉˢˢ(5,3,3,0.3im)
@show less[4]
less[4] = 0.1 - 0.2im
@show less[4]
```

See also: [`gˡᵉˢˢ`](@ref).
"""
function Base.setindex!(less::gˡᵉˢˢ{S}, v::S, i::I64) where {S}
    # Sanity check
    @assert 1 ≤ i ≤ less.tstp

    # G^{<}(tᵢ, tⱼ ≡ tstp) .= v
    fill!(less.data[i], v)
end


"""
    Base.setindex!(cfm::ℱ{S}, cfv::𝒻{S}, tstp::I64)

Setup contour-ordered Green's function at given time step `tstp`.

See also: [`ℱ`](@ref), [`𝒻`](@ref).
"""
function Base.setindex!(cfm::ℱ{S}, cfv::𝒻{S}, tstp::I64) where {S}
    # Sanity check
    @assert tstp == gettstp(cfv)
    @assert 0 ≤ tstp ≤ getntime(cfm)

    # Copy data from `𝒻` object to `ℱ` object
    memcpy!(cfv, cfm, tstp)
end


"""
    Base.setindex!(pgf::PseudoGF{S}, pgv::PseudoGFStep{S}, tstp::I64)

Setup pseudo-particle Green's function at given time step `tstp`.

See also: [`PseudoGF`](@ref), [`PseudoGFStep`](@ref).
"""
function Base.setindex!(pgf::PseudoGF{S}, pgv::PseudoGFStep{S}, tstp::I64) where {S}
    # Sanity check
    @assert tstp == gettstp(pgv)
    @assert 0 ≤ tstp ≤ getntime(pgf)

    # Copy data from `PseudoGFStep` object to `PseudoGF` object
    memcpy!(pgv, pgf, tstp)
end


"""
    Base.setindex!(A::FFTArray{T}, M::AbstractMatrix, n::I64, dom::FFTDomain)

Copy the matrix `M` to the 1-based storage index `n` in the domain `dom`.

See also: [`FFTArray`](@ref).
"""
function Base.setindex!(A::FFTArray{T}, M::AbstractMatrix, n::I64, dom::FFTDomain) where {T}
    1 ≤ n ≤ getnft(A) || throw(BoundsError(1:getnft(A), n))
    size(M) == getdims(A) ||
        throw(DimensionMismatch("the matrix must have the size $(getdims(A))"))
    copyto!(view(_fft_data(A, dom), :, :, n), M)
    return A
end


"""
    Base.setindex!(A::FFTArray{T}, x::Number, n::I64, dom::FFTDomain)

Set the scalar `x` at the 1-based storage index `n` in the domain `dom`.
It is for 1 × 1 orbital matrices only.

See also: [`FFTArray`](@ref).
"""
function Base.setindex!(A::FFTArray{T}, x::Number, n::I64, dom::FFTDomain) where {T}
    getdims(A) == (1, 1) ||
        throw(ArgumentError("a scalar can only be set for 1 × 1 matrices"))
    1 ≤ n ≤ getnft(A) || throw(BoundsError(1:getnft(A), n))
    _fft_data(A, dom)[1, 1, n] = x
    return A
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


"""
    Base.setindex!(v::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ}, x::AbstractMatrix, c::Symbol,
                   deltj::I64)

Set the block `deltj` of the component `c` of a moving time step or view.
A view writes through to its parent.
"""
function Base.setindex!(v::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ}, x::AbstractMatrix,
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


