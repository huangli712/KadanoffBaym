#
# Project : Lavender
# Source  : elemcpy.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Element Copy Operations*
=#

"""
    elemcpy!(
        cz1::CopyZone,
        src::ℱ{T},
        cz2::CopyZone,
        dst::ℱ{T}
    ) where {T}

Copy matrix elements between two contour-ordered Green's functions within
specified zones for all time steps.

### Arguments
* cz1 -> Source zone in the source Green's function.
* src -> source contour-ordered Green's function (ℱ).
* cz2 -> Destination zone in the destination Green's function.
* dst -> Destination contour-ordered Green's function (ℱ).

### Returns
* `dst` should be modified.

### Notes
This function performs element-wise copy for all time steps. It copies the
Matsubara, retarded, left-mixing, and lesser components. The source and
destination must have the same number of time steps. Both copy zones must
be valid and compatible with each other.

See also: [`CopyZone`](@ref).
"""
function elemcpy!(
    cz1::CopyZone,
    src::ℱ{T},
    cz2::CopyZone,
    dst::ℱ{T}
) where {T}
    # Extract parameters
    ntime = getntime(src)
    
    # Sanity check
    @assert getntime(src) == getntime(dst)
    @assert iscompatible(cz1, cz2)

    elemcpy!(cz1, src.mat, cz2, dst.mat)
    for tstp = 1:ntime
        elemcpy!(tstp, cz1, src.ret, cz2, dst.ret)
        elemcpy!(tstp, cz1, src.lmix, cz2, dst.lmix)
        elemcpy!(tstp, cz1, src.less, cz2, dst.less)
    end
end

"""
    elemcpy!(
        tstp::I64,
        cz1::CopyZone,
        src::ℱ{T},
        cz2::CopyZone,
        dst::ℱ{T}
    ) where {T}

Copy matrix elements between two contour-ordered Green's functions within
specified zones at a given time step.

### Arguments
* tstp -> Time step index.
* cz1 -> Source zone in the source Green's function.
* src -> Source contour-ordered Green's function (ℱ).
* cz2 -> Destination zone in the destination Green's function.
* dst -> Destination contour-ordered Green's function (ℱ).

### Returns
* `dst` should be modified.

### Notes
This function performs element-wise copy only at a specific time step
(`t = tstp`). It copies the retarded, left-mixing, and lesser components.
The source and destination must have the same number of time steps. Both
copy zones must be valid and compatible with each other.

See also: [`CopyZone`](@ref).
"""
function elemcpy!(
    tstp::I64,
    cz1::CopyZone,
    src::ℱ{T},
    cz2::CopyZone,
    dst::ℱ{T}
) where {T}
    # Extract parameters
    ntime = getntime(src)
    
    # Sanity check
    @assert getntime(src) == getntime(dst)
    @assert iscompatible(cz1, cz2)
    @assert ntime ≥ tstp ≥ 1

    elemcpy!(tstp, cz1, src.ret, cz2, dst.ret)
    elemcpy!(tstp, cz1, src.lmix, cz2, dst.lmix)
    elemcpy!(tstp, cz1, src.less, cz2, dst.less)
end

"""
    elemcpy!(
        cz1::CopyZone,
        src::Gᵐᵃᵗ{T},
        cz2::CopyZone,
        dst::Gᵐᵃᵗ{T}
    ) where {T}

Copy matrix elements between Matsubara components of two contour-ordered
Green's functions within specified zones.

### Arguments
* cz1 -> Source zone in the source Green's function.
* src -> Source Matsubara Green's function (Gᵐᵃᵗ).
* cz2 -> Destination zone in the destination Green's function.
* dst -> Destination Matsubara Green's function (Gᵐᵃᵗ).

### Returns
* `dst` should be modified.

### Notes
This function performs element-wise copy for all imaginary time points.
The source and destination must have the same number of imaginary time
points. Both copy zones must be valid and compatible with their respective
objects.

See also: [`CopyZone`](@ref).
"""
function elemcpy!(
    cz1::CopyZone,
    src::Gᵐᵃᵗ{T},
    cz2::CopyZone,
    dst::Gᵐᵃᵗ{T}
) where {T}
    # Extract parameters
    ntau = getntau(src)

    # Sanity check
    @assert getntau(src) == getntau(dst)
    @assert iscompatible(cz1, src)
    @assert iscompatible(cz2, dst)
    @assert iscompatible(cz1, cz2)
    @assert isvalid(cz1)
    @assert isvalid(cz2)

    # Copy elements
    for i = 1:ntau
        _block(dst, i)[cz2.x₁:cz2.x₂, cz2.y₁:cz2.y₂] .=
            _block(src, i)[cz1.x₁:cz1.x₂, cz1.y₁:cz1.y₂]
    end
end

"""
    elemcpy!(
        tstp::I64,
        cz1::CopyZone,
        src::Gʳᵉᵗ{T},
        cz2::CopyZone,
        dst::Gʳᵉᵗ{T}
    ) where {T}

Copy matrix elements between retarded components of two contour-ordered
Green's functions within specified zones at a given time step.

### Arguments
* tstp -> Time step index.
* cz1 -> Source zone in the source Green's function.
* src -> Source retarded Green's function (Gʳᵉᵗ).
* cz2 -> Destination zone in the destination Green's function.
* dst -> Destination retarded Green's function (Gʳᵉᵗ).

### Returns
* `dst` should be modified.

### Notes
This function performs element-wise copy for `t = tstp` and `t' < tstp`.
The source and destination must have the same number of time points. Both
copy zones must be valid and compatible with their respective objects.

See also: [`CopyZone`](@ref).
"""
function elemcpy!(
    tstp::I64,
    cz1::CopyZone,
    src::Gʳᵉᵗ{T},
    cz2::CopyZone,
    dst::Gʳᵉᵗ{T}
) where {T}
    # Extract parameters
    ntime = getntime(src)
    
    # Sanity check
    @assert getntime(src) == getntime(dst)
    @assert iscompatible(cz1, src)
    @assert iscompatible(cz2, dst)
    @assert iscompatible(cz1, cz2)
    @assert isvalid(cz1)
    @assert isvalid(cz2)
    @assert ntime ≥ tstp ≥ 1

    # Copy elements
    for i = 1:tstp
        _block(dst, tstp, i)[cz2.x₁:cz2.x₂, cz2.y₁:cz2.y₂] .=
            _block(src, tstp, i)[cz1.x₁:cz1.x₂, cz1.y₁:cz1.y₂]
    end
end

"""
    elemcpy!(
        tstp::I64,
        cz1::CopyZone,
        src::Gˡᵐⁱˣ{T},
        cz2::CopyZone,
        dst::Gˡᵐⁱˣ{T}
    ) where {T}

Copy matrix elements between left-mixing components of two contour-ordered
Green's functions within specified zones at a given time step.

### Arguments
* tstp -> Time step index.
* cz1 -> Source zone in the source Green's function.
* src -> Source left-mixing Green's function (Gˡᵐⁱˣ).
* cz2 -> Destination zone in the destination Green's function.
* dst -> Destination left-mixing Green's function (Gˡᵐⁱˣ).

### Returns
* `dst` should be modified.

### Notes
This function performs element-wise copy for all imaginary time points at
a given time step (`t = tstp`). The source and destination must have the
same number of time points and imaginary time points. Both copy zones must
be valid and compatible with their respective objects.

See also: [`CopyZone`](@ref).
"""
function elemcpy!(
    tstp::I64,
    cz1::CopyZone,
    src::Gˡᵐⁱˣ{T},
    cz2::CopyZone,
    dst::Gˡᵐⁱˣ{T}
) where {T}
    # Extract parameters
    ntime = getntime(src)
    ntau = getntau(src)

    # Sanity check
    @assert getntime(src) == getntime(dst)
    @assert getntau(src) == getntau(dst)
    @assert iscompatible(cz1, src)
    @assert iscompatible(cz2, dst)
    @assert iscompatible(cz1, cz2)
    @assert isvalid(cz1)
    @assert isvalid(cz2)
    @assert ntime ≥ tstp ≥ 1

    # Copy elements
    for i = 1:ntau
        _block(dst, tstp, i)[cz2.x₁:cz2.x₂, cz2.y₁:cz2.y₂] .=
            _block(src, tstp, i)[cz1.x₁:cz1.x₂, cz1.y₁:cz1.y₂]
    end
end

"""
    elemcpy!(
        tstp::I64,
        cz1::CopyZone,
        src::Gˡᵉˢˢ{T},
        cz2::CopyZone,
        dst::Gˡᵉˢˢ{T}
    ) where {T}

Copy matrix elements between lesser components of two contour-ordered
Green's functions within specified zones at a given time step.

### Arguments
* tstp -> Time step index.
* cz1 -> Source zone in the source Green's function.
* src -> Source lesser Green's function (Gˡᵉˢˢ).
* cz2 -> Destination zone in the destination Green's function.
* dst -> Destination lesser Green's function (Gˡᵉˢˢ).

### Returns
* `dst` should be modified.

### Notes
This function performs element-wise copy for `t < tstp` and `t' = tstp`.
The source and destination must have the same number of time points. Both
copy zones must be valid and compatible with their respective objects.

See also: [`CopyZone`](@ref).
"""
function elemcpy!(
    tstp::I64,
    cz1::CopyZone,
    src::Gˡᵉˢˢ{T},
    cz2::CopyZone,
    dst::Gˡᵉˢˢ{T}
) where {T}
    # Extract parameters
    ntime = getntime(src)

    # Sanity check
    @assert getntime(src) == getntime(dst)
    @assert iscompatible(cz1, src)
    @assert iscompatible(cz2, dst)
    @assert iscompatible(cz1, cz2)
    @assert isvalid(cz1)
    @assert isvalid(cz2)
    @assert ntime ≥ tstp ≥ 1

    # Copy elements
    for i = 1:tstp
        _block(dst, i, tstp)[cz2.x₁:cz2.x₂, cz2.y₁:cz2.y₂] .=
            _block(src, i, tstp)[cz1.x₁:cz1.x₂, cz1.y₁:cz1.y₂]
    end
end

#=
### *Element Copy Operations* : *Index Maps*
=#

"""
    elemcpy!(
        rows1::AbstractVector{I64},
        cols1::AbstractVector{I64},
        src::ℱ{T},
        rows2::AbstractVector{I64},
        cols2::AbstractVector{I64},
        dst::ℱ{T}
    ) where {T}

Copy matrix elements between two contour-ordered Green's functions for
all time steps, according to the index maps
`dst[rows2[k], cols2[k]] = src[rows1[k], cols1[k]]`.

### Arguments
* rows1 -> Row indices in the source Green's function.
* cols1 -> Column indices in the source Green's function.
* src   -> Source contour-ordered Green's function (ℱ).
* rows2 -> Row indices in the destination Green's function.
* cols2 -> Column indices in the destination Green's function.
* dst   -> Destination contour-ordered Green's function (ℱ).

### Returns
* `dst` should be modified.

### Notes
The index maps need not form a rectangular zone. The elements of `dst`
which are not addressed by the maps are not touched. It is equivalent to
the `set_submatrix()` function in `NESSi`.

See also: [`CopyZone`](@ref).
"""
function elemcpy!(
    rows1::AbstractVector{I64},
    cols1::AbstractVector{I64},
    src::ℱ{T},
    rows2::AbstractVector{I64},
    cols2::AbstractVector{I64},
    dst::ℱ{T}
) where {T}
    # Sanity check
    @assert length(rows1) == length(cols1) == length(rows2) == length(cols2)

    for k in eachindex(rows1)
        cz1 = CopyZone(rows1[k], cols1[k])
        cz2 = CopyZone(rows2[k], cols2[k])
        elemcpy!(cz1, src, cz2, dst)
    end
end

"""
    elemcpy!(
        tstp::I64,
        rows1::AbstractVector{I64},
        cols1::AbstractVector{I64},
        src::ℱ{T},
        rows2::AbstractVector{I64},
        cols2::AbstractVector{I64},
        dst::ℱ{T}
    ) where {T}

Copy matrix elements between two contour-ordered Green's functions at a
given time step, according to the index maps
`dst[rows2[k], cols2[k]] = src[rows1[k], cols1[k]]`. If `tstp = 0`, the
Matsubara component is copied. Otherwise, the retarded, left-mixing, and
lesser components at `t = tstp` are copied.

### Arguments
* tstp  -> Time step index.
* rows1 -> Row indices in the source Green's function.
* cols1 -> Column indices in the source Green's function.
* src   -> Source contour-ordered Green's function (ℱ).
* rows2 -> Row indices in the destination Green's function.
* cols2 -> Column indices in the destination Green's function.
* dst   -> Destination contour-ordered Green's function (ℱ).

### Returns
* `dst` should be modified.

See also: [`CopyZone`](@ref).
"""
function elemcpy!(
    tstp::I64,
    rows1::AbstractVector{I64},
    cols1::AbstractVector{I64},
    src::ℱ{T},
    rows2::AbstractVector{I64},
    cols2::AbstractVector{I64},
    dst::ℱ{T}
) where {T}
    # Sanity check
    @assert length(rows1) == length(cols1) == length(rows2) == length(cols2)
    @assert getntime(src) ≥ tstp ≥ 0

    for k in eachindex(rows1)
        cz1 = CopyZone(rows1[k], cols1[k])
        cz2 = CopyZone(rows2[k], cols2[k])
        if tstp == 0
            elemcpy!(cz1, src.mat, cz2, dst.mat)
        else
            elemcpy!(tstp, cz1, src, cz2, dst)
        end
    end
end
