#
# Project : Lavender
# Source  : element.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/01/13
#

#=
### *CopyZone Struct*
=#

"""
    CopyZone(x₁::I64, y₁::I64, x₂::I64, y₂::I64)

Struct to define a rectangular zone for element-wise copy operations.

### Arguments
* x₁ -> Starting row index (1-based).
* y₁ -> Starting column index (1-based).
* x₂ -> Ending row index (1-based).
* y₂ -> Ending column index (1-based).

### Notes

This struct defines a rectangular region `[x₁:x₂, y₁:y₂]` for copying
elements between two different matrices. All indices are 1-based and
must satisfy `x₂ ≥ x₁ ≥ 1` and `y₂ ≥ y₁ ≥ 1`.

See also: [`elemcpy!`](@ref).
"""
struct CopyZone
    x₁::I64
    y₁::I64
    x₂::I64
    y₂::I64
end

"""
    CopyZone(x::I64, y::I64)

Create a single-point copy zone.

### Arguments
* x -> Row index (1-based).
* y -> Column index (1-based).

### Returns
* CopyZone object representing a single cell at position (x, y).

See also: [`elemcpy!`](@ref).
"""
function CopyZone(x::I64, y::I64)
    return CopyZone(x, y, x, y)
end

"""
    CopyZone(x::I64, y::I64, δ::I64)

Create a square-like copy zone starting from position (x, y).

### Arguments
* x -> Starting row index (1-based).
* y -> Starting column index (1-based).
* δ -> Width (or height) of the square region (must be ≥ 1).

### Returns
* CopyZone object representing a square region `[x:x+δ-1, y:y+δ-1]`.

See also: [`elemcpy!`](@ref).
"""
function CopyZone(x::I64, y::I64, δ::I64)
    @assert δ ≥ 1
    return CopyZone(x, y, x + δ - 1, y + δ - 1)
end

"""
    isvalid(cz::CopyZone)

Check if a copy zone is valid.

### Arguments
* cz -> CopyZone object to validate.

### Returns
* `true` if the copy zone is valid, `false` otherwise.

### Notes

A copy zone is valid if all indices satisfy `x₂ ≥ x₁ ≥ 1` and `y₂ ≥ y₁ ≥ 1`.

See also: [`CopyZone`](@ref).
"""
function isvalid(cz::CopyZone)
    return cz.x₂ ≥ cz.x₁ ≥ 1 && cz.y₂ ≥ cz.y₁ ≥ 1
end

"""
    iscompatible(cz1::CopyZone, cz2::CopyZone)

Check if two copy zones have compatible dimensions.

### Arguments
* cz1 -> First CopyZone object.
* cz2 -> Second CopyZone object.

### Returns
* `true` if the zones have the same dimensions, `false` otherwise.

### Notes

Two copy zones are compatible if they have the same width and height.

See also: [`CopyZone`](@ref).
"""
function iscompatible(cz1::CopyZone, cz2::CopyZone)
    return (cz1.x₂ - cz1.x₁) == (cz2.x₂ - cz2.x₁) &&
           (cz1.y₂ - cz1.y₁) == (cz2.y₂ - cz2.y₁)
end

"""
    iscompatible(cz::CopyZone, obj::CnAbstractMatrix{T}) where {T}

Check if a copy zone is compatible with a matrix object. Here, `obj` marks
the components of contour-ordered Green's functions.

### Arguments
* cz -> CopyZone object.
* obj -> Matrix object (CnAbstractMatrix).

### Returns
* `true` if the zone fits within the matrix dimensions, `false` otherwise.

See also: [`CopyZone`](@ref).
"""
function iscompatible(cz::CopyZone, obj::CnAbstractMatrix{T}) where {T}
    return (cz.x₁, cz.y₁) ≤ getdims(obj) &&
           (cz.x₂, cz.y₂) ≤ getdims(obj)
end

"""
    iscompatible(obj::CnAbstractMatrix{T}, cz::CopyZone) where {T}

Check if a matrix object is compatible with a copy zone. Here, `obj` marks
the components of contour-ordered Green's functions.

### Arguments
* obj -> Matrix object (CnAbstractMatrix).
* cz -> CopyZone object.

### Returns
* `true` if the zone fits within the matrix dimensions, `false` otherwise.

See also: [`CopyZone`](@ref).
"""
function iscompatible(obj::CnAbstractMatrix{T}, cz::CopyZone) where {T}
    return iscompatible(cz, obj)
end

"""
    iscompatible(cz::CopyZone, obj::CnAbstractVector{T}) where {T}

Check if a copy zone is compatible with a vector object. Here, `obj` marks
the components of contour-ordered Green's functions at given time step.

### Arguments
* cz -> CopyZone object.
* obj -> Vector object (CnAbstractVector).

### Returns
* `true` if the zone fits within the matrix dimensions, `false` otherwise.

See also: [`CopyZone`](@ref).
"""
function iscompatible(cz::CopyZone, obj::CnAbstractVector{T}) where {T}
    return (cz.x₁, cz.y₁) ≤ getdims(obj) &&
           (cz.x₂, cz.y₂) ≤ getdims(obj)
end

"""
    iscompatible(obj::CnAbstractVector{T}, cz::CopyZone) where {T}

Check if a vector object is compatible with a copy zone. Here, `obj` marks
the components of contour-ordered Green's functions at given time step.

### Arguments
* obj -> Vector object (CnAbstractVector).
* cz -> CopyZone object.

### Returns
* `true` if the zone fits within the matrix dimensions, `false` otherwise.

See also: [`CopyZone`](@ref).
"""
function iscompatible(obj::CnAbstractVector{T}, cz::CopyZone) where {T}
    return iscompatible(cz, obj)
end

#=
### *Element Copy Operations*
=#


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

Copy elements between Matsubara components of two contour-ordered Green's
functions within specified zones.

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
        dst.data[i,1][cz2.x₁:cz2.x₂, cz2.y₁:cz2.y₂] .=
            src.data[i,1][cz1.x₁:cz1.x₂, cz1.y₁:cz1.y₂]
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

Copy elements between retarded components of two contour-ordered Green's
functions within specified zones at a given time step.

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
        dst.data[tstp,i][cz2.x₁:cz2.x₂, cz2.y₁:cz2.y₂] .=
            src.data[tstp,i][cz1.x₁:cz1.x₂, cz1.y₁:cz1.y₂]
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

Copy elements between left-mixing components of two contour-ordered
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
        dst.data[tstp,i][cz2.x₁:cz2.x₂, cz2.y₁:cz2.y₂] .=
            src.data[tstp,i][cz1.x₁:cz1.x₂, cz1.y₁:cz1.y₂]
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

Copy elements between lesser components of two contour-ordered Green's
functions within specified zones at a given time step.

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
        dst.data[i,tstp][cz2.x₁:cz2.x₂, cz2.y₁:cz2.y₂] .=
            src.data[i,tstp][cz1.x₁:cz1.x₂, cz1.y₁:cz1.y₂]
    end
end
