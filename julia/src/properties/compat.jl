#
# Project : Lavender
# Source  : compat.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Compatibility*
=#

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

"""
    iscompatible(cf1::Cf{T}, cf2::Cf{T})

Judge whether two `Cf` objects are compatible.
"""
function iscompatible(cf1::Cf{T}, cf2::Cf{T}) where {T}
    getsize(cf1) == getsize(cf2) &&
    getdims(cf1) == getdims(cf2)
end

"""
    iscompatible(C::Cn, cf::Cf{T})

Judge whether `C` (which is a `Cn` object) is compatible with `cf`
(which is a `Cf{T}` object).
"""
function iscompatible(C::Cn, cf::Cf{T}) where {T}
    getntime(C) == getntime(cf) &&
    getdims(C) == getdims(cf)
end

"""
    iscompatible(cf::Cf{T}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `cf`
(which is a `Cf{T}` object).
"""
iscompatible(cf::Cf{T}, C::Cn) where {T} = iscompatible(C, cf)

"""
    iscompatible(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T})

Judge whether two `Gᵐᵃᵗ` objects are compatible.
"""
function iscompatible(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T}) where {T}
    getsize(mat1) == getsize(mat2) &&
    getdims(mat1) == getdims(mat2)
end

"""
    iscompatible(C::Cn, mat::Gᵐᵃᵗ{T})

Judge whether `C` (which is a `Cn` object) is compatible with `mat`
(which is a `Gᵐᵃᵗ{T}` object).
"""
function iscompatible(C::Cn, mat::Gᵐᵃᵗ{T}) where {T}
    getntau(C) == getntau(mat) &&
    getdims(C) == getdims(mat)
end

"""
    iscompatible(mat::Gᵐᵃᵗ{T}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `mat`
(which is a `Gᵐᵃᵗ{T}` object).
"""
iscompatible(mat::Gᵐᵃᵗ{T}, C::Cn) where {T} = iscompatible(C, mat)

"""
    iscompatible(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T})

Judge whether two `Gʳᵉᵗ` objects are compatible.
"""
function iscompatible(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}) where {T}
    getsize(ret1) == getsize(ret2) &&
    getdims(ret1) == getdims(ret2)
end

"""
    iscompatible(C::Cn, ret::Gʳᵉᵗ{T})

Judge whether `C` (which is a `Cn` object) is compatible with `ret`
(which is a `Gʳᵉᵗ{T}` object).
"""
function iscompatible(C::Cn, ret::Gʳᵉᵗ{T}) where {T}
    getntime(C) == getntime(ret) &&
    getdims(C) == getdims(ret)
end

"""
    iscompatible(ret::Gʳᵉᵗ{T}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `ret`
(which is a `Gʳᵉᵗ{T}` object).
"""
iscompatible(ret::Gʳᵉᵗ{T}, C::Cn) where {T} = iscompatible(C, ret)

"""
    iscompatible(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T})

Judge whether two `Gˡᵐⁱˣ` objects are compatible.
"""
function iscompatible(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}) where {T}
    getsize(lmix1) == getsize(lmix2) &&
    getdims(lmix1) == getdims(lmix2)
end

"""
    iscompatible(C::Cn, lmix::Gˡᵐⁱˣ{T})

Judge whether `C` (which is a `Cn` object) is compatible with `lmix`
(which is a `Gˡᵐⁱˣ{T}` object).
"""
function iscompatible(C::Cn, lmix::Gˡᵐⁱˣ{T}) where {T}
    getsize(C) == getsize(lmix) &&
    getdims(C) == getdims(lmix)
end

"""
    iscompatible(lmix::Gˡᵐⁱˣ{T}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `lmix`
(which is a `Gˡᵐⁱˣ{T}` object).
"""
iscompatible(lmix::Gˡᵐⁱˣ{T}, C::Cn) where {T} = iscompatible(C, lmix)

"""
    iscompatible(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T})

Judge whether two `Gˡᵉˢˢ` objects are compatible.
"""
function iscompatible(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}) where {T}
    getsize(less1) == getsize(less2) &&
    getdims(less1) == getdims(less2)
end

"""
    iscompatible(C::Cn, less::Gˡᵉˢˢ{T})

Judge whether `C` (which is a `Cn` object) is compatible with `less`
(which is a `Gˡᵉˢˢ{T}` object).
"""
function iscompatible(C::Cn, less::Gˡᵉˢˢ{T}) where {T}
    getntime(C) == getntime(less) &&
    getdims(C) == getdims(less)
end

"""
    iscompatible(less::Gˡᵉˢˢ{T}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `less`
(which is a `Gˡᵉˢˢ{T}` object).
"""
iscompatible(less::Gˡᵉˢˢ{T}, C::Cn) where {T} = iscompatible(C, less)

"""
    iscompatible(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S})

Judge whether two `gᵐᵃᵗ` objects are compatible.
"""
function iscompatible(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}) where {S}
    getsize(mat1) == getsize(mat2) &&
    getdims(mat1) == getdims(mat2)
end

"""
    iscompatible(mat1::gᵐᵃᵗ{S}, mat2::Gᵐᵃᵗ{S})

Judge whether the `gᵐᵃᵗ` and `Gᵐᵃᵗ` objects are compatible.
"""
function iscompatible(mat1::gᵐᵃᵗ{S}, mat2::Gᵐᵃᵗ{S}) where {S}
    getsize(mat1) == getsize(mat2) &&
    getdims(mat1) == getdims(mat2)
end

"""
    iscompatible(mat1::Gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S})

Judge whether the `gᵐᵃᵗ` and `Gᵐᵃᵗ` objects are compatible.
"""
iscompatible(mat1::Gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}) where {S} = iscompatible(mat2, mat1)

"""
    iscompatible(C::Cn, mat::gᵐᵃᵗ{S})

Judge whether `C` (which is a `Cn` object) is compatible with `mat`
(which is a `gᵐᵃᵗ{S}` object).
"""
function iscompatible(C::Cn, mat::gᵐᵃᵗ{S}) where {S}
    getntau(C) == getntau(mat) &&
    getdims(C) == getdims(mat)
end

"""
    iscompatible(mat::gᵐᵃᵗ{S}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `mat`
(which is a `gᵐᵃᵗ{S}` object).
"""
iscompatible(mat::gᵐᵃᵗ{S}, C::Cn) where {S} = iscompatible(C, mat)

"""
    iscompatible(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S})

Judge whether two `gʳᵉᵗ` objects are compatible.
"""
function iscompatible(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}) where {S}
    getsize(ret1) == getsize(ret2) &&
    getdims(ret1) == getdims(ret2)
end

"""
    iscompatible(ret1::gʳᵉᵗ{S}, ret2::Gʳᵉᵗ{S})

Judge whether the `gʳᵉᵗ` and `Gʳᵉᵗ` objects are compatible.
"""
function iscompatible(ret1::gʳᵉᵗ{S}, ret2::Gʳᵉᵗ{S}) where {S}
    getsize(ret1) ≤ getsize(ret2) &&
    getdims(ret1) == getdims(ret2)
end

"""
    iscompatible(ret1::Gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S})

Judge whether the `gʳᵉᵗ` and `Gʳᵉᵗ` objects are compatible.
"""
iscompatible(ret1::Gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}) where {S} = iscompatible(ret2, ret1)

"""
    iscompatible(C::Cn, ret::gʳᵉᵗ{S})

Judge whether `C` (which is a `Cn` object) is compatible with `ret`
(which is a `gʳᵉᵗ{S}` object).
"""
function iscompatible(C::Cn, ret::gʳᵉᵗ{S}) where {S}
    getntime(C) ≥ gettstp(ret) &&
    getdims(C) == getdims(ret)
end

"""
    iscompatible(ret::gʳᵉᵗ{S}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `ret`
(which is a `gʳᵉᵗ{S}` object).
"""
iscompatible(ret::gʳᵉᵗ{S}, C::Cn) where {S} = iscompatible(C, ret)

"""
    iscompatible(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S})

Judge whether two `gˡᵐⁱˣ` objects are compatible.
"""
function iscompatible(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}) where {S}
    getsize(lmix1) == getsize(lmix2) &&
    getdims(lmix1) == getdims(lmix2)
end

"""
    iscompatible(lmix1::gˡᵐⁱˣ{S}, lmix2::Gˡᵐⁱˣ{S})

Judge whether the `gˡᵐⁱˣ` and `Gˡᵐⁱˣ` objects are compatible.
"""
function iscompatible(lmix1::gˡᵐⁱˣ{S}, lmix2::Gˡᵐⁱˣ{S}) where {S}
    getntau(lmix1) == getntau(lmix2) &&
    getdims(lmix1) == getdims(lmix2)
end

"""
    iscompatible(lmix1::Gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S})

Judge whether the `gˡᵐⁱˣ` and `Gˡᵐⁱˣ` objects are compatible.
"""
iscompatible(lmix1::Gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}) where {S} = iscompatible(lmix2, lmix1)

"""
    iscompatible(C::Cn, lmix::gˡᵐⁱˣ{S})

Judge whether `C` (which is a `Cn` object) is compatible with `lmix`
(which is a `gˡᵐⁱˣ{S}` object).
"""
function iscompatible(C::Cn, lmix::gˡᵐⁱˣ{S}) where {S}
    getntau(C) == getntau(lmix) &&
    getdims(C) == getdims(lmix)
end

"""
    iscompatible(lmix::gˡᵐⁱˣ{S}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `lmix`
(which is a `gˡᵐⁱˣ{S}` object).
"""
iscompatible(lmix::gˡᵐⁱˣ{S}, C::Cn) where {S} = iscompatible(C, lmix)

"""
    iscompatible(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S})

Judge whether two `gˡᵉˢˢ` objects are compatible.
"""
function iscompatible(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}) where {S}
    getsize(less1) == getsize(less2) &&
    getdims(less1) == getdims(less2)
end

"""
    iscompatible(less1::gˡᵉˢˢ{S}, less2::Gˡᵉˢˢ{S})

Judge whether the `gˡᵉˢˢ` and `Gˡᵉˢˢ` objects are compatible.
"""
function iscompatible(less1::gˡᵉˢˢ{S}, less2::Gˡᵉˢˢ{S}) where {S}
    getsize(less1) ≤ getsize(less2) &&
    getdims(less1) == getdims(less2)
end

"""
    iscompatible(less1::Gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S})

Judge whether the `gˡᵉˢˢ` and `Gˡᵉˢˢ` objects are compatible.
"""
iscompatible(less1::Gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}) where {S} = iscompatible(less2, less1)

"""
    iscompatible(C::Cn, less::gˡᵉˢˢ{S})

Judge whether `C` (which is a `Cn` object) is compatible with `less`
(which is a `gˡᵉˢˢ{S}` object).
"""
function iscompatible(C::Cn, less::gˡᵉˢˢ{S}) where {S}
    getntime(C) ≥ getsize(less) &&
    getdims(C) == getdims(less)
end

"""
    iscompatible(less::gˡᵉˢˢ{S}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `less`
(which is a `gˡᵉˢˢ{S}` object).
"""
iscompatible(less::gˡᵉˢˢ{S}, C::Cn) where {S} = iscompatible(C, less)

"""
    iscompatible(cfm1::ℱ{T}, cfm2::ℱ{T})

Judge whether two `gˡᵉˢˢ` objects are compatible.
"""
function iscompatible(cfm1::ℱ{T}, cfm2::ℱ{T}) where {T}
    getsign(cfm1) == getsign(cfm2)     &&
    iscompatible(cfm1.mat, cfm2.mat)   &&
    iscompatible(cfm1.ret, cfm2.ret)   &&
    iscompatible(cfm1.lmix, cfm2.lmix) &&
    iscompatible(cfm1.less, cfm2.less)
end

"""
    iscompatible(pgf1::ℱᵖˢ{T}, pgf2::ℱᵖˢ{T})

Judge whether two `ℱᵖˢ` objects are compatible.
"""
function iscompatible(pgf1::ℱᵖˢ{T}, pgf2::ℱᵖˢ{T}) where {T}
    getsign(pgf1) == getsign(pgf2)     &&
    iscompatible(pgf1.mat, pgf2.mat)   &&
    iscompatible(pgf1.ret, pgf2.ret)   &&
    iscompatible(pgf1.lmix, pgf2.lmix) &&
    iscompatible(pgf1.less, pgf2.less)
end

"""
    iscompatible(G₁::NessHermMat, G₂::NessHermMat)

Judge whether two NessHermMat structs have the same number of grid
points and the same orbital size.
"""
function iscompatible(G₁::NessHermMat, G₂::NessHermMat)
    return getnft(G₁) == getnft(G₂) && getdims(G₁) == getdims(G₂)
end

"""
    iscompatible(G₁::ℱᵐᵒᵛ, G₂::ℱᵐᵒᵛ)

Whether two moving Green's functions can be combined: both initialized,
the same `tc`, `tstp`, matrix size, and statistics.
"""
function iscompatible(G₁::ℱᵐᵒᵛ, G₂::ℱᵐᵒᵛ)
    return G₁.initialized && G₂.initialized &&
           G₁.tc == G₂.tc && G₁.tstp == G₂.tstp &&
           getdims(G₁) == getdims(G₂) && G₁.sign == G₂.sign
end

"""
    iscompatible(G::ℱᵐᵒᵛ, f::Cfᵐᵒᵛ)

Whether a moving Green's function and a moving single-time function can
be combined: both initialized, the same `tc`, `tstp`, and matrix size.
"""
function iscompatible(G::ℱᵐᵒᵛ, f::Cfᵐᵒᵛ)
    return G.initialized && f.initialized && G.tc == f.tc &&
           G.tstp == f.tstp && getdims(G) == getdims(f)
end

"""
    iscompatible(x::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ}, y::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ})

Whether two moving time steps (owning or views) can be combined: the
same number of blocks, matrix size, and statistics.
"""
function iscompatible(x::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ}, y::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ})
    return getsize(x) == getsize(y) && getdims(x) == getdims(y) &&
           getsign(x) == getsign(y)
end

"""
    _check_tstp(G::ℱ, tstp::I64)
    _check_tstp(G::𝒻, tstp::I64)

Check whether the time step `tstp` is valid for `G`. It is a private
function.
"""
_check_tstp(G::Union{ℱ,ℱᵖˢ}, tstp::I64) = @assert 0 ≤ tstp ≤ getntime(G)

_check_tstp(G::Union{𝒻,𝒻ᵖˢ}, tstp::I64) = @assert tstp == gettstp(G)

