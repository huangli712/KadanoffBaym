#
# Project : Lavender
# Source  : compat.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Equality and Compatibility*
=#

"""
    Base.:(==)(C₁::Cn, C₂::Cn)

Compare two `Cn` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(C₁::Cn, C₂::Cn)
    return (
        ( C₁.ntime, C₁.ntau, C₁.ndim1, C₁.ndim2, C₁.tmax, C₁.beta )
        ==
        ( C₂.ntime, C₂.ntau, C₂.ndim1, C₂.ndim2, C₂.tmax, C₂.beta )
    )
end

"""
    Base.:(==)(cf₁::Cf, cf₂::Cf)

Compare two `Cf` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(cf₁::Cf, cf₂::Cf)
    return (
        ( cf₁.ntime, cf₁.ndim1, cf₁.ndim2, cf₁.data )
        ==
        ( cf₂.ntime, cf₂.ndim1, cf₂.ndim2, cf₂.data )
    )
end

"""
    Base.:(==)(mat₁::Gᵐᵃᵗ{T}, mat₂::Gᵐᵃᵗ{T})

Compare two `Gᵐᵃᵗ` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(mat₁::Gᵐᵃᵗ{T}, mat₂::Gᵐᵃᵗ{T}) where {T}
    return (
        ( mat₁.type, mat₁.ntau, mat₁.ndim1, mat₁.ndim2, mat₁.data )
        ==
        ( mat₂.type, mat₂.ntau, mat₂.ndim1, mat₂.ndim2, mat₂.data )
    )
end

"""
    Base.:(==)(ret₁::Gʳᵉᵗ{T}, ret₂::Gʳᵉᵗ{T})

Compare two `Gʳᵉᵗ` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(ret₁::Gʳᵉᵗ{T}, ret₂::Gʳᵉᵗ{T}) where {T}
    return (
        ( ret₁.type, ret₁.ntime, ret₁.ndim1, ret₁.ndim2, ret₁.data )
        ==
        ( ret₂.type, ret₂.ntime, ret₂.ndim1, ret₂.ndim2, ret₂.data )
    )
end

"""
    Base.:(==)(lmix₁::Gˡᵐⁱˣ{T}, lmix₂::Gˡᵐⁱˣ{T})

Compare two `Gˡᵐⁱˣ` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(lmix₁::Gˡᵐⁱˣ{T}, lmix₂::Gˡᵐⁱˣ{T}) where {T}
    return (
        ( lmix₁.type, lmix₁.ntime, lmix₁.ntau, lmix₁.ndim1, lmix₁.ndim2, lmix₁.data )
        ==
        ( lmix₂.type, lmix₂.ntime, lmix₂.ntau, lmix₂.ndim1, lmix₂.ndim2, lmix₂.data )
    )
end

"""
    Base.:(==)(less₁::Gˡᵉˢˢ{T}, less₂::Gˡᵉˢˢ{T})

Compare two `Gˡᵉˢˢ` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(less₁::Gˡᵉˢˢ{T}, less₂::Gˡᵉˢˢ{T}) where {T}
    return (
        ( less₁.type, less₁.ntime, less₁.ndim1, less₁.ndim2, less₁.data )
        ==
        ( less₂.type, less₂.ntime, less₂.ndim1, less₂.ndim2, less₂.data )
    )
end

"""
    Base.:(==)(matm₁::Gᵐᵃᵗᵐ{T}, matm₂::Gᵐᵃᵗᵐ{T})

Compare two `Gᵐᵃᵗᵐ` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(matm₁::Gᵐᵃᵗᵐ{T}, matm₂::Gᵐᵃᵗᵐ{T}) where {T}
    return (
        ( matm₁.type, matm₁.ntau, matm₁.ndim1, matm₁.ndim2, matm₁.dataM[] )
        ==
        ( matm₂.type, matm₂.ntau, matm₂.ndim1, matm₂.ndim2, matm₂.dataM[] )
    )
end

"""
    Base.:(==)(adv₁::Gᵃᵈᵛ{T}, adv₂::Gᵃᵈᵛ{T})

Compare two `Gᵃᵈᵛ` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(adv₁::Gᵃᵈᵛ{T}, adv₂::Gᵃᵈᵛ{T}) where {T}
    return (
        ( adv₁.type, adv₁.ntime, adv₁.ndim1, adv₁.ndim2, adv₁.dataR[] )
        ==
        ( adv₂.type, adv₂.ntime, adv₂.ndim1, adv₂.ndim2, adv₂.dataR[] )
    )
end

"""
    Base.:(==)(rmix₁::Gʳᵐⁱˣ{T}, rmix₂::Gʳᵐⁱˣ{T})

Compare two `Gʳᵐⁱˣ` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(rmix₁::Gʳᵐⁱˣ{T}, rmix₂::Gʳᵐⁱˣ{T}) where {T}
    return (
        ( rmix₁.type, rmix₁.sign, rmix₁.ntime, rmix₁.ntau, rmix₁.ndim1, rmix₁.ndim2, rmix₁.dataL[] )
        ==
        ( rmix₂.type, rmix₂.sign, rmix₂.ntime, rmix₂.ntau, rmix₂.ndim1, rmix₂.ndim2, rmix₂.dataL[] )
    )
end

"""
    Base.:(==)(gtr₁::Gᵍᵗʳ{T}, gtr₂::Gᵍᵗʳ{T})

Compare two `Gᵍᵗʳ` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(gtr₁::Gᵍᵗʳ{T}, gtr₂::Gᵍᵗʳ{T}) where {T}
    return (
        ( gtr₁.type, gtr₁.ntime, gtr₁.ndim1, gtr₁.ndim2, gtr₁.dataL[], gtr₁.dataR[] )
        ==
        ( gtr₂.type, gtr₂.ntime, gtr₂.ndim1, gtr₂.ndim2, gtr₂.dataL[], gtr₂.dataR[] )
    )
end

"""
    Base.:(==)(mat₁::gᵐᵃᵗ{S}, mat₂::gᵐᵃᵗ{S})

Compare two `gᵐᵃᵗ` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(mat₁::gᵐᵃᵗ{S}, mat₂::gᵐᵃᵗ{S}) where {S}
    return (
        ( mat₁.type, mat₁.ntau, mat₁.ndim1, mat₁.ndim2, mat₁.data )
        ==
        ( mat₂.type, mat₂.ntau, mat₂.ndim1, mat₂.ndim2, mat₂.data )
    )
end

"""
    Base.:(==)(ret₁::gʳᵉᵗ{S}, ret₂::gʳᵉᵗ{S})

Compare two `gʳᵉᵗ` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(ret₁::gʳᵉᵗ{S}, ret₂::gʳᵉᵗ{S}) where {S}
    return (
        ( ret₁.type, ret₁.tstp, ret₁.ndim1, ret₁.ndim2, ret₁.data )
        ==
        ( ret₂.type, ret₂.tstp, ret₂.ndim1, ret₂.ndim2, ret₂.data )
    )
end

"""
    Base.:(==)(lmix₁::gˡᵐⁱˣ{S}, lmix₂::gˡᵐⁱˣ{S})

Compare two `gˡᵐⁱˣ` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(lmix₁::gˡᵐⁱˣ{S}, lmix₂::gˡᵐⁱˣ{S}) where {S}
    return (
        ( lmix₁.type, lmix₁.ntau, lmix₁.ndim1, lmix₁.ndim2, lmix₁.data )
        ==
        ( lmix₂.type, lmix₂.ntau, lmix₂.ndim1, lmix₂.ndim2, lmix₂.data )
    )
end

"""
    Base.:(==)(less₁::gˡᵉˢˢ{S}, less₂::gˡᵉˢˢ{S})

Compare two `gˡᵉˢˢ` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(less₁::gˡᵉˢˢ{S}, less₂::gˡᵉˢˢ{S}) where {S}
    return (
        ( less₁.type, less₁.tstp, less₁.ndim1, less₁.ndim2, less₁.data )
        ==
        ( less₂.type, less₂.tstp, less₂.ndim1, less₂.ndim2, less₂.data )
    )
end

"""
    Base.:(==)(matm₁::gᵐᵃᵗᵐ{S}, matm₂::gᵐᵃᵗᵐ{S})

Compare two `gᵐᵃᵗ` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(matm₁::gᵐᵃᵗᵐ{S}, matm₂::gᵐᵃᵗᵐ{S}) where {S}
    return (
        ( matm₁.type, matm₁.sign, matm₁.ntau, matm₁.ndim1, matm₁.ndim2, matm₁.dataM[] )
        ==
        ( matm₂.type, matm₂.sign, matm₂.ntau, matm₂.ndim1, matm₂.ndim2, matm₂.dataM[] )
    )
end

"""
    Base.:(==)(adv₁::gᵃᵈᵛ{S}, adv₂::gᵃᵈᵛ{S})

Compare two `gᵃᵈᵛ` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(adv₁::gᵃᵈᵛ{S}, adv₂::gᵃᵈᵛ{S}) where {S}
    return (
        ( adv₁.type, adv₁.tstp, adv₁.ndim1, adv₁.ndim2, adv₁.dataR[] )
        ==
        ( adv₂.type, adv₂.tstp, adv₂.ndim1, adv₂.ndim2, adv₂.dataR[] )
    )
end

"""
    Base.:(==)(rmix₁::gʳᵐⁱˣ{S}, rmix₂::gʳᵐⁱˣ{S})

Compare two `gʳᵐⁱˣ` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(rmix₁::gʳᵐⁱˣ{S}, rmix₂::gʳᵐⁱˣ{S}) where {S}
    return (
        ( rmix₁.type, rmix₁.sign, rmix₁.ntau, rmix₁.ndim1, rmix₁.ndim2, rmix₁.dataL[] )
        ==
        ( rmix₂.type, rmix₂.sign, rmix₂.ntau, rmix₂.ndim1, rmix₂.ndim2, rmix₂.dataL[] )
    )
end

"""
    Base.:(==)(gtr₁::gᵍᵗʳ{S}, gtr₂::gᵍᵗʳ{S})

Compare two `gᵍᵗʳ` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(gtr₁::gᵍᵗʳ{S}, gtr₂::gᵍᵗʳ{S}) where {S}
    return (
        ( gtr₁.type, gtr₁.tstp, gtr₁.ndim1, gtr₁.ndim2, gtr₁.dataL[], gtr₁.dataR[] )
        ==
        ( gtr₂.type, gtr₂.tstp, gtr₂.ndim1, gtr₂.ndim2, gtr₂.dataL[], gtr₂.dataR[] )
    )
end

"""
    Base.:(==)(cfm₁::ℱ{T}, cfm₂::ℱ{T})

Compare two `ℱ` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(cfm₁::ℱ{T}, cfm₂::ℱ{T}) where {T}
    return (
        ( cfm₁.sign, cfm₁.mat, cfm₁.ret, cfm₁.lmix, cfm₁.less )
        ==
        ( cfm₂.sign, cfm₂.mat, cfm₂.ret, cfm₂.lmix, cfm₂.less )
    )
end

"""
    Base.:(==)(cfv₁::𝒻{S}, cfv₂::𝒻{S})

Compare two `𝒻` structs, and judge whether the two structs are equal.
"""
function Base.:(==)(cfv₁::𝒻{S}, cfv₂::𝒻{S}) where {S}
    return (
        ( cfv₁.sign, cfv₁.tstp, cfv₁.mat, cfv₁.ret, cfv₁.lmix, cfv₁.less )
        ==
        ( cfv₂.sign, cfv₂.tstp, cfv₂.mat, cfv₂.ret, cfv₂.lmix, cfv₂.less )
    )
end

"""
    Base.:(==)(pgf₁::PseudoGF{T}, pgf₂::PseudoGF{T})

Compare two `PseudoGF` structs, and judge whether the two structs are
equal.
"""
function Base.:(==)(pgf₁::PseudoGF{T}, pgf₂::PseudoGF{T}) where {T}
    return (
        ( pgf₁.sign, pgf₁.mat, pgf₁.ret, pgf₁.lmix, pgf₁.less )
        ==
        ( pgf₂.sign, pgf₂.mat, pgf₂.ret, pgf₂.lmix, pgf₂.less )
    )
end

"""
    Base.:(==)(pgv₁::PseudoGFStep{S}, pgv₂::PseudoGFStep{S})

Compare two `PseudoGFStep` structs, and judge whether the two structs
are equal.
"""
function Base.:(==)(pgv₁::PseudoGFStep{S}, pgv₂::PseudoGFStep{S}) where {S}
    return (
        ( pgv₁.sign, pgv₁.tstp, pgv₁.mat, pgv₁.ret, pgv₁.lmix, pgv₁.less )
        ==
        ( pgv₂.sign, pgv₂.tstp, pgv₂.mat, pgv₂.ret, pgv₂.lmix, pgv₂.less )
    )
end

"""
    Base.:(==)(A::FFTArray{T}, B::FFTArray{T})

Compare the time and frequency data of two `FFTArray` structs.
"""
function Base.:(==)(A::FFTArray{T}, B::FFTArray{T}) where {T}
    return A.time == B.time && A.freq == B.freq
end

"""
    Base.:(==)(G₁::HermMatrixNESS{T}, G₂::HermMatrixNESS{T})

Compare two `HermMatrixNESS` structs.
"""
function Base.:(==)(G₁::HermMatrixNESS{T}, G₂::HermMatrixNESS{T}) where {T}
    return G₁.grid == G₂.grid && G₁.ret == G₂.ret && G₁.les == G₂.les
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
