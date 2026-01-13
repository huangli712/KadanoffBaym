#
# Project : Lavender
# Source  : element.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/01/13
#

struct CopyZone
    x₁::I64
    y₁::I64
    x₂::I64
    y₂::I64
end

function CopyZone(x::I64, y::I64)
    return CopyZone(x, y, x, y)
end

function CopyZone(x::I64, y::I64, δ::I64)
    @assert δ ≥ 1
    return CopyZone(x, y, x + δ - 1, y + δ - 1)
end

function isvalid(cz::CopyZone)
    return cz.x₂ ≥ cz.x₁ ≥ 1 && cz.y₂ ≥ cz.y₁ ≥ 1
end

function iscompatible(cz1::CopyZone, cz2::CopyZone)
    return (cz1.x₂ - cz1.x₁) == (cz2.x₂ - cz2.x₁) &&
           (cz1.y₂ - cz1.y₁) == (cz2.y₂ - cz2.y₁)
end

function iscompatible(cz::CopyZone, obj::CnAbstractMatrix{T}) where {T}
    return (cz.x₁, cz.y₁) < getdims(obj) &&
           (cz.x₂, cz.y₂) < getdims(obj)
end

function iscompatible(obj::CnAbstractMatrix{T}, cz::CopyZone) where {T}
    return iscompatible(cz, obj)
end

function iscompatible(cz::CopyZone, obj::CnAbstractVector{T}) where {T}
    return (cz.x₁, cz.y₁) < getdims(obj) &&
           (cz.x₂, cz.y₂) < getdims(obj)
end

function iscompatible(obj::CnAbstractVector{T}, cz::CopyZone) where {T}
    return iscompatible(cz, obj)
end

function elemcpy!(
    cz1::CopyZone,
    src::Gᵐᵃᵗ{T},
    cz2::CopyZone,
    dst::Gᵐᵃᵗ{T}
) where T
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

function elemcpy!(
    tstp::I64,
    cz1::CopyZone,
    src::Gʳᵉᵗ{T},
    cz2::CopyZone,
    dst::Gʳᵉᵗ{T}
) where T
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
        dst.data[tstp,1][cz2.x₁:cz2.x₂, cz2.y₁:cz2.y₂] .=
            src.data[tstp,1][cz1.x₁:cz1.x₂, cz1.y₁:cz1.y₂]
    end
end

function elemcpy!(
    tstp::I64,
    cz1::CopyZone,
    src::Gˡᵐⁱˣ{T},
    cz2::CopyZone,
    dst::Gˡᵐⁱˣ{T}
) where T

end

function elemcpy!(
    tstp::I64,
    cz1::CopyZone,
    src::Gˡᵉˢˢ{T},
    cz2::CopyZone,
    dst::Gˡᵉˢˢ{T}
) where T

end
