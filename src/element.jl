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

function elemcpy!(
    cz1::CopyZone,
    src::Gᵐᵃᵗ{T},
    cz2::CopyZone,
    dst::Gᵐᵃᵗ{T}
) where T

end

function elemcpy!(
    tstp::I64,
    cz1::CopyZone,
    src::Gʳᵉᵗ{T},
    cz2::CopyZone,
    dst::Gʳᵉᵗ{T}
) where T

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