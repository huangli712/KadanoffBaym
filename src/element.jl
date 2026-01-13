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