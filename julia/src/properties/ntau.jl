#
# Project : Lavender
# Source  : ntau.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    getntau(C::Cn)

Return the `ntau` parameter of contour. `ntau` means number of time
slices in imaginary time axis.

See also: [`Cn`](@ref).
"""
function getntau(C::Cn)
    return C.ntau
end

"""
    getntau(mat::Gᵐᵃᵗ{T})

Return the `ntau` parameter of contour-ordered Green's function. `ntau`
means number of time slices in imaginary time axis.

See also: [`Gᵐᵃᵗ`](@ref).
"""
getntau(mat::Gᵐᵃᵗ{T}) where {T} = getsize(mat)

"""
    getntau(lmix::Gˡᵐⁱˣ{T})

Return the `ntau` parameter of contour-ordered Green's function. `ntau`
means number of time slices in imaginary time axis.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function getntau(lmix::Gˡᵐⁱˣ{T}) where {T}
    return lmix.ntau
end

"""
    getntau(mat::gᵐᵃᵗ{S})

Return the `ntau` parameter of contour-ordered Green's function. `ntau`
means number of time slices in imaginary time axis.

See also: [`gᵐᵃᵗ`](@ref).
"""
getntau(mat::gᵐᵃᵗ{S}) where {S} = getsize(mat)

"""
    getntau(lmix::gˡᵐⁱˣ{S})

Return the `ntau` parameter of contour-ordered Green's function. `ntau`
means number of time slices in imaginary time axis.

See also: [`gˡᵐⁱˣ`](@ref).
"""
getntau(lmix::gˡᵐⁱˣ{S}) where {S} = getsize(lmix)

"""
    getntau(cfm::ℱ{T})

Return the `ntau` parameter of contour-ordered Green's function.
"""
function getntau(cfm::ℱ{T}) where {T}
    return getsize(cfm.mat)
end

"""
    getntau(cfv::𝒻{S})

Return the `ntau` parameter of contour-ordered Green's function.
"""
function getntau(cfv::𝒻{S}) where {S}
    return getsize(cfv.mat)
end

"""
    getntau(pgf::ℱᵖˢ{T})

Return the `ntau` parameter of pseudo-particle Green's function.
"""
function getntau(pgf::ℱᵖˢ{T}) where {T}
    return getsize(pgf.mat)
end

"""
    getntau(pgv::𝒻ᵖˢ{S})

Return the `ntau` parameter of pseudo-particle Green's function.
"""
function getntau(pgv::𝒻ᵖˢ{S}) where {S}
    return getsize(pgv.mat)
end

