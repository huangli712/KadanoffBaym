#
# Project : Lavender
# Source  : dims.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    getdims(C::Cn)

Return the dimensional parameters of contour.

See also: [`Cn`](@ref).
"""
function getdims(C::Cn)
    return (C.ndim1, C.ndim2)
end

"""
    getdims(cf::Cf{T})

Return the dimensional parameters of contour function.

See also: [`Cf`](@ref).
"""
function getdims(cf::Cf{T}) where {T}
    return (cf.ndim1, cf.ndim2)
end

"""
    getdims(mat::Gᵐᵃᵗ{T})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function getdims(mat::Gᵐᵃᵗ{T}) where {T}
    return (mat.ndim1, mat.ndim2)
end

"""
    getdims(ret::Gʳᵉᵗ{T})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`Gʳᵉᵗ`](@ref).
"""
function getdims(ret::Gʳᵉᵗ{T}) where {T}
    return (ret.ndim1, ret.ndim2)
end

"""
    getdims(lmix::Gˡᵐⁱˣ{T})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function getdims(lmix::Gˡᵐⁱˣ{T}) where {T}
    return (lmix.ndim1, lmix.ndim2)
end

"""
    getdims(less::Gˡᵉˢˢ{T})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function getdims(less::Gˡᵉˢˢ{T}) where {T}
    return (less.ndim1, less.ndim2)
end

"""
    getdims(mat::gᵐᵃᵗ{S})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`gᵐᵃᵗ`](@ref).
"""
function getdims(mat::gᵐᵃᵗ{S}) where {S}
    return (mat.ndim1, mat.ndim2)
end

"""
    getdims(ret::gʳᵉᵗ{S})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`gʳᵉᵗ`](@ref).
"""
function getdims(ret::gʳᵉᵗ{S}) where {S}
    return (ret.ndim1, ret.ndim2)
end

"""
    getdims(lmix::gˡᵐⁱˣ{S})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function getdims(lmix::gˡᵐⁱˣ{S}) where {S}
    return (lmix.ndim1, lmix.ndim2)
end

"""
    getdims(less::gˡᵉˢˢ{S})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`gˡᵉˢˢ`](@ref).
"""
function getdims(less::gˡᵉˢˢ{S}) where {S}
    return (less.ndim1, less.ndim2)
end

"""
    getdims(cfm::ℱ{T})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`ℱ`](@ref).
"""
function getdims(cfm::ℱ{T}) where {T}
    return getdims(cfm.less)
end

"""
    getdims(cfv::𝒻{S})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`𝒻`](@ref).
"""
function getdims(cfv::𝒻{S}) where {S}
    return getdims(cfv.less)
end

"""
    getdims(pgf::PseudoGF{T})

Return the dimensional parameters of pseudo-particle Green's function.

See also: [`PseudoGF`](@ref).
"""
function getdims(pgf::PseudoGF{T}) where {T}
    return getdims(pgf.less)
end

"""
    getdims(pgv::PseudoGFStep{S})

Return the dimensional parameters of pseudo-particle Green's function.

See also: [`PseudoGFStep`](@ref).
"""
function getdims(pgv::PseudoGFStep{S}) where {S}
    return getdims(pgv.less)
end

"""
    getdims(A::FFTArray)

Return the size of the orbital matrices of an FFTArray struct.
"""
getdims(A::FFTArray) = (size(A.time, 1), size(A.time, 2))

"""
    getdims(G::HermMatrixNESS)

Return the size of the orbital matrices of a HermMatrixNESS struct.
"""
getdims(G::HermMatrixNESS) = getdims(G.ret)

"""
    getdims(G::Union{ℱᵐᵒᵛ,𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ,Cfᵐᵒᵛ})

Return the dimensional parameters of a moving object.
"""
getdims(G::ℱᵐᵒᵛ) = (G.ret.ndim1, G.ret.ndim2)

getdims(G::𝒻ᵐᵒᵛ) = (G.ret.ndim1, G.ret.ndim2)

getdims(v::𝒻ᵛⁱᵉʷᵐᵒᵛ) = getdims(v.parent)

getdims(f::Cfᵐᵒᵛ) = (f.ndim1, f.ndim2)

