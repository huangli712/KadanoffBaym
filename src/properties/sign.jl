#
# Project : Lavender
# Source  : sign.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *ℱ* : *Properties*
=#

"""
    getsign(cfm::ℱ{T})

Return the `sign` parameter of contour-ordered Green's function.
"""
function getsign(cfm::ℱ{T}) where {T}
    return cfm.sign
end

#=
### *𝒻* : *Properties*
=#

"""
    getsign(cfv::𝒻{S})

Return the `sign` parameter of contour-ordered Green's function.
"""
function getsign(cfv::𝒻{S}) where {S}
    return cfv.sign
end

#=
### *ℱᵖˢ* : *Properties*
=#

"""
    getsign(pgf::ℱᵖˢ{T})

Return the `sign` parameter of pseudo-particle Green's function.
"""
function getsign(pgf::ℱᵖˢ{T}) where {T}
    return pgf.sign
end

#=
### *𝒻ᵖˢ* : *Properties*
=#

"""
    getsign(pgv::𝒻ᵖˢ{S})

Return the `sign` parameter of pseudo-particle Green's function.
"""
function getsign(pgv::𝒻ᵖˢ{S}) where {S}
    return pgv.sign
end

"""
    getsign(G::Union{ℱᵐᵒᵛ,𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ})

Return the statistics (`FERMI` or `BOSE`) of a moving Green's function.
"""
getsign(G::Union{ℱᵐᵒᵛ,𝒻ᵐᵒᵛ}) = G.sign

getsign(v::𝒻ᵛⁱᵉʷᵐᵒᵛ) = v.parent.sign

