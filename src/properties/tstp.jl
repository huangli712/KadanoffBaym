#
# Project : Lavender
# Source  : tstp.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    gettstp(ret::gʳᵉᵗ{S})

Return the `tstp` parameter of contour-ordered Green's function. `tstp`
means current time step in real time axis.

See also: [`gʳᵉᵗ`](@ref).
"""
gettstp(ret::gʳᵉᵗ{S}) where {S} = getsize(ret)

"""
    gettstp(less::gˡᵉˢˢ{S})

Return the `tstp` parameter of contour-ordered Green's function. `tstp`
means current time step in real time axis.

See also: [`gˡᵉˢˢ`](@ref).
"""
gettstp(less::gˡᵉˢˢ{S}) where {S} = getsize(less)

"""
    gettstp(cfv::𝒻{S})

Return the `tstp` parameter of contour-ordered Green's function.
"""
function gettstp(cfv::𝒻{S}) where {S}
    return cfv.tstp # getsize(cfv.less) is wrong when cfv.tstp = 0!
end

"""
    gettstp(pgv::𝒻ᵖˢ{S})

Return the `tstp` parameter of pseudo-particle Green's function.
"""
function gettstp(pgv::𝒻ᵖˢ{S}) where {S}
    return pgv.tstp
end

"""
    gettstp(G::Union{ℱᵐᵒᵛ,𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ,Cfᵐᵒᵛ})

Return the KB time step of the leading row of a moving object (for a
view: the time step of its row). An uninitialized object throws an
`ArgumentError`; its internal placeholder is never a real time step.
"""
function gettstp(G::Union{ℱᵐᵒᵛ,Cfᵐᵒᵛ})
    _check_initialized(G)
    return G.tstp
end

gettstp(G::𝒻ᵐᵒᵛ) = G.tstp

function gettstp(v::𝒻ᵛⁱᵉʷᵐᵒᵛ)
    _check_view(v)
    return v.parent.tstp - v.delti
end

