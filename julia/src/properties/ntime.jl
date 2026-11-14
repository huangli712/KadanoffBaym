#
# Project : Lavender
# Source  : ntime.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    getntime(C::Cn)

Return the `ntime` parameter of contour. `ntime` means number of time
slices in real time axis.

See also: [`Cn`](@ref).
"""
function getntime(C::Cn)
    return C.ntime
end

"""
    getntime(cf::Cf{T})

Return the `ntime` parameter of contour function. `ntime` means number of
time slices in real time axis.

See also: [`Cf`](@ref).
"""
getntime(cf::Cf{T}) where {T} = getsize(cf)

"""
    getntime(ret::Gʳᵉᵗ{T})

Return the `ntime` parameter of contour-ordered Green's function. `ntime`
means number of time slices in real time axis.

See also: [`Gʳᵉᵗ`](@ref).
"""
getntime(ret::Gʳᵉᵗ{T}) where {T} = getsize(ret)

"""
    getntime(lmix::Gˡᵐⁱˣ{T})

Return the `ntime` parameter of contour-ordered Green's function. `ntime`
means number of time slices in real time axis.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function getntime(lmix::Gˡᵐⁱˣ{T}) where {T}
    return lmix.ntime
end

"""
    getntime(less::Gˡᵉˢˢ{T})

Return the `ntime` parameter of contour-ordered Green's function. `ntime`
means number of time slices in real time axis.

See also: [`Gˡᵉˢˢ`](@ref).
"""
getntime(less::Gˡᵉˢˢ{T}) where {T} = getsize(less)

"""
    getntime(cfm::ℱ{T})

Return the `ntime` parameter of contour-ordered Green's function.
"""
function getntime(cfm::ℱ{T}) where {T}
    return getsize(cfm.less)
end

"""
    getntime(pgf::ℱᵖˢ{T})

Return the `ntime` parameter of pseudo-particle Green's function.
"""
function getntime(pgf::ℱᵖˢ{T}) where {T}
    return getsize(pgf.less)
end

