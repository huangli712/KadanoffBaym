#
# Project : Lavender
# Source  : size.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Cn* : *Properties*
=#

"""
    getsize(C::Cn)

Return the size of contour. Here, it should return a tuple of `ntime` and
`ntau` parameters. `ntime` means number of time slices in real time axis,
and `ntau` means number of time slices in imaginary time axis.

See also: [`Cn`](@ref).
"""
function getsize(C::Cn)
    return (C.ntime, C.ntau)
end

#=
### *Cf* : *Properties*
=#

"""
    getsize(cf::Cf{T})

Return the nominal size of contour function, i.e `ntime`. Actually, the
true size of contour function should be `ntime + 1`. Here, `ntime` means
number of time slices in real time axis.

See also: [`Cf`](@ref).
"""
function getsize(cf::Cf{T}) where {T}
    return cf.ntime
end

#=
### *Gᵐᵃᵗ* : *Properties*
=#

"""
    getsize(mat::Gᵐᵃᵗ{T})

Return the size of contour-ordered Green's function. Here, it should be
`ntau`. `ntau` means number of time slices in imaginary time axis.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function getsize(mat::Gᵐᵃᵗ{T}) where {T}
    return mat.ntau
end

#=
### *Gʳᵉᵗ* : *Properties*
=#

"""
    getsize(ret::Gʳᵉᵗ{T})

Return the size of contour-ordered Green's function. Here, it should be
`ntime`. `ntime` means number of time slices in real time axis.

See also: [`Gʳᵉᵗ`](@ref).
"""
function getsize(ret::Gʳᵉᵗ{T}) where {T}
    return ret.ntime
end

#=
### *Gˡᵐⁱˣ* : *Properties*
=#

"""
    getsize(lmix::Gˡᵐⁱˣ{T})

Return the size of contour-ordered Green's function. Here, it should
return a tuple of `ntime` and `ntau` parameters. `ntime` means number
of time slices in real time axis, and `ntau` means number of time slices
in imaginary time axis.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function getsize(lmix::Gˡᵐⁱˣ{T}) where {T}
    return (lmix.ntime, lmix.ntau)
end

#=
### *Gˡᵉˢˢ* : *Properties*
=#

"""
    getsize(less::Gˡᵉˢˢ{T})

Return the size of contour-ordered Green's function. Here, it should be
`ntime`. `ntime` means number of time slices in real time axis.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function getsize(less::Gˡᵉˢˢ{T}) where {T}
    return less.ntime
end

#=
### *gᵐᵃᵗ* : *Properties*
=#

"""
    getsize(mat::gᵐᵃᵗ{S})

Return the size of contour-ordered Green's function. Here, it should be
`ntau`. `ntau` means number of time slices in imaginary time axis.

See also: [`gᵐᵃᵗ`](@ref).
"""
function getsize(mat::gᵐᵃᵗ{S}) where {S}
    return mat.ntau
end

#=
### *gʳᵉᵗ* : *Properties*
=#

"""
    getsize(ret::gʳᵉᵗ{S})

Return the size of contour-ordered Green's function. Here, it should be
`tstp`. `tstp` means current time step in real time axis.

See also: [`gʳᵉᵗ`](@ref).
"""
function getsize(ret::gʳᵉᵗ{S}) where {S}
    return ret.tstp
end

#=
### *gˡᵐⁱˣ* : *Properties*
=#

"""
    getsize(lmix::gˡᵐⁱˣ{S})

Return the size of contour-ordered Green's function. Here, it should be
`ntau`. `ntau` means number of time slices in imaginary time axis.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function getsize(lmix::gˡᵐⁱˣ{S}) where {S}
    return lmix.ntau
end

#=
### *gˡᵉˢˢ* : *Properties*
=#

"""
    getsize(less::gˡᵉˢˢ{S})

Return the size of contour-ordered Green's function. Here, it should be
`tstp`. `tstp` means current time step in real time axis.

See also: [`gˡᵉˢˢ`](@ref).
"""
function getsize(less::gˡᵉˢˢ{S}) where {S}
    return less.tstp
end

#=
### *Moving Window* : *Properties*
=#

"""
    getsize(G::Union{ℱᵐᵒᵛ,𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ,Cfᵐᵒᵛ})

Return the cut-off `tc` of a moving object (the window contains `tc + 1`
time steps).
"""
getsize(G::Union{ℱᵐᵒᵛ,𝒻ᵐᵒᵛ,Cfᵐᵒᵛ}) = G.tc

getsize(v::𝒻ᵛⁱᵉʷᵐᵒᵛ) = v.parent.tc

