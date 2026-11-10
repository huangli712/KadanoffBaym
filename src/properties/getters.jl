#
# Project : Lavender
# Source  : getters.jl
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
    getntau(C::Cn)

Return the `ntau` parameter of contour. `ntau` means number of time
slices in imaginary time axis.

See also: [`Cn`](@ref).
"""
function getntau(C::Cn)
    return C.ntau
end

"""
    getdims(C::Cn)

Return the dimensional parameters of contour.

See also: [`Cn`](@ref).
"""
function getdims(C::Cn)
    return (C.ndim1, C.ndim2)
end

"""
    gettmax(C::Cn)

Return the `tmax` parameter of contour. `tmax` means maximum evolution time.

See also: [`Cn`](@ref).
"""
function gettmax(C::Cn)
    return C.tmax
end

"""
    getbeta(C::Cn)

Return the `beta` parameter of contour. `beta` means inverse temperature.

See also: [`Cn`](@ref).
"""
function getbeta(C::Cn)
    return C.beta
end

"""
    getdt(C::Cn)

Return the `dt` parameter of contour. `dt` means time step (interval) in
real time axis.

See also: [`Cn`](@ref).
"""
function getdt(C::Cn)
    return C.dt
end

"""
    getdtau(C::Cn)

Return the `dtau` parameter of contour. `dtau` means time step (interval)
in imaginary time axis.

See also: [`Cn`](@ref).
"""
function getdtau(C::Cn)
    return C.dtau
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

"""
    getntime(cf::Cf{T})

Return the `ntime` parameter of contour function. `ntime` means number of
time slices in real time axis.

See also: [`Cf`](@ref).
"""
getntime(cf::Cf{T}) where {T} = getsize(cf)

"""
    getdims(cf::Cf{T})

Return the dimensional parameters of contour function.

See also: [`Cf`](@ref).
"""
function getdims(cf::Cf{T}) where {T}
    return (cf.ndim1, cf.ndim2)
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

"""
    getntau(mat::Gᵐᵃᵗ{T})

Return the `ntau` parameter of contour-ordered Green's function. `ntau`
means number of time slices in imaginary time axis.

See also: [`Gᵐᵃᵗ`](@ref).
"""
getntau(mat::Gᵐᵃᵗ{T}) where {T} = getsize(mat)

"""
    getdims(mat::Gᵐᵃᵗ{T})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function getdims(mat::Gᵐᵃᵗ{T}) where {T}
    return (mat.ndim1, mat.ndim2)
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

"""
    getntime(ret::Gʳᵉᵗ{T})

Return the `ntime` parameter of contour-ordered Green's function. `ntime`
means number of time slices in real time axis.

See also: [`Gʳᵉᵗ`](@ref).
"""
getntime(ret::Gʳᵉᵗ{T}) where {T} = getsize(ret)

"""
    getdims(ret::Gʳᵉᵗ{T})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`Gʳᵉᵗ`](@ref).
"""
function getdims(ret::Gʳᵉᵗ{T}) where {T}
    return (ret.ndim1, ret.ndim2)
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
    getntau(lmix::Gˡᵐⁱˣ{T})

Return the `ntau` parameter of contour-ordered Green's function. `ntau`
means number of time slices in imaginary time axis.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function getntau(lmix::Gˡᵐⁱˣ{T}) where {T}
    return lmix.ntau
end

"""
    getdims(lmix::Gˡᵐⁱˣ{T})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function getdims(lmix::Gˡᵐⁱˣ{T}) where {T}
    return (lmix.ndim1, lmix.ndim2)
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

"""
    getntime(less::Gˡᵉˢˢ{T})

Return the `ntime` parameter of contour-ordered Green's function. `ntime`
means number of time slices in real time axis.

See also: [`Gˡᵉˢˢ`](@ref).
"""
getntime(less::Gˡᵉˢˢ{T}) where {T} = getsize(less)

"""
    getdims(less::Gˡᵉˢˢ{T})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function getdims(less::Gˡᵉˢˢ{T}) where {T}
    return (less.ndim1, less.ndim2)
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

"""
    getntau(mat::gᵐᵃᵗ{S})

Return the `ntau` parameter of contour-ordered Green's function. `ntau`
means number of time slices in imaginary time axis.

See also: [`gᵐᵃᵗ`](@ref).
"""
getntau(mat::gᵐᵃᵗ{S}) where {S} = getsize(mat)

"""
    getdims(mat::gᵐᵃᵗ{S})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`gᵐᵃᵗ`](@ref).
"""
function getdims(mat::gᵐᵃᵗ{S}) where {S}
    return (mat.ndim1, mat.ndim2)
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

"""
    gettstp(ret::gʳᵉᵗ{S})

Return the `tstp` parameter of contour-ordered Green's function. `tstp`
means current time step in real time axis.

See also: [`gʳᵉᵗ`](@ref).
"""
gettstp(ret::gʳᵉᵗ{S}) where {S} = getsize(ret)

"""
    getdims(ret::gʳᵉᵗ{S})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`gʳᵉᵗ`](@ref).
"""
function getdims(ret::gʳᵉᵗ{S}) where {S}
    return (ret.ndim1, ret.ndim2)
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

"""
    getntau(lmix::gˡᵐⁱˣ{S})

Return the `ntau` parameter of contour-ordered Green's function. `ntau`
means number of time slices in imaginary time axis.

See also: [`gˡᵐⁱˣ`](@ref).
"""
getntau(lmix::gˡᵐⁱˣ{S}) where {S} = getsize(lmix)

"""
    getdims(lmix::gˡᵐⁱˣ{S})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function getdims(lmix::gˡᵐⁱˣ{S}) where {S}
    return (lmix.ndim1, lmix.ndim2)
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

"""
    gettstp(less::gˡᵉˢˢ{S})

Return the `tstp` parameter of contour-ordered Green's function. `tstp`
means current time step in real time axis.

See also: [`gˡᵉˢˢ`](@ref).
"""
gettstp(less::gˡᵉˢˢ{S}) where {S} = getsize(less)

"""
    getdims(less::gˡᵉˢˢ{S})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`gˡᵉˢˢ`](@ref).
"""
function getdims(less::gˡᵉˢˢ{S}) where {S}
    return (less.ndim1, less.ndim2)
end

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

"""
    getntime(cfm::ℱ{T})

Return the `ntime` parameter of contour-ordered Green's function.
"""
function getntime(cfm::ℱ{T}) where {T}
    return getsize(cfm.less)
end

"""
    getntau(cfm::ℱ{T})

Return the `ntau` parameter of contour-ordered Green's function.
"""
function getntau(cfm::ℱ{T}) where {T}
    return getsize(cfm.mat)
end

"""
    getdims(cfm::ℱ{T})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`ℱ`](@ref).
"""
function getdims(cfm::ℱ{T}) where {T}
    return getdims(cfm.less)
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

"""
    gettstp(cfv::𝒻{S})

Return the `tstp` parameter of contour-ordered Green's function.
"""
function gettstp(cfv::𝒻{S}) where {S}
    return cfv.tstp # getsize(cfv.less) is wrong when cfv.tstp = 0!
end

"""
    getntau(cfv::𝒻{S})

Return the `ntau` parameter of contour-ordered Green's function.
"""
function getntau(cfv::𝒻{S}) where {S}
    return getsize(cfv.mat)
end

"""
    getdims(cfv::𝒻{S})

Return the dimensional parameters of contour-ordered Green's function.

See also: [`𝒻`](@ref).
"""
function getdims(cfv::𝒻{S}) where {S}
    return getdims(cfv.less)
end

#=
### *PseudoGF* : *Properties*
=#

"""
    getsign(pgf::PseudoGF{T})

Return the `sign` parameter of pseudo-particle Green's function.
"""
function getsign(pgf::PseudoGF{T}) where {T}
    return pgf.sign
end

"""
    getntime(pgf::PseudoGF{T})

Return the `ntime` parameter of pseudo-particle Green's function.
"""
function getntime(pgf::PseudoGF{T}) where {T}
    return getsize(pgf.less)
end

"""
    getntau(pgf::PseudoGF{T})

Return the `ntau` parameter of pseudo-particle Green's function.
"""
function getntau(pgf::PseudoGF{T}) where {T}
    return getsize(pgf.mat)
end

"""
    getdims(pgf::PseudoGF{T})

Return the dimensional parameters of pseudo-particle Green's function.

See also: [`PseudoGF`](@ref).
"""
function getdims(pgf::PseudoGF{T}) where {T}
    return getdims(pgf.less)
end

#=
### *PseudoGFStep* : *Properties*
=#

"""
    getsign(pgv::PseudoGFStep{S})

Return the `sign` parameter of pseudo-particle Green's function.
"""
function getsign(pgv::PseudoGFStep{S}) where {S}
    return pgv.sign
end

"""
    gettstp(pgv::PseudoGFStep{S})

Return the `tstp` parameter of pseudo-particle Green's function.
"""
function gettstp(pgv::PseudoGFStep{S}) where {S}
    return pgv.tstp
end

"""
    getntau(pgv::PseudoGFStep{S})

Return the `ntau` parameter of pseudo-particle Green's function.
"""
function getntau(pgv::PseudoGFStep{S}) where {S}
    return getsize(pgv.mat)
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

"""
    getdims(G::Union{ℱᵐᵒᵛ,𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ,Cfᵐᵒᵛ})

Return the dimensional parameters of a moving object.
"""
getdims(G::ℱᵐᵒᵛ) = (G.ret.ndim1, G.ret.ndim2)

getdims(G::𝒻ᵐᵒᵛ) = (G.ret.ndim1, G.ret.ndim2)

getdims(v::𝒻ᵛⁱᵉʷᵐᵒᵛ) = getdims(v.parent)

getdims(f::Cfᵐᵒᵛ) = (f.ndim1, f.ndim2)

"""
    getsign(G::Union{ℱᵐᵒᵛ,𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ})

Return the statistics (`FERMI` or `BOSE`) of a moving Green's function.
"""
getsign(G::Union{ℱᵐᵒᵛ,𝒻ᵐᵒᵛ}) = G.sign

getsign(v::𝒻ᵛⁱᵉʷᵐᵒᵛ) = v.parent.sign

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

#=
*Remarks* : *Pseudo-Particle Convention*

The standard (`ℱ`) and pseudo-particle (`PseudoGF`) Green's functions
share the same storage layout. They differ in the meaning of the stored
real time triangle (``G^{R}`` versus ``G^{>}``) and in the Matsubara and
left-mixing algorithms. The convention is a static property of the type,
and it is used to select the algorithms by dispatch.

*References* :

Please see [`NESSi`] `cntr_herm_pseudo_decl.hpp`.
=#

"""
    gf_convention(::Type{<:ℱ})
    gf_convention(::Type{<:PseudoGF})

Return the convention of the contour-ordered Green's function, namely
`Val(:standard)` or `Val(:pseudo)`. It is used to select the Matsubara
and left-mixing algorithms only.

See also: [`ℱ`](@ref), [`PseudoGF`](@ref).
"""
gf_convention(::Type{<:ℱ}) = Val(:standard)

gf_convention(::Type{<:PseudoGF}) = Val(:pseudo)
#

gf_convention(G::Union{ℱ,PseudoGF}) = gf_convention(typeof(G))

