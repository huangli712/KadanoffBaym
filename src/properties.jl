#
# Project : Lavender
# Source  : properties.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/10/22
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

"""
    equaldims(C::Cn)

Return whether the dimensional parameters are equal.

See also: [`Cn`](@ref).
"""
function equaldims(C::Cn)
    return C.ndim1 == C.ndim2
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

"""
    equaldims(cf::Cf{T})

Return whether the dimensional parameters are equal.

See also: [`Cf`](@ref).
"""
function equaldims(cf::Cf{T}) where {T}
    return cf.ndim1 == cf.ndim2
end

"""
    iscompatible(cf1::Cf{T}, cf2::Cf{T})

Judge whether two `Cf` objects are compatible.
"""
function iscompatible(cf1::Cf{T}, cf2::Cf{T}) where {T}
    getsize(cf1) == getsize(cf2) &&
    getdims(cf1) == getdims(cf2)
end

"""
    iscompatible(C::Cn, cf::Cf{T})

Judge whether `C` (which is a `Cn` object) is compatible with `cf`
(which is a `Cf{T}` object).
"""
function iscompatible(C::Cn, cf::Cf{T}) where {T}
    getntime(C) == getntime(cf) &&
    getdims(C) == getdims(cf)
end

"""
    iscompatible(cf::Cf{T}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `cf`
(which is a `Cf{T}` object).
"""
iscompatible(cf::Cf{T}, C::Cn) where {T} = iscompatible(C, cf)

"""
    distance(cf1::Cf{T}, cf2::Cf{T})

Calculate distance between two `Cf` objects.
"""
function distance(cf1::Cf{T}, cf2::Cf{T}) where {T}
    @assert iscompatible(cf1, cf2)

    err = 0.0
    #
    for m = 1:getsize(cf1)+1
        err = err + abs(sum(cf1.data[m] - cf2.data[m]))
    end
    #
    return err
end

#=
### *Gᵐᵃᵗ* : *Properties*
=#

"""
    getsize(mat::Gᵐᵃᵗ{T})

Return the size of contour Green's function. Here, it should be `ntau`.
`ntau` means number of time slices in imaginary time axis.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function getsize(mat::Gᵐᵃᵗ{T}) where {T}
    return mat.ntau
end

"""
    getntau(mat::Gᵐᵃᵗ{T})

Return the `ntau` parameter of contour Green's function. `ntau` means
number of time slices in imaginary time axis.

See also: [`Gᵐᵃᵗ`](@ref).
"""
getntau(mat::Gᵐᵃᵗ{T}) where {T} = getsize(mat)

"""
    getdims(mat::Gᵐᵃᵗ{T})

Return the dimensional parameters of contour Green's function.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function getdims(mat::Gᵐᵃᵗ{T}) where {T}
    return (mat.ndim1, mat.ndim2)
end

"""
    equaldims(mat::Gᵐᵃᵗ{T})

Return whether the dimensional parameters are equal.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function equaldims(mat::Gᵐᵃᵗ{T}) where {T}
    return mat.ndim1 == mat.ndim2
end

"""
    iscompatible(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T})

Judge whether two `Gᵐᵃᵗ` objects are compatible.
"""
function iscompatible(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T}) where {T}
    getsize(mat1) == getsize(mat2) &&
    getdims(mat1) == getdims(mat2)
end

"""
    iscompatible(C::Cn, mat::Gᵐᵃᵗ{T})

Judge whether `C` (which is a `Cn` object) is compatible with `mat`
(which is a `Gᵐᵃᵗ{T}` object).
"""
function iscompatible(C::Cn, mat::Gᵐᵃᵗ{T}) where {T}
    getntau(C) == getntau(mat) &&
    getdims(C) == getdims(mat)
end

"""
    iscompatible(mat::Gᵐᵃᵗ{T}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `mat`
(which is a `Gᵐᵃᵗ{T}` object).
"""
iscompatible(mat::Gᵐᵃᵗ{T}, C::Cn) where {T} = iscompatible(C, mat)

"""
    distance(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T})

Calculate distance between two `Gᵐᵃᵗ` objects.
"""
function distance(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T}) where {T}
    @assert iscompatible(mat1, mat2)

    err = 0.0
    #
    for m = 1:getsize(mat1)
        err = err + abs(sum(mat1.data[m,1] - mat2.data[m,1]))
    end
    #
    return err
end

#=
### *Gʳᵉᵗ* : *Properties*
=#

"""
    getsize(ret::Gʳᵉᵗ{T})

Return the size of contour Green's function. Here, it should be `ntime`.
`ntime` means number of time slices in real time axis.

See also: [`Gʳᵉᵗ`](@ref).
"""
function getsize(ret::Gʳᵉᵗ{T}) where {T}
    return ret.ntime
end

"""
    getntime(ret::Gʳᵉᵗ{T})

Return the `ntime` parameter of contour Green's function. `ntime` means
number of time slices in real time axis.

See also: [`Gʳᵉᵗ`](@ref).
"""
getntime(ret::Gʳᵉᵗ{T}) where {T} = getsize(ret)

"""
    getdims(ret::Gʳᵉᵗ{T})

Return the dimensional parameters of contour Green's function.

See also: [`Gʳᵉᵗ`](@ref).
"""
function getdims(ret::Gʳᵉᵗ{T}) where {T}
    return (ret.ndim1, ret.ndim2)
end

"""
    equaldims(ret::Gʳᵉᵗ{T})

Return whether the dimensional parameters are equal.

See also: [`Gʳᵉᵗ`](@ref).
"""
function equaldims(ret::Gʳᵉᵗ{T}) where {T}
    return ret.ndim1 == ret.ndim2
end

"""
    iscompatible(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T})

Judge whether two `Gʳᵉᵗ` objects are compatible.
"""
function iscompatible(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}) where {T}
    getsize(ret1) == getsize(ret2) &&
    getdims(ret1) == getdims(ret2)
end

"""
    iscompatible(C::Cn, ret::Gʳᵉᵗ{T})

Judge whether `C` (which is a `Cn` object) is compatible with `ret`
(which is a `Gʳᵉᵗ{T}` object).
"""
function iscompatible(C::Cn, ret::Gʳᵉᵗ{T}) where {T}
    getntime(C) == getntime(ret) &&
    getdims(C) == getdims(ret)
end

"""
    iscompatible(ret::Gʳᵉᵗ{T}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `ret`
(which is a `Gʳᵉᵗ{T}` object).
"""
iscompatible(ret::Gʳᵉᵗ{T}, C::Cn) where {T} = iscompatible(C, ret)

"""
    distance(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}, tstp::I64)

Calculate distance between two `Gʳᵉᵗ` objects at given time step `tstp`.
"""
function distance(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}, tstp::I64) where {T}
    # Sanity check
    @assert 1 ≤ tstp ≤ getsize(ret1)
    @assert 1 ≤ tstp ≤ getsize(ret2)

    err = 0
    #
    for i = 1:tstp
        err = err + abs(sum(ret1.data[tstp,i] - ret2.data[tstp,i]))
    end
    #
    return err
end

#=
### *Gˡᵐⁱˣ* : *Properties*
=#

"""
    getsize(lmix::Gˡᵐⁱˣ{T})

Return the size of contour Green's function. Here, it should return a
tuple of `ntime` and `ntau` parameters. `ntime` means number of time
slices in real time axis, and `ntau` means number of time slices in
imaginary time axis.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function getsize(lmix::Gˡᵐⁱˣ{T}) where {T}
    return (lmix.ntime, lmix.ntau)
end

"""
    getdims(lmix::Gˡᵐⁱˣ{T})

Return the dimensional parameters of contour Green's function.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function getdims(lmix::Gˡᵐⁱˣ{T}) where {T}
    return (lmix.ndim1, lmix.ndim2)
end

"""
    equaldims(lmix::Gˡᵐⁱˣ{T})

Return whether the dimensional parameters are equal.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function equaldims(lmix::Gˡᵐⁱˣ{T}) where {T}
    return lmix.ndim1 == lmix.ndim2
end

"""
    iscompatible(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T})

Judge whether two `Gˡᵐⁱˣ` objects are compatible.
"""
function iscompatible(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}) where {T}
    getsize(lmix1) == getsize(lmix2) &&
    getdims(lmix1) == getdims(lmix2)
end

"""
    iscompatible(C::Cn, lmix::Gˡᵐⁱˣ{T})

Judge whether `C` (which is a `Cn` object) is compatible with `lmix`
(which is a `Gˡᵐⁱˣ{T}` object).
"""
function iscompatible(C::Cn, lmix::Gˡᵐⁱˣ{T}) where {T}
    (getntime(C), getntau(C)) == getsize(lmix) &&
    getdims(C) == getdims(lmix)
end

"""
    iscompatible(lmix::Gˡᵐⁱˣ{T}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `lmix`
(which is a `Gˡᵐⁱˣ{T}` object).
"""
iscompatible(lmix::Gˡᵐⁱˣ{T}, C::Cn) where {T} = iscompatible(C, lmix)

"""
    distance(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}, tstp::I64)

Calculate distance between two `Gˡᵐⁱˣ` objects at given time step `tstp`.
"""
function distance(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}, tstp::I64) where {T}
    # Sanity check
    @assert 1 ≤ tstp ≤ lmix1.ntime
    @assert 1 ≤ tstp ≤ lmix2.ntime

    err = 0
    #
    for i = 1:lmix1.ntau
        err = err + abs(sum(lmix1.data[tstp,i] - lmix2.data[tstp,i]))
    end
    #
    return err
end

#=
### *Gˡᵉˢˢ* : *Properties*
=#

"""
    getdims(less::Gˡᵉˢˢ{T})

Return the dimensional parameters of contour function.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function getdims(less::Gˡᵉˢˢ{T}) where {T}
    return (less.ndim1, less.ndim2)
end

"""
    getsize(less::Gˡᵉˢˢ{T})

Return the size of contour function.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function getsize(less::Gˡᵉˢˢ{T}) where {T}
    return less.ntime
end

"""
    equaldims(less::Gˡᵉˢˢ{T})

Return whether the dimensional parameters are equal.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function equaldims(less::Gˡᵉˢˢ{T}) where {T}
    return less.ndim1 == less.ndim2
end

"""
    iscompatible(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T})

Judge whether two `Gˡᵉˢˢ` objects are compatible.
"""
function iscompatible(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}) where {T}
    getsize(less1) == getsize(less2) &&
    getdims(less1) == getdims(less2)
end

"""
    iscompatible(C::Cn, less::Gˡᵉˢˢ{T})

Judge whether `C` (which is a `Cn` object) is compatible with `less`
(which is a `Gˡᵉˢˢ{T}` object).
"""
function iscompatible(C::Cn, less::Gˡᵉˢˢ{T}) where {T}
    C.ntime == getsize(less) &&
    getdims(C) == getdims(less)
end

"""
    iscompatible(less::Gˡᵉˢˢ{T}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `less`
(which is a `Gˡᵉˢˢ{T}` object).
"""
iscompatible(less::Gˡᵉˢˢ{T}, C::Cn) where {T} = iscompatible(C, less)

"""
    distance(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}, tstp::I64)

Calculate distance between two `Gˡᵉˢˢ` objects at given time step `tstp`.
"""
function distance(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}, tstp::I64) where {T}
    # Sanity check
    @assert 1 ≤ tstp ≤ less1.ntime

    err = 0
    #
    for i = 1:tstp
        err = err + abs(sum(less1.data[i,tstp] - less2.data[i,tstp]))
    end
    #
    return err
end

#=
### *gᵐᵃᵗ* : *Properties*
=#

"""
    getdims(mat::gᵐᵃᵗ{S})

Return the dimensional parameters of contour function.

See also: [`gᵐᵃᵗ`](@ref).
"""
function getdims(mat::gᵐᵃᵗ{S}) where {S}
    return (mat.ndim1, mat.ndim2)
end

"""
    getsize(mat::gᵐᵃᵗ{S})

Return the size of contour function.

See also: [`gᵐᵃᵗ`](@ref).
"""
function getsize(mat::gᵐᵃᵗ{S}) where {S}
    return mat.ntau
end

"""
    equaldims(mat::gᵐᵃᵗ{S})

Return whether the dimensional parameters are equal.

See also: [`gᵐᵃᵗ`](@ref).
"""
function equaldims(mat::gᵐᵃᵗ{S}) where {S}
    return mat.ndim1 == mat.ndim2
end

"""
    iscompatible(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S})

Judge whether two `gᵐᵃᵗ` objects are compatible.
"""
function iscompatible(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}) where {S}
    getsize(mat1) == getsize(mat2) &&
    getdims(mat1) == getdims(mat2)
end

"""
    iscompatible(mat1::gᵐᵃᵗ{S}, mat2::Gᵐᵃᵗ{S})

Judge whether the `gᵐᵃᵗ` and `Gᵐᵃᵗ` objects are compatible.
"""
function iscompatible(mat1::gᵐᵃᵗ{S}, mat2::Gᵐᵃᵗ{S}) where {S}
    getsize(mat1) == getsize(mat2) &&
    getdims(mat1) == getdims(mat2)
end

"""
    iscompatible(mat1::Gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S})

Judge whether the `gᵐᵃᵗ` and `Gᵐᵃᵗ` objects are compatible.
"""
iscompatible(mat1::Gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}) where {S} = iscompatible(mat2, mat1)

"""
    iscompatible(C::Cn, mat::gᵐᵃᵗ{S})

Judge whether `C` (which is a `Cn` object) is compatible with `mat`
(which is a `gᵐᵃᵗ{S}` object).
"""
function iscompatible(C::Cn, mat::gᵐᵃᵗ{S}) where {S}
    C.ntau == getsize(mat) &&
    getdims(C) == getdims(mat)
end

"""
    iscompatible(mat::gᵐᵃᵗ{S}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `mat`
(which is a `gᵐᵃᵗ{S}` object).
"""
iscompatible(mat::gᵐᵃᵗ{S}, C::Cn) where {S} = iscompatible(C, mat)

"""
    distance(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S})

Calculate distance between two `gᵐᵃᵗ` objects.
"""
function distance(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}) where {S}
    @assert iscompatible(mat1, mat2)

    err = 0.0
    #
    for m = 1:mat1.ntau
        err = err + abs(sum(mat1.data[m] - mat2.data[m]))
    end
    #
    return err
end

"""
    distance(mat1::gᵐᵃᵗ{S}, mat2::Gᵐᵃᵗ{S})

Calculate distance between a `gᵐᵃᵗ` object and a `Gᵐᵃᵗ` object.
"""
function distance(mat1::gᵐᵃᵗ{S}, mat2::Gᵐᵃᵗ{S}) where {S}
    @assert iscompatible(mat1, mat2)

    err = 0.0
    #
    for m = 1:mat1.ntau
        err = err + abs(sum(mat1.data[m] - mat2.data[m,1]))
    end
    #
    return err
end

"""
    distance(mat1::Gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S})

Calculate distance between a `gᵐᵃᵗ` object and a `Gᵐᵃᵗ` object.
"""
distance(mat1::Gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}) where {S} = distance(mat2, mat1)

#=
### *gʳᵉᵗ* : *Properties*
=#

"""
    getdims(ret::gʳᵉᵗ{S})

Return the dimensional parameters of contour function.

See also: [`gʳᵉᵗ`](@ref).
"""
function getdims(ret::gʳᵉᵗ{S}) where {S}
    return (ret.ndim1, ret.ndim2)
end

"""
    getsize(ret::gʳᵉᵗ{S})

Return the size of contour function.

See also: [`gʳᵉᵗ`](@ref).
"""
function getsize(ret::gʳᵉᵗ{S}) where {S}
    return ret.tstp
end

"""
    equaldims(ret::gʳᵉᵗ{S})

Return whether the dimensional parameters are equal.

See also: [`gʳᵉᵗ`](@ref).
"""
function equaldims(ret::gʳᵉᵗ{S}) where {S}
    return ret.ndim1 == ret.ndim2
end

"""
    iscompatible(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S})

Judge whether two `gʳᵉᵗ` objects are compatible.
"""
function iscompatible(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}) where {S}
    getsize(ret1) == getsize(ret2) &&
    getdims(ret1) == getdims(ret2)
end

"""
    iscompatible(ret1::gʳᵉᵗ{S}, ret2::Gʳᵉᵗ{S})

Judge whether the `gʳᵉᵗ` and `Gʳᵉᵗ` objects are compatible.
"""
function iscompatible(ret1::gʳᵉᵗ{S}, ret2::Gʳᵉᵗ{S}) where {S}
    getsize(ret1) ≤ getsize(ret2) &&
    getdims(ret1) == getdims(ret2)
end

"""
    iscompatible(ret1::Gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S})

Judge whether the `gʳᵉᵗ` and `Gʳᵉᵗ` objects are compatible.
"""
iscompatible(ret1::Gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}) where {S} = iscompatible(ret2, ret1)

"""
    iscompatible(C::Cn, ret::gʳᵉᵗ{S})

Judge whether `C` (which is a `Cn` object) is compatible with `ret`
(which is a `gʳᵉᵗ{S}` object).
"""
function iscompatible(C::Cn, ret::gʳᵉᵗ{S}) where {S}
    C.ntime ≥ getsize(ret) &&
    getdims(C) == getdims(ret)
end

"""
    iscompatible(ret::gʳᵉᵗ{S}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `ret`
(which is a `gʳᵉᵗ{S}` object).
"""
iscompatible(ret::gʳᵉᵗ{S}, C::Cn) where {S} = iscompatible(C, ret)

"""
    distance(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S})

Calculate distance between two `gʳᵉᵗ` objects.
"""
function distance(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}) where {S}
    @assert iscompatible(ret1, ret2)

    err = 0.0
    #
    for m = 1:ret1.tstp
        err = err + abs(sum(ret1.data[m] - ret2.data[m]))
    end
    #
    return err
end

"""
    distance(ret1::gʳᵉᵗ{S}, ret2::Gʳᵉᵗ{S}, tstp::I64)

Calculate distance between a `gʳᵉᵗ` object and a `Gʳᵉᵗ` object at
given time step `tstp`.
"""
function distance(ret1::gʳᵉᵗ{S}, ret2::Gʳᵉᵗ{S}, tstp::I64) where {S}
    @assert iscompatible(ret1, ret2)
    @assert ret1.tstp == tstp

    err = 0.0
    #
    for m = 1:ret1.tstp
        err = err + abs(sum(ret1.data[m] - ret2.data[tstp,m]))
    end
    #
    return err
end

"""
    distance(ret1::Gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}, tstp::I64)

Calculate distance between a `gʳᵉᵗ` object and a `Gʳᵉᵗ` object at
given time step `tstp`.
"""
distance(ret1::Gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}, tstp::I64) where {S} = distance(ret2, ret1, tstp)

#=
### *gˡᵐⁱˣ* : *Properties*
=#

"""
    getdims(lmix::gˡᵐⁱˣ{S})

Return the dimensional parameters of contour function.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function getdims(lmix::gˡᵐⁱˣ{S}) where {S}
    return (lmix.ndim1, lmix.ndim2)
end

"""
    getsize(lmix::gˡᵐⁱˣ{S})

Return the size of contour function.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function getsize(lmix::gˡᵐⁱˣ{S}) where {S}
    return lmix.ntau
end

"""
    equaldims(lmix::gˡᵐⁱˣ{S})

Return whether the dimensional parameters are equal.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function equaldims(lmix::gˡᵐⁱˣ{S}) where {S}
    return lmix.ndim1 == lmix.ndim2
end

"""
    iscompatible(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S})

Judge whether two `gˡᵐⁱˣ` objects are compatible.
"""
function iscompatible(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}) where {S}
    getsize(lmix1) == getsize(lmix2) &&
    getdims(lmix1) == getdims(lmix2)
end

"""
    iscompatible(lmix1::gˡᵐⁱˣ{S}, lmix2::Gˡᵐⁱˣ{S})

Judge whether the `gˡᵐⁱˣ` and `Gˡᵐⁱˣ` objects are compatible.
"""
function iscompatible(lmix1::gˡᵐⁱˣ{S}, lmix2::Gˡᵐⁱˣ{S}) where {S}
    getsize(lmix1) == lmix2.ntau &&
    getdims(lmix1) == getdims(lmix2)
end

"""
    iscompatible(lmix1::Gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S})

Judge whether the `gˡᵐⁱˣ` and `Gˡᵐⁱˣ` objects are compatible.
"""
iscompatible(lmix1::Gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}) where {S} = iscompatible(lmix2, lmix1)

"""
    iscompatible(C::Cn, lmix::gˡᵐⁱˣ{S})

Judge whether `C` (which is a `Cn` object) is compatible with `lmix`
(which is a `gˡᵐⁱˣ{S}` object).
"""
function iscompatible(C::Cn, lmix::gˡᵐⁱˣ{S}) where {S}
    C.ntau == getsize(lmix) &&
    getdims(C) == getdims(lmix)
end

"""
    iscompatible(lmix::gˡᵐⁱˣ{S}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `lmix`
(which is a `gˡᵐⁱˣ{S}` object).
"""
iscompatible(lmix::gˡᵐⁱˣ{S}, C::Cn) where {S} = iscompatible(C, lmix)

"""
    distance(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S})

Calculate distance between two `gˡᵐⁱˣ` objects.
"""
function distance(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}) where {S}
    @assert iscompatible(lmix1, lmix2)

    err = 0.0
    #
    for m = 1:lmix1.ntau
        err = err + abs(sum(lmix1.data[m] - lmix2.data[m]))
    end
    #
    return err
end

"""
    distance(lmix1::gˡᵐⁱˣ{S}, lmix2::Gˡᵐⁱˣ{S}, tstp::I64)

Calculate distance between a `gˡᵐⁱˣ` object and a `Gˡᵐⁱˣ` object at
given time step `tstp`.
"""
function distance(lmix1::gˡᵐⁱˣ{S}, lmix2::Gˡᵐⁱˣ{S}, tstp::I64) where {S}
    @assert iscompatible(lmix1, lmix2)

    err = 0.0
    #
    for m = 1:lmix1.ntau
        err = err + abs(sum(lmix1.data[m] - lmix2.data[tstp,m]))
    end
    #
    return err
end

"""
    distance(lmix1::Gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}, tstp::I64)

Calculate distance between a `gˡᵐⁱˣ` object and a `Gˡᵐⁱˣ` object at
given time step `tstp`.
"""
distance(lmix1::Gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}, tstp::I64) where {S} = distance(lmix2, lmix1, tstp)

#=
### *gˡᵉˢˢ* : *Properties*
=#

"""
    getdims(less::gˡᵉˢˢ{S})

Return the dimensional parameters of contour function.

See also: [`gˡᵉˢˢ`](@ref).
"""
function getdims(less::gˡᵉˢˢ{S}) where {S}
    return (less.ndim1, less.ndim2)
end

"""
    getsize(less::gˡᵉˢˢ{S})

Return the size of contour function.

See also: [`gˡᵉˢˢ`](@ref).
"""
function getsize(less::gˡᵉˢˢ{S}) where {S}
    return less.tstp
end

"""
    equaldims(less::gˡᵉˢˢ{S})

Return whether the dimensional parameters are equal.

See also: [`gˡᵉˢˢ`](@ref).
"""
function equaldims(less::gˡᵉˢˢ{S}) where {S}
    return less.ndim1 == less.ndim2
end

"""
    iscompatible(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S})

Judge whether two `gˡᵉˢˢ` objects are compatible.
"""
function iscompatible(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}) where {S}
    getsize(less1) == getsize(less2) &&
    getdims(less1) == getdims(less2)
end

"""
    iscompatible(less1::gˡᵉˢˢ{S}, less2::Gˡᵉˢˢ{S})

Judge whether the `gˡᵉˢˢ` and `Gˡᵉˢˢ` objects are compatible.
"""
function iscompatible(less1::gˡᵉˢˢ{S}, less2::Gˡᵉˢˢ{S}) where {S}
    getsize(less1) ≤ getsize(less2) &&
    getdims(less1) == getdims(less2)
end

"""
    iscompatible(less1::Gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S})

Judge whether the `gˡᵉˢˢ` and `Gˡᵉˢˢ` objects are compatible.
"""
iscompatible(less1::Gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}) where {S} = iscompatible(less2, less1)

"""
    iscompatible(C::Cn, less::gˡᵉˢˢ{S})

Judge whether `C` (which is a `Cn` object) is compatible with `less`
(which is a `gˡᵉˢˢ{S}` object).
"""
function iscompatible(C::Cn, less::gˡᵉˢˢ{S}) where {S}
    C.ntime ≥ getsize(less) &&
    getdims(C) == getdims(less)
end

"""
    iscompatible(less::gˡᵉˢˢ{S}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `less`
(which is a `gˡᵉˢˢ{S}` object).
"""
iscompatible(less::gˡᵉˢˢ{S}, C::Cn) where {S} = iscompatible(C, less)

"""
    distance(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S})

Calculate distance between two `gˡᵉˢˢ` objects.
"""
function distance(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}) where {S}
    @assert iscompatible(less1, less2)

    err = 0.0
    #
    for m = 1:less1.tstp
        err = err + abs(sum(less1.data[m] - less2.data[m]))
    end
    #
    return err
end

"""
    distance(less1::gˡᵉˢˢ{S}, less2::Gˡᵉˢˢ{S}, tstp::I64)

Calculate distance between a `gˡᵉˢˢ` object and a `Gˡᵉˢˢ` object at
given time step `tstp`.
"""
function distance(less1::gˡᵉˢˢ{S}, less2::Gˡᵉˢˢ{S}, tstp::I64) where {S}
    @assert iscompatible(less1, less2)
    @assert tstp == less1.tstp

    err = 0.0
    #
    for m = 1:less1.tstp
        err = err + abs(sum(less1.data[m] - less2.data[m,tstp]))
    end
    #
    return err
end

"""
    distance(less1::Gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}, tstp::I64)

Calculate distance between a `gˡᵉˢˢ` object and a `Gˡᵉˢˢ` object at
given time step `tstp`.
"""
distance(less1::Gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}, tstp::I64) where {S} = distance(less2, less1, tstp)

#=
### *ℱ* : *Properties*
=#

"""
    getdims(cfm::ℱ{T})

Return the dimensional parameters of contour Green's function.

See also: [`ℱ`](@ref).
"""
function getdims(cfm::ℱ{T}) where {T}
    return getdims(cfm.less)
end

"""
    getntime(cfm::ℱ{T})

Return the `ntime` parameter of contour Green's function.
"""
function getntime(cfm::ℱ{T}) where {T}
    return getsize(cfm.less)
end

"""
    getntau(cfm::ℱ{T})

Return the `ntau` parameter of contour Green's function.
"""
function getntau(cfm::ℱ{T}) where {T}
    return getsize(cfm.mat)
end

"""
    getsign(cfm::ℱ{T})

Return the `sign` parameter of contour Green's function.
"""
function getsign(cfm::ℱ{T}) where {T}
    return cfm.sign
end

"""
    equaldims(cfm::ℱ{T})

Return whether the dimensional parameters are equal.

See also: [`ℱ`](@ref).
"""
function equaldims(cfm::ℱ{T}) where {T}
    return equaldims(cfm.less)
end

"""
    distance(cfm1::ℱ{T}, cfm2::ℱ{T}, tstp::I64)

Calculate distance between two `ℱ` objects at given time step `tstp`.
"""
function distance(cfm1::ℱ{T}, cfm2::ℱ{T}, tstp::I64) where {T}
    # Sanity check
    @assert 0 ≤ tstp ≤ getntime(cfm1)

    err = 0.0
    #
    if tstp == 0
        err = err + distance(cfm1.mat, cfm2.mat)
    else
        err = err + distance(cfm1.ret, cfm2.ret, tstp)
        err = err + distance(cfm1.lmix, cfm2.lmix, tstp)
        err = err + distance(cfm1.less, cfm2.less, tstp)
    end
    #
    return err
end

#=
### *ℱ* : *Traits*
=#

"""
    Base.getproperty(cfm::ℱ{T}, symbol::Symbol)

Visit the properties stored in `ℱ` object. It provides access to
the Matsubara (minus, `matm`), advanced (`adv`), right-mixing (`rmix`),
and greater (`gtr`) components of the contour-ordered Green's function.
"""
function Base.getproperty(cfm::ℱ{T}, symbol::Symbol) where {T}
    if symbol === :matm
        return Gᵐᵃᵗᵐ(cfm.sign, cfm.mat)
    #
    elseif symbol === :adv
        error("Sorry, this feature has not been implemented")
    #
    elseif symbol === :rmix
        return Gʳᵐⁱˣ(cfm.sign, cfm.lmix)
    #
    elseif symbol === :gtr
        return Gᵍᵗʳ(cfm.less, cfm.ret)
    #
    else # Fallback to getfield()
        return getfield(cfm, symbol)
    end
end

#=
### *𝒻* : *Properties*
=#

"""
    getdims(cfv::𝒻{S})

Return the dimensional parameters of contour Green's function.

See also: [`𝒻`](@ref).
"""
function getdims(cfv::𝒻{S}) where {S}
    return getdims(cfv.less)
end

"""
    getntau(cfv::𝒻{S})

Return the `ntau` parameter of contour Green's function.
"""
function getntau(cfv::𝒻{S}) where {S}
    return getsize(cfv.mat)
end

"""
    gettstp(cfv::𝒻{S})

Return the `tstp` parameter of contour Green's function.
"""
function gettstp(cfv::𝒻{S}) where {S}
    return cfv.tstp # getsize(cfv.less) is wrong when cfv.tstp = 0!
end

"""
    getsign(cfv::𝒻{S})

Return the `sign` parameter of contour Green's function.
"""
function getsign(cfv::𝒻{S}) where {S}
    return cfv.sign
end

"""
    equaldims(cfv::𝒻{S})

Return whether the dimensional parameters are equal.

See also: [`𝒻`](@ref).
"""
function equaldims(cfv::𝒻{S}) where {S}
    return equaldims(cfv.less)
end

"""
    distance(cfv1::𝒻{S}, cfv2::𝒻{S}, tstp::I64)

Calculate distance between two `𝒻` objects at given time step `tstp`.
"""
function distance(cfv1::𝒻{S}, cfv2::𝒻{S}, tstp::I64) where {S}
    # Sanity check
    @assert tstp == gettstp(cfv1)

    err = 0.0
    #
    if tstp == 0
        err = err + distance(cfv1.mat, cfv2.mat)
    else
        err = err + distance(cfv1.ret, cfv2.ret)
        err = err + distance(cfv1.lmix, cfv2.lmix)
        err = err + distance(cfv1.less, cfv2.less)
    end
    #
    return err
end

"""
    distance(cfv1::𝒻{S}, cfm2::ℱ{S}, tstp::I64)

Calculate distance between a `𝒻` object and a `ℱ` object at
given time step `tstp`.
"""
function distance(cfv1::𝒻{S}, cfm2::ℱ{S}, tstp::I64) where {S}
    # Sanity check
    @assert tstp == gettstp(cfv1)

    err = 0.0
    #
    if tstp == 0
        err = err + distance(cfv1.mat, cfm2.mat)
    else
        err = err + distance(cfv1.ret, cfm2.ret, tstp)
        err = err + distance(cfv1.lmix, cfm2.lmix, tstp)
        err = err + distance(cfv1.less, cfm2.less, tstp)
    end
    #
    return err
end

"""
    distance(cfm1::ℱ{S}, cfv2::𝒻{S}, tstp::I64)

Calculate distance between a `𝒻` object and a `ℱ` object at
given time step `tstp`.
"""
distance(cfm1::ℱ{S}, cfv2::𝒻{S}, tstp::I64) where {S} = distance(cfv2, cfm1, tstp)

#=
### *𝒻* : *Traits*
=#

"""
    Base.getproperty(cfv::𝒻{S}, symbol::Symbol)

Visit the properties stored in `𝒻` object. It provides access to
the Matsubara (minus, `matm`), advanced (`adv`), right-mixing (`rmix`),
and greater (`gtr`) components of the contour-ordered Green's function
at given time step `tstp`..
"""
function Base.getproperty(cfv::𝒻{S}, symbol::Symbol) where {S}
    if symbol === :matm
        return gᵐᵃᵗᵐ(cfv.sign, cfv.mat)
    #
    elseif symbol === :adv
        error("Sorry, this feature has not been implemented")
    #
    elseif symbol === :rmix
        return gʳᵐⁱˣ(cfv.sign, cfv.lmix)
    #
    elseif symbol === :gtr
        return gᵍᵗʳ(cfv.less, cfv.ret)
    #
    else # Fallback to getfield()
        return getfield(cfv, symbol)
    end
end
