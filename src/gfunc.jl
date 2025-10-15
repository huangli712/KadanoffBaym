#
# Project : Lavender
# Source  : green3.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/10/15
#

#=
*Remarks : Full Contour Green's Functions*

As mentioned before, there are six linearly independent ''physical''
Green's functions. Assuming the hermitian symmetry, the number of
independent components is limited to four. Hence, in this package,
we just use ``{G^{M},\ G^{R},\ G^{\rceil},\ G^{<}}`` as the minimal
set of independent contour-ordered Green's functions. We call them
as `mat`, `ret`, `lmix`, and `less` components throughout the package.
=#

#=
### *ℱ* : *Struct*
=#

"""
    ℱ{T}

Standard contour-ordered Green's function. It includes four independent
components, namely `mat`, `ret`, `lmix`, and `less`.
"""
mutable struct ℱ{T} <: CnAbstractFunction{T}
    sign :: I64 # Used to distinguish fermions and bosons
    mat  :: Gᵐᵃᵗ{T}
    ret  :: Gʳᵉᵗ{T}
    lmix :: Gˡᵐⁱˣ{T}
    less :: Gˡᵉˢˢ{T}
end

#=
### *ℱ* : *Constructors*
=#

"""
    ℱ(C::Cn, v::T, sign::I64)

Standard constructor. This function is initialized by `v`.
"""
function ℱ(C::Cn, v::T, sign::I64) where {T}
    # Sanity check
    @assert sign in (BOSE, FERMI)

    # Create mat, ret, lmix, and less.
    mat  = Gᵐᵃᵗ(C, v)
    ret  = Gʳᵉᵗ(C, v)
    lmix = Gˡᵐⁱˣ(C, v)
    less = Gˡᵉˢˢ(C, v)

    # Call the default constructor
    ℱ(sign, mat, ret, lmix, less)
end

"""
    ℱ(C::Cn, sign::I64 = FERMI)

Constructor. Create a contour Green's function with zero initial values.
"""
function ℱ(C::Cn, sign::I64 = FERMI)
    # Setup sign
    @assert sign in (BOSE, FERMI)

    # Create mat, ret, lmix, and less.
    mat  = Gᵐᵃᵗ(C)
    ret  = Gʳᵉᵗ(C)
    lmix = Gˡᵐⁱˣ(C)
    less = Gˡᵉˢˢ(C)

    # Call the default constructor
    ℱ(sign, mat, ret, lmix, less)
end

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
    density(cfm::ℱ{T}, tstp::I64)

Returns the density matrix at given time step `tstp`. If `tstp = 0`,
it denotes the equilibrium state. However, when `tstp > 0`, it means
the nonequilibrium state.

See also: [`Gᵐᵃᵗ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
function density(cfm::ℱ{T}, tstp::I64) where {T}
    # Sanity check
    @assert 0 ≤ tstp ≤ getntime(cfm)

    if tstp == 0
        return -cfm.mat[getntime(cfm)]
    else
        return cfm.less[tstp, tstp] * getsign(cfm) * im
    end
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
### *ℱ* : *Operations*
=#

"""
    memset!(cfm::ℱ{T}, x)

Reset all the matrix elements of `cfm` to `x`. `x` should be a
scalar number.
"""
function memset!(cfm::ℱ{T}, x) where {T}
    memset!(cfm.mat, x)
    memset!(cfm.ret, x)
    memset!(cfm.lmix, x)
    memset!(cfm.less, x)
end

"""
    memset!(cfm::ℱ{T}, tstp::I64, x)

Reset the matrix elements of `cfm` at given time step `tstp` to `x`. `x`
should be a scalar number. Note that `tstp = 0` means the equilibrium
state, at this time this function will reset the Matsubara component
only (`mat`). However, when `tstp > 0`, the `ret`, `lmix`, and `less`
components will be changed.
"""
function memset!(cfm::ℱ{T}, tstp::I64, x) where {T}
    @assert 0 ≤ tstp ≤ getntime(cfm)
    if tstp > 0
        memset!(cfm.ret, tstp, x)
        memset!(cfm.lmix, tstp, x)
        memset!(cfm.less, tstp, x)
    else
        @assert tstp == 0
        memset!(cfm.mat, x)
    end
end

"""
    zeros!(cfm::ℱ{T})

Reset all the matrix elements of `cfm` to `zero`.
"""
zeros!(cfm::ℱ{T}) where {T} = memset!(cfm, zero(T))

"""
    zeros!(cfm::ℱ{T}, tstp::I64)

Reset the matrix elements of `cfm` at given time step `tstp` to `zero`.
"""
zeros!(cfm::ℱ{T}, tstp::I64) where {T} = memset!(cfm, tstp, zero(T))

"""
    memcpy!(src::ℱ{T}, dst::ℱ{T}, tstp::I64)

Copy contour Green's function at given time step `tstp`. Note that
`tstp = 0` means the equilibrium state, at this time this function
will copy the Matsubara component only (`mat`). However, when `tstp > 0`,
the `ret`, `lmix`, and `less` components will be copied.
"""
function memcpy!(src::ℱ{T}, dst::ℱ{T}, tstp::I64) where {T}
    @assert 0 ≤ tstp ≤ getntime(src)
    if tstp > 0
        memcpy!(src.ret, dst.ret, tstp)
        memcpy!(src.lmix, dst.lmix, tstp)
        memcpy!(src.less, dst.less, tstp)
    else
        @assert tstp == 0
        memcpy!(src.mat, dst.mat)
    end
end

"""
    incr!(cfm1::ℱ{T}, cfm2::ℱ{T}, tstp::I64, α)

Adds a `ℱ` with given weight (`α`) to another `ℱ` (at given
time step `tstp`).
"""
function incr!(cfm1::ℱ{T}, cfm2::ℱ{T}, tstp::I64, α) where {T}
    @assert 0 ≤ tstp ≤ getntime(cfm2)
    cα = convert(T, α)
    if tstp > 0
        incr!(cfm1.ret, cfm2.ret, tstp, cα)
        incr!(cfm1.lmix, cfm2.lmix, tstp, cα)
        incr!(cfm1.less, cfm2.less, tstp, cα)
    else
        @assert tstp == 0
        incr!(cfm1.mat, cfm2.mat, cα)
    end
end

"""
    incr!(cfm1::ℱ{T}, cfm2::ℱ{T}, α)

Adds a `ℱ` with given weight (`α`) to another `ℱ` (at all
possible time step `tstp`).
"""
function incr!(cfm1::ℱ{T}, cfm2::ℱ{T}, α) where {T}
    for tstp = 0:getntime(cfm2)
        incr!(cfm1, cfm2, tstp, α)
    end
end

"""
    smul!(cfm::ℱ{T}, tstp::I64, α)

Multiply a `ℱ` with given weight (`α`) at given time
step `tstp`.
"""
function smul!(cfm::ℱ{T}, tstp::I64, α) where {T}
    @assert 0 ≤ tstp ≤ getntime(cfm)
    cα = convert(T, α)
    if tstp > 0
        smul!(cfm.ret, tstp, cα)
        smul!(cfm.lmix, tstp, cα)
        smul!(cfm.less, tstp, cα)
    else
        @assert tstp == 0
        smul!(cfm.mat, cα)
    end
end

"""
    smul!(cff::Cf{T}, cfm::ℱ{T}, tstp::I64)

Left multiply a `ℱ` with given weight (`Cf`) at given time
step `tstp`.
"""
function smul!(cff::Cf{T}, cfm::ℱ{T}, tstp::I64) where {T}
    @assert 0 ≤ tstp ≤ getntime(cfm)
    if tstp > 0
        smul!(cff[tstp], cfm.ret, tstp)
        smul!(cff[tstp], cfm.lmix, tstp)
        smul!(cff, cfm.less, tstp)
    else
        @assert tstp == 0
        smul!(cff[0], cfm.mat)
    end
end

"""
    smul!(cfm::ℱ{T}, cff::Cf{T}, tstp::I64)

Right multiply a `ℱ` with given weight (`Cf`) at given time
step `tstp`.
"""
function smul!(cfm::ℱ{T}, cff::Cf{T}, tstp::I64) where {T}
    @assert 0 ≤ tstp ≤ getntime(cfm)
    if tstp > 0
        smul!(cfm.ret, cff, tstp)
        smul!(cfm.lmix, cff[0], tstp)
        smul!(cfm.less, cff[tstp], tstp)
    else
        @assert tstp == 0
        smul!(cfm.mat, cff[0])
    end
end

#=
### *ℱ* : *I/O*
=#

"""
    read!(fname::AbstractString, cfm::ℱ{T})

Read the contour Green's functions from given file.
"""
function read!(fname::AbstractString, cfm::ℱ{T}) where {T}
    sorry()
end

"""
    write(fname::AbstractString, cfm::ℱ{T})

Write the contour Green's functions to given file.
"""
function write(fname::AbstractString, cfm::ℱ{T}) where {T}
    sorry()
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
*Remarks : Full Contour Green's Functions at Given Time Step `tstp`*

In general, it can be viewed as a slice of the contour Green's function
at time axis. It includes four independent components.

* ``G^{M}(\tau)``
* ``G^{R}(t_i \equiv tstp, t_j)``, where ``t_j \le tstp``
* ``G^{⌉}(t_i \equiv tstp, \tau_j)``
* ``G^{<}(t_i, t_j \equiv tstp)``, where ``t_i \le tstp``

We also name them as `mat`, `ret`, `lmix`, and `less`, respectively.
=#

#=
### *𝒻* : *Struct*
=#

"""
    𝒻{S}

Standard contour-ordered Green's function at given time step `tstp`. It
includes four independent components, namely `mat`, `ret`, `lmix`, and
`less`. If `tstp = 0`, it denotes the equilibrium state (only the `mat`
component is valid). On the other hand, `tstp > 0` means nonequilibrium
state.
"""
mutable struct 𝒻{S} <: CnAbstractFunction{S}
    sign :: I64 # Used to distinguish fermions and bosons
    tstp :: I64
    mat  :: gᵐᵃᵗ{S}
    ret  :: gʳᵉᵗ{S}
    lmix :: gˡᵐⁱˣ{S}
    less :: gˡᵉˢˢ{S}
end

#=
### *𝒻* : *Constructors*
=#

"""
    𝒻(C::Cn, tstp::I64, v::S, sign::I64 = FERMI)

Standard constructor. This function is initialized by `v`.
"""
function 𝒻(C::Cn, tstp::I64, v::S, sign::I64 = FERMI) where {S}
    # Sanity check
    @assert sign in (BOSE, FERMI)
    @assert C.ntime ≥ tstp ≥ 0

    # Create mat, ret, lmix, and less.
    mat = gᵐᵃᵗ(C.ntau, C.ndim1, C.ndim2, v)
    #
    if tstp == 0
        # Actually, at this time this component should not be accessed.
        ret = gʳᵉᵗ(tstp + 1, C.ndim1, C.ndim2, v)
    else
        ret = gʳᵉᵗ(tstp, C.ndim1, C.ndim2, v)
    end
    #
    lmix = gˡᵐⁱˣ(C.ntau, C.ndim1, C.ndim2, v)
    #
    if tstp == 0
        # Actually, at this time this component should not be accessed.
        less = gˡᵉˢˢ(tstp + 1, C.ndim1, C.ndim2, v)
    else
        less = gˡᵉˢˢ(tstp, C.ndim1, C.ndim2, v)
    end

    # Call the default constructor
    𝒻(sign, tstp, mat, ret, lmix, less)
end

"""
    𝒻(C::Cn, tstp::I64, sign::I64 = FERMI)

Constructor. Create a fermionic contour function with zero initial values.
"""
function 𝒻(C::Cn, tstp::I64, sign::I64 = FERMI)
    # Sanity check
    @assert sign in (BOSE, FERMI)
    @assert C.ntime ≥ tstp ≥ 0

    # Create mat, ret, lmix, and less.
    mat = gᵐᵃᵗ(C.ntau, C.ndim1, C.ndim2)
    #
    if tstp == 0
        # Actually, at this time this component should not be accessed.
        ret = gʳᵉᵗ(tstp + 1, C.ndim1, C.ndim2)
    else
        ret = gʳᵉᵗ(tstp, C.ndim1, C.ndim2)
    end
    #
    lmix = gˡᵐⁱˣ(C.ntau, C.ndim1, C.ndim2)
    #
    if tstp == 0
        # Actually, at this time this component should not be accessed.
        less = gˡᵉˢˢ(tstp + 1, C.ndim1, C.ndim2)
    else
        less = gˡᵉˢˢ(tstp, C.ndim1, C.ndim2)
    end

    # Call the default constructor
    𝒻(sign, tstp, mat, ret, lmix, less)
end

"""
    𝒻(tstp::I64, ntau::I64, ndim1::I64, ndim2::I64, sign::I64 = FERMI)

Constructor. Create a fermionic contour function with zero initial values.
"""
function 𝒻(tstp::I64, ntau::I64, ndim1::I64, ndim2::I64, sign::I64 = FERMI)
    # Sanity check
    @assert sign in (BOSE, FERMI)
    @assert tstp  ≥ 0
    @assert ntau  ≥ 2
    @assert ndim1 ≥ 1
    @assert ndim2 ≥ 1

    # Create mat, ret, lmix, and less.
    mat = gᵐᵃᵗ(ntau, ndim1, ndim2)
    #
    if tstp == 0
        # Actually, at this time this component should not be accessed.
        ret = gʳᵉᵗ(tstp + 1, ndim1, ndim2)
    else
        ret = gʳᵉᵗ(tstp, ndim1, ndim2)
    end
    #
    lmix = gˡᵐⁱˣ(ntau, ndim1, ndim2)
    #
    if tstp == 0
        # Actually, at this time this component should not be accessed.
        less = gˡᵉˢˢ(tstp + 1, ndim1, ndim2)
    else
        less = gˡᵉˢˢ(tstp, ndim1, ndim2)
    end

    # Call the default constructor
    𝒻(sign, tstp, mat, ret, lmix, less)
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
    density(cfv::𝒻{S}, tstp::I64)

Returns the density matrix at given time step `tstp`. If `tstp = 0`,
it denotes the equilibrium state. However, when `tstp > 0`, it means
the nonequilibrium state.

See also: [`gᵐᵃᵗ`](@ref), [`gˡᵉˢˢ`](@ref).
"""
function density(cfv::𝒻{S}, tstp::I64) where {S}
    # Sanity check
    @assert tstp == gettstp(cfv)

    if tstp == 0
        return -cfv.mat[getntau(cfv)]
    else
        return cfv.less[tstp] * getsign(cfv) * im
    end
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
### *𝒻* : *Indexing*
=#

"""
    Base.getindex(cfm::ℱ{T}, tstp::I64)

Return contour Green's function at given time step `tstp`.

See also: [`ℱ`](@ref), [`𝒻`](@ref).
"""
function Base.getindex(cfm::ℱ{T}, tstp::I64) where {T}
    # Sanity check
    @assert getntime(cfm) ≥ tstp ≥ 0

    # Get key parameters
    sign = getsign(cfm)
    ntau = getntau(cfm)
    ndim1, ndim2 = getdims(cfm)

    # Construct an empty `𝒻` struct
    cfv = 𝒻(tstp, ntau, ndim1, ndim2, sign)

    # Extract data at time step `tstp` from `ℱ` object, then copy
    # them to `𝒻` object.
    memcpy!(cfm, cfv)

    # Return the desired struct
    return cfv
end

"""
    Base.setindex!(cfm::ℱ{S}, cfv::𝒻{S}, tstp::I64)

Setup contout Green's function at given time step `tstp`.

See also: [`ℱ`](@ref), [`𝒻`](@ref).
"""
function Base.setindex!(cfm::ℱ{S}, cfv::𝒻{S}, tstp::I64) where {S}
    # Sanity check
    @assert tstp == gettstp(cfv)
    @assert 0 ≤ tstp ≤ getntime(cfm)

    # Copy data from `𝒻` object to `ℱ` object
    memcpy!(cfv, cfm)
end

#=
### *𝒻* : *Operations*
=#

"""
    memset!(cfv::𝒻{S}, x)

Reset all the matrix elements of `cfv` to `x`. `x` should be a
scalar number.
"""
function memset!(cfv::𝒻{S}, x) where {S}
    memset!(cfv.mat, x)
    memset!(cfv.ret, x)
    memset!(cfv.lmix, x)
    memset!(cfv.less, x)
end

"""
    memset!(cfv::𝒻{S}, tstp::I64, x)

Reset all the matrix elements of `cfv` to `x`. `x` should be a
scalar number. If `tstp = 0`, only the `mat` component is updated.
On the other hand, if `tstp > 0`, the `ret`, `lmix`, and `less`
components will be updated.
"""
function memset!(cfv::𝒻{S}, tstp::I64, x) where {S}
    @assert tstp == gettstp(cfv)
    if tstp > 0
        memset!(cfv.ret, x)
        memset!(cfv.lmix, x)
        memset!(cfv.less, x)
    else
        memset!(cfv.mat, x)
    end
end

"""
    zeros!(cfv::𝒻{S})

Reset all the matrix elements of `cfv` to `zero`.
"""
zeros!(cfv::𝒻{S}) where {S} = memset!(cfv, zero(S))

"""
    zeros!(cfv::𝒻{S}, tstp::I64)

Reset all the matrix elements of `cfv` to `zero` at given time step `tstp`.
"""
zeros!(cfv::𝒻{S}, tstp::I64) where {S} = memset!(cfv, tstp, zero(S))

"""
    memcpy!(src::𝒻{S}, dst::𝒻{S}, tstp::I64)

Extract data from a `𝒻` object (at given time step `tstp`), then
copy them to another `𝒻` object.

See also: [`𝒻`](@ref).
"""
function memcpy!(src::𝒻{S}, dst::𝒻{S}, tstp::I64) where {S}
    @assert tstp == gettstp(src)
    if tstp > 0
        memcpy!(src.ret, dst.ret)
        memcpy!(src.lmix, dst.lmix)
        memcpy!(src.less, dst.less)
    else
        memcpy!(src.mat, dst.mat)
    end
end

"""
    memcpy!(cfm::ℱ{S}, cfv::𝒻{S}, tstp::I64)

Extract data from a `ℱ` object (at given time step `tstp`), then
copy them to a `𝒻` object.

See also: [`ℱ`](@ref), [`𝒻`](@ref).
"""
function memcpy!(cfm::ℱ{S}, cfv::𝒻{S}, tstp::I64) where {S}
    @assert tstp == gettstp(cfv)
    if tstp > 0
        memcpy!(cfm.ret, cfv.ret)
        memcpy!(cfm.lmix, cfv.lmix, cfv.tstp)
        memcpy!(cfm.less, cfv.less)
    else
        memcpy!(cfm.mat, cfv.mat)
    end
end

"""
    memcpy!(cfv::𝒻{S}, cfm::ℱ{S}, tstp::I64)

Extract data from a `𝒻` object, then copy them to a `ℱ` object
(at given time step `tstp`).

See also: [`ℱ`](@ref), [`𝒻`](@ref).
"""
function memcpy!(cfv::𝒻{S}, cfm::ℱ{S}, tstp::I64) where {S}
    @assert tstp == gettstp(cfv)
    if tstp > 0
        memcpy!(cfv.ret, cfm.ret)
        memcpy!(cfv.lmix, cfm.lmix, cfv.tstp)
        memcpy!(cfv.less, cfm.less)
    else
        memcpy!(cfv.mat, cfm.mat)
    end
end

"""
    incr!(cfv1::𝒻{S}, cfv2::𝒻{S}, tstp::I64, α)

Adds a `𝒻` with given weight (`α`) to another `𝒻` (at given
time step `tstp`).
"""
function incr!(cfv1::𝒻{S}, cfv2::𝒻{S}, tstp::I64, α) where {S}
    @assert gettstp(cfv1) == gettstp(cfv2) == tstp
    cα = convert(S, α)
    if tstp > 0
        incr!(cfv1.ret, cfv2.ret, cα)
        incr!(cfv1.lmix, cfv2.lmix, cα)
        incr!(cfv1.less, cfv2.less, cα)
    else
        incr!(cfv1.mat, cfv2.mat, cα)
    end
end

"""
    incr!(cfm::ℱ{S}, cfv::𝒻{S}, tstp::I64, α)

Adds a `𝒻` with given weight (`α`) to a `ℱ` (at given
time step `tstp`).
"""
function incr!(cfm::ℱ{S}, cfv::𝒻{S}, tstp::I64, α) where {S}
    @assert 0 ≤ tstp ≤ getntime(cfm)
    @assert tstp == gettstp(cfv)
    cα = convert(S, α)
    if tstp > 0
        incr!(cfm.ret, cfv.ret, cα)
        incr!(cfm.lmix, cfv.lmix, tstp, cα)
        incr!(cfm.less, cfv.less, cα)
    else
        incr!(cfm.mat, cfv.mat, cα)
    end
end

"""
    incr!(cfv::𝒻{S}, cfm::ℱ{S}, tstp::I64, α)

Adds a `ℱ` with given weight (`α`) to a `𝒻` (at given
time step `tstp`).
"""
function incr!(cfv::𝒻{S}, cfm::ℱ{S}, tstp::I64, α) where {S}
    @assert 0 ≤ tstp ≤ getntime(cfm)
    @assert tstp == gettstp(cfv)
    cα = convert(S, α)
    if tstp > 0
        incr!(cfv.ret, cfm.ret, cα)
        incr!(cfv.lmix, cfm.lmix, tstp, cα)
        incr!(cfv.less, cfm.less, cα)
    else
        incr!(cfv.mat, cfm.mat, cα)
    end
end

"""
    smul!(cfv::𝒻{S}, tstp::I64, α)

Multiply a `𝒻` with given weight (`α`) at given time
step `tstp`.
"""
function smul!(cfv::𝒻{S}, tstp::I64, α) where {S}
    @assert tstp == gettstp(cfv)
    cα = convert(S, α)
    if tstp > 0
        smul!(cfv.ret, cα)
        smul!(cfv.lmix, cα)
        smul!(cfv.less, cα)
    else
        smul!(cfv.mat, cα)
    end
end

"""
    smul!(cff::Cf{S}, cfv::𝒻{S}, tstp::I64)

Left multiply a `𝒻` with given weight (`Cf`) at given time
step `tstp`.
"""
function smul!(cff::Cf{S}, cfv::𝒻{S}, tstp::I64) where {S}
    @assert tstp == gettstp(cfv)
    @assert tstp ≤ getsize(cff)
    if tstp > 0
        smul!(cff[tstp], cfv.ret)
        smul!(cff[tstp], cfv.lmix)
        smul!(cff, cfv.less)
    else
        smul!(cff[0], cfv.mat)
    end
end

"""
    smul!(cfv::𝒻{S}, cff::Cf{S}, tstp::I64)

Right multiply a `𝒻` with given weight (`Cf`) at given time
step `tstp`.
"""
function smul!(cfv::𝒻{S}, cff::Cf{S}, tstp::I64) where {S}
    @assert tstp == gettstp(cfv)
    @assert tstp ≤ getsize(cff)
    if tstp > 0
        smul!(cfv.ret, cff)
        smul!(cfv.lmix, cff[0])
        smul!(cfv.less, cff[tstp])
    else
        smul!(cfv.mat, cff[0])
    end
end

#=
### *𝒻* : *I/O*
=#

"""
    read!(fname::AbstractString, cfv::𝒻{S})

Read the contour Green's functions from given file.
"""
function read!(fname::AbstractString, cfv::𝒻{S}) where {S}
    sorry()
end

"""
    write(fname::AbstractString, cfv::𝒻{S})

Write the contour Green's functions to given file.
"""
function write(fname::AbstractString, cfv::𝒻{S}) where {S}
    sorry()
end

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
