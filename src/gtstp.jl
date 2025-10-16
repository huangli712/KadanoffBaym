


#=
### *gˡᵐⁱˣ* : *Struct*
=#

"""
    gˡᵐⁱˣ{S}

Left-mixing component (``G^{⌉}``) of contour Green's function at given
time step `tstp`. Actually, it denotes ``G^{⌉}(tᵢ ≡ tstp, τⱼ)``.

See also: [`gᵐᵃᵗ`](@ref), [`gʳᵉᵗ`](@ref), [`gˡᵉˢˢ`](@ref).
"""
mutable struct gˡᵐⁱˣ{S} <: CnAbstractVector{S}
    type  :: String
    ntau  :: I64
    ndim1 :: I64
    ndim2 :: I64
    data  :: VecArray{S}
end

#=
### *gˡᵐⁱˣ* : *Constructors*
=#

"""
    gˡᵐⁱˣ(ntau::I64, ndim1::I64, ndim2::I64, v::S)

Constructor. All the matrix elements are set to be `v`.
"""
function gˡᵐⁱˣ(ntau::I64, ndim1::I64, ndim2::I64, v::S) where {S}
    # Sanity check
    @assert ntau  ≥ 2
    @assert ndim1 ≥ 1
    @assert ndim2 ≥ 1

    # Create Element{S}
    element = fill(v, ndim1, ndim2)

    # Create VecArray{S}, whose size is indeed (ntau,).
    data = VecArray{S}(undef, ntau)
    for i = 1:ntau
        data[i] = copy(element)
    end

    # Call the default constructor
    gˡᵐⁱˣ("lmix", ntau, ndim1, ndim2, data)
end

"""
    gˡᵐⁱˣ(ntau::I64, ndim1::I64, ndim2::I64)

Constructor. All the matrix elements are set to be complex zero.
"""
function gˡᵐⁱˣ(ntau::I64, ndim1::I64, ndim2::I64)
    gˡᵐⁱˣ(ntau, ndim1, ndim2, zero(C64))
end

"""
    gˡᵐⁱˣ(ntau::I64, ndim1::I64)

Constructor. All the matrix elements are set to be complex.
"""
function gˡᵐⁱˣ(ntau::I64, ndim1::I64)
    gˡᵐⁱˣ(ntau, ndim1, ndim1, zero(C64))
end

"""
    gˡᵐⁱˣ(ntau::I64, x::Element{S})

Constructor. The matrix is initialized by `x`.
"""
function gˡᵐⁱˣ(ntau::I64, x::Element{S}) where {S}
    # Sanity check
    @assert ntau ≥ 2

    ndim1, ndim2 = size(x)
    data = VecArray{S}(undef, ntau)
    for i = 1:ntau
        data[i] = copy(x)
    end

    # Call the default constructor
    gˡᵐⁱˣ("lmix", ntau, ndim1, ndim2, data)
end

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
### *gˡᵐⁱˣ* : *Indexing*
=#

"""
    Base.getindex(lmix::gˡᵐⁱˣ{S}, j::I64)

Visit the element stored in `gˡᵐⁱˣ` object.
"""
function Base.getindex(lmix::gˡᵐⁱˣ{S}, j::I64) where {S}
    # Sanity check
    @assert 1 ≤ j ≤ lmix.ntau

    # Return G^{⌉}(tᵢ ≡ tstp, τⱼ)
    lmix.data[j]
end

"""
    Base.setindex!(lmix::gˡᵐⁱˣ{S}, x::Element{S}, j::I64)

Setup the element in `gˡᵐⁱˣ` object.
"""
function Base.setindex!(lmix::gˡᵐⁱˣ{S}, x::Element{S}, j::I64) where {S}
    # Sanity check
    @assert size(x) == getdims(lmix)
    @assert 1 ≤ j ≤ lmix.ntau

    # G^{⌉}(tᵢ ≡ tstp, τⱼ) = x
    lmix.data[j] = copy(x)
end

"""
    Base.setindex!(lmix::gˡᵐⁱˣ{S}, v::S, j::I64)

Setup the element in `gˡᵐⁱˣ` object.
"""
function Base.setindex!(lmix::gˡᵐⁱˣ{S}, v::S, j::I64) where {S}
    # Sanity check
    @assert 1 ≤ j ≤ lmix.ntau

    # G^{⌉}(tᵢ ≡ tstp, τⱼ) .= v
    fill!(lmix.data[j], v)
end

#=
### *gˡᵐⁱˣ* : *Operations*
=#

"""
    memset!(lmix::gˡᵐⁱˣ{S}, x)

Reset all the matrix elements of `lmix` to `x`. `x` should be a
scalar number.
"""
function memset!(lmix::gˡᵐⁱˣ{S}, x) where {S}
    cx = convert(S, x)
    for i=1:lmix.ntau
        fill!(lmix.data[i], cx)
    end
end

"""
    zeros!(lmix::gˡᵐⁱˣ{S})

Reset all the matrix elements of `lmix` to `zero`.
"""
zeros!(lmix::gˡᵐⁱˣ{S}) where {S} = memset!(lmix, zero(S))

"""
    memcpy!(src::gˡᵐⁱˣ{S}, dst::gˡᵐⁱˣ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::gˡᵐⁱˣ{S}, dst::gˡᵐⁱˣ{S}) where {S}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memcpy!(src::Gˡᵐⁱˣ{S}, dst::gˡᵐⁱˣ{S}, tstp::I64)

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::Gˡᵐⁱˣ{S}, dst::gˡᵐⁱˣ{S}, tstp::I64) where {S}
    @assert iscompatible(src, dst)
    @assert 1 ≤ tstp ≤ src.ntime
    @. dst.data = copy(src.data[tstp,:])
end

"""
    memcpy!(src::gˡᵐⁱˣ{S}, dst::Gˡᵐⁱˣ{S}, tstp::I64)

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::gˡᵐⁱˣ{S}, dst::Gˡᵐⁱˣ{S}, tstp::I64) where {S}
    @assert iscompatible(src, dst)
    @assert 1 ≤ tstp ≤ dst.ntime
    @. dst.data[tstp,:] = copy(src.data)
end

"""
    incr!(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}, α::S)

Add a `gˡᵐⁱˣ` with given weight (`α`) to another `gˡᵐⁱˣ`.
"""
function incr!(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}, α::S) where {S}
    @assert iscompatible(lmix1, lmix2)
    for i = 1:lmix2.ntau
        @. lmix1.data[i] = lmix1.data[i] + lmix2.data[i] * α
    end
end

"""
    incr!(lmix1::Gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}, tstp::I64, α::S)

Add a `gˡᵐⁱˣ` with given weight (`α`) to a `Gˡᵐⁱˣ`.
"""
function incr!(lmix1::Gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}, tstp::I64, α::S) where {S}
    @assert iscompatible(lmix1, lmix2)
    @assert 1 ≤ tstp ≤ lmix1.ntime
    for i = 1:lmix2.ntau
        @. lmix1.data[tstp,i] = lmix1.data[tstp,i] + lmix2.data[i] * α
    end
end

"""
    incr!(lmix1::gˡᵐⁱˣ{S}, lmix2::Gˡᵐⁱˣ{S}, tstp::I64, α::S)

Add a `Gˡᵐⁱˣ` with given weight (`α`) to a `gˡᵐⁱˣ`.
"""
function incr!(lmix1::gˡᵐⁱˣ{S}, lmix2::Gˡᵐⁱˣ{S}, tstp::I64, α::S) where {S}
    @assert iscompatible(lmix1, lmix2)
    @assert 1 ≤ tstp ≤ lmix2.ntime
    for i = 1:lmix1.ntau
        @. lmix1.data[i] = lmix1.data[i] + lmix2.data[tstp,i] * α
    end
end

"""
    smul!(lmix::gˡᵐⁱˣ{S}, α::S)

Multiply a `gˡᵐⁱˣ` with given weight (`α`).
"""
function smul!(lmix::gˡᵐⁱˣ{S}, α::S) where {S}
    for i = 1:lmix.ntau
        @. lmix.data[i] = lmix.data[i] * α
    end
end

"""
    smul!(x::Element{S}, lmix::gˡᵐⁱˣ{S})

Left multiply a `gˡᵐⁱˣ` with given weight (`x`).
"""
function smul!(x::Element{S}, lmix::gˡᵐⁱˣ{S}) where {S}
    for i = 1:lmix.ntau
        lmix.data[i] = x * lmix.data[i]
    end
end

"""
    smul!(lmix::gˡᵐⁱˣ{S}, x::Element{S})

Right multiply a `gˡᵐⁱˣ` with given weight (`x`).
"""
function smul!(lmix::gˡᵐⁱˣ{S}, x::Element{S}) where {S}
    for i = 1:lmix.ntau
        lmix.data[i] = lmix.data[i] * x
    end
end

#=
### *gˡᵐⁱˣ* : *Traits*
=#

"""
    Base.:+(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S})

Operation `+` for two `gˡᵐⁱˣ` objects.
"""
function Base.:+(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}) where {S}
    # Sanity check
    @assert getsize(lmix1) == getsize(lmix2)
    @assert getdims(lmix1) == getdims(lmix2)

    gˡᵐⁱˣ(lmix1.type, lmix1.ntau, lmix1.ndim1, lmix1.ndim2, lmix1.data + lmix2.data)
end

"""
    Base.:-(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S})

Operation `-` for two `gˡᵐⁱˣ` objects.
"""
function Base.:-(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}) where {S}
    # Sanity check
    @assert getsize(lmix1) == getsize(lmix2)
    @assert getdims(lmix1) == getdims(lmix2)

    gˡᵐⁱˣ(lmix1.type, lmix1.ntau, lmix1.ndim1, lmix1.ndim2, lmix1.data - lmix2.data)
end

"""
    Base.:*(lmix::gˡᵐⁱˣ{S}, x)

Operation `*` for a `gˡᵐⁱˣ` object and a scalar value.
"""
function Base.:*(lmix::gˡᵐⁱˣ{S}, x) where {S}
    cx = convert(S, x)
    gˡᵐⁱˣ(lmix.type, lmix.ntau, lmix.ndim1, lmix.ndim2, lmix.data * cx)
end

"""
    Base.:*(x, lmix::gˡᵐⁱˣ{S})

Operation `*` for a scalar value and a `gˡᵐⁱˣ` object.
"""
Base.:*(x, lmix::gˡᵐⁱˣ{S}) where {S} = Base.:*(lmix, x)

#=
### *gˡᵉˢˢ* : *Struct*
=#

"""
    gˡᵉˢˢ{S}

Lesser component (``G^{<}``) of contour Green's function at given
time step `tstp`. Actually, it denotes ``G^{<}(tᵢ, tⱼ ≡ tstp)``.

See also: [`gᵐᵃᵗ`](@ref), [`gʳᵉᵗ`](@ref), [`gˡᵐⁱˣ`](@ref).
"""
mutable struct gˡᵉˢˢ{S} <: CnAbstractVector{S}
    type  :: String
    tstp  :: I64
    ndim1 :: I64
    ndim2 :: I64
    data  :: VecArray{S}
end

#=
### *gˡᵉˢˢ* : *Constructors*
=#

"""
    gˡᵉˢˢ(tstp::I64, ndim1::I64, ndim2::I64, v::S)

Constructor. All the matrix elements are set to be `v`.
"""
function gˡᵉˢˢ(tstp::I64, ndim1::I64, ndim2::I64, v::S) where {S}
    # Sanity check
    @assert tstp  ≥ 1
    @assert ndim1 ≥ 1
    @assert ndim2 ≥ 1

    # Create Element{S}
    element = fill(v, ndim1, ndim2)

    # Create VecArray{S}, whose size is indeed (tstp,).
    data = VecArray{S}(undef, tstp)
    for i = 1:tstp
        data[i] = copy(element)
    end

    # Call the default constructor
    gˡᵉˢˢ("less", tstp, ndim1, ndim2, data)
end

"""
    gˡᵉˢˢ(tstp::I64, ndim1::I64, ndim2::I64)

Constructor. All the matrix elements are set to be complex zero.
"""
function gˡᵉˢˢ(tstp::I64, ndim1::I64, ndim2::I64)
    gˡᵉˢˢ(tstp, ndim1, ndim2, zero(C64))
end

"""
    gˡᵉˢˢ(tstp::I64, ndim1::I64)

Constructor. All the matrix elements are set to be complex zero.
"""
function gˡᵉˢˢ(tstp::I64, ndim1::I64)
    gˡᵉˢˢ(tstp, ndim1, ndim1, zero(C64))
end

"""
    gˡᵉˢˢ(tstp::I64, x::Element{S})

Constructor. The matrix is initialized by `x`.
"""
function gˡᵉˢˢ(tstp::I64, x::Element{S}) where {S}
    # Sanity check
    @assert tstp ≥ 1

    ndim1, ndim2 = size(x)
    data = VecArray{S}(undef, tstp)
    for i = 1:tstp
        data[i] = copy(x)
    end

    # Call the default constructor
    gˡᵉˢˢ("less", tstp, ndim1, ndim2, data)
end

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
### *gˡᵉˢˢ* : *Indexing*
=#

"""
    Base.getindex(less::gˡᵉˢˢ{S}, i::I64)

Visit the element stored in `gˡᵉˢˢ` object.
"""
function Base.getindex(less::gˡᵉˢˢ{S}, i::I64) where {S}
    # Sanity check
    @assert 1 ≤ i ≤ less.tstp

    # Return G^{<}(tᵢ, tⱼ ≡ tstp)
    less.data[i]
end

"""
    Base.getindex(less::gˡᵉˢˢ{S}, tstp::I64, j::I64)

Visit the element stored in `gˡᵉˢˢ` object.
"""
function Base.getindex(less::gˡᵉˢˢ{S}, tstp::I64, j::I64) where {S}
    # Sanity check
    @assert tstp == less.tstp
    @assert 1 ≤ j ≤ less.tstp

    # Return G^{<}(tᵢ ≡ tstp, tⱼ)
    -(less.data[i])'
end

"""
    Base.setindex!(less::gˡᵉˢˢ{S}, x::Element{S}, i::I64)

Setup the element in `gˡᵉˢˢ` object.
"""
function Base.setindex!(less::gˡᵉˢˢ{S}, x::Element{S}, i::I64) where {S}
    # Sanity check
    @assert size(x) == getdims(less)
    @assert 1 ≤ i ≤ less.tstp

    # G^{<}(tᵢ, tⱼ ≡ tstp) = x
    less.data[i] = copy(x)
end

"""
    Base.setindex!(less::gˡᵉˢˢ{S}, v::S, i::I64)

Setup the element in `gˡᵉˢˢ` object.
"""
function Base.setindex!(less::gˡᵉˢˢ{S}, v::S, i::I64) where {S}
    # Sanity check
    @assert 1 ≤ i ≤ less.tstp

    # G^{<}(tᵢ, tⱼ ≡ tstp) .= v
    fill!(less.data[i], v)
end

#=
### *gˡᵉˢˢ* : *Operations*
=#

"""
    memset!(less::gˡᵉˢˢ{S}, x)

Reset all the matrix elements of `less` to `x`. `x` should be a
scalar number.
"""
function memset!(less::gˡᵉˢˢ{S}, x) where {S}
    cx = convert(S, x)
    for i=1:less.tstp
        fill!(less.data[i], cx)
    end
end

"""
    zeros!(less::gˡᵉˢˢ{S})

Reset all the matrix elements of `less` to `zero`.
"""
zeros!(less::gˡᵉˢˢ{S}) where {S} = memset!(less, zero(S))

"""
    memcpy!(src::gˡᵉˢˢ{S}, dst::gˡᵉˢˢ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::gˡᵉˢˢ{S}, dst::gˡᵉˢˢ{S}) where {S}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memcpy!(src::Gˡᵉˢˢ{S}, dst::gˡᵉˢˢ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::Gˡᵉˢˢ{S}, dst::gˡᵉˢˢ{S}) where {S}
    @assert iscompatible(src, dst)
    tstp = dst.tstp
    @. dst.data = copy(src.data[1:tstp,tstp])
end

"""
    memcpy!(src::gˡᵉˢˢ{S}, dst::Gˡᵉˢˢ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::gˡᵉˢˢ{S}, dst::Gˡᵉˢˢ{S}) where {S}
    @assert iscompatible(src, dst)
    tstp = src.tstp
    @. dst.data[1:tstp,tstp] = copy(src.data)
end

"""
    incr!(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}, α::S)

Add a `gˡᵉˢˢ` with given weight (`α`) to another `gˡᵉˢˢ`.
"""
function incr!(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}, α::S) where {S}
    @assert iscompatible(less1, less2)
    tstp = less2.tstp
    for i = 1:tstp
        @. less1.data[i] = less1.data[i] + less2.data[i] * α
    end
end

"""
    incr!(less1::Gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}, α::S)

Add a `gˡᵉˢˢ` with given weight (`α`) to a `Gˡᵉˢˢ`.
"""
function incr!(less1::Gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}, α::S) where {S}
    @assert iscompatible(less1, less2)
    tstp = less2.tstp
    for i = 1:tstp
        @. less1.data[i,tstp] = less1.data[i,tstp] + less2.data[i] * α
    end
end

"""
    incr!(less1::gˡᵉˢˢ{S}, less2::Gˡᵉˢˢ{S}, α::S)

Add a `Gˡᵉˢˢ` with given weight (`α`) to a `gˡᵉˢˢ`.
"""
function incr!(less1::gˡᵉˢˢ{S}, less2::Gˡᵉˢˢ{S}, α::S) where {S}
    @assert iscompatible(less1, less2)
    tstp = less1.tstp
    for i = 1:tstp
        @. less1.data[i] = less1.data[i] + less2.data[i,tstp] * α
    end
end

"""
    smul!(less::gˡᵉˢˢ{S}, α::S)

Multiply a `gˡᵉˢˢ` with given weight (`α`).
"""
function smul!(less::gˡᵉˢˢ{S}, α::S) where {S}
    for i = 1:less.tstp
        @. less.data[i] = less.data[i] * α
    end
end

"""
    smul!(x::Cf{S}, less::gˡᵉˢˢ{S})

Left multiply a `gˡᵉˢˢ` with given weight (`x`).
"""
function smul!(x::Cf{S}, less::gˡᵉˢˢ{S}) where {S}
    for i = 1:less.tstp
        less.data[i] = x[i] * less.data[i]
    end
end

"""
    smul!(less::gˡᵉˢˢ{S}, x::Element{S})

Right multiply a `gˡᵉˢˢ` with given weight (`x`).
"""
function smul!(less::gˡᵉˢˢ{S}, x::Element{S}) where {S}
    for i = 1:less.tstp
        less.data[i] = less.data[i] * x
    end
end

#=
### *gˡᵉˢˢ* : *Traits*
=#

"""
    Base.:+(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S})

Operation `+` for two `gˡᵉˢˢ` objects.
"""
function Base.:+(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}) where {S}
    # Sanity check
    @assert getsize(less1) == getsize(less2)
    @assert getdims(less1) == getdims(less2)

    gˡᵉˢˢ(less1.type, less1.tstp, less1.ndim1, less1.ndim2, less1.data + less2.data)
end

"""
    Base.:-(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S})

Operation `-` for two `gˡᵉˢˢ` objects.
"""
function Base.:-(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}) where {S}
    # Sanity check
    @assert getsize(less1) == getsize(less2)
    @assert getdims(less1) == getdims(less2)

    gˡᵉˢˢ(less1.type, less1.tstp, less1.ndim1, less1.ndim2, less1.data - less2.data)
end

"""
    Base.:*(less::gˡᵉˢˢ{S}, x)

Operation `*` for a `gˡᵉˢˢ` object and a scalar value.
"""
function Base.:*(less::gˡᵉˢˢ{S}, x) where {S}
    cx = convert(S, x)
    gˡᵉˢˢ(less.type, less.tstp, less.ndim1, less.ndim2, less.data * cx)
end

"""
    Base.:*(x, less::gˡᵉˢˢ{S})

Operation `*` for a scalar value and a `gˡᵉˢˢ` object.
"""
Base.:*(x, less::gˡᵉˢˢ{S}) where {S} = Base.:*(less, x)

#=
### *gᵐᵃᵗᵐ* : *Struct*
=#

"""
    gᵐᵃᵗᵐ{S}

Matsubara component (``G^M``) of contour Green's function at given time
step `tstp = 0`. It is designed for ``\tau < 0`` case. It is not an
independent component. It can be constructed from the `gᵐᵃᵗ{T}` struct.

See also: [`gʳᵉᵗ`](@ref), [`gˡᵐⁱˣ`](@ref), [`gˡᵉˢˢ`](@ref).
"""
mutable struct gᵐᵃᵗᵐ{S} <: CnAbstractVector{S}
    type  :: String
    sign  :: I64 # Used to distinguish fermions and bosons
    ntau  :: I64
    ndim1 :: I64
    ndim2 :: I64
    dataV :: Ref{gᵐᵃᵗ{S}}
end

#=
### *gᵐᵃᵗᵐ* : *Constructors*
=#

"""
    gᵐᵃᵗᵐ(sign::I64, mat::gᵐᵃᵗ{S})

Constructor. Note that the `matm` component is not independent. We use
the `mat` component to initialize it.
"""
function gᵐᵃᵗᵐ(sign::I64, mat::gᵐᵃᵗ{S}) where {S}
    # Sanity check
    @assert sign in (BOSE, FERMI)

    # Setup properties
    # Extract parameters from `mat`
    ntau = mat.ntau
    ndim1 = mat.ndim1
    ndim2 = mat.ndim2
    #
    # We don't allocate memory for `dataV` directly, but let it point to
    # the `mat` object.
    dataV = Ref(mat)

    # Call the default constructor
    gᵐᵃᵗᵐ("matm", sign, ntau, ndim1, ndim2, dataV)
end

#=
### *gᵐᵃᵗᵐ* : *Indexing*
=#

"""
    Base.getindex(matm::gᵐᵃᵗᵐ{S}, ind::I64)

Visit the element stored in `gᵐᵃᵗᵐ` object.
"""
function Base.getindex(matm::gᵐᵃᵗᵐ{S}, ind::I64) where {S}
    # Sanity check
    @assert 1 ≤ ind ≤ matm.ntau

    # Return G^{M}(τᵢ < 0)
    matm.dataV[][matm.ntau - ind + 1] * matm.sign
end

#=
### *gᵃᵈᵛ* : *Struct*
=#

"""
    gᵃᵈᵛ{S}

Advanced component (``G^{A}``) of contour Green's function at given
time step `tstp`.

Note that currently we do not need this component explicitly. However,
for the sake of completeness, we still define an empty struct for it.

See also: [`gᵐᵃᵗ`](@ref), [`gˡᵐⁱˣ`](@ref), [`gˡᵉˢˢ`](@ref).
"""
mutable struct gᵃᵈᵛ{S} <: CnAbstractVector{S}
    type  :: String
end

#=
### *gᵃᵈᵛ* : *Constructors*
=#

"""
    gᵃᵈᵛ()

Constructor. Note that the `adv` component is not independent. We use
the `ret` component to initialize it.
"""
function gᵃᵈᵛ()
    # Call the default constructor
    gᵃᵈᵛ("adv")
end

#=
### *gᵃᵈᵛ* : *Indexing*
=#

"""
    Base.getindex(adv::gᵃᵈᵛ{S}, ind::I64)

Visit the element stored in `gᵃᵈᵛ` object.
"""
function Base.getindex(adv::gᵃᵈᵛ{S}, ind::I64) where {S}
    sorry()
end

#=
### *gʳᵐⁱˣ* : *Struct*
=#

"""
    gʳᵐⁱˣ{S}

Right-mixing component (``G^{⌈}``) of contour Green's function at given
time step `tstp`. Actually, it denotes ``G^{⌈}(τᵢ, tⱼ ≡ tstp)``

See also: [`gᵐᵃᵗ`](@ref), [`gʳᵉᵗ`](@ref), [`gˡᵉˢˢ`](@ref).
"""
mutable struct gʳᵐⁱˣ{S} <: CnAbstractVector{S}
    type  :: String
    sign  :: I64 # Used to distinguish fermions and bosons
    ntau  :: I64
    ndim1 :: I64
    ndim2 :: I64
    dataL :: Ref{gˡᵐⁱˣ{S}}
end

#=
### *gʳᵐⁱˣ* : *Constructors*
=#

"""
    gʳᵐⁱˣ(sign::I64, lmix::gˡᵐⁱˣ{S})

Constructor. Note that the `rmix` component is not independent. We use
the `lmix` component to initialize it.
"""
function gʳᵐⁱˣ(sign::I64, lmix::gˡᵐⁱˣ{S}) where {S}
    # Sanity check
    @assert sign in (BOSE, FERMI)

    # Setup properties
    # Extract parameters from `lmix`
    ntau  = lmix.ntau
    ndim1 = lmix.ndim1
    ndim2 = lmix.ndim2
    #
    # We don't allocate memory for `dataL` directly, but let it point to
    # the `lmix` object.
    dataL = Ref(lmix)

    # Call the default constructor
    gʳᵐⁱˣ("rmix", sign, ntau, ndim1, ndim2, dataL)
end

#=
### *gʳᵐⁱˣ* : *Indexing*
=#

"""
    Base.getindex(rmix::gʳᵐⁱˣ{S}, i::I64)

Visit the element stored in `gʳᵐⁱˣ` object.
"""
function Base.getindex(rmix::gʳᵐⁱˣ{S}, i::I64) where {S}
    # Sanity check
    @assert 1 ≤ i ≤ rmix.ntau

    # Return G^{⌈}(τᵢ, tⱼ ≡ tstp)
    (rmix.dataL[])[rmix.ntau - i + 1]' * (-rmix.sign)
end

#=
### *gᵍᵗʳ* : *Struct*
=#

"""
    gᵍᵗʳ{S}

Greater component (``G^{>}``) of contour Green's function at given
time step `tstp`.

See also: [`gʳᵉᵗ`](@ref), [`gˡᵐⁱˣ`](@ref), [`gˡᵉˢˢ`](@ref).
"""
mutable struct gᵍᵗʳ{S} <: CnAbstractVector{S}
    type  :: String
    tstp  :: I64
    ndim1 :: I64
    ndim2 :: I64
    dataL :: Ref{gˡᵉˢˢ{S}}
    dataR :: Ref{gʳᵉᵗ{S}}
end

#=
### *gᵍᵗʳ* : *Constructors*
=#

"""
    gᵍᵗʳ(less::gˡᵉˢˢ{S}, ret::gʳᵉᵗ{S})

Constructor. Note that the `gtr` component is not independent. We use
the `less` and `ret` components to initialize it.
"""
function gᵍᵗʳ(less::gˡᵉˢˢ{S}, ret::gʳᵉᵗ{S}) where {S}
    # Setup properties
    # Extract parameters from `less`
    tstp  = less.tstp
    ndim1 = less.ndim1
    ndim2 = less.ndim2
    #
    # We don't allocate memory for `dataL` and `dataR` directly, but
    # let them point to  `less` and `ret` objects, respectively.
    dataL = Ref(less)
    dataR = Ref(ret)

    # Call the default constructor
    gᵍᵗʳ("gtr", tstp, ndim1, ndim2, dataL, dataR)
end

#=
### *gᵍᵗʳ* : *Indexing*
=#

"""
    Base.getindex(gtr::gᵍᵗʳ{S}, i::I64)

Visit the element stored in `gᵍᵗʳ` object.
"""
function Base.getindex(gtr::gᵍᵗʳ{S}, i::I64) where {S}
    # Sanity check
    @assert 1 ≤ i ≤ gtr.tstp

    # Return G^{>}(tᵢ, tⱼ ≡ tstp)
    gtr.dataL[][i] + gtr.dataR[][i, gtr.tstp]
end

"""
    Base.getindex(gtr::gᵍᵗʳ{S}, tstp::I64, j::I64)

Visit the element stored in `gᵍᵗʳ` object.
"""
function Base.getindex(gtr::gᵍᵗʳ{S}, tstp::I64, j::I64) where {S}
    # Sanity check
    @assert tstp == gtr.tstp
    @assert 1 ≤ j ≤ gtr.tstp

    # Return G^{>}(tᵢ ≡ tstp, tⱼ)
    gtr.dataL[][tstp, j] + gtr.dataR[][j]
end
