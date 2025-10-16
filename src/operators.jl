#=
### *Gᵐᵃᵗ* : *Traits*
=#

"""
    Base.:+(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T})

Operation `+` for two `Gᵐᵃᵗ` objects.
"""
function Base.:+(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T}) where {T}
    # Sanity check
    @assert getsize(mat1) == getsize(mat2)
    @assert getdims(mat1) == getdims(mat2)

    Gᵐᵃᵗ(mat1.type, mat1.ntau, mat1.ndim1, mat1.ndim2, mat1.data + mat2.data)
end

"""
    Base.:-(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T})

Operation `-` for two `Gᵐᵃᵗ` objects.
"""
function Base.:-(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T}) where {T}
    # Sanity check
    @assert getsize(mat1) == getsize(mat2)
    @assert getdims(mat1) == getdims(mat2)

    Gᵐᵃᵗ(mat1.type, mat1.ntau, mat1.ndim1, mat1.ndim2, mat1.data - mat2.data)
end

"""
    Base.:*(mat::Gᵐᵃᵗ{T}, x)

Operation `*` for a `Gᵐᵃᵗ` object and a scalar value.
"""
function Base.:*(mat::Gᵐᵃᵗ{T}, x) where {T}
    cx = convert(T, x)
    Gᵐᵃᵗ(mat.type, mat.ntau, mat.ndim1, mat.ndim2, mat.data * cx)
end

"""
    Base.:*(x, mat::Gᵐᵃᵗ{T})

Operation `*` for a scalar value and a `Gᵐᵃᵗ` object.
"""
Base.:*(x, mat::Gᵐᵃᵗ{T}) where {T} = Base.:*(mat, x)

#=
### *Gʳᵉᵗ* : *Traits*
=#

"""
    Base.:+(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T})

Operation `+` for two `Gʳᵉᵗ` objects.
"""
function Base.:+(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}) where {T}
    # Sanity check
    @assert getsize(ret1) == getsize(ret2)
    @assert getdims(ret1) == getdims(ret2)

    Gʳᵉᵗ(ret1.type, ret1.ntime, ret1.ndim1, ret1.ndim2, ret1.data + ret2.data)
end

"""
    Base.:-(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T})

Operation `-` for two `Gʳᵉᵗ` objects.
"""
function Base.:-(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}) where {T}
    # Sanity check
    @assert getsize(ret1) == getsize(ret2)
    @assert getdims(ret1) == getdims(ret2)

    Gʳᵉᵗ(ret1.type, ret1.ntime, ret1.ndim1, ret1.ndim2, ret1.data - ret2.data)
end

"""
    Base.:*(ret::Gʳᵉᵗ{T}, x)

Operation `*` for a `Gʳᵉᵗ` object and a scalar value.
"""
function Base.:*(ret::Gʳᵉᵗ{T}, x) where {T}
    cx = convert(T, x)
    Gʳᵉᵗ(ret.type, ret.ntime, ret.ndim1, ret.ndim2, ret.data * cx)
end

"""
    Base.:*(x, ret::Gʳᵉᵗ{T})

Operation `*` for a scalar value and a `Gʳᵉᵗ` object.
"""
Base.:*(x, ret::Gʳᵉᵗ{T}) where {T} = Base.:*(ret, x)

#=
### *Gˡᵐⁱˣ* : *Traits*
=#

"""
    Base.:+(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T})

Operation `+` for two `Gˡᵐⁱˣ` objects.
"""
function Base.:+(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}) where {T}
    # Sanity check
    @assert getsize(lmix1) == getsize(lmix2)
    @assert getdims(lmix1) == getdims(lmix2)

    Gˡᵐⁱˣ(lmix1.type, lmix1.ntime, lmix1.ntau, lmix1.ndim1, lmix1.ndim2, lmix1.data + lmix2.data)
end

"""
    Base.:-(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T})

Operation `-` for two `Gˡᵐⁱˣ` objects.
"""
function Base.:-(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}) where {T}
    # Sanity check
    @assert getsize(lmix1) == getsize(lmix2)
    @assert getdims(lmix1) == getdims(lmix2)

    Gˡᵐⁱˣ(lmix1.type, lmix1.ntime, lmix1.ntau, lmix1.ndim1, lmix1.ndim2, lmix1.data - lmix2.data)
end

"""
    Base.:*(lmix::Gˡᵐⁱˣ{T}, x)

Operation `*` for a `Gˡᵐⁱˣ` object and a scalar value.
"""
function Base.:*(lmix::Gˡᵐⁱˣ{T}, x) where {T}
    cx = convert(T, x)
    Gˡᵐⁱˣ(lmix.type, lmix.ntime, lmix.ntau, lmix.ndim1, lmix.ndim2, lmix.data * cx)
end

"""
    Base.:*(x, lmix::Gˡᵐⁱˣ{T})

Operation `*` for a scalar value and a `Gˡᵐⁱˣ` object.
"""
Base.:*(x, lmix::Gˡᵐⁱˣ{T}) where {T} = Base.:*(lmix, x)

#=
### *Gˡᵉˢˢ* : *Traits*
=#

"""
    Base.:+(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T})

Operation `+` for two `Gˡᵉˢˢ` objects.
"""
function Base.:+(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}) where {T}
    # Sanity check
    @assert getsize(less1) == getsize(less2)
    @assert getdims(less1) == getdims(less2)

    Gˡᵉˢˢ(less1.type, less1.ntime, less1.ndim1, less1.ndim2, less1.data + less2.data)
end

"""
    Base.:-(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T})

Operation `-` for two `Gˡᵉˢˢ` objects.
"""
function Base.:-(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}) where {T}
    # Sanity check
    @assert getsize(less1) == getsize(less2)
    @assert getdims(less1) == getdims(less2)

    Gˡᵉˢˢ(less1.type, less1.ntime, less1.ndim1, less1.ndim2, less1.data - less2.data)
end

"""
    Base.:*(less::Gˡᵉˢˢ{T}, x)

Operation `*` for a `Gˡᵉˢˢ` object and a scalar value.
"""
function Base.:*(less::Gˡᵉˢˢ{T}, x) where {T}
    cx = convert(T, x)
    Gˡᵉˢˢ(less.type, less.ntime, less.ndim1, less.ndim2, less.data * cx)
end

"""
    Base.:*(x, less::Gˡᵉˢˢ{T})

Operation `*` for a scalar value and a `Gˡᵉˢˢ` object.
"""
Base.:*(x, less::Gˡᵉˢˢ{T}) where {T} = Base.:*(less, x)

#=
### *gᵐᵃᵗ* : *Traits*
=#

"""
    Base.:+(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S})

Operation `+` for two `gᵐᵃᵗ` objects.
"""
function Base.:+(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}) where {S}
    # Sanity check
    @assert getsize(mat1) == getsize(mat2)
    @assert getdims(mat1) == getdims(mat2)

    gᵐᵃᵗ(mat1.type, mat1.ntau, mat1.ndim1, mat1.ndim2, mat1.data + mat2.data)
end

"""
    Base.:-(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S})

Operation `-` for two `gᵐᵃᵗ` objects.
"""
function Base.:-(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}) where {S}
    # Sanity check
    @assert getsize(mat1) == getsize(mat2)
    @assert getdims(mat1) == getdims(mat2)

    gᵐᵃᵗ(mat1.type, mat1.ntau, mat1.ndim1, mat1.ndim2, mat1.data - mat2.data)
end

"""
    Base.:*(mat::gᵐᵃᵗ{S}, x)

Operation `*` for a `gᵐᵃᵗ` object and a scalar value.
"""
function Base.:*(mat::gᵐᵃᵗ{S}, x) where {S}
    cx = convert(S, x)
    gᵐᵃᵗ(mat.type, mat.ntau, mat.ndim1, mat.ndim2, mat.data * cx)
end

"""
    Base.:*(x, mat::gᵐᵃᵗ{S})

Operation `*` for a scalar value and a `gᵐᵃᵗ` object.
"""
Base.:*(x, mat::gᵐᵃᵗ{S}) where {S} = Base.:*(mat, x)
