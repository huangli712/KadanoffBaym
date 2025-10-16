#
# Project : Lavender
# Source  : traits.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/10/15
#

#=
### *Gᵐᵃᵗ* : *Operations*
=#

"""
    memset!(mat::Gᵐᵃᵗ{T}, x)

Reset all the matrix elements of `mat` to `x`. `x` should be a
scalar number.
"""
function memset!(mat::Gᵐᵃᵗ{T}, x) where {T}
    cx = convert(T, x)
    for i = 1:mat.ntau
        fill!(mat.data[i,1], cx)
    end
end

"""
    zeros!(mat::Gᵐᵃᵗ{T})

Reset all the matrix elements of `mat` to `zero`.
"""
zeros!(mat::Gᵐᵃᵗ{T}) where {T} = memset!(mat, zero(T))

"""
    memcpy!(src::Gᵐᵃᵗ{T}, dst::Gᵐᵃᵗ{T})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::Gᵐᵃᵗ{T}, dst::Gᵐᵃᵗ{T}) where {T}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    incr!(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T}, α::T)

Add a `Gᵐᵃᵗ` with given weight (`α`) to another `Gᵐᵃᵗ`.

```math
G^M_1 ⟶ G^M_1 + α * G^M_2.
```
"""
function incr!(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T}, α::T) where {T}
    @assert iscompatible(mat1, mat2)
    for i = 1:mat2.ntau
        @. mat1.data[i,1] = mat1.data[i,1] + mat2.data[i,1] * α
    end
end

"""
    smul!(mat::Gᵐᵃᵗ{T}, α::T)

Multiply a `Gᵐᵃᵗ` with given weight (`α`).

```math
G^M ⟶ α * G^M.
```
"""
function smul!(mat::Gᵐᵃᵗ{T}, α::T) where {T}
    for i = 1:mat.ntau
        @. mat.data[i,1] = mat.data[i,1] * α
    end
end

"""
    smul!(x::Element{T}, mat::Gᵐᵃᵗ{T})

Left multiply a `Gᵐᵃᵗ` with given weight (`x`), which is actually a
matrix.
"""
function smul!(x::Element{T}, mat::Gᵐᵃᵗ{T}) where {T}
    for i = 1:mat.ntau
        mat.data[i,1] = x * mat.data[i,1]
    end
end

"""
    smul!(mat::Gᵐᵃᵗ{T}, x::Element{T})

Right multiply a `Gᵐᵃᵗ` with given weight (`x`), which is actually a
matrix.
"""
function smul!(mat::Gᵐᵃᵗ{T}, x::Element{T}) where {T}
    for i = 1:mat.ntau
        mat.data[i,1] = mat.data[i,1] * x
    end
end

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
### *Gʳᵉᵗ* : *Operations*
=#

"""
    memset!(ret::Gʳᵉᵗ{T}, x)

Reset all the matrix elements of `ret` to `x`. `x` should be a
scalar number.
"""
function memset!(ret::Gʳᵉᵗ{T}, x) where {T}
    cx = convert(T, x)
    for i=1:ret.ntime
        for j=1:ret.ntime
            fill!(ret.data[j,i], cx)
        end
    end
end

"""
    memset!(ret::Gʳᵉᵗ{T}, tstp::I64, x)

Reset the matrix elements of `ret` at given time step `tstp` (and at all
`t` where `t < tstp`) to `x`. `x` should be a scalar number.
"""
function memset!(ret::Gʳᵉᵗ{T}, tstp::I64, x) where {T}
    @assert 1 ≤ tstp ≤ ret.ntime
    cx = convert(T, x)
    for i=1:tstp
        fill!(ret.data[tstp,i], cx)
    end
end

"""
    zeros!(ret::Gʳᵉᵗ{T})

Reset all the matrix elements of `ret` to `zero`.
"""
zeros!(ret::Gʳᵉᵗ{T}) where {T} = memset!(ret, zero(T))

"""
    zeros!(ret::Gʳᵉᵗ{T}, tstp::I64)

Reset the matrix elements of `ret` at given time step `tstp` (and at all
`t` where `t < tstp`) to `zero`.
"""
zeros!(ret::Gʳᵉᵗ{T}, tstp::I64) where {T} = memset!(ret, tstp, zero(T))

"""
    memcpy!(src::Gʳᵉᵗ{T}, dst::Gʳᵉᵗ{T})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::Gʳᵉᵗ{T}, dst::Gʳᵉᵗ{T}) where {T}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memcpy!(src::Gʳᵉᵗ{T}, dst::Gʳᵉᵗ{T}, tstp::I64)

Copy some matrix elements from `src` to `dst`. Only the matrix elements
at given time step `tstp` (and at all `t` where `t < tstp`) are copied.
"""
function memcpy!(src::Gʳᵉᵗ{T}, dst::Gʳᵉᵗ{T}, tstp::I64) where {T}
    @assert iscompatible(src, dst)
    @assert 1 ≤ tstp ≤ src.ntime
    for i=1:tstp
        dst.data[tstp,i] = copy(src.data[tstp,i])
    end
end

"""
    incr!(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}, tstp::I64, α::T)

Add a `Gʳᵉᵗ` with given weight (`α`) at given time step `tstp` (and at all
`t` where `t < tstp`) to another `Gʳᵉᵗ`.
"""
function incr!(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}, tstp::I64, α::T) where {T}
    @assert iscompatible(ret1, ret2)
    @assert 1 ≤ tstp ≤ ret2.ntime
    for i = 1:tstp
        @. ret1.data[tstp,i] = ret1.data[tstp,i] + ret2.data[tstp,i] * α
    end
end

"""
    smul!(ret::Gʳᵉᵗ{T}, tstp::I64, α::T)

Multiply a `Gʳᵉᵗ` with given weight (`α`) at given time step `tstp` (and
at all `t` where `t < tstp`).
"""
function smul!(ret::Gʳᵉᵗ{T}, tstp::I64, α::T) where {T}
    @assert 1 ≤ tstp ≤ ret.ntime
    for i = 1:tstp
        @. ret.data[tstp,i] = ret.data[tstp,i] * α
    end
end

"""
    smul!(x::Element{T}, ret::Gʳᵉᵗ{T}, tstp::I64)

Left multiply a `Gʳᵉᵗ` with given weight (`x`) at given time step `tstp`
(and at all `t` where `t < tstp`).
"""
function smul!(x::Element{T}, ret::Gʳᵉᵗ{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ ret.ntime
    for i = 1:tstp
        ret.data[tstp,i] = x * ret.data[tstp,i]
    end
end

"""
    smul!(ret::Gʳᵉᵗ{T}, x::Cf{T}, tstp::I64)

Right multiply a `Gʳᵉᵗ` with given weight (`x`) at given time step `tstp`
(and at all `t` where `t < tstp`).
"""
function smul!(ret::Gʳᵉᵗ{T}, x::Cf{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ ret.ntime
    for i = 1:tstp
        ret.data[tstp,i] = ret.data[tstp,i] * x[i]
    end
end

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
### *Gˡᵐⁱˣ* : *Operations*
=#

"""
    memset!(lmix::Gˡᵐⁱˣ{T}, x)

Reset all the matrix elements of `lmix` to `x`. `x` should be a
scalar number.
"""
function memset!(lmix::Gˡᵐⁱˣ{T}, x) where {T}
    cx = convert(T, x)
    for i=1:lmix.ntau
        for j=1:lmix.ntime
            fill!(lmix.data[j,i], cx)
        end
    end
end

"""
    memset!(lmix::Gˡᵐⁱˣ{T}, tstp::I64, x)

Reset the matrix elements of `lmix` at given time step `tstp` to `x`. `x`
should be a scalar number.
"""
function memset!(lmix::Gˡᵐⁱˣ{T}, tstp::I64, x) where {T}
    @assert 1 ≤ tstp ≤ lmix.ntime
    cx = convert(T, x)
    for i=1:lmix.ntau
        fill!(lmix.data[tstp,i], cx)
    end
end

"""
    zeros!(lmix::Gˡᵐⁱˣ{T})

Reset all the matrix elements of `lmix` to `zero`.
"""
zeros!(lmix::Gˡᵐⁱˣ{T}) where {T} = memset!(lmix, zero(T))

"""
    zeros!(lmix::Gˡᵐⁱˣ{T}, tstp::I64)

Reset the matrix elements of `lmix` at given time step `tstp` to `zero`.
"""
zeros!(lmix::Gˡᵐⁱˣ{T}, tstp::I64) where {T} = memset!(lmix, tstp, zero(T))

"""
    memcpy!(src::Gˡᵐⁱˣ{T}, dst::Gˡᵐⁱˣ{T})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::Gˡᵐⁱˣ{T}, dst::Gˡᵐⁱˣ{T}) where {T}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memcpy!(src::Gˡᵐⁱˣ{T}, dst::Gˡᵐⁱˣ{T}, tstp::I64)

Copy some matrix elements from `src` to `dst`. Only the matrix elements
at given time step `tstp` are copied.
"""
function memcpy!(src::Gˡᵐⁱˣ{T}, dst::Gˡᵐⁱˣ{T}, tstp::I64) where {T}
    @assert iscompatible(src, dst)
    @assert 1 ≤ tstp ≤ src.ntime
    for i=1:src.ntau
        dst.data[tstp,i] = copy(src.data[tstp,i])
    end
end

"""
    incr!(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}, tstp::I64, α::T)

Add a `Gˡᵐⁱˣ` with given weight (`α`) at given time step `tstp` to
another `Gˡᵐⁱˣ`.
"""
function incr!(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}, tstp::I64, α::T) where {T}
    @assert iscompatible(lmix1, lmix2)
    @assert 1 ≤ tstp ≤ lmix2.ntime
    for i = 1:lmix2.ntau
        @. lmix1.data[tstp,i] = lmix1.data[tstp,i] + lmix2.data[tstp,i] * α
    end
end

"""
    smul!(lmix::Gˡᵐⁱˣ{T}, tstp::I64, α::T)

Multiply a `Gˡᵐⁱˣ` with given weight (`α`) at given time
step `tstp`.
"""
function smul!(lmix::Gˡᵐⁱˣ{T}, tstp::I64, α::T) where {T}
    @assert 1 ≤ tstp ≤ lmix.ntime
    for i = 1:lmix.ntau
        @. lmix.data[tstp,i] = lmix.data[tstp,i] * α
    end
end

"""
    smul!(x::Element{T}, lmix::Gˡᵐⁱˣ{T}, tstp::I64)

Left multiply a `Gˡᵐⁱˣ` with given weight (`x`) at given time
step `tstp`.
"""
function smul!(x::Element{T}, lmix::Gˡᵐⁱˣ{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ lmix.ntime
    for i = 1:lmix.ntau
        lmix.data[tstp,i] = x * lmix.data[tstp,i]
    end
end

"""
    smul!(lmix::Gˡᵐⁱˣ{T}, x::Element{T}, tstp::I64)

Right multiply a `Gˡᵐⁱˣ` with given weight (`x`) at given time
step `tstp`.
"""
function smul!(lmix::Gˡᵐⁱˣ{T}, x::Element{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ lmix.ntime
    for i = 1:lmix.ntau
        lmix.data[tstp,i] = lmix.data[tstp,i] * x
    end
end

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
### *Gˡᵉˢˢ* : *Operations*
=#

"""
    memset!(less::Gˡᵉˢˢ{T}, x)

Reset all the matrix elements of `less` to `x`. `x` should be a
scalar number.
"""
function memset!(less::Gˡᵉˢˢ{T}, x) where {T}
    cx = convert(T, x)
    for i=1:less.ntime
        for j=1:less.ntime
            fill!(less.data[j,i], cx)
        end
    end
end

"""
    memset!(less::Gˡᵉˢˢ{T}, tstp::I64, x)

Reset the matrix elements of `less` at given time step `tstp` (and at all
`t` where `t < tstp`) to `x`. `x` should be a scalar number.
"""
function memset!(less::Gˡᵉˢˢ{T}, tstp::I64, x) where {T}
    @assert 1 ≤ tstp ≤ less.ntime
    cx = convert(T, x)
    for i=1:tstp
        fill!(less.data[i,tstp], cx)
    end
end

"""
    zeros!(less::Gˡᵉˢˢ{T})

Reset all the matrix elements of `less` to `zero`.
"""
zeros!(less::Gˡᵉˢˢ{T}) where {T} = memset!(less, zero(T))

"""
    zeros!(less::Gˡᵉˢˢ{T}, tstp::I64)

Reset the matrix elements of `less` at given time step `tstp` (and at all
`t` where `t < tstp`) to `zero`.
"""
zeros!(less::Gˡᵉˢˢ{T}, tstp::I64) where {T} = memset!(less, tstp, zero(T))

"""
    memcpy!(src::Gˡᵉˢˢ{T}, dst::Gˡᵉˢˢ{T})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::Gˡᵉˢˢ{T}, dst::Gˡᵉˢˢ{T}) where {T}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memcpy!(src::Gˡᵉˢˢ{T}, dst::Gˡᵉˢˢ{T}, tstp::I64)

Copy some matrix elements from `src` to `dst`. Only the matrix elements
at given time step `tstp` (and at all `t` where `t < tstp`) are copied.
"""
function memcpy!(src::Gˡᵉˢˢ{T}, dst::Gˡᵉˢˢ{T}, tstp::I64) where {T}
    @assert iscompatible(src, dst)
    @assert 1 ≤ tstp ≤ src.ntime
    for i=1:tstp
        dst.data[i,tstp] = copy(src.data[i,tstp])
    end
end

"""
    incr!(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}, tstp::I64, α::T)

Add a `Gˡᵉˢˢ` with given weight (`α`) at given time step `tstp` (and at
all `t` where `t < tstp`) to another `Gˡᵉˢˢ`.
"""
function incr!(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}, tstp::I64, α::T) where {T}
    @assert iscompatible(less1, less2)
    @assert 1 ≤ tstp ≤ less2.ntime
    for i = 1:tstp
        @. less1.data[i,tstp] = less1.data[i,tstp] + less2.data[i,tstp] * α
    end
end

"""
    smul!(less::Gˡᵉˢˢ{T}, tstp::I64, α::T)

Multiply a `Gˡᵉˢˢ` with given weight (`α`) at given time step `tstp` (and
at all `t` where `t < tstp`).
"""
function smul!(less::Gˡᵉˢˢ{T}, tstp::I64, α::T) where {T}
    @assert 1 ≤ tstp ≤ less.ntime
    for i = 1:tstp
        @. less.data[i,tstp] = less.data[i,tstp] * α
    end
end

"""
    smul!(x::Cf{T}, less::Gˡᵉˢˢ{T}, tstp::I64)

Left multiply a `Gˡᵉˢˢ` with given weight (`x`) at given time step `tstp`
(and at all `t` where `t < tstp`).
"""
function smul!(x::Cf{T}, less::Gˡᵉˢˢ{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ less.ntime
    for i = 1:tstp
        less.data[i,tstp] = x[i] * less.data[i,tstp]
    end
end

"""
    smul!(less::Gˡᵉˢˢ{T}, x::Element{T}, tstp::I64)

Right multiply a `Gˡᵉˢˢ` with given weight (`x`) at given time step `tstp`
(and at all `t` where `t < tstp`).
"""
function smul!(less::Gˡᵉˢˢ{T}, x::Element{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ less.ntime
    for i = 1:tstp
        less.data[i,tstp] = less.data[i,tstp] * x
    end
end

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
### *gᵐᵃᵗ* : *Indexing*
=#

"""
    Base.getindex(mat::gᵐᵃᵗ{S}, ind::I64)

Visit the element stored in `gᵐᵃᵗ` object.
"""
function Base.getindex(mat::gᵐᵃᵗ{S}, ind::I64) where {S}
    # Sanity check
    @assert 1 ≤ ind ≤ mat.ntau

    # Return G^{M}(τᵢ)
    mat.data[ind]
end

"""
    Base.setindex!(mat::gᵐᵃᵗ{S}, x::Element{S}, ind::I64)

Setup the element in `gᵐᵃᵗ` object.
"""
function Base.setindex!(mat::gᵐᵃᵗ{S}, x::Element{S}, ind::I64) where {S}
    # Sanity check
    @assert size(x) == getdims(mat)
    @assert 1 ≤ ind ≤ mat.ntau

    # G^{M}(τᵢ) = x
    mat.data[ind] = copy(x)
end

"""
    Base.setindex!(mat::gᵐᵃᵗ{S}, v::S, ind::I64)

Setup the element in `gᵐᵃᵗ` object.
"""
function Base.setindex!(mat::gᵐᵃᵗ{S}, v::S, ind::I64) where {S}
    # Sanity check
    @assert 1 ≤ ind ≤ mat.ntau

    # G^{M}(τᵢ) .= v
    fill!(mat.data[ind], v)
end

#=
### *gᵐᵃᵗ* : *Operations*
=#

"""
    memset!(mat::gᵐᵃᵗ{S}, x)

Reset all the vector elements of `mat` to `x`. `x` should be a
scalar number.
"""
function memset!(mat::gᵐᵃᵗ{S}, x) where {S}
    cx = convert(S, x)
    for i = 1:mat.ntau
        fill!(mat.data[i], cx)
    end
end

"""
    zeros!(mat::gᵐᵃᵗ{S})

Reset all the vector elements of `mat` to `zero`.
"""
zeros!(mat::gᵐᵃᵗ{S}) where {S} = memset!(mat, zero(S))

"""
    memcpy!(src::gᵐᵃᵗ{S}, dst::gᵐᵃᵗ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::gᵐᵃᵗ{S}, dst::gᵐᵃᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memcpy!(src::Gᵐᵃᵗ{S}, dst::gᵐᵃᵗ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::Gᵐᵃᵗ{S}, dst::gᵐᵃᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data[:,1])
end

"""
    memcpy!(src::gᵐᵃᵗ{S}, dst::Gᵐᵃᵗ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::gᵐᵃᵗ{S}, dst::Gᵐᵃᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    @. dst.data[:,1] = copy(src.data)
end

"""
    incr!(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}, α::S)

Add a `gᵐᵃᵗ` with given weight (`α`) to another `gᵐᵃᵗ`.
"""
function incr!(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}, α::S) where {S}
    @assert iscompatible(mat1, mat2)
    for i = 1:mat2.ntau
        @. mat1.data[i] = mat1.data[i] + mat2.data[i] * α
    end
end

"""
    incr!(mat1::Gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}, α::S)

Add a `gᵐᵃᵗ` with given weight (`α`) to a `Gᵐᵃᵗ`.
"""
function incr!(mat1::Gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}, α::S) where {S}
    @assert iscompatible(mat1, mat2)
    for i = 1:mat2.ntau
        @. mat1.data[i,1] = mat1.data[i,1] + mat2.data[i] * α
    end
end

"""
    incr!(mat1::gᵐᵃᵗ{S}, mat2::Gᵐᵃᵗ{S}, α::S)

Add a `Gᵐᵃᵗ` with given weight (`α`) to a `gᵐᵃᵗ`.
"""
function incr!(mat1::gᵐᵃᵗ{S}, mat2::Gᵐᵃᵗ{S}, α::S) where {S}
    @assert iscompatible(mat1, mat2)
    for i = 1:mat1.ntau
        @. mat1.data[i] = mat1.data[i] + mat2.data[i,1] * α
    end
end

"""
    smul!(mat::gᵐᵃᵗ{S}, α::S)

Multiply a `gᵐᵃᵗ` with given weight (`α`).
"""
function smul!(mat::gᵐᵃᵗ{S}, α::S) where {S}
    for i = 1:mat.ntau
        @. mat.data[i] = mat.data[i] * α
    end
end

"""
    smul!(x::Element{S}, mat::gᵐᵃᵗ{S})

Left multiply a `gᵐᵃᵗ` with given weight (`x`).
"""
function smul!(x::Element{S}, mat::gᵐᵃᵗ{S}) where {S}
    for i = 1:mat.ntau
        mat.data[i] = x * mat.data[i]
    end
end

"""
    smul!(mat::gᵐᵃᵗ{S}, x::Element{S})

Right multiply a `gᵐᵃᵗ` with given weight (`x`).
"""
function smul!(mat::gᵐᵃᵗ{S}, x::Element{S}) where {S}
    for i = 1:mat.ntau
        mat.data[i] = mat.data[i] * x
    end
end

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
### *gʳᵉᵗ* : *Indexing*
=#

"""
    Base.getindex(ret::gʳᵉᵗ{S}, j::I64)

Visit the element stored in `gʳᵉᵗ` object. Here `j` is index for
real times.
"""
function Base.getindex(ret::gʳᵉᵗ{S}, j::I64) where {S}
    # Sanity check
    @assert 1 ≤ j ≤ ret.tstp

    # Return G^{R}(tᵢ ≡ tstp, tⱼ)
    ret.data[j]
end

"""
    Base.getindex(ret::gʳᵉᵗ{S}, i::I64, tstp::I64)

Visit the element stored in `gʳᵉᵗ` object. Here `i` is index for
real times.
"""
function Base.getindex(ret::gʳᵉᵗ{S}, i::I64, tstp::I64) where {S}
    # Sanity check
    @assert tstp == ret.tstp
    @assert 1 ≤ i ≤ ret.tstp

    # Return G^{R}(tᵢ, tⱼ ≡ tstp)
    -(ret.data[j])'
end

"""
    Base.setindex!(ret::gʳᵉᵗ{S}, x::Element{S}, j::I64)

Setup the element in `gʳᵉᵗ` object.
"""
function Base.setindex!(ret::gʳᵉᵗ{S}, x::Element{S}, j::I64) where {S}
    # Sanity check
    @assert size(x) == getdims(ret)
    @assert 1 ≤ j ≤ ret.tstp

    # G^{R}(tᵢ ≡ tstp, tⱼ) = x
    ret.data[j] = copy(x)
end

"""
    Base.setindex!(ret::gʳᵉᵗ{S}, v::S, j::I64)

Setup the element in `gʳᵉᵗ` object.
"""
function Base.setindex!(ret::gʳᵉᵗ{S}, v::S, j::I64) where {S}
    # Sanity check
    @assert 1 ≤ j ≤ ret.tstp

    # G^{R}(tᵢ ≡ tstp, tⱼ) .= v
    fill!(ret.data[j], v)
end

#=
### *gʳᵉᵗ* : *Operations*
=#

"""
    memset!(ret::gʳᵉᵗ{S}, x)

Reset all the vector elements of `ret` to `x`. `x` should be a
scalar number.
"""
function memset!(ret::gʳᵉᵗ{S}, x) where {S}
    cx = convert(T, x)
    for i=1:ret.tstp
        fill!(ret.data[i], cx)
    end
end

"""
    zeros!(ret::gʳᵉᵗ{S})

Reset all the vector elements of `ret` to `zero`.
"""
zeros!(ret::gʳᵉᵗ{S}) where {S} = memset!(ret, zero(S))

"""
    memcpy!(src::gʳᵉᵗ{S}, dst::gʳᵉᵗ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::gʳᵉᵗ{S}, dst::gʳᵉᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memcpy!(src::Gʳᵉᵗ{S}, dst::gʳᵉᵗ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::Gʳᵉᵗ{S}, dst::gʳᵉᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    tstp = dst.tstp
    @. dst.data = copy(src.data[tstp,1:tstp])
end

"""
    memcpy!(src::gʳᵉᵗ{S}, dst::Gʳᵉᵗ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::gʳᵉᵗ{S}, dst::Gʳᵉᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    tstp = src.tstp
    @. dst.data[tstp,1:tstp] = copy(src.data)
end

"""
    incr!(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}, α::S)

Add a `gʳᵉᵗ` with given weight (`α`) to another `gʳᵉᵗ`.
"""
function incr!(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}, α::S) where {S}
    @assert iscompatible(ret1, ret2)
    tstp = ret2.tstp
    for i = 1:tstp
        @. ret1.data[i] = ret1.data[i] + ret2.data[i] * α
    end
end

"""
    incr!(ret1::Gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}, α::S)

Add a `gʳᵉᵗ` with given weight (`α`) to a `Gʳᵉᵗ`.
"""
function incr!(ret1::Gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}, α::S) where {S}
    @assert iscompatible(ret1, ret2)
    tstp = ret2.tstp
    for i = 1:tstp
        @. ret1.data[tstp,i] = ret1.data[tstp,i] + ret2.data[i] * α
    end
end

"""
    incr!(ret1::gʳᵉᵗ{S}, ret2::Gʳᵉᵗ{S}, α::S)

Add a `Gʳᵉᵗ` with given weight (`α`) to a `gʳᵉᵗ`.
"""
function incr!(ret1::gʳᵉᵗ{S}, ret2::Gʳᵉᵗ{S}, α::S) where {S}
    @assert iscompatible(ret1, ret2)
    tstp = ret1.tstp
    for i = 1:tstp
        @. ret1.data[i] = ret1.data[i] + ret2.data[tstp,i] * α
    end
end

"""
    smul!(ret::gʳᵉᵗ{S}, α::S)

Multiply a `gʳᵉᵗ` with given weight (`α`).
"""
function smul!(ret::gʳᵉᵗ{S}, α::S) where {S}
    for i = 1:ret.tstp
        @. ret.data[i] = ret.data[i] * α
    end
end

"""
    smul!(x::Element{S}, ret::gʳᵉᵗ{S})

Left multiply a `gʳᵉᵗ` with given weight (`x`).
"""
function smul!(x::Element{S}, ret::gʳᵉᵗ{S}) where {S}
    for i = 1:ret.tstp
        ret.data[i] = x * ret.data[i]
    end
end

"""
    smul!(ret::gʳᵉᵗ{S}, x::Cf{S})

Right multiply a `gʳᵉᵗ` with given weight (`x`).
"""
function smul!(ret::gʳᵉᵗ{S}, x::Cf{S}) where {S}
    for i = 1:ret.tstp
        ret.data[i] = ret.data[i] * x[i]
    end
end

#=
### *gʳᵉᵗ* : *Traits*
=#

"""
    Base.:+(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S})

Operation `+` for two `gʳᵉᵗ` objects.
"""
function Base.:+(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}) where {S}
    # Sanity check
    @assert getsize(ret1) == getsize(ret2)
    @assert getdims(ret1) == getdims(ret2)

    gʳᵉᵗ(ret1.type, ret1.tstp, ret1.ndim1, ret1.ndim2, ret1.data + ret2.data)
end

"""
    Base.:-(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S})

Operation `-` for two `gʳᵉᵗ` objects.
"""
function Base.:-(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}) where {S}
    # Sanity check
    @assert getsize(ret1) == getsize(ret2)
    @assert getdims(ret1) == getdims(ret2)

    gʳᵉᵗ(ret1.type, ret1.tstp, ret1.ndim1, ret1.ndim2, ret1.data - ret2.data)
end

"""
    Base.:*(ret::gʳᵉᵗ{S}, x)

Operation `*` for a `gʳᵉᵗ` object and a scalar value.
"""
function Base.:*(ret::gʳᵉᵗ{S}, x) where {S}
    cx = convert(S, x)
    gʳᵉᵗ(ret.type, ret.tstp, ret.ndim1, ret.ndim2, ret.data * cx)
end

"""
    Base.:*(x, ret::gʳᵉᵗ{S})

Operation `*` for a scalar value and a `gʳᵉᵗ` object.
"""
Base.:*(x, ret::gʳᵉᵗ{S}) where {S} = Base.:*(ret, x)

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
