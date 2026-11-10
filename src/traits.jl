#
# Project : Lavender
# Source  : traits.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/12/02
#

#=
### *Cn* : *Traits*
=#

"""
    refresh!(C::Cn)

Update the `dt` and `dtau` parameters of contour.

See also: [`Cn`](@ref).
"""
function refresh!(C::Cn)
    # Sanity check
    @assert C.ntime ≥ 2
    @assert C.ntau ≥ 2

    # Evaluate `dt` and `dtau` again
    C.dt = C.tmax / ( C.ntime - 1 )
    C.dtau = C.beta / ( C.ntau - 1 )
end

#=
### *Cf* : *Traits*
=#

"""
    memcpy!(src::Cf{T}, dst::Cf{T})

Copy all the matrix elements from `src` to `dst`. It is for the `Cf`
struct only.

See also: [`Cf`](@ref).
"""
function memcpy!(src::Cf{T}, dst::Cf{T}) where {T}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memset!(cf::Cf{T}, x)

Reset all the matrix elements of `cf` to `x`. `x` should be a scalar
number. It is for the `Cf` struct only.

See also: [`Cf`](@ref).
"""
function memset!(cf::Cf{T}, x) where {T}
    cx = convert(T, x)
    for i = 1:cf.ntime + 1
        fill!(cf.data[i], cx)
    end
end

"""
    zeros!(cf::Cf{T})

Reset all the matrix elements of `cf` to `zero`. It is for the `Cf`
struct only.

See also: [`Cf`](@ref).
"""
zeros!(cf::Cf{T}) where {T} = memset!(cf, zero(T))

"""
    incr!(cf1::Cf{T}, cf2::Cf{T}, α::T)

Add a `Cf` struct (`cf2`) with the given weight (`α`) to another `Cf`
struct (`cf1`). Finally, `cf1` will be changed and `cf2` won't be changed.

See also: [`Cf`](@ref).
"""
function incr!(cf1::Cf{T}, cf2::Cf{T}, α::T) where {T}
    @assert iscompatible(cf1, cf2)
    for i = 1:cf1.ntime + 1
        @. cf1.data[i] = cf1.data[i] + cf2.data[i] * α
    end
end

"""
    smul!(cf::Cf{T}, α::T)

Multiply a `Cf` struct with the given weight (`α`).

See also: [`Cf`](@ref).
"""
function smul!(cf::Cf{T}, α::T) where {T}
    for i = 1:cf.ntime + 1
        @. cf.data[i] = cf.data[i] * α
    end
end

"""
    smul!(x::Element{T}, cf::Cf{T})

Left multiply a `Cf` struct with the given weight (`x`). `x` should be a
2D array.

See also: [`Cf`](@ref).
"""
function smul!(x::Element{T}, cf::Cf{T}) where {T}
    for i = 1:cf.ntime + 1
        cf.data[i] = x * cf.data[i]
    end
end

"""
    smul!(cf::Cf{T}, x::Element{T})

Right multiply a `Cf` struct with the given weight (`x`). `x` should be a
2D array.

See also: [`Cf`](@ref).
"""
function smul!(cf::Cf{T}, x::Element{T}) where {T}
    for i = 1:cf.ntime + 1
        cf.data[i] = cf.data[i] * x
    end
end

#=
### *Gᵐᵃᵗ* : *Traits*
=#

"""
    memcpy!(src::Gᵐᵃᵗ{T}, dst::Gᵐᵃᵗ{T})

Copy all the matrix elements from `src` to `dst`. It is for the `Gᵐᵃᵗ`
struct only.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function memcpy!(src::Gᵐᵃᵗ{T}, dst::Gᵐᵃᵗ{T}) where {T}
    @assert iscompatible(src, dst)
    copyto!(dst.data, src.data)
end

"""
    memset!(mat::Gᵐᵃᵗ{T}, x)

Reset all the matrix elements of `mat` to `x`. `x` should be a scalar
number. It is for the `Gᵐᵃᵗ` struct only.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function memset!(mat::Gᵐᵃᵗ{T}, x) where {T}
    cx = convert(T, x)
    for i = 1:mat.ntau
        fill!(_block(mat, i), cx)
    end
end

"""
    zeros!(mat::Gᵐᵃᵗ{T})

Reset all the matrix elements of `mat` to `zero`. It is for the `Gᵐᵃᵗ`
struct only.

See also: [`Gᵐᵃᵗ`](@ref).
"""
zeros!(mat::Gᵐᵃᵗ{T}) where {T} = memset!(mat, zero(T))

"""
    incr!(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T}, α::T)

Add a `Gᵐᵃᵗ` struct (`mat2`) with the given weight (`α`) to another
`Gᵐᵃᵗ` struct (`mat1`). Finally, `mat1` will be changed and `mat2` won't
be changed.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function incr!(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T}, α::T) where {T}
    @assert iscompatible(mat1, mat2)
    for i = 1:mat2.ntau
        _block(mat1, i) .+= _block(mat2, i) .* α
    end
end

"""
    smul!(mat::Gᵐᵃᵗ{T}, α::T)

Multiply a `Gᵐᵃᵗ` struct with the given weight (`α`).

See also: [`Gᵐᵃᵗ`](@ref).
"""
function smul!(mat::Gᵐᵃᵗ{T}, α::T) where {T}
    for i = 1:mat.ntau
        _block(mat, i) .*= α
    end
end

"""
    smul!(x::Element{T}, mat::Gᵐᵃᵗ{T})

Left multiply a `Gᵐᵃᵗ` struct with the given weight (`x`). `x` should be
a 2D array.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function smul!(x::Element{T}, mat::Gᵐᵃᵗ{T}) where {T}
    for i = 1:mat.ntau
        _block(mat, i) .= x * _block(mat, i)
    end
end

"""
    smul!(mat::Gᵐᵃᵗ{T}, x::Element{T})

Right multiply a `Gᵐᵃᵗ` struct with the given weight (`x`). `x` should be
a 2D array.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function smul!(mat::Gᵐᵃᵗ{T}, x::Element{T}) where {T}
    for i = 1:mat.ntau
        _block(mat, i) .= _block(mat, i) * x
    end
end

#=
### *Gʳᵉᵗ* : *Traits*
=#

"""
    memcpy!(src::Gʳᵉᵗ{T}, dst::Gʳᵉᵗ{T})

Copy all the matrix elements from `src` to `dst`. It is for the `Gʳᵉᵗ`
struct only.

See also: [`Gʳᵉᵗ`](@ref).
"""
function memcpy!(src::Gʳᵉᵗ{T}, dst::Gʳᵉᵗ{T}) where {T}
    @assert iscompatible(src, dst)
    copyto!(dst.data, src.data)
end

"""
    memcpy!(src::Gʳᵉᵗ{T}, dst::Gʳᵉᵗ{T}, tstp::I64)

Copy some matrix elements from `src` to `dst`. Only the matrix elements
at given time step `tstp` (and at all `t` where `t < tstp`) are copied.
It is for the `Gʳᵉᵗ` struct only.

See also: [`Gʳᵉᵗ`](@ref).
"""
function memcpy!(src::Gʳᵉᵗ{T}, dst::Gʳᵉᵗ{T}, tstp::I64) where {T}
    @assert iscompatible(src, dst)
    @assert 1 ≤ tstp ≤ src.ntime
    for i = 1:tstp
        _block(dst, tstp, i) .= _block(src, tstp, i)
    end
end

"""
    memset!(ret::Gʳᵉᵗ{T}, x)

Reset all the matrix elements of `ret` to `x`. `x` should be a scalar
number. It is for the `Gʳᵉᵗ` struct only.

See also: [`Gʳᵉᵗ`](@ref).
"""
function memset!(ret::Gʳᵉᵗ{T}, x) where {T}
    cx = convert(T, x)
    fill!(ret.data, cx)
end

"""
    memset!(ret::Gʳᵉᵗ{T}, tstp::I64, x)

Reset the matrix elements of `ret` at given time step `tstp` (and at all
`t` where `t < tstp`) to `x`. `x` should be a scalar number. It is for
the `Gʳᵉᵗ` struct only.

See also: [`Gʳᵉᵗ`](@ref).
"""
function memset!(ret::Gʳᵉᵗ{T}, tstp::I64, x) where {T}
    @assert 1 ≤ tstp ≤ ret.ntime
    cx = convert(T, x)
    for i = 1:tstp
        fill!(_block(ret, tstp, i), cx)
    end
end

"""
    zeros!(ret::Gʳᵉᵗ{T})

Reset all the matrix elements of `ret` to `zero`. It is for the `Gʳᵉᵗ`
struct only.

See also: [`Gʳᵉᵗ`](@ref).
"""
zeros!(ret::Gʳᵉᵗ{T}) where {T} = memset!(ret, zero(T))

"""
    zeros!(ret::Gʳᵉᵗ{T}, tstp::I64)

Reset the matrix elements of `ret` at given time step `tstp` (and at all
`t` where `t < tstp`) to `zero`. It is for the `Gʳᵉᵗ` struct only.

See also: [`Gʳᵉᵗ`](@ref).
"""
zeros!(ret::Gʳᵉᵗ{T}, tstp::I64) where {T} = memset!(ret, tstp, zero(T))

"""
    incr!(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}, tstp::I64, α::T)

Add a `Gʳᵉᵗ` struct (`ret2`) with the given weight (`α`) at given time
step `tstp` (and at all `t` where `t < tstp`) to another `Gʳᵉᵗ` struct
(`ret1`). Finally, `ret1` will be changed and `ret2` won't be changed.

See also: [`Gʳᵉᵗ`](@ref).
"""
function incr!(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}, tstp::I64, α::T) where {T}
    @assert iscompatible(ret1, ret2)
    @assert 1 ≤ tstp ≤ ret2.ntime
    for i = 1:tstp
        _block(ret1, tstp, i) .+= _block(ret2, tstp, i) .* α
    end
end

"""
    smul!(ret::Gʳᵉᵗ{T}, tstp::I64, α::T)

Multiply a `Gʳᵉᵗ` struct with the given weight (`α`) at given time step
`tstp` (and at all `t` where `t < tstp`).

See also: [`Gʳᵉᵗ`](@ref).
"""
function smul!(ret::Gʳᵉᵗ{T}, tstp::I64, α::T) where {T}
    @assert 1 ≤ tstp ≤ ret.ntime
    for i = 1:tstp
        _block(ret, tstp, i) .*= α
    end
end

"""
    smul!(x::Element{T}, ret::Gʳᵉᵗ{T}, tstp::I64)

Left multiply a `Gʳᵉᵗ` struct with the given weight (`x`) at given time
step `tstp` (and at all `t` where `t < tstp`). `x` should be a 2D array.

See also: [`Gʳᵉᵗ`](@ref).
"""
function smul!(x::Element{T}, ret::Gʳᵉᵗ{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ ret.ntime
    for i = 1:tstp
        _block(ret, tstp, i) .= x * _block(ret, tstp, i)
    end
end

"""
    smul!(ret::Gʳᵉᵗ{T}, x::Element{T}, tstp::I64)

Right multiply a `Gʳᵉᵗ` struct with the given weight (`x`) at given time
step `tstp` (and at all `t` where `t < tstp`). `x` should be a 2D array.

See also: [`Gʳᵉᵗ`](@ref).
"""
function smul!(ret::Gʳᵉᵗ{T}, x::Element{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ ret.ntime
    for i = 1:tstp
        _block(ret, tstp, i) .= _block(ret, tstp, i) * x
    end
end

"""
    smul!(x::Cf{T}, ret::Gʳᵉᵗ{T}, tstp::I64)

Left multiply a `Gʳᵉᵗ` struct with the given weight (`x`) at given time
step `tstp` (and at all `t` where `t < tstp`). `x` should be a `Cf` struct.

See also: [`Gʳᵉᵗ`](@ref).
"""
function smul!(x::Cf{T}, ret::Gʳᵉᵗ{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ ret.ntime
    @assert 1 ≤ tstp ≤ x.ntime
    for i = 1:tstp
        _block(ret, tstp, i) .= x[i] * _block(ret, tstp, i)
    end
end

"""
    smul!(ret::Gʳᵉᵗ{T}, x::Cf{T}, tstp::I64)

Right multiply a `Gʳᵉᵗ` struct with the given weight (`x`) at given time
step `tstp` (and at all `t` where `t < tstp`). `x` should be a `Cf` struct.

See also: [`Gʳᵉᵗ`](@ref).
"""
function smul!(ret::Gʳᵉᵗ{T}, x::Cf{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ ret.ntime
    @assert 1 ≤ tstp ≤ x.ntime
    for i = 1:tstp
        _block(ret, tstp, i) .= _block(ret, tstp, i) * x[i]
    end
end

#=
### *Gˡᵐⁱˣ* : *Traits*
=#

"""
    memcpy!(src::Gˡᵐⁱˣ{T}, dst::Gˡᵐⁱˣ{T})

Copy all the matrix elements from `src` to `dst`. It is for the `Gˡᵐⁱˣ`
struct only.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function memcpy!(src::Gˡᵐⁱˣ{T}, dst::Gˡᵐⁱˣ{T}) where {T}
    @assert iscompatible(src, dst)
    copyto!(dst.data, src.data)
end

"""
    memcpy!(src::Gˡᵐⁱˣ{T}, dst::Gˡᵐⁱˣ{T}, tstp::I64)

Copy some matrix elements from `src` to `dst`. Only the matrix elements
at given time step `tstp` are copied. It is for the `Gˡᵐⁱˣ` struct only.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function memcpy!(src::Gˡᵐⁱˣ{T}, dst::Gˡᵐⁱˣ{T}, tstp::I64) where {T}
    @assert iscompatible(src, dst)
    @assert 1 ≤ tstp ≤ src.ntime
    for i = 1:src.ntau
        _block(dst, tstp, i) .= _block(src, tstp, i)
    end
end

"""
    memset!(lmix::Gˡᵐⁱˣ{T}, x)

Reset all the matrix elements of `lmix` to `x`. `x` should be a scalar
number. It is for the `Gˡᵐⁱˣ` struct only.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function memset!(lmix::Gˡᵐⁱˣ{T}, x) where {T}
    cx = convert(T, x)
    fill!(lmix.data, cx)
end

"""
    memset!(lmix::Gˡᵐⁱˣ{T}, tstp::I64, x)

Reset the matrix elements of `lmix` at given time step `tstp` to `x`. `x`
should be a scalar number. It is for the `Gˡᵐⁱˣ` struct only.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function memset!(lmix::Gˡᵐⁱˣ{T}, tstp::I64, x) where {T}
    @assert 1 ≤ tstp ≤ lmix.ntime
    cx = convert(T, x)
    for i = 1:lmix.ntau
        fill!(_block(lmix, tstp, i), cx)
    end
end

"""
    zeros!(lmix::Gˡᵐⁱˣ{T})

Reset all the matrix elements of `lmix` to `zero`. It is for the `Gˡᵐⁱˣ`
struct only.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
zeros!(lmix::Gˡᵐⁱˣ{T}) where {T} = memset!(lmix, zero(T))

"""
    zeros!(lmix::Gˡᵐⁱˣ{T}, tstp::I64)

Reset the matrix elements of `lmix` at given time step `tstp` to `zero`.
It is for the `Gˡᵐⁱˣ` struct only.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
zeros!(lmix::Gˡᵐⁱˣ{T}, tstp::I64) where {T} = memset!(lmix, tstp, zero(T))

"""
    incr!(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}, tstp::I64, α::T)

Add a `Gˡᵐⁱˣ` struct (`lmix2`) with the given weight (`α`) at given time
step `tstp` to another `Gˡᵐⁱˣ` struct (`lmix1`). Finally, `lmix1` will be
changed and `lmix2` won't be changed.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function incr!(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}, tstp::I64, α::T) where {T}
    @assert iscompatible(lmix1, lmix2)
    @assert 1 ≤ tstp ≤ lmix2.ntime
    for i = 1:lmix2.ntau
        _block(lmix1, tstp, i) .+= _block(lmix2, tstp, i) .* α
    end
end

"""
    smul!(lmix::Gˡᵐⁱˣ{T}, tstp::I64, α::T)

Multiply a `Gˡᵐⁱˣ` struct with the given weight (`α`) at given time step
`tstp`.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function smul!(lmix::Gˡᵐⁱˣ{T}, tstp::I64, α::T) where {T}
    @assert 1 ≤ tstp ≤ lmix.ntime
    for i = 1:lmix.ntau
        _block(lmix, tstp, i) .*= α
    end
end

"""
    smul!(x::Element{T}, lmix::Gˡᵐⁱˣ{T}, tstp::I64)

Left multiply a `Gˡᵐⁱˣ` struct with the given weight (`x`) at given time
step `tstp`. `x` should be a 2D array.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function smul!(x::Element{T}, lmix::Gˡᵐⁱˣ{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ lmix.ntime
    for i = 1:lmix.ntau
        _block(lmix, tstp, i) .= x * _block(lmix, tstp, i)
    end
end

"""
    smul!(lmix::Gˡᵐⁱˣ{T}, x::Element{T}, tstp::I64)

Right multiply a `Gˡᵐⁱˣ` struct with the given weight (`x`) at given time
step `tstp`. `x` should be a 2D array.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function smul!(lmix::Gˡᵐⁱˣ{T}, x::Element{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ lmix.ntime
    for i = 1:lmix.ntau
        _block(lmix, tstp, i) .= _block(lmix, tstp, i) * x
    end
end

#=
### *Gˡᵉˢˢ* : *Traits*
=#

"""
    memcpy!(src::Gˡᵉˢˢ{T}, dst::Gˡᵉˢˢ{T})

Copy all the matrix elements from `src` to `dst`. It is for the `Gˡᵉˢˢ`
struct only.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function memcpy!(src::Gˡᵉˢˢ{T}, dst::Gˡᵉˢˢ{T}) where {T}
    @assert iscompatible(src, dst)
    copyto!(dst.data, src.data)
end

"""
    memcpy!(src::Gˡᵉˢˢ{T}, dst::Gˡᵉˢˢ{T}, tstp::I64)

Copy some matrix elements from `src` to `dst`. Only the matrix elements
at given time step `tstp` (and at all `t` where `t < tstp`) are copied.
It is for the `Gˡᵉˢˢ` struct only.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function memcpy!(src::Gˡᵉˢˢ{T}, dst::Gˡᵉˢˢ{T}, tstp::I64) where {T}
    @assert iscompatible(src, dst)
    @assert 1 ≤ tstp ≤ src.ntime
    for i = 1:tstp
        _block(dst, i, tstp) .= _block(src, i, tstp)
    end
end

"""
    memset!(less::Gˡᵉˢˢ{T}, x)

Reset all the matrix elements of `less` to `x`. `x` should be a scalar
number. It is for the `Gˡᵉˢˢ` struct only.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function memset!(less::Gˡᵉˢˢ{T}, x) where {T}
    cx = convert(T, x)
    fill!(less.data, cx)
end

"""
    memset!(less::Gˡᵉˢˢ{T}, tstp::I64, x)

Reset the matrix elements of `less` at given time step `tstp` (and at all
`t` where `t < tstp`) to `x`. `x` should be a scalar number. It is for
the `Gˡᵉˢˢ` struct only.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function memset!(less::Gˡᵉˢˢ{T}, tstp::I64, x) where {T}
    @assert 1 ≤ tstp ≤ less.ntime
    cx = convert(T, x)
    for i = 1:tstp
        fill!(_block(less, i, tstp), cx)
    end
end

"""
    zeros!(less::Gˡᵉˢˢ{T})

Reset all the matrix elements of `less` to `zero`. It is for the `Gˡᵉˢˢ`
struct only.

See also: [`Gˡᵉˢˢ`](@ref).
"""
zeros!(less::Gˡᵉˢˢ{T}) where {T} = memset!(less, zero(T))

"""
    zeros!(less::Gˡᵉˢˢ{T}, tstp::I64)

Reset the matrix elements of `less` at given time step `tstp` (and at all
`t` where `t < tstp`) to `zero`. It is for the `Gˡᵉˢˢ` struct only.

See also: [`Gˡᵉˢˢ`](@ref).
"""
zeros!(less::Gˡᵉˢˢ{T}, tstp::I64) where {T} = memset!(less, tstp, zero(T))

"""
    incr!(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}, tstp::I64, α::T)

Add a `Gˡᵉˢˢ` struct (`less2`) with the given weight (`α`) at given time
step `tstp` (and at all `t` where `t < tstp`) to another `Gˡᵉˢˢ` struct
(`less1`). Finally, `less1` will be changed and `less2` won't be changed.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function incr!(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}, tstp::I64, α::T) where {T}
    @assert iscompatible(less1, less2)
    @assert 1 ≤ tstp ≤ less2.ntime
    for i = 1:tstp
        _block(less1, i, tstp) .+= _block(less2, i, tstp) .* α
    end
end

"""
    smul!(less::Gˡᵉˢˢ{T}, tstp::I64, α::T)

Multiply a `Gˡᵉˢˢ` struct with the given weight (`α`) at given time step
`tstp` (and at all `t` where `t < tstp`).

See also: [`Gˡᵉˢˢ`](@ref).
"""
function smul!(less::Gˡᵉˢˢ{T}, tstp::I64, α::T) where {T}
    @assert 1 ≤ tstp ≤ less.ntime
    for i = 1:tstp
        _block(less, i, tstp) .*= α
    end
end

"""
    smul!(x::Cf{T}, less::Gˡᵉˢˢ{T}, tstp::I64)

Left multiply a `Gˡᵉˢˢ` struct with the given weight (`x`) at given time
step `tstp` (and at all `t` where `t < tstp`). `x` should be a `Cf` struct.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function smul!(x::Cf{T}, less::Gˡᵉˢˢ{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ less.ntime
    @assert 1 ≤ tstp ≤ x.ntime
    for i = 1:tstp
        _block(less, i, tstp) .= x[i] * _block(less, i, tstp)
    end
end

"""
    smul!(less::Gˡᵉˢˢ{T}, x::Cf{T}, tstp::I64)

Right multiply a `Gˡᵉˢˢ` struct with the given weight (`x`) at given time
step `tstp` (and at all `t` where `t < tstp`). `x` should be a `Cf` struct.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function smul!(less::Gˡᵉˢˢ{T}, x::Cf{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ less.ntime
    @assert 1 ≤ tstp ≤ x.ntime
    for i = 1:tstp
        _block(less, i, tstp) .= _block(less, i, tstp) * x[i]
    end
end

"""
    smul!(x::Element{T}, less::Gˡᵉˢˢ{T}, tstp::I64)

Left multiply a `Gˡᵉˢˢ` struct with the given weight (`x`) at given time
step `tstp` (and at all `t` where `t < tstp`). `x` should be a 2D array.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function smul!(x::Element{T}, less::Gˡᵉˢˢ{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ less.ntime
    for i = 1:tstp
        _block(less, i, tstp) .= x * _block(less, i, tstp)
    end
end

"""
    smul!(less::Gˡᵉˢˢ{T}, x::Element{T}, tstp::I64)

Right multiply a `Gˡᵉˢˢ` struct with the given weight (`x`) at given time
step `tstp` (and at all `t` where `t < tstp`). `x` should be a 2D array.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function smul!(less::Gˡᵉˢˢ{T}, x::Element{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ less.ntime
    for i = 1:tstp
        _block(less, i, tstp) .= _block(less, i, tstp) * x
    end
end

#=
### *gᵐᵃᵗ* : *Traits*
=#

"""
    memcpy!(src::gᵐᵃᵗ{S}, dst::gᵐᵃᵗ{S})

Copy all the vector elements from `src` to `dst`. It is for the `gᵐᵃᵗ`
struct only.

See also: [`gᵐᵃᵗ`](@ref).
"""
function memcpy!(src::gᵐᵃᵗ{S}, dst::gᵐᵃᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memcpy!(src::Gᵐᵃᵗ{S}, dst::gᵐᵃᵗ{S})

Copy all the matrix elements from `src` to `dst`. Note that `src` is a
`Gᵐᵃᵗ` struct, while `dst` is a `gᵐᵃᵗ` struct.

See also: [`gᵐᵃᵗ`](@ref).
"""
function memcpy!(src::Gᵐᵃᵗ{S}, dst::gᵐᵃᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    for i = 1:dst.ntau
        copyto!(dst.data[i], _block(src, i))
    end
end

"""
    memcpy!(src::gᵐᵃᵗ{S}, dst::Gᵐᵃᵗ{S})

Copy all the vector elements from `src` to `dst`. Note that `src` is a
`gᵐᵃᵗ` struct, while `dst` is a `Gᵐᵃᵗ` struct.

See also: [`gᵐᵃᵗ`](@ref).
"""
function memcpy!(src::gᵐᵃᵗ{S}, dst::Gᵐᵃᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    for i = 1:src.ntau
        copyto!(_block(dst, i), src.data[i])
    end
end

"""
    memset!(mat::gᵐᵃᵗ{S}, x)

Reset all the vector elements of `mat` to `x`. `x` should be a scalar
number. It is for the `gᵐᵃᵗ` struct only.

See also: [`gᵐᵃᵗ`](@ref).
"""
function memset!(mat::gᵐᵃᵗ{S}, x) where {S}
    cx = convert(S, x)
    for i = 1:mat.ntau
        fill!(mat.data[i], cx)
    end
end

"""
    zeros!(mat::gᵐᵃᵗ{S})

Reset all the vector elements of `mat` to `zero`. It is for the `gᵐᵃᵗ`
struct only.

See also: [`gᵐᵃᵗ`](@ref).
"""
zeros!(mat::gᵐᵃᵗ{S}) where {S} = memset!(mat, zero(S))

"""
    incr!(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}, α::S)

Add a `gᵐᵃᵗ` struct (`mat2`) with the given weight (`α`) to another
`gᵐᵃᵗ` struct (`mat1`). Finally, `mat1` will be changed and `mat2` won't
be changed.

See also: [`gᵐᵃᵗ`](@ref).
"""
function incr!(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}, α::S) where {S}
    @assert iscompatible(mat1, mat2)
    for i = 1:mat2.ntau
        @. mat1.data[i] = mat1.data[i] + mat2.data[i] * α
    end
end

"""
    incr!(mat1::Gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}, α::S)

Add a `gᵐᵃᵗ` struct (`mat2`) with the given weight (`α`) to another
`Gᵐᵃᵗ` struct (`mat1`). Finally, `mat1` will be changed and `mat2` won't
be changed.

See also: [`gᵐᵃᵗ`](@ref).
"""
function incr!(mat1::Gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}, α::S) where {S}
    @assert iscompatible(mat1, mat2)
    for i = 1:mat2.ntau
        _block(mat1, i) .+= mat2.data[i] .* α
    end
end

"""
    incr!(mat1::gᵐᵃᵗ{S}, mat2::Gᵐᵃᵗ{S}, α::S)

Add a `Gᵐᵃᵗ` struct (`mat2`) with the given weight (`α`) to another
`gᵐᵃᵗ` struct (`mat1`). Finally, `mat1` will be changed and `mat2` won't
be changed.

See also: [`gᵐᵃᵗ`](@ref).
"""
function incr!(mat1::gᵐᵃᵗ{S}, mat2::Gᵐᵃᵗ{S}, α::S) where {S}
    @assert iscompatible(mat1, mat2)
    for i = 1:mat1.ntau
        mat1.data[i] .+= _block(mat2, i) .* α
    end
end

"""
    smul!(mat::gᵐᵃᵗ{S}, α::S)

Multiply a `gᵐᵃᵗ` struct with the given weight (`α`).

See also: [`gᵐᵃᵗ`](@ref).
"""
function smul!(mat::gᵐᵃᵗ{S}, α::S) where {S}
    for i = 1:mat.ntau
        @. mat.data[i] = mat.data[i] * α
    end
end

"""
    smul!(x::Element{S}, mat::gᵐᵃᵗ{S})

Left multiply a `gᵐᵃᵗ` struct with the given weight (`x`). `x` should be
a 2D array.

See also: [`gᵐᵃᵗ`](@ref).
"""
function smul!(x::Element{S}, mat::gᵐᵃᵗ{S}) where {S}
    for i = 1:mat.ntau
        mat.data[i] = x * mat.data[i]
    end
end

"""
    smul!(mat::gᵐᵃᵗ{S}, x::Element{S})

Right multiply a `gᵐᵃᵗ` struct with the given weight (`x`). `x` should be
a 2D array.

See also: [`gᵐᵃᵗ`](@ref).
"""
function smul!(mat::gᵐᵃᵗ{S}, x::Element{S}) where {S}
    for i = 1:mat.ntau
        mat.data[i] = mat.data[i] * x
    end
end

#=
### *gʳᵉᵗ* : *Traits*
=#

"""
    memcpy!(src::gʳᵉᵗ{S}, dst::gʳᵉᵗ{S})

Copy all the vector elements from `src` to `dst`. It is for the `gʳᵉᵗ`
struct only.

See also: [`gʳᵉᵗ`](@ref).
"""
function memcpy!(src::gʳᵉᵗ{S}, dst::gʳᵉᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memcpy!(src::Gʳᵉᵗ{S}, dst::gʳᵉᵗ{S})

Copy part of the matrix elements from `src` to `dst`. Note that `src` is
a `Gʳᵉᵗ` struct, while `dst` is a `gʳᵉᵗ` struct. For a given `Gʳᵉᵗ(tᵢ,tⱼ)`,
only those elements with `tᵢ = tstp` and `1 ≤ tⱼ ≤ tstp` are copied.

See also: [`gʳᵉᵗ`](@ref).
"""
function memcpy!(src::Gʳᵉᵗ{S}, dst::gʳᵉᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    tstp = dst.tstp
    for i = 1:tstp
        copyto!(dst.data[i], _block(src, tstp, i))
    end
end

"""
    memcpy!(src::gʳᵉᵗ{S}, dst::Gʳᵉᵗ{S})

Copy all the vector elements from `src` to `dst`. Note that `src` is a
`gʳᵉᵗ` struct, while `dst` is a `Gʳᵉᵗ` struct. For a given `Gʳᵉᵗ(tᵢ,tⱼ)`,
only those elements with `tᵢ = tstp` and `1 ≤ tⱼ ≤ tstp` are updated.

See also: [`gʳᵉᵗ`](@ref).
"""
function memcpy!(src::gʳᵉᵗ{S}, dst::Gʳᵉᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    tstp = src.tstp
    for i = 1:tstp
        copyto!(_block(dst, tstp, i), src.data[i])
    end
end

"""
    memset!(ret::gʳᵉᵗ{S}, x)

Reset all the vector elements of `ret` to `x`. `x` should be a scalar
number. It is for the `gʳᵉᵗ` struct only.

See also: [`gʳᵉᵗ`](@ref).
"""
function memset!(ret::gʳᵉᵗ{S}, x) where {S}
    cx = convert(S, x)
    for i = 1:ret.tstp
        fill!(ret.data[i], cx)
    end
end

"""
    zeros!(ret::gʳᵉᵗ{S})

Reset all the vector elements of `ret` to `zero`. It is for the `gʳᵉᵗ`
struct only.

See also: [`gʳᵉᵗ`](@ref).
"""
zeros!(ret::gʳᵉᵗ{S}) where {S} = memset!(ret, zero(S))

"""
    incr!(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}, α::S)

Add a `gʳᵉᵗ` struct (`ret2`) with the given weight (`α`) to another
`gʳᵉᵗ` struct (`ret1`). Finally, `ret1` will be changed and `ret2` won't
be changed.

See also: [`gʳᵉᵗ`](@ref).
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

Add a `gʳᵉᵗ` struct (`ret2`) with the given weight (`α`) to another
`Gʳᵉᵗ` struct (`ret1`). Finally, `ret1` will be changed and `ret2` won't
be changed. For a given `Gʳᵉᵗ(tᵢ,tⱼ)`, only those elements with `tᵢ = tstp`
and `1 ≤ tⱼ ≤ tstp` are updated.

See also: [`gʳᵉᵗ`](@ref).
"""
function incr!(ret1::Gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}, α::S) where {S}
    @assert iscompatible(ret1, ret2)
    tstp = ret2.tstp
    for i = 1:tstp
        _block(ret1, tstp, i) .+= ret2.data[i] .* α
    end
end

"""
    incr!(ret1::gʳᵉᵗ{S}, ret2::Gʳᵉᵗ{S}, α::S)

Add a `Gʳᵉᵗ` struct (`ret2`) with the given weight (`α`) to another
`gʳᵉᵗ` struct (`ret1`). Finally, `ret1` will be changed and `ret2` won't
be changed. For a given `Gʳᵉᵗ(tᵢ,tⱼ)`, only those elements with `tᵢ = tstp`
and `1 ≤ tⱼ ≤ tstp` are utilized.

See also: [`gʳᵉᵗ`](@ref).
"""
function incr!(ret1::gʳᵉᵗ{S}, ret2::Gʳᵉᵗ{S}, α::S) where {S}
    @assert iscompatible(ret1, ret2)
    tstp = ret1.tstp
    for i = 1:tstp
        ret1.data[i] .+= _block(ret2, tstp, i) .* α
    end
end

"""
    smul!(ret::gʳᵉᵗ{S}, α::S)

Multiply a `gʳᵉᵗ` struct with the given weight (`α`).

See also: [`gʳᵉᵗ`](@ref).
"""
function smul!(ret::gʳᵉᵗ{S}, α::S) where {S}
    for i = 1:ret.tstp
        @. ret.data[i] = ret.data[i] * α
    end
end

"""
    smul!(x::Element{S}, ret::gʳᵉᵗ{S})

Left multiply a `gʳᵉᵗ` struct with the given weight (`x`). `x` should be
a 2D array.

See also: [`gʳᵉᵗ`](@ref).
"""
function smul!(x::Element{S}, ret::gʳᵉᵗ{S}) where {S}
    for i = 1:ret.tstp
        ret.data[i] = x * ret.data[i]
    end
end

"""
    smul!(ret::gʳᵉᵗ{S}, x::Element{S})

Right multiply a `gʳᵉᵗ` struct with the given weight (`x`). `x` should be
a 2D array.

See also: [`gʳᵉᵗ`](@ref).
"""
function smul!(ret::gʳᵉᵗ{S}, x::Element{S}) where {S}
    for i = 1:ret.tstp
        ret.data[i] = ret.data[i] * x
    end
end

"""
    smul!(x::Cf{S}, ret::gʳᵉᵗ{S})

Left multiply a `gʳᵉᵗ` struct with the given weight (`x`). `x` should be
a `Cf` struct.

See also: [`gʳᵉᵗ`](@ref).
"""
function smul!(x::Cf{S}, ret::gʳᵉᵗ{S}) where {S}
    @assert 1 ≤ ret.tstp ≤ x.ntime
    for i = 1:ret.tstp
        ret.data[i] = x[i] * ret.data[i]
    end
end

"""
    smul!(ret::gʳᵉᵗ{S}, x::Cf{S})

Right multiply a `gʳᵉᵗ` struct with the given weight (`x`). `x` should be
a `Cf` struct.

See also: [`gʳᵉᵗ`](@ref).
"""
function smul!(ret::gʳᵉᵗ{S}, x::Cf{S}) where {S}
    @assert 1 ≤ ret.tstp ≤ x.ntime
    for i = 1:ret.tstp
        ret.data[i] = ret.data[i] * x[i]
    end
end

#=
### *gˡᵐⁱˣ* : *Traits*
=#

"""
    memcpy!(src::gˡᵐⁱˣ{S}, dst::gˡᵐⁱˣ{S})

Copy all the vector elements from `src` to `dst`. It is for the `gˡᵐⁱˣ`
struct only.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function memcpy!(src::gˡᵐⁱˣ{S}, dst::gˡᵐⁱˣ{S}) where {S}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memcpy!(src::Gˡᵐⁱˣ{S}, dst::gˡᵐⁱˣ{S}, tstp::I64)

Copy part of the matrix elements from `src` to `dst`. Note that `src` is
a `Gˡᵐⁱˣ` struct, while `dst` is a `gˡᵐⁱˣ` struct. For a given
`Gˡᵐⁱˣ(tᵢ,τⱼ)`, only those elements with `tᵢ = tstp` are copied.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function memcpy!(src::Gˡᵐⁱˣ{S}, dst::gˡᵐⁱˣ{S}, tstp::I64) where {S}
    @assert iscompatible(src, dst)
    @assert 1 ≤ tstp ≤ src.ntime
    for i = 1:dst.ntau
        copyto!(dst.data[i], _block(src, tstp, i))
    end
end

"""
    memcpy!(src::gˡᵐⁱˣ{S}, dst::Gˡᵐⁱˣ{S}, tstp::I64)

Copy all the vector elements from `src` to `dst`. Note that `src` is a
`gˡᵐⁱˣ` struct, while `dst` is a `Gˡᵐⁱˣ` struct. For a given
`Gˡᵐⁱˣ(tᵢ,τⱼ)`, only those elements with `tᵢ = tstp` are updated.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function memcpy!(src::gˡᵐⁱˣ{S}, dst::Gˡᵐⁱˣ{S}, tstp::I64) where {S}
    @assert iscompatible(src, dst)
    @assert 1 ≤ tstp ≤ dst.ntime
    for i = 1:src.ntau
        copyto!(_block(dst, tstp, i), src.data[i])
    end
end

"""
    memset!(lmix::gˡᵐⁱˣ{S}, x)

Reset all the vector elements of `lmix` to `x`. `x` should be a scalar
number. It is for the `gˡᵐⁱˣ` struct only.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function memset!(lmix::gˡᵐⁱˣ{S}, x) where {S}
    cx = convert(S, x)
    for i = 1:lmix.ntau
        fill!(lmix.data[i], cx)
    end
end

"""
    zeros!(lmix::gˡᵐⁱˣ{S})

Reset all the vector elements of `lmix` to `zero`. It is for the `gˡᵐⁱˣ`
struct only.

See also: [`gˡᵐⁱˣ`](@ref).
"""
zeros!(lmix::gˡᵐⁱˣ{S}) where {S} = memset!(lmix, zero(S))

"""
    incr!(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}, α::S)

Add a `gˡᵐⁱˣ` struct (`lmix2`) with the given weight (`α`) to another
`gˡᵐⁱˣ` struct (`lmix1`). Finally, `lmix1` will be changed and `lmix2`
won't be changed.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function incr!(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}, α::S) where {S}
    @assert iscompatible(lmix1, lmix2)
    for i = 1:lmix2.ntau
        @. lmix1.data[i] = lmix1.data[i] + lmix2.data[i] * α
    end
end

"""
    incr!(lmix1::Gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}, tstp::I64, α::S)

Add a `gˡᵐⁱˣ` struct (`lmix2`) with the given weight (`α`) to another
`Gˡᵐⁱˣ` struct (`lmix1`). Finally, `lmix1` will be changed and `lmix2`
won't be changed. For a given `Gˡᵐⁱˣ(tᵢ,τⱼ)`, only those elements with
`tᵢ = tstp` are updated.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function incr!(lmix1::Gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}, tstp::I64, α::S) where {S}
    @assert iscompatible(lmix1, lmix2)
    @assert 1 ≤ tstp ≤ lmix1.ntime
    for i = 1:lmix2.ntau
        _block(lmix1, tstp, i) .+= lmix2.data[i] .* α
    end
end

"""
    incr!(lmix1::gˡᵐⁱˣ{S}, lmix2::Gˡᵐⁱˣ{S}, tstp::I64, α::S)

Add a `Gˡᵐⁱˣ` struct (`lmix2`) with the given weight (`α`) to another
`gˡᵐⁱˣ` struct (`lmix1`). Finally, `lmix1` will be changed and `lmix2`
won't be changed. For a given `Gˡᵐⁱˣ(tᵢ,τⱼ)`, only those elements with
`tᵢ = tstp` are utilized.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function incr!(lmix1::gˡᵐⁱˣ{S}, lmix2::Gˡᵐⁱˣ{S}, tstp::I64, α::S) where {S}
    @assert iscompatible(lmix1, lmix2)
    @assert 1 ≤ tstp ≤ lmix2.ntime
    for i = 1:lmix1.ntau
        lmix1.data[i] .+= _block(lmix2, tstp, i) .* α
    end
end

"""
    smul!(lmix::gˡᵐⁱˣ{S}, α::S)

Multiply a `gˡᵐⁱˣ` struct with the given weight (`α`).

See also: [`gˡᵐⁱˣ`](@ref).
"""
function smul!(lmix::gˡᵐⁱˣ{S}, α::S) where {S}
    for i = 1:lmix.ntau
        @. lmix.data[i] = lmix.data[i] * α
    end
end

"""
    smul!(x::Element{S}, lmix::gˡᵐⁱˣ{S})

Left multiply a `gˡᵐⁱˣ` struct with the given weight (`x`). `x` should
be a 2D array.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function smul!(x::Element{S}, lmix::gˡᵐⁱˣ{S}) where {S}
    for i = 1:lmix.ntau
        lmix.data[i] = x * lmix.data[i]
    end
end

"""
    smul!(lmix::gˡᵐⁱˣ{S}, x::Element{S})

Right multiply a `gˡᵐⁱˣ` struct with the given weight (`x`). `x` should
be a 2D array.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function smul!(lmix::gˡᵐⁱˣ{S}, x::Element{S}) where {S}
    for i = 1:lmix.ntau
        lmix.data[i] = lmix.data[i] * x
    end
end

#=
### *gˡᵉˢˢ* : *Traits*
=#

"""
    memcpy!(src::gˡᵉˢˢ{S}, dst::gˡᵉˢˢ{S})

Copy all the vector elements from `src` to `dst`. It is for the `gˡᵉˢˢ`
struct only.

See also: [`gˡᵉˢˢ`](@ref).
"""
function memcpy!(src::gˡᵉˢˢ{S}, dst::gˡᵉˢˢ{S}) where {S}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memcpy!(src::Gˡᵉˢˢ{S}, dst::gˡᵉˢˢ{S})

Copy part of the matrix elements from `src` to `dst`. Note that `src` is
a `Gˡᵉˢˢ` struct, while `dst` is a `gˡᵉˢˢ` struct. For a given
`Gˡᵉˢˢ(tᵢ,tⱼ)`, only those elements with `1 ≤ tᵢ ≤ tstp` and `tⱼ = tstp`
are copied.

See also: [`gˡᵉˢˢ`](@ref).
"""
function memcpy!(src::Gˡᵉˢˢ{S}, dst::gˡᵉˢˢ{S}) where {S}
    @assert iscompatible(src, dst)
    tstp = dst.tstp
    for i = 1:tstp
        copyto!(dst.data[i], _block(src, i, tstp))
    end
end

"""
    memcpy!(src::gˡᵉˢˢ{S}, dst::Gˡᵉˢˢ{S})

Copy all the vector elements from `src` to `dst`. Note that `src` is a
`gˡᵉˢˢ` struct, while `dst` is a `Gˡᵉˢˢ` struct. For a given
`Gˡᵉˢˢ(tᵢ,tⱼ)`, only those elements with `1 ≤ tᵢ ≤ tstp` and `tⱼ = tstp`
are updated.

See also: [`gˡᵉˢˢ`](@ref).
"""
function memcpy!(src::gˡᵉˢˢ{S}, dst::Gˡᵉˢˢ{S}) where {S}
    @assert iscompatible(src, dst)
    tstp = src.tstp
    for i = 1:tstp
        copyto!(_block(dst, i, tstp), src.data[i])
    end
end

"""
    memset!(less::gˡᵉˢˢ{S}, x)

Reset all the vector elements of `less` to `x`. `x` should be a scalar
number. It is for the `gˡᵉˢˢ` struct only.

See also: [`gˡᵉˢˢ`](@ref).
"""
function memset!(less::gˡᵉˢˢ{S}, x) where {S}
    cx = convert(S, x)
    for i = 1:less.tstp
        fill!(less.data[i], cx)
    end
end

"""
    zeros!(less::gˡᵉˢˢ{S})

Reset all the vector elements of `less` to `zero`. It is for the `gˡᵉˢˢ`
struct only.

See also: [`gˡᵉˢˢ`](@ref).
"""
zeros!(less::gˡᵉˢˢ{S}) where {S} = memset!(less, zero(S))

"""
    incr!(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}, α::S)

Add a `gˡᵉˢˢ` struct (`less2`) with the given weight (`α`) to another
`gˡᵉˢˢ` struct (`less1`). Finally, `less1` will be changed and `less2`
won't be changed.

See also: [`gˡᵉˢˢ`](@ref).
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

Add a `gˡᵉˢˢ` struct (`less2`) with the given weight (`α`) to another
`Gˡᵉˢˢ` struct (`less1`). Finally, `less1` will be changed and `less2`
won't be changed. For a given `Gˡᵉˢˢ(tᵢ,tⱼ)`, only those elements with
`1 ≤ tᵢ ≤ tstp` and `tⱼ = tstp` are updated.

See also: [`gˡᵉˢˢ`](@ref).
"""
function incr!(less1::Gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}, α::S) where {S}
    @assert iscompatible(less1, less2)
    tstp = less2.tstp
    for i = 1:tstp
        _block(less1, i, tstp) .+= less2.data[i] .* α
    end
end

"""
    incr!(less1::gˡᵉˢˢ{S}, less2::Gˡᵉˢˢ{S}, α::S)

Add a `Gˡᵉˢˢ` struct (`less2`) with the given weight (`α`) to another
`gˡᵉˢˢ` struct (`less1`). Finally, `less1` will be changed and `less2`
won't be changed. For a given `Gˡᵉˢˢ(tᵢ,tⱼ)`, only those elements with
`1 ≤ tᵢ ≤ tstp` and `tⱼ = tstp` are utilized.

See also: [`gˡᵉˢˢ`](@ref).
"""
function incr!(less1::gˡᵉˢˢ{S}, less2::Gˡᵉˢˢ{S}, α::S) where {S}
    @assert iscompatible(less1, less2)
    tstp = less1.tstp
    for i = 1:tstp
        less1.data[i] .+= _block(less2, i, tstp) .* α
    end
end

"""
    smul!(less::gˡᵉˢˢ{S}, α::S)

Multiply a `gˡᵉˢˢ` struct with the given weight (`α`).

See also: [`gˡᵉˢˢ`](@ref).
"""
function smul!(less::gˡᵉˢˢ{S}, α::S) where {S}
    for i = 1:less.tstp
        @. less.data[i] = less.data[i] * α
    end
end

"""
    smul!(x::Cf{S}, less::gˡᵉˢˢ{S})

Left multiply a `gˡᵉˢˢ` struct with the given weight (`x`). `x` should
be a 2D array.

See also: [`gˡᵉˢˢ`](@ref).
"""
function smul!(x::Cf{S}, less::gˡᵉˢˢ{S}) where {S}
    for i = 1:less.tstp
        less.data[i] = x[i] * less.data[i]
    end
end

"""
    smul!(less::gˡᵉˢˢ{S}, x::Element{S})

Right multiply a `gˡᵉˢˢ` struct with the given weight (`x`). `x` should
be a 2D array.

See also: [`gˡᵉˢˢ`](@ref).
"""
function smul!(less::gˡᵉˢˢ{S}, x::Element{S}) where {S}
    for i = 1:less.tstp
        less.data[i] = less.data[i] * x
    end
end

#=
### *ℱ* : *Traits*
=#

"""
    memcpy!(src::ℱ{T}, dst::ℱ{T}) where {T}

Copy full standard contour-ordered Green's function.

See also: [`ℱ`](@ref).
"""
function memcpy!(src::ℱ{T}, dst::ℱ{T}) where {T}
    memcpy!(src.mat, dst.mat)
    memcpy!(src.ret, dst.ret)
    memcpy!(src.lmix, dst.lmix)
    memcpy!(src.less, dst.less)
end

"""
    memcpy!(src::ℱ{T}, dst::ℱ{T}, tstp::I64)

Copy standard contour-ordered Green's function at given time step `tstp`.
Note that `tstp = 0` denotes the equilibrium state, so this function will
copy the Matsubara component only (`mat`). However, when `tstp > 0`, the
`ret`, `lmix`, and `less` components will be copied. It is for the `ℱ`
struct only.

See also: [`ℱ`](@ref).
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
    memset!(cfm::ℱ{T}, x)

Reset all the matrix elements of `cfm` to `x`. `x` should be a scalar
number. It is for the `ℱ` struct only.

See also: [`ℱ`](@ref).
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
components will be changed. It is for the `ℱ` struct only.

See also: [`ℱ`](@ref).
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

Reset all the matrix elements of `cfm` to `zero`. It is for the `ℱ`
struct only.

See also: [`ℱ`](@ref).
"""
zeros!(cfm::ℱ{T}) where {T} = memset!(cfm, zero(T))

"""
    zeros!(cfm::ℱ{T}, tstp::I64)

Reset the matrix elements of `cfm` at given time step `tstp` to `zero`.
It is for the `ℱ` struct only.

See also: [`ℱ`](@ref).
"""
zeros!(cfm::ℱ{T}, tstp::I64) where {T} = memset!(cfm, tstp, zero(T))

"""
    incr!(cfm1::ℱ{T}, cfm2::ℱ{T}, tstp::I64, α)

Adds a `ℱ` struct (`cfm2`) with the given weight (`α`) to another `ℱ`
struct (`cfm1`) at given time step `tstp`. Finally, `cfm1` will be
changed and `cfm2` won't be changed.

See also: [`ℱ`](@ref).
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

Adds a `ℱ` struct (`cfm2`) with the given weight (`α`) to another `ℱ`
struct (`cfm1`) at all possible time step `tstp`. Finally, `cfm1` will be
changed and `cfm2` won't be changed.

See also: [`ℱ`](@ref).
"""
function incr!(cfm1::ℱ{T}, cfm2::ℱ{T}, α) where {T}
    for tstp = 0:getntime(cfm2)
        incr!(cfm1, cfm2, tstp, α)
    end
end

"""
    smul!(cfm::ℱ{T}, tstp::I64, α)

Multiply a `ℱ` struct with the given weight (`α`) at given time step
`tstp`. When `tstp = 0`, only the Matsubara component (`mat`) is involved.
When `tstp > 0`, the `ret`, `lmix`, and `less` components are changed.

See also: [`ℱ`](@ref).
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

Left multiply a `ℱ` struct with the given weight (`Cf`) at given time
step `tstp`. When `tstp = 0`, only the Matsubara component (`mat`) is
involved. When `tstp > 0`, the `ret`, `lmix`, and `less` components are
changed.

See also: [`ℱ`](@ref).
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

Right multiply a `ℱ` struct with the given weight (`Cf`) at given time
step `tstp`. When `tstp = 0`, only the Matsubara component (`mat`) is
involved. When `tstp > 0`, the `ret`, `lmix`, and `less` components are
changed.

See also: [`ℱ`](@ref).
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
### *𝒻* : *Traits*
=#

"""
    memcpy!(src::𝒻{S}, dst::𝒻{S}, tstp::I64)

Extract data from a `𝒻` struct (at given time step `tstp`), then copy
them to another `𝒻` struct. When `tstp = 0`, only the Matsubara component
(`mat`) is copied. When `tstp > 0`, the `ret`, `lmix`, and `less`
component are changed.

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

Extract data from a `ℱ` struct (at given time step `tstp`), then copy
them to a `𝒻` struct. If `tstp = 0`, only the Matsubara component (`mat`)
is involved. If `tstp > 0`, the `ret`, `lmix`, and `less` components are
changed.

See also: [`𝒻`](@ref).
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

Extract data from a `𝒻` struct, then copy them to a `ℱ` struct (at given
time step `tstp`). If `tstp = 0`, only the Matsubara component (`mat`)
is involved. If `tstp > 0`, the `ret`, `lmix`, and `less` components are
changed.

See also: [`𝒻`](@ref).
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
    memset!(cfv::𝒻{S}, x)

Reset all the matrix elements of `cfv` to `x`. `x` should be a scalar
number. It is for the `𝒻` struct only.

See also: [`𝒻`](@ref).
"""
function memset!(cfv::𝒻{S}, x) where {S}
    memset!(cfv.mat, x)
    memset!(cfv.ret, x)
    memset!(cfv.lmix, x)
    memset!(cfv.less, x)
end

"""
    memset!(cfv::𝒻{S}, tstp::I64, x)

Reset all the matrix elements of `cfv` to `x`. `x` should be a scalar
number. If `tstp = 0`, only the `mat` component is updated. On the other
hand, if `tstp > 0`, the `ret`, `lmix`, and `less` components will be
updated. It is for the `𝒻` struct only.

See also: [`𝒻`](@ref).
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

Reset all the matrix elements of `cfv` to `zero`. It is for the `𝒻`
struct only.

See also: [`𝒻`](@ref).
"""
zeros!(cfv::𝒻{S}) where {S} = memset!(cfv, zero(S))

"""
    zeros!(cfv::𝒻{S}, tstp::I64)

Reset all the matrix elements of `cfv` at given time step `tstp` to `zero`.
It is for the `𝒻` struct only.

See also: [`𝒻`](@ref).
"""
zeros!(cfv::𝒻{S}, tstp::I64) where {S} = memset!(cfv, tstp, zero(S))

"""
    incr!(cfv1::𝒻{S}, cfv2::𝒻{S}, tstp::I64, α)

Adds a `𝒻` struct (`cfv2`) with the given weight (`α`) to another `𝒻`
struct (`cfv1`) at given time step `tstp`. Finally, `cfv1` will be
changed and `cfv2` won't be changed. If `tstp = 0`, only the Matsubara
component (`mat`) will be updated. If `tstp > 0`, the `ret`, `lmix`, and
`less` components will be updated.

See also: [`𝒻`](@ref).
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

Adds a `𝒻` struct (`cfv`) with the given weight (`α`) to another `ℱ`
struct (`cfm`) at given time step `tstp`. Finally, `cfm` will be changed
and `cfv` won't be changed. If `tstp = 0`, only the Matsubara component
(`mat`) will be updated. If `tstp > 0`, the `ret`, `lmix`, and `less`
components will be updated.

See also: [`𝒻`](@ref).
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

Adds a `ℱ` struct (`cfm`) with the given weight (`α`) to another `𝒻`
struct (`cfv`) at given time step `tstp`. Finally, `cfv` will be changed
and `cfm` won't be changed. If `tstp = 0`, only the Matsubara component
(`mat`) will be updated. If `tstp > 0`, the `ret`, `lmix`, and `less`
components will be updated.

See also: [`𝒻`](@ref).
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

Multiply a `𝒻` struct with the given weight (`α`) at given time step
`tstp`. When `tstp = 0`, only the Matsubara component (`mat`) is involved.
When `tstp > 0`, the `ret`, `lmix`, and `less` components are changed.

See also: [`𝒻`](@ref).
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

Left multiply a `𝒻` struct with the given weight (`Cf`) at given time
step `tstp`. When `tstp = 0`, only the Matsubara component (`mat`) is
involved. When `tstp > 0`, the `ret`, `lmix`, and `less` components are
changed.
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

Right multiply a `𝒻` struct with the given weight (`Cf`) at given time
step `tstp`. When `tstp = 0`, only the Matsubara component (`mat`) is
involved. When `tstp > 0`, the `ret`, `lmix`, and `less` components are
changed.

See also: [`𝒻`](@ref).
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
### *PseudoGF* : *Traits*
=#

#=
*Remarks* :

The following functions act on the four stored components only. Since the
`ret` component of `PseudoGF` stores the greater component, no derived
quantity (such as ``G^{>} = G^{<} + G^{R}``) is evaluated here. See
[`NESSi`] `cntr_herm_pseudo_impl.hpp`.
=#

"""
    memcpy!(src::PseudoGF{T}, dst::PseudoGF{T})

Copy full pseudo-particle Green's function.

See also: [`PseudoGF`](@ref).
"""
function memcpy!(src::PseudoGF{T}, dst::PseudoGF{T}) where {T}
    memcpy!(src.mat, dst.mat)
    memcpy!(src.ret, dst.ret)
    memcpy!(src.lmix, dst.lmix)
    memcpy!(src.less, dst.less)
end

"""
    memcpy!(src::PseudoGF{T}, dst::PseudoGF{T}, tstp::I64)

Copy pseudo-particle Green's function at given time step `tstp`. If
`tstp = 0`, only the Matsubara component (`mat`) is copied. Otherwise,
the `ret`, `lmix`, and `less` components are copied.

See also: [`PseudoGF`](@ref).
"""
function memcpy!(src::PseudoGF{T}, dst::PseudoGF{T}, tstp::I64) where {T}
    @assert 0 ≤ tstp ≤ getntime(src)
    if tstp > 0
        memcpy!(src.ret, dst.ret, tstp)
        memcpy!(src.lmix, dst.lmix, tstp)
        memcpy!(src.less, dst.less, tstp)
    else
        memcpy!(src.mat, dst.mat)
    end
end

"""
    memset!(pgf::PseudoGF{T}, x)

Reset all the matrix elements of `pgf` to `x`. `x` should be a scalar
number.

See also: [`PseudoGF`](@ref).
"""
function memset!(pgf::PseudoGF{T}, x) where {T}
    memset!(pgf.mat, x)
    memset!(pgf.ret, x)
    memset!(pgf.lmix, x)
    memset!(pgf.less, x)
end

"""
    memset!(pgf::PseudoGF{T}, tstp::I64, x)

Reset the matrix elements of `pgf` at given time step `tstp` to `x`. If
`tstp = 0`, only the Matsubara component (`mat`) is changed. Otherwise,
the `ret`, `lmix`, and `less` components are changed.

See also: [`PseudoGF`](@ref).
"""
function memset!(pgf::PseudoGF{T}, tstp::I64, x) where {T}
    @assert 0 ≤ tstp ≤ getntime(pgf)
    if tstp > 0
        memset!(pgf.ret, tstp, x)
        memset!(pgf.lmix, tstp, x)
        memset!(pgf.less, tstp, x)
    else
        memset!(pgf.mat, x)
    end
end

"""
    zeros!(pgf::PseudoGF{T})

Reset all the matrix elements of `pgf` to `zero`.

See also: [`PseudoGF`](@ref).
"""
zeros!(pgf::PseudoGF{T}) where {T} = memset!(pgf, zero(T))

"""
    zeros!(pgf::PseudoGF{T}, tstp::I64)

Reset the matrix elements of `pgf` at given time step `tstp` to `zero`.

See also: [`PseudoGF`](@ref).
"""
zeros!(pgf::PseudoGF{T}, tstp::I64) where {T} = memset!(pgf, tstp, zero(T))

"""
    incr!(pgf1::PseudoGF{T}, pgf2::PseudoGF{T}, tstp::I64, α)

Adds a `PseudoGF` struct (`pgf2`) with the given weight (`α`) to another
`PseudoGF` struct (`pgf1`) at given time step `tstp`.

See also: [`PseudoGF`](@ref).
"""
function incr!(pgf1::PseudoGF{T}, pgf2::PseudoGF{T}, tstp::I64, α) where {T}
    @assert iscompatible(pgf1, pgf2)
    @assert 0 ≤ tstp ≤ getntime(pgf2)
    cα = convert(T, α)
    if tstp > 0
        incr!(pgf1.ret, pgf2.ret, tstp, cα)
        incr!(pgf1.lmix, pgf2.lmix, tstp, cα)
        incr!(pgf1.less, pgf2.less, tstp, cα)
    else
        incr!(pgf1.mat, pgf2.mat, cα)
    end
end

"""
    incr!(pgf1::PseudoGF{T}, pgf2::PseudoGF{T}, α)

Adds a `PseudoGF` struct (`pgf2`) with the given weight (`α`) to another
`PseudoGF` struct (`pgf1`) at all possible time steps.

See also: [`PseudoGF`](@ref).
"""
function incr!(pgf1::PseudoGF{T}, pgf2::PseudoGF{T}, α) where {T}
    for tstp = 0:getntime(pgf2)
        incr!(pgf1, pgf2, tstp, α)
    end
end

"""
    smul!(pgf::PseudoGF{T}, tstp::I64, α)

Multiply a `PseudoGF` struct with the given weight (`α`) at given time
step `tstp`.

See also: [`PseudoGF`](@ref).
"""
function smul!(pgf::PseudoGF{T}, tstp::I64, α) where {T}
    @assert 0 ≤ tstp ≤ getntime(pgf)
    cα = convert(T, α)
    if tstp > 0
        smul!(pgf.ret, tstp, cα)
        smul!(pgf.lmix, tstp, cα)
        smul!(pgf.less, tstp, cα)
    else
        smul!(pgf.mat, cα)
    end
end

"""
    smul!(pgf::PseudoGF{T}, α)

Multiply a `PseudoGF` struct with the given weight (`α`) at all possible
time steps.

See also: [`PseudoGF`](@ref).
"""
function smul!(pgf::PseudoGF{T}, α) where {T}
    for tstp = 0:getntime(pgf)
        smul!(pgf, tstp, α)
    end
end

"""
    smul!(cff::Cf{T}, pgf::PseudoGF{T}, tstp::I64)

Left multiply a `PseudoGF` struct with the given weight (`Cf`) at given
time step `tstp`. The `ret` (greater) and `lmix` components are multiplied
by `cff[tstp]`, while ``G^{<}(t_i,t)`` is multiplied by `cff[i]`.

See also: [`PseudoGF`](@ref).
"""
function smul!(cff::Cf{T}, pgf::PseudoGF{T}, tstp::I64) where {T}
    @assert 0 ≤ tstp ≤ getntime(pgf)
    if tstp > 0
        smul!(cff[tstp], pgf.ret, tstp)
        smul!(cff[tstp], pgf.lmix, tstp)
        smul!(cff, pgf.less, tstp)
    else
        smul!(cff[0], pgf.mat)
    end
end

"""
    smul!(pgf::PseudoGF{T}, cff::Cf{T}, tstp::I64)

Right multiply a `PseudoGF` struct with the given weight (`Cf`) at given
time step `tstp`. ``G^{>}(t,t_j)`` is multiplied by `cff[j]`, the `lmix`
component by `cff[0]`, and the `less` component by `cff[tstp]`.

See also: [`PseudoGF`](@ref).
"""
function smul!(pgf::PseudoGF{T}, cff::Cf{T}, tstp::I64) where {T}
    @assert 0 ≤ tstp ≤ getntime(pgf)
    if tstp > 0
        smul!(pgf.ret, cff, tstp)
        smul!(pgf.lmix, cff[0], tstp)
        smul!(pgf.less, cff[tstp], tstp)
    else
        smul!(pgf.mat, cff[0])
    end
end

#=
### *PseudoGFStep* : *Traits*
=#

"""
    memcpy!(src::PseudoGFStep{S}, dst::PseudoGFStep{S}, tstp::I64)

Copy a `PseudoGFStep` struct to another one at given time step `tstp`.

See also: [`PseudoGFStep`](@ref).
"""
function memcpy!(src::PseudoGFStep{S}, dst::PseudoGFStep{S}, tstp::I64) where {S}
    @assert tstp == gettstp(src) == gettstp(dst)
    if tstp > 0
        memcpy!(src.ret, dst.ret)
        memcpy!(src.lmix, dst.lmix)
        memcpy!(src.less, dst.less)
    else
        memcpy!(src.mat, dst.mat)
    end
end

"""
    memcpy!(pgf::PseudoGF{S}, pgv::PseudoGFStep{S}, tstp::I64)

Extract data from a `PseudoGF` struct (at given time step `tstp`), then
copy them to a `PseudoGFStep` struct.

See also: [`PseudoGFStep`](@ref).
"""
function memcpy!(pgf::PseudoGF{S}, pgv::PseudoGFStep{S}, tstp::I64) where {S}
    @assert tstp == gettstp(pgv)
    if tstp > 0
        memcpy!(pgf.ret, pgv.ret)
        memcpy!(pgf.lmix, pgv.lmix, tstp)
        memcpy!(pgf.less, pgv.less)
    else
        memcpy!(pgf.mat, pgv.mat)
    end
end

"""
    memcpy!(pgv::PseudoGFStep{S}, pgf::PseudoGF{S}, tstp::I64)

Extract data from a `PseudoGFStep` struct, then copy them to a `PseudoGF`
struct (at given time step `tstp`).

See also: [`PseudoGFStep`](@ref).
"""
function memcpy!(pgv::PseudoGFStep{S}, pgf::PseudoGF{S}, tstp::I64) where {S}
    @assert tstp == gettstp(pgv)
    if tstp > 0
        memcpy!(pgv.ret, pgf.ret)
        memcpy!(pgv.lmix, pgf.lmix, tstp)
        memcpy!(pgv.less, pgf.less)
    else
        memcpy!(pgv.mat, pgf.mat)
    end
end

"""
    memset!(pgv::PseudoGFStep{S}, x)

Reset all the matrix elements of `pgv` to `x`.

See also: [`PseudoGFStep`](@ref).
"""
function memset!(pgv::PseudoGFStep{S}, x) where {S}
    memset!(pgv.mat, x)
    memset!(pgv.ret, x)
    memset!(pgv.lmix, x)
    memset!(pgv.less, x)
end

"""
    zeros!(pgv::PseudoGFStep{S})

Reset all the matrix elements of `pgv` to `zero`.

See also: [`PseudoGFStep`](@ref).
"""
zeros!(pgv::PseudoGFStep{S}) where {S} = memset!(pgv, zero(S))

"""
    smul!(pgv::PseudoGFStep{S}, tstp::I64, α)

Multiply a `PseudoGFStep` struct with the given weight (`α`).

See also: [`PseudoGFStep`](@ref).
"""
function smul!(pgv::PseudoGFStep{S}, tstp::I64, α) where {S}
    @assert tstp == gettstp(pgv)
    cα = convert(S, α)
    if tstp > 0
        smul!(pgv.ret, cα)
        smul!(pgv.lmix, cα)
        smul!(pgv.less, cα)
    else
        smul!(pgv.mat, cα)
    end
end

function _submatrix_indices(G::ℱ, rows, cols)
    row_indices = rows isa Integer ? [rows] : collect(rows)
    col_indices = cols isa Integer ? [cols] : collect(cols)
    if isempty(row_indices) || isempty(col_indices)
        throw(DimensionMismatch("Orbital index selections must be non-empty."))
    end
    if !all(index -> index isa Integer, row_indices) ||
       !all(index -> index isa Integer, col_indices)
        throw(BoundsError(G, (rows, cols)))
    end

    ndim1, ndim2 = getdims(G)
    if any(index -> index < 1 || index > ndim1, row_indices) ||
       any(index -> index < 1 || index > ndim2, col_indices)
        throw(BoundsError(G, (rows, cols)))
    end
    return I64.(row_indices), I64.(col_indices)
end

"""
    submatrix(G::ℱ, rows, cols)

Create a contour-ordered Green's function containing the selected orbital
rows and columns of `G` in the requested order.

### Arguments

* `G`: Source contour-ordered Green's function.
* `rows`: Orbital row indices.
* `cols`: Orbital column indices.

### Returns

A new `ℱ` with the same sign and time grids as `G`.
"""
function submatrix(G::ℱ{T}, rows, cols) where {T}
    row_indices, col_indices = _submatrix_indices(G, rows, cols)
    ndim1 = length(row_indices)
    ndim2 = length(col_indices)
    ntime = getntime(G)
    ntau = getntau(G)
    result = ℱ(
        getsign(G),
        Gᵐᵃᵗ(ntau, ndim1, ndim2, zero(T)),
        Gʳᵉᵗ(ntime, ndim1, ndim2, zero(T)),
        Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, zero(T)),
        Gˡᵉˢˢ(ntime, ndim1, ndim2, zero(T)),
    )

    for q = 1:ntau
        result.mat[q] = G.mat[q][row_indices,col_indices]
    end
    for i = 1:ntime
        for j = 1:i
            result.ret[i,j] = G.ret[i,j][row_indices,col_indices]
        end
        for q = 1:ntau
            result.lmix[i,q] = G.lmix[i,q][row_indices,col_indices]
        end
        for j = i:ntime
            result.less[i,j] = G.less[i,j][row_indices,col_indices]
        end
    end
    return result
end

"""
    set_submatrix!(G::ℱ, rows, cols, X::ℱ)

Write every independent contour block of `X` into the selected orbital rows
and columns of `G`.

### Arguments

* `G`: Destination contour-ordered Green's function.
* `rows`: Destination orbital row indices.
* `cols`: Destination orbital column indices.
* `X`: Source contour-ordered Green's function.

### Returns

The modified destination `G`.
"""
function set_submatrix!(G::ℱ, rows, cols, X::ℱ)
    row_indices, col_indices = _submatrix_indices(G, rows, cols)
    expected_dims = (length(row_indices), length(col_indices))
    if getdims(X) != expected_dims ||
       getntime(X) != getntime(G) ||
       getntau(X) != getntau(G)
        throw(DimensionMismatch("Source dimensions and time grids must match the selected destination."))
    end

    for q = 1:getntau(G)
        G.mat[q][row_indices,col_indices] = X.mat[q]
    end
    for i = 1:getntime(G)
        for j = 1:i
            G.ret[i,j][row_indices,col_indices] = X.ret[i,j]
        end
        for q = 1:getntau(G)
            G.lmix[i,q][row_indices,col_indices] = X.lmix[i,q]
        end
        for j = i:getntime(G)
            G.less[i,j][row_indices,col_indices] = X.less[i,j]
        end
    end
    return G
end

#=
### *ℱ* and *𝒻* : *Hermitian Conjugate Multiplication*
=#

#
# Private accessors to the stored blocks at given time step `tstp` are
# defined in properties.jl (`_mat_at`, `_ret_at`, `_lmix_at`, `_less_at`).
#

"""
    smul!(cff::Cf{T}, G::Union{ℱ{T},𝒻{T}}, tstp::I64, ::Val{:adjoint}, weight::Real = 1)

Left multiply a contour-ordered Green's function by the hermitian
conjugate of a contour function at given time step `tstp`, i.e.,
``G(t,t') \\to w f(t)^{\\dagger} G(t,t')``. When `tstp = 0`, only the
Matsubara component is changed (``f(0^-)`` is used). When `tstp > 0`,
the `ret`, `lmix`, and `less` components are changed.

### Arguments
* cff    -> A contour function.
* G      -> A `ℱ` or `𝒻` object.
* tstp   -> Index of time step.
* weight -> A real weight.

### Returns
N/A
"""
function smul!(cff::Cf{T}, G::Union{ℱ{T},𝒻{T}}, tstp::I64,
               ::Val{:adjoint}, weight::Real = 1) where {T}
    # Sanity check
    _check_tstp(G, tstp)
    @assert equaldims(G)
    @assert getdims(cff) == getdims(G)
    @assert cff.ntime ≥ tstp

    w = convert(F64, weight)
    if tstp == 0
        fcc = copy(cff[0]')
        for m = 1:getntau(G)
            x = _mat_at(G, m)
            x .= w .* (fcc * x)
        end
    else
        fcc = copy(cff[tstp]')
        for j = 1:tstp
            x = _ret_at(G, tstp, j)
            x .= w .* (fcc * x)
        end
        for m = 1:getntau(G)
            x = _lmix_at(G, tstp, m)
            x .= w .* (fcc * x)
        end
        for i = 1:tstp
            x = _less_at(G, tstp, i)
            x .= w .* (cff[i]' * x)
        end
    end
end

"""
    smul!(G::Union{ℱ{T},𝒻{T}}, cff::Cf{T}, tstp::I64, ::Val{:adjoint}, weight::Real = 1)

Right multiply a contour-ordered Green's function by the hermitian
conjugate of a contour function at given time step `tstp`, i.e.,
``G(t,t') \\to w G(t,t') f(t')^{\\dagger}``. When `tstp = 0`, only the
Matsubara component is changed (``f(0^-)`` is used). When `tstp > 0`,
the `ret`, `lmix`, and `less` components are changed.

### Arguments
* G      -> A `ℱ` or `𝒻` object.
* cff    -> A contour function.
* tstp   -> Index of time step.
* weight -> A real weight.

### Returns
N/A
"""
function smul!(G::Union{ℱ{T},𝒻{T}}, cff::Cf{T}, tstp::I64,
               ::Val{:adjoint}, weight::Real = 1) where {T}
    # Sanity check
    _check_tstp(G, tstp)
    @assert equaldims(G)
    @assert getdims(cff) == getdims(G)
    @assert cff.ntime ≥ tstp

    w = convert(F64, weight)
    fcc = copy(cff[0]')
    if tstp == 0
        for m = 1:getntau(G)
            x = _mat_at(G, m)
            x .= w .* (x * fcc)
        end
    else
        for j = 1:tstp
            x = _ret_at(G, tstp, j)
            x .= w .* (x * cff[j]')
        end
        for m = 1:getntau(G)
            x = _lmix_at(G, tstp, m)
            x .= w .* (x * fcc)
        end
        fcc = copy(cff[tstp]')
        for i = 1:tstp
            x = _less_at(G, tstp, i)
            x .= w .* (x * fcc)
        end
    end
end

#=
### *Traits* : *FFT Array*
=#

"""
    memcpy!(src::FFTArray{T}, dst::FFTArray{T})

Copy the time and frequency data of `src` to `dst`.

See also: [`FFTArray`](@ref).
"""
function memcpy!(src::FFTArray{T}, dst::FFTArray{T}) where {T}
    size(src.time) == size(dst.time) ||
        throw(DimensionMismatch("src and dst must have the same size"))
    copyto!(dst.time, src.time)
    copyto!(dst.freq, src.freq)
    return dst
end

"""
    memset!(A::FFTArray{T}, dom::FFTDomain, x)

Set all the matrix elements of the data in the domain `dom` to `x`.

See also: [`FFTArray`](@ref).
"""
function memset!(A::FFTArray{T}, dom::FFTDomain, x) where {T}
    fill!(_fft_data(A, dom), convert(T, x))
    return A
end

"""
    zeros!(A::FFTArray{T}, dom::FFTDomain)

Reset the data in the domain `dom` to zero.

See also: [`FFTArray`](@ref).
"""
zeros!(A::FFTArray{T}, dom::FFTDomain) where {T} = memset!(A, dom, zero(T))

"""
    incr!(A::FFTArray{T}, B::FFTArray{T}, α, dom::FFTDomain)

Add `α B` to `A` in the domain `dom`.

*References* :

Please see [`NESSi`] `ness2_fft_impl.hpp`, `fft_array::incr`.
"""
function incr!(A::FFTArray{T}, B::FFTArray{T}, α, dom::FFTDomain) where {T}
    size(A.time) == size(B.time) ||
        throw(DimensionMismatch("A and B must have the same size"))
    X = _fft_data(A, dom)
    Y = _fft_data(B, dom)
    a = convert(T, α)
    @. X = X + a * Y
    return A
end

"""
    smul!(A::FFTArray{T}, α, dom::FFTDomain)

Multiply the data in the domain `dom` by the scalar `α`.

*References* :

Please see [`NESSi`] `ness2_fft_impl.hpp`, `fft_array::smul`.
"""
function smul!(A::FFTArray{T}, α, dom::FFTDomain) where {T}
    X = _fft_data(A, dom)
    a = convert(T, α)
    @. X = a * X
    return A
end

"""
    left_multiply!(A::FFTArray{T}, M::AbstractMatrix, dom::FFTDomain;
                   hermconj::Bool = false)

Replace the matrix of every mode in the domain `dom` by ``M A`` (or
``M^\\dagger A`` if `hermconj` is true).

*References* :

Please see [`NESSi`] `ness2_fft_decl.hpp`, `fft_array::left_multiply`.
"""
function left_multiply!(A::FFTArray{T}, M::AbstractMatrix, dom::FFTDomain;
                        hermconj::Bool = false) where {T}
    size(M) == getdims(A) ||
        throw(DimensionMismatch("M must have the size $(getdims(A))"))
    X = _fft_data(A, dom)
    L = hermconj ? Matrix{T}(M') : Matrix{T}(M)
    tmp = similar(L)
    for k in axes(X, 3)
        Z = view(X, :, :, k)
        mul!(tmp, L, Z)
        copyto!(Z, tmp)
    end
    return A
end

"""
    right_multiply!(A::FFTArray{T}, M::AbstractMatrix, dom::FFTDomain;
                    hermconj::Bool = false)

Replace the matrix of every mode in the domain `dom` by ``A M`` (or
``A M^\\dagger`` if `hermconj` is true).

*References* :

Please see [`NESSi`] `ness2_fft_decl.hpp`, `fft_array::right_multiply`.
"""
function right_multiply!(A::FFTArray{T}, M::AbstractMatrix, dom::FFTDomain;
                         hermconj::Bool = false) where {T}
    size(M) == getdims(A) ||
        throw(DimensionMismatch("M must have the size $(getdims(A))"))
    X = _fft_data(A, dom)
    R = hermconj ? Matrix{T}(M') : Matrix{T}(M)
    tmp = similar(R)
    for k in axes(X, 3)
        Z = view(X, :, :, k)
        mul!(tmp, Z, R)
        copyto!(Z, tmp)
    end
    return A
end

"""
    set_matrixelement!(A::FFTArray{T}, i₁::I64, i₂::I64, B::FFTArray{T},
                       j₁::I64, j₂::I64, dom::FFTDomain)

Copy the matrix element `(j₁, j₂)` of `B` to the matrix element `(i₁, i₂)`
of `A` for all modes in the domain `dom`. The indices are 1-based.

*References* :

Please see [`NESSi`] `ness2_fft_impl.hpp`, `fft_array::set_matrixelement`.
"""
function set_matrixelement!(A::FFTArray{T}, i₁::I64, i₂::I64, B::FFTArray{T},
                            j₁::I64, j₂::I64, dom::FFTDomain) where {T}
    getnft(A) == getnft(B) ||
        throw(DimensionMismatch("A and B must have the same number of grid points"))
    X = _fft_data(A, dom)
    Y = _fft_data(B, dom)
    for k in axes(X, 3)
        X[i₁, i₂, k] = Y[j₁, j₂, k]
    end
    return A
end

#=
### *Traits* : *Steady-state Green's Function*
=#

"""
    memcpy!(src::HermMatrixNESS{T}, dst::HermMatrixNESS{T})

Copy the retarded and lesser data (both domains) of `src` to `dst`.
"""
function memcpy!(src::HermMatrixNESS{T}, dst::HermMatrixNESS{T}) where {T}
    memcpy!(src.ret, dst.ret)
    memcpy!(src.les, dst.les)
    dst.grid = src.grid
    return dst
end

"""
    memset!(G::HermMatrixNESS{T}, dom::FFTDomain, x)

Set all the retarded and lesser matrix elements in the domain `dom` to `x`.
"""
function memset!(G::HermMatrixNESS{T}, dom::FFTDomain, x) where {T}
    memset!(G.ret, dom, x)
    memset!(G.les, dom, x)
    return G
end

"""
    zeros!(G::HermMatrixNESS{T}, dom::FFTDomain)

Reset the retarded and lesser data in the domain `dom` to zero.
"""
zeros!(G::HermMatrixNESS{T}, dom::FFTDomain) where {T} = memset!(G, dom, zero(T))

"""
    incr!(A::HermMatrixNESS{T}, B::HermMatrixNESS{T}, α, dom::FFTDomain)

Add `α B` to `A` in the domain `dom`.
"""
function incr!(A::HermMatrixNESS{T}, B::HermMatrixNESS{T}, α, dom::FFTDomain) where {T}
    incr!(A.ret, B.ret, α, dom)
    incr!(A.les, B.les, α, dom)
    return A
end

"""
    smul!(G::HermMatrixNESS{T}, α, dom::FFTDomain)

Multiply the retarded and lesser data in the domain `dom` by `α`.
"""
function smul!(G::HermMatrixNESS{T}, α, dom::FFTDomain) where {T}
    smul!(G.ret, α, dom)
    smul!(G.les, α, dom)
    return G
end

"""
    left_multiply!(G::HermMatrixNESS{T}, M::AbstractMatrix, dom::FFTDomain;
                   hermconj::Bool = false)

Left multiplication of the retarded and lesser components by `M` (or
``M^\\dagger``).
"""
function left_multiply!(G::HermMatrixNESS{T}, M::AbstractMatrix, dom::FFTDomain;
                        hermconj::Bool = false) where {T}
    left_multiply!(G.ret, M, dom; hermconj = hermconj)
    left_multiply!(G.les, M, dom; hermconj = hermconj)
    return G
end

"""
    right_multiply!(G::HermMatrixNESS{T}, M::AbstractMatrix, dom::FFTDomain;
                    hermconj::Bool = false)

Right multiplication of the retarded and lesser components by `M` (or
``M^\\dagger``).
"""
function right_multiply!(G::HermMatrixNESS{T}, M::AbstractMatrix, dom::FFTDomain;
                         hermconj::Bool = false) where {T}
    right_multiply!(G.ret, M, dom; hermconj = hermconj)
    right_multiply!(G.les, M, dom; hermconj = hermconj)
    return G
end

"""
    set_matrixelement!(A::HermMatrixNESS{T}, i₁::I64, i₂::I64,
                       B::HermMatrixNESS{T}, j₁::I64, j₂::I64, dom::FFTDomain)

Copy the matrix element `(j₁, j₂)` of `B` to the element `(i₁, i₂)` of `A`
for both components. The indices are 1-based.
"""
function set_matrixelement!(A::HermMatrixNESS{T}, i₁::I64, i₂::I64,
                            B::HermMatrixNESS{T}, j₁::I64, j₂::I64,
                            dom::FFTDomain) where {T}
    set_matrixelement!(A.ret, i₁, i₂, B.ret, j₁, j₂, dom)
    set_matrixelement!(A.les, i₁, i₂, B.les, j₁, j₂, dom)
    return A
end
