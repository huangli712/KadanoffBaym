#
# Project : Lavender
# Source  : traits.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/11/07
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
    @. dst.data = copy(src.data)
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
        fill!(mat.data[i,1], cx)
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
        @. mat1.data[i,1] = mat1.data[i,1] + mat2.data[i,1] * α
    end
end

"""
    smul!(mat::Gᵐᵃᵗ{T}, α::T)

Multiply a `Gᵐᵃᵗ` struct with the given weight (`α`).

See also: [`Gᵐᵃᵗ`](@ref).
"""
function smul!(mat::Gᵐᵃᵗ{T}, α::T) where {T}
    for i = 1:mat.ntau
        @. mat.data[i,1] = mat.data[i,1] * α
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
        mat.data[i,1] = x * mat.data[i,1]
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
        mat.data[i,1] = mat.data[i,1] * x
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
    @. dst.data = copy(src.data)
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
        dst.data[tstp,i] = copy(src.data[tstp,i])
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
    for i = 1:ret.ntime
        for j = 1:ret.ntime
            fill!(ret.data[j,i], cx)
        end
    end
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
        fill!(ret.data[tstp,i], cx)
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
        @. ret1.data[tstp,i] = ret1.data[tstp,i] + ret2.data[tstp,i] * α
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
        @. ret.data[tstp,i] = ret.data[tstp,i] * α
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
        ret.data[tstp,i] = x * ret.data[tstp,i]
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
        ret.data[tstp,i] = ret.data[tstp,i] * x
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
        ret.data[tstp,i] = x[i] * ret.data[tstp,i]
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
        ret.data[tstp,i] = ret.data[tstp,i] * x[i]
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
    @. dst.data = copy(src.data)
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
        dst.data[tstp,i] = copy(src.data[tstp,i])
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
    for i = 1:lmix.ntau
        for j = 1:lmix.ntime
            fill!(lmix.data[j,i], cx)
        end
    end
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
        fill!(lmix.data[tstp,i], cx)
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
        @. lmix1.data[tstp,i] = lmix1.data[tstp,i] + lmix2.data[tstp,i] * α
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
        @. lmix.data[tstp,i] = lmix.data[tstp,i] * α
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
        lmix.data[tstp,i] = x * lmix.data[tstp,i]
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
        lmix.data[tstp,i] = lmix.data[tstp,i] * x
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
    @. dst.data = copy(src.data)
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
        dst.data[i,tstp] = copy(src.data[i,tstp])
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
    for i = 1:less.ntime
        for j = 1:less.ntime
            fill!(less.data[j,i], cx)
        end
    end
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
        fill!(less.data[i,tstp], cx)
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
        @. less1.data[i,tstp] = less1.data[i,tstp] + less2.data[i,tstp] * α
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
        @. less.data[i,tstp] = less.data[i,tstp] * α
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
        less.data[i,tstp] = x[i] * less.data[i,tstp]
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
        less.data[i,tstp] = less.data[i,tstp] * x[i]
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
        less.data[i,tstp] = x * less.data[i,tstp]
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
        less.data[i,tstp] = less.data[i,tstp] * x
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
    @. dst.data = copy(src.data[:,1])
end

"""
    memcpy!(src::gᵐᵃᵗ{S}, dst::Gᵐᵃᵗ{S})

Copy all the vector elements from `src` to `dst`. Note that `src` is a
`gᵐᵃᵗ` struct, while `dst` is a `Gᵐᵃᵗ` struct.

See also: [`gᵐᵃᵗ`](@ref).
"""
function memcpy!(src::gᵐᵃᵗ{S}, dst::Gᵐᵃᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    @. dst.data[:,1] = copy(src.data)
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
        @. mat1.data[i,1] = mat1.data[i,1] + mat2.data[i] * α
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
        @. mat1.data[i] = mat1.data[i] + mat2.data[i,1] * α
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
    @. dst.data = copy(src.data[tstp,1:tstp])
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
    @. dst.data[tstp,1:tstp] = copy(src.data)
end

"""
    memset!(ret::gʳᵉᵗ{S}, x)

Reset all the vector elements of `ret` to `x`. `x` should be a scalar
number. It is for the `gʳᵉᵗ` struct only.

See also: [`gʳᵉᵗ`](@ref).
"""
function memset!(ret::gʳᵉᵗ{S}, x) where {S}
    cx = convert(T, x)
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
        @. ret1.data[tstp,i] = ret1.data[tstp,i] + ret2.data[i] * α
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
        @. ret1.data[i] = ret1.data[i] + ret2.data[tstp,i] * α
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
    @. dst.data = copy(src.data[tstp,:])
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
    @. dst.data[tstp,:] = copy(src.data)
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
        @. lmix1.data[tstp,i] = lmix1.data[tstp,i] + lmix2.data[i] * α
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
        @. lmix1.data[i] = lmix1.data[i] + lmix2.data[tstp,i] * α
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
    memset!(less::gˡᵉˢˢ{S}, x)

Reset all the matrix elements of `less` to `x`. `x` should be a
scalar number.
"""
function memset!(less::gˡᵉˢˢ{S}, x) where {S}
    cx = convert(S, x)
    for i = 1:less.tstp
        fill!(less.data[i], cx)
    end
end

"""
    zeros!(less::gˡᵉˢˢ{S})

Reset all the matrix elements of `less` to `zero`.
"""
zeros!(less::gˡᵉˢˢ{S}) where {S} = memset!(less, zero(S))

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
### *ℱ* : *Traits*
=#

"""
    memcpy!(src::ℱ{T}, dst::ℱ{T}, tstp::I64)

Copy contour-ordered Green's function at given time step `tstp`. Note that
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
### *𝒻* : *Traits*
=#

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
