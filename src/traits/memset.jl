#
# Project : Lavender
# Source  : memset.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

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
    memset!(pgf::ℱᵖˢ{T}, x)

Reset all the matrix elements of `pgf` to `x`. `x` should be a scalar
number.

See also: [`ℱᵖˢ`](@ref).
"""
function memset!(pgf::ℱᵖˢ{T}, x) where {T}
    memset!(pgf.mat, x)
    memset!(pgf.ret, x)
    memset!(pgf.lmix, x)
    memset!(pgf.less, x)
end

"""
    memset!(pgf::ℱᵖˢ{T}, tstp::I64, x)

Reset the matrix elements of `pgf` at given time step `tstp` to `x`. If
`tstp = 0`, only the Matsubara component (`mat`) is changed. Otherwise,
the `ret`, `lmix`, and `less` components are changed.

See also: [`ℱᵖˢ`](@ref).
"""
function memset!(pgf::ℱᵖˢ{T}, tstp::I64, x) where {T}
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
    memset!(pgv::𝒻ᵖˢ{S}, x)

Reset all the matrix elements of `pgv` to `x`.

See also: [`𝒻ᵖˢ`](@ref).
"""
function memset!(pgv::𝒻ᵖˢ{S}, x) where {S}
    memset!(pgv.mat, x)
    memset!(pgv.ret, x)
    memset!(pgv.lmix, x)
    memset!(pgv.less, x)
end

"""
    memset!(A::NessArray{T}, dom::FFTDomain, x)

Set all the matrix elements of the data in the domain `dom` to `x`.

See also: [`NessArray`](@ref).
"""
function memset!(A::NessArray{T}, dom::FFTDomain, x) where {T}
    fill!(_fft_data(A, dom), convert(T, x))
    return A
end

"""
    memset!(G::NessHermMat{T}, dom::FFTDomain, x)

Set all the retarded and lesser matrix elements in the domain `dom` to `x`.
"""
function memset!(G::NessHermMat{T}, dom::FFTDomain, x) where {T}
    memset!(G.ret, dom, x)
    memset!(G.les, dom, x)
    return G
end

