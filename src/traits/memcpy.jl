#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

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
    memcpy!(src::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView}, dst::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView})

Copy the retarded and lesser components of the moving time step `src`
to `dst`.
"""
function memcpy!(src::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView}, dst::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView})
    _check_mrow(src, dst)
    for c in (:ret, :less)
        copyto!(_mrows(dst, c), _mrows(src, c))
    end
    return dst
end


"""
    memcpy!(src::ℱᵐᵒᵛ{T}, dst::ℱᵐᵒᵛ{T})

Copy the whole moving Green's function `src` (data and state) to `dst`.
Every existing view of `dst` becomes stale.
"""
function memcpy!(src::ℱᵐᵒᵛ{T}, dst::ℱᵐᵒᵛ{T}) where {T}
    (src.tc == dst.tc && getdims(src) == getdims(dst)) ||
        throw(DimensionMismatch("the moving Green's functions are not compatible"))
    copyto!(dst.ret.data, src.ret.data)
    copyto!(dst.less.data, src.less.data)
    dst.ret.head = src.ret.head
    dst.less.head = src.less.head
    dst.tstp = src.tstp
    dst.initialized = src.initialized
    dst.sign = src.sign
    dst.generation += 1
    return dst
end


