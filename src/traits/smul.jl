#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

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
    smul!(G::HermMatrixNESS{T}, α, dom::FFTDomain)

Multiply the retarded and lesser data in the domain `dom` by `α`.
"""
function smul!(G::HermMatrixNESS{T}, α, dom::FFTDomain) where {T}
    smul!(G.ret, α, dom)
    smul!(G.les, α, dom)
    return G
end


"""
    smul!(x::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView}, α::Number)

Multiply the retarded and lesser components of a moving time step by
the scalar `α`.
"""
function smul!(x::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView}, α::Number)
    a = C64(α)
    for c in (:ret, :less)
        D = _mrows(x, c)
        @inbounds for i in eachindex(D)
            D[i] = a * D[i]
        end
    end
    return x
end


"""
    smul!(f::Cfᵐᵒᵛ{T}, x::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView})
    smul!(f::Cfᵐᵒᵛ{T}, x::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView}, ::Val{:adjoint})

Left multiplication of the leading time step of a moving Green's
function, ``G(t, t - \\delta_j) \\leftarrow f(t) G(t, t - \\delta_j)``,
where ``f(t)`` is the slot 0 of `f`. With `Val(:adjoint)`,
``f(t)^{\\dagger}`` is used. The time step `x` must be the leading row
(the time step of the slot 0 of `f`).

*References* :

Please see [`NESSi`] `herm_matrix_timestep_moving_view::left_multiply`
and `left_multiply_hermconj`.
"""
function smul!(f::Cfᵐᵒᵛ{T}, x::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView}) where {T}
    _moving_lmul!(f, x, false)
end
#

function smul!(f::Cfᵐᵒᵛ{T}, x::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView}, ::Val{:adjoint}) where {T}
    _moving_lmul!(f, x, true)
end


"""
    smul!(x::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView}, f::Cfᵐᵒᵛ{T})
    smul!(x::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView}, f::Cfᵐᵒᵛ{T}, ::Val{:adjoint})

Right multiplication of the leading time step of a moving Green's
function, ``G(t, t - \\delta_j) \\leftarrow G(t, t - \\delta_j)
f(t - \\delta_j)``, where ``f(t - \\delta_j)`` is the slot `deltj` of `f`.
With `Val(:adjoint)`, ``f(t - \\delta_j)^{\\dagger}`` is used.

*References* :

Please see [`NESSi`] `herm_matrix_timestep_moving_view::right_multiply`
and `right_multiply_hermconj`.
"""
function smul!(x::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView}, f::Cfᵐᵒᵛ{T}) where {T}
    _moving_rmul!(x, f, false)
end
#

function smul!(x::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView}, f::Cfᵐᵒᵛ{T}, ::Val{:adjoint}) where {T}
    _moving_rmul!(x, f, true)
end


"""
    _check_mfunc(x, f::Cfᵐᵒᵛ)

Check that the single-time function `f` fits to the moving time step `x`:
initialized, the same matrix size, the window large enough, and the
leading time steps agree. It is a private function.
"""
function _check_mfunc(x, f::Cfᵐᵒᵛ)
    _check_initialized(f)
    getdims(x) == getdims(f) ||
        throw(DimensionMismatch("the matrix sizes differ"))
    f.tc ≥ getsize(x) ||
        throw(ArgumentError("the cut-off of f must be ≥ the one of the time step"))
    gettstp(x) == f.tstp ||
        throw(ArgumentError("the time step $(gettstp(x)) is not the leading step $(f.tstp) of f"))
    return nothing
end


"""
    _moving_lmul!(f::Cfᵐᵒᵛ, x, adj::Bool)

Left multiplication by `f(t)` or `f(t)^†`. It is a private function.
"""
function _moving_lmul!(f::Cfᵐᵒᵛ, x, adj::Bool)
    _check_mfunc(x, f)
    F = adj ? Matrix(_fslot(f, 0)') : Matrix(_fslot(f, 0))
    tmp = similar(F)
    for c in (:ret, :less)
        D = _mrows(x, c)
        for j = 1:size(D, 3)
            B = view(D, :, :, j)
            mul!(tmp, F, B)
            copyto!(B, tmp)
        end
    end
    return x
end


"""
    _moving_rmul!(x, f::Cfᵐᵒᵛ, adj::Bool)

Right multiplication by `f(t - δⱼ)` or `f(t - δⱼ)^†`. It is a private
function.
"""
function _moving_rmul!(x, f::Cfᵐᵒᵛ, adj::Bool)
    _check_mfunc(x, f)
    d = getdims(x)
    tmp = zeros(C64, d)
    Fj = zeros(C64, d)
    for c in (:ret, :less)
        D = _mrows(x, c)
        for j = 1:size(D, 3)
            S = _fslot(f, j - 1)
            adj ? adjoint!(Fj, S) : copyto!(Fj, S)
            B = view(D, :, :, j)
            mul!(tmp, B, Fj)
            copyto!(B, tmp)
        end
    end
    return x
end

