#
# Project : Lavender
# Source  : incr.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

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
    incr!(A::HermMatrixNESS{T}, B::HermMatrixNESS{T}, α, dom::FFTDomain)

Add `α B` to `A` in the domain `dom`.
"""
function incr!(A::HermMatrixNESS{T}, B::HermMatrixNESS{T}, α, dom::FFTDomain) where {T}
    incr!(A.ret, B.ret, α, dom)
    incr!(A.les, B.les, α, dom)
    return A
end


"""
    incr!(dst::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ}, src::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ}, α::Number)

``dst \\leftarrow dst + α\\ src`` for the retarded and lesser components
of a moving time step. `src` and `dst` may be the same row.

*References* :

Please see [`NESSi`] `herm_matrix_timestep_moving_view::incr_timestep`.
"""
function incr!(dst::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ}, src::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ}, α::Number)
    _check_mrow(src, dst)
    a = C64(α)
    for c in (:ret, :less)
        D = _mrows(dst, c)
        S = _mrows(src, c)
        @inbounds for i in eachindex(D, S)
            D[i] = D[i] + a * S[i]
        end
    end
    return dst
end


