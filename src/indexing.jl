#=
### *Gᵐᵃᵗ* : *Indexing*
=#

"""
    Base.getindex(mat::Gᵐᵃᵗ{T}, ind::I64)

Visit the element stored in `Gᵐᵃᵗ` object.
"""
function Base.getindex(mat::Gᵐᵃᵗ{T}, ind::I64) where {T}
    # Sanity check
    @assert 1 ≤ ind ≤ mat.ntau

    # Return G^{M}(τᵢ ≥ 0)
    mat.data[ind,1]
end

"""
    Base.setindex!(mat::Gᵐᵃᵗ{T}, x::Element{T}, ind::I64)

Setup the element in `Gᵐᵃᵗ` object.
"""
function Base.setindex!(mat::Gᵐᵃᵗ{T}, x::Element{T}, ind::I64) where {T}
    # Sanity check
    @assert size(x) == getdims(mat)
    @assert 1 ≤ ind ≤ mat.ntau

    # G^{M}(τᵢ) = x
    mat.data[ind,1] = copy(x)
end

"""
    Base.setindex!(mat::Gᵐᵃᵗ{T}, v::T, ind::I64)

Setup the element in `Gᵐᵃᵗ` object.
"""
function Base.setindex!(mat::Gᵐᵃᵗ{T}, v::T, ind::I64) where {T}
    # Sanity check
    @assert 1 ≤ ind ≤ mat.ntau

    # G^{M}(τᵢ) .= v
    fill!(mat.data[ind,1], v)
end

#=
### *Gʳᵉᵗ* : *Indexing*
=#

#=
*Remarks* :

In principle, when ``t < t'``, ``G^{R}(t,t') \equiv 0``. Here, we assume
that the modified retarded component also fulfills the following hermitian
conjugate relation:

```math
\begin{equation}
\tilde{G}^{R}(t,t') = - \tilde{G}^{R}(t',t)^{*}
\end{equation}
```

See [`NESSi`] Eq.~(20) for more details.
=#

"""
    Base.getindex(ret::Gʳᵉᵗ{T}, i::I64, j::I64)

Visit the element stored in `Gʳᵉᵗ` object. Here `i` and `j` are indices
for real times.
"""
function Base.getindex(ret::Gʳᵉᵗ{T}, i::I64, j::I64) where {T}
    # Sanity check
    @assert 1 ≤ i ≤ ret.ntime
    @assert 1 ≤ j ≤ ret.ntime

    # Return G^{R}(tᵢ, tⱼ)
    if i ≥ j
        ret.data[i,j]
    else
        -ret.data'[i,j]
    end
end

"""
    Base.setindex!(ret::Gʳᵉᵗ{T}, x::Element{T}, i::I64, j::I64)

Setup the element in `Gʳᵉᵗ` object.
"""
function Base.setindex!(ret::Gʳᵉᵗ{T}, x::Element{T}, i::I64, j::I64) where {T}
    # Sanity check
    @assert size(x) == getdims(ret)
    @assert 1 ≤ i ≤ ret.ntime
    @assert 1 ≤ j ≤ ret.ntime

    # G^{R}(tᵢ, tⱼ) = x
    ret.data[i,j] = copy(x)
end

"""
    Base.setindex!(ret::Gʳᵉᵗ{T}, v::T, i::I64, j::I64)

Setup the element in `Gʳᵉᵗ` object.
"""
function Base.setindex!(ret::Gʳᵉᵗ{T}, v::T, i::I64, j::I64) where {T}
    # Sanity check
    @assert 1 ≤ i ≤ ret.ntime
    @assert 1 ≤ j ≤ ret.ntime

    # G^{R}(tᵢ, tⱼ) .= v
    fill!(ret.data[i,j], v)
end

#=
### *Gˡᵐⁱˣ* : *Indexing*
=#

"""
    Base.getindex(lmix::Gˡᵐⁱˣ{T}, i::I64, j::I64)

Visit the element stored in `Gˡᵐⁱˣ` object.
"""
function Base.getindex(lmix::Gˡᵐⁱˣ{T}, i::I64, j::I64) where {T}
    # Sanity check
    @assert 1 ≤ i ≤ lmix.ntime
    @assert 1 ≤ j ≤ lmix.ntau

    # Return G^{⌉}(tᵢ, τⱼ)
    lmix.data[i,j]
end

"""
    Base.setindex!(lmix::Gˡᵐⁱˣ{T}, x::Element{T}, i::I64, j::I64)

Setup the element in `Gˡᵐⁱˣ` object.
"""
function Base.setindex!(lmix::Gˡᵐⁱˣ{T}, x::Element{T}, i::I64, j::I64) where {T}
    # Sanity check
    @assert size(x) == getdims(lmix)
    @assert 1 ≤ i ≤ lmix.ntime
    @assert 1 ≤ j ≤ lmix.ntau

    # G^{⌉}(tᵢ, τⱼ) = x
    lmix.data[i,j] = copy(x)
end

"""
    Base.setindex!(lmix::Gˡᵐⁱˣ{T}, v::T, i::I64, j::I64)

Setup the element in `Gˡᵐⁱˣ` object.
"""
function Base.setindex!(lmix::Gˡᵐⁱˣ{T}, v::T, i::I64, j::I64) where {T}
    # Sanity check
    @assert 1 ≤ i ≤ lmix.ntime
    @assert 1 ≤ j ≤ lmix.ntau

    # G^{⌉}(tᵢ, τⱼ) .= v
    fill!(lmix.data[i,j], v)
end

#=
### *Gˡᵉˢˢ* : *Indexing*
=#

#=
*Remarks* :

Here we apply the following hermitian conjugate relation:

```math
\begin{equation}
G^{<}(t,t') = -G^{<}(t',t)^{*}
\end{equation}
```

See [`NESSi`] Eq.~(18a) for more details.
=#

"""
    Base.getindex(less::Gˡᵉˢˢ{T}, i::I64, j::I64)

Visit the element stored in `Gˡᵉˢˢ` object.
"""
function Base.getindex(less::Gˡᵉˢˢ{T}, i::I64, j::I64) where {T}
    # Sanity check
    @assert 1 ≤ i ≤ less.ntime
    @assert 1 ≤ j ≤ less.ntime

    # Return G^{<}(tᵢ, tⱼ)
    if i ≤ j
        less.data[i,j]
    else
        -less.data'[i,j]
    end
end

"""
    Base.setindex!(less::Gˡᵉˢˢ{T}, x::Element{T}, i::I64, j::I64)

Setup the element in `Gˡᵉˢˢ` object.
"""
function Base.setindex!(less::Gˡᵉˢˢ{T}, x::Element{T}, i::I64, j::I64) where {T}
    # Sanity check
    @assert size(x) == getdims(less)
    @assert 1 ≤ i ≤ less.ntime
    @assert 1 ≤ j ≤ less.ntime

    # G^{<}(tᵢ, tⱼ) = x
    less.data[i,j] = copy(x)
end

"""
    Base.setindex!(less::Gˡᵉˢˢ{T}, v::T, i::I64, j::I64)

Setup the element in `Gˡᵉˢˢ` object.
"""
function Base.setindex!(less::Gˡᵉˢˢ{T}, v::T, i::I64, j::I64) where {T}
    # Sanity check
    @assert 1 ≤ i ≤ less.ntime
    @assert 1 ≤ j ≤ less.ntime

    # G^{<}(tᵢ, tⱼ) .= v
    fill!(less.data[i,j], v)
end

#=
### *Gᵐᵃᵗᵐ* : *Indexing*
=#

"""
    Base.getindex(matm::Gᵐᵃᵗᵐ{T}, ind::I64)

Visit the element stored in `Gᵐᵃᵗᵐ` object.
"""
function Base.getindex(matm::Gᵐᵃᵗᵐ{T}, ind::I64) where {T}
    # Sanity check
    @assert 1 ≤ ind ≤ matm.ntau

    # Return G^{M}(τᵢ < 0)
    matm.dataM[][matm.ntau - ind + 1] * matm.sign
end

#=
### *Gᵃᵈᵛ* : *Indexing*
=#

"""
    Base.getindex(adv::Gᵃᵈᵛ{T}, ind::I64)

Visit the element stored in `Gᵃᵈᵛ` object.
"""
function Base.getindex(adv::Gᵃᵈᵛ{T}, ind::I64) where {T}
    sorry()
end

#=
### *Gʳᵐⁱˣ* : *Indexing*
=#

"""
    Base.getindex(rmix::Gʳᵐⁱˣ{T}, i::I64, j::I64)

Visit the element stored in `Gʳᵐⁱˣ` object.
"""
function Base.getindex(rmix::Gʳᵐⁱˣ{T}, i::I64, j::I64) where {T}
    # Sanity check
    @assert 1 ≤ i ≤ rmix.ntau
    @assert 1 ≤ j ≤ rmix.ntime

    # Return G^{⌈}(τᵢ, tⱼ)
    (rmix.dataL[])[j,rmix.ntau - i + 1]' * (-rmix.sign)
end

#=
### *Gᵍᵗʳ* : *Indexing*
=#

"""
    Base.getindex(gtr::Gᵍᵗʳ{T}, i::I64, j::I64)

Visit the element stored in `Gᵍᵗʳ` object.
"""
function Base.getindex(gtr::Gᵍᵗʳ{T}, i::I64, j::I64) where {T}
    # Sanity check
    @assert 1 ≤ i ≤ gtr.ntime
    @assert 1 ≤ j ≤ gtr.ntime

    # Return G^{>}(tᵢ, tⱼ)
    gtr.dataL[][i,j] + gtr.dataR[][i,j]
end

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