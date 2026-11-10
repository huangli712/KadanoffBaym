#
# Project : Lavender
# Source  : getindex.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Cf* : *Indexing*
=#

"""
    Base.getindex(cf::Cf{T}, i::I64)

Visit the element stored in `Cf` object. If `i = 0`, it returns the
element at Matsubara axis. On the other hand, if `i > 0`, it will
return elements at real time axis.

See also: [`Cf`](@ref).
"""
function Base.getindex(cf::Cf{T}, i::I64) where {T}
    # Sanity check
    @assert 0 ≤ i ≤ cf.ntime

    # Return 𝑓(𝑡ᵢ)
    if i == 0 # Matsubara axis
        cf.data[end]
    else      # Real time axis
        cf.data[i]
    end
end


#=
### *Gᵐᵃᵗ* : *Indexing*
=#

"""
    Base.getindex(mat::Gᵐᵃᵗ{T}, i::I64)

Visit the element stored in `Gᵐᵃᵗ` object. `i` is index for imaginary time.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function Base.getindex(mat::Gᵐᵃᵗ{T}, i::I64) where {T}
    # Sanity check
    @assert 1 ≤ i ≤ mat.ntau

    # Return G^{M}(τᵢ ≥ 0)
    _block(mat, i)
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

Visit the element stored in `Gʳᵉᵗ` object. `i` and `j` are indices for
real time.

See also: [`Gʳᵉᵗ`](@ref).
"""
function Base.getindex(ret::Gʳᵉᵗ{T}, i::I64, j::I64) where {T}
    # Sanity check
    @assert 1 ≤ i ≤ ret.ntime
    @assert 1 ≤ j ≤ ret.ntime

    # Return G^{R}(tᵢ, tⱼ)
    #
    # The stored block is returned as a write-through view, while the
    # reconstructed one is a fresh dense matrix.
    if i ≥ j
        _block(ret, i, j)
    else
        -Matrix(_block(ret, j, i)')
    end
end


#=
### *Gˡᵐⁱˣ* : *Indexing*
=#

"""
    Base.getindex(lmix::Gˡᵐⁱˣ{T}, i::I64, j::I64)

Visit the element stored in `Gˡᵐⁱˣ` object. `i` is index for real time,
and `j` is index for imaginary time.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function Base.getindex(lmix::Gˡᵐⁱˣ{T}, i::I64, j::I64) where {T}
    # Sanity check
    @assert 1 ≤ i ≤ lmix.ntime
    @assert 1 ≤ j ≤ lmix.ntau

    # Return G^{⌉}(tᵢ, τⱼ)
    _block(lmix, i, j)
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

Visit the element stored in `Gˡᵉˢˢ` object. `i` and `j` are indices for
real time.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function Base.getindex(less::Gˡᵉˢˢ{T}, i::I64, j::I64) where {T}
    # Sanity check
    @assert 1 ≤ i ≤ less.ntime
    @assert 1 ≤ j ≤ less.ntime

    # Return G^{<}(tᵢ, tⱼ)
    #
    # The stored block is returned as a write-through view, while the
    # reconstructed one is a fresh dense matrix.
    if i ≤ j
        _block(less, i, j)
    else
        -Matrix(_block(less, j, i)')
    end
end


#=
### *Gᵐᵃᵗᵐ* : *Indexing*
=#

#=
*Remarks* :

The following Kubo-Martin-Schwinger boundary condition is used:

```math
\begin{equation}
G(\tau) = \pm G(\tau + \beta)
\end{equation}
```

where `+` is for bosonic system, while `-` is for fermionic system.

=#

"""
    Base.getindex(matm::Gᵐᵃᵗᵐ{T}, i::I64)

Visit the element stored in `Gᵐᵃᵗᵐ` object. `i` is index for imaginary time.

See also: [`Gᵐᵃᵗᵐ`](@ref).
"""
function Base.getindex(matm::Gᵐᵃᵗᵐ{T}, i::I64) where {T}
    # Sanity check
    @assert 1 ≤ i ≤ matm.ntau

    # Return G^{M}(τᵢ < 0)
    matm.dataM[][matm.ntau - i + 1] * matm.sign
end


#=
### *Gᵃᵈᵛ* : *Indexing*
=#

"""
    Base.getindex(adv::Gᵃᵈᵛ{T}, i::I64, j::I64)

Visit the element stored in `Gᵃᵈᵛ` object. `i` and `j` are indices for
real time.

See also: [`Gᵃᵈᵛ`](@ref).
"""
function Base.getindex(adv::Gᵃᵈᵛ{T}, i::I64, j::I64) where {T}
    # Sanity check
    @assert 1 ≤ i ≤ adv.ntime
    @assert 1 ≤ j ≤ adv.ntime

    # Return G^{A}(tᵢ, tⱼ) = G^{R}(tⱼ, tᵢ)†
    copy((adv.dataR[])[j,i]')
end


#=
### *Gʳᵐⁱˣ* : *Indexing*
=#

"""
    Base.getindex(rmix::Gʳᵐⁱˣ{T}, i::I64, j::I64)

Visit the element stored in `Gʳᵐⁱˣ` object. `i` is index for imaginary
time, and `j` is index for real time.

See also: [`Gʳᵐⁱˣ`](@ref).
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

Visit the element stored in `Gᵍᵗʳ` object. `i` and `j` are indices for
real time.

See also: [`Gᵍᵗʳ`](@ref).
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
    Base.getindex(mat::gᵐᵃᵗ{S}, i::I64)

Visit the element stored in `gᵐᵃᵗ` object. `i` is index for imaginary time.

See also: [`gᵐᵃᵗ`](@ref).
"""
function Base.getindex(mat::gᵐᵃᵗ{S}, i::I64) where {S}
    # Sanity check
    @assert 1 ≤ i ≤ mat.ntau

    # Return G^{M}(τᵢ)
    mat.data[i]
end


#=
### *gʳᵉᵗ* : *Indexing*
=#

"""
    Base.getindex(ret::gʳᵉᵗ{S}, j::I64)

Visit the element stored in `gʳᵉᵗ` object. `j` is index for real time.

See also: [`gʳᵉᵗ`](@ref).
"""
function Base.getindex(ret::gʳᵉᵗ{S}, j::I64) where {S}
    # Sanity check
    @assert 1 ≤ j ≤ ret.tstp

    # Return G^{R}(tᵢ ≡ tstp, tⱼ)
    ret.data[j]
end


"""
    Base.getindex(ret::gʳᵉᵗ{S}, i::I64, tstp::I64)

Visit the element stored in `gʳᵉᵗ` object. `i` is index for real time.

See also: [`gʳᵉᵗ`](@ref).
"""
function Base.getindex(ret::gʳᵉᵗ{S}, i::I64, tstp::I64) where {S}
    # Sanity check
    @assert tstp == ret.tstp
    @assert 1 ≤ i ≤ ret.tstp

    # Return G^{R}(tᵢ, tⱼ ≡ tstp)
    -(ret.data[i])'
end


#=
### *gˡᵐⁱˣ* : *Indexing*
=#

"""
    Base.getindex(lmix::gˡᵐⁱˣ{S}, j::I64)

Visit the element stored in `gˡᵐⁱˣ` object. `j` is index for imaginary
time.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function Base.getindex(lmix::gˡᵐⁱˣ{S}, j::I64) where {S}
    # Sanity check
    @assert 1 ≤ j ≤ lmix.ntau

    # Return G^{⌉}(tᵢ ≡ tstp, τⱼ)
    lmix.data[j]
end


#=
### *gˡᵉˢˢ* : *Indexing*
=#

"""
    Base.getindex(less::gˡᵉˢˢ{S}, i::I64)

Visit the element stored in `gˡᵉˢˢ` object. `i` is index for real time.

See also: [`gˡᵉˢˢ`](@ref).
"""
function Base.getindex(less::gˡᵉˢˢ{S}, i::I64) where {S}
    # Sanity check
    @assert 1 ≤ i ≤ less.tstp

    # Return G^{<}(tᵢ, tⱼ ≡ tstp)
    less.data[i]
end


"""
    Base.getindex(less::gˡᵉˢˢ{S}, tstp::I64, j::I64)

Visit the element stored in `gˡᵉˢˢ` object. `j` is index for real time.

See also: [`gˡᵉˢˢ`](@ref).
"""
function Base.getindex(less::gˡᵉˢˢ{S}, tstp::I64, j::I64) where {S}
    # Sanity check
    @assert tstp == less.tstp
    @assert 1 ≤ j ≤ less.tstp

    # Return G^{<}(tᵢ ≡ tstp, tⱼ)
    -(less.data[j])'
end


#=
### *gᵐᵃᵗᵐ* : *Indexing*
=#

"""
    Base.getindex(matm::gᵐᵃᵗᵐ{S}, i::I64)

Visit the element stored in `gᵐᵃᵗᵐ` object. `i` is index for imaginary time.

See also: [`gᵐᵃᵗᵐ`](@ref).
"""
function Base.getindex(matm::gᵐᵃᵗᵐ{S}, i::I64) where {S}
    # Sanity check
    @assert 1 ≤ i ≤ matm.ntau

    # Return G^{M}(τᵢ < 0)
    matm.dataM[][matm.ntau - i + 1] * matm.sign
end


#=
### *gᵃᵈᵛ* : *Indexing*
=#

"""
    Base.getindex(adv::gᵃᵈᵛ{S}, i::I64)

Visit the element stored in `gᵃᵈᵛ` object. `i` is index for real time.

See also: [`gᵃᵈᵛ`](@ref).
"""
function Base.getindex(adv::gᵃᵈᵛ{S}, i::I64) where {S}
    # Sanity check
    @assert 1 ≤ i ≤ adv.tstp

    # Return G^{A}(tᵢ, tstp) = G^{R}(tstp, tᵢ)†
    copy((adv.dataR[])[i]')
end


#=
### *gʳᵐⁱˣ* : *Indexing*
=#

"""
    Base.getindex(rmix::gʳᵐⁱˣ{S}, i::I64)

Visit the element stored in `gʳᵐⁱˣ` object. `i` is index for imaginary
time.

See also: [`gʳᵐⁱˣ`](@ref).
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

Visit the element stored in `gᵍᵗʳ` object. `i` is index for real time.

See also: [`gᵍᵗʳ`](@ref).
"""
function Base.getindex(gtr::gᵍᵗʳ{S}, i::I64) where {S}
    # Sanity check
    @assert 1 ≤ i ≤ gtr.tstp

    # Return G^{>}(tᵢ, tⱼ ≡ tstp)
    gtr.dataL[][i] + gtr.dataR[][i, gtr.tstp]
end


"""
    Base.getindex(gtr::gᵍᵗʳ{S}, tstp::I64, j::I64)

Visit the element stored in `gᵍᵗʳ` object. `j` is index for real time.

See also: [`gᵍᵗʳ`](@ref).
"""
function Base.getindex(gtr::gᵍᵗʳ{S}, tstp::I64, j::I64) where {S}
    # Sanity check
    @assert tstp == gtr.tstp
    @assert 1 ≤ j ≤ gtr.tstp

    # Return G^{>}(tᵢ ≡ tstp, tⱼ)
    gtr.dataL[][tstp, j] + gtr.dataR[][j]
end


#=
### *ℱ* : *Indexing*
=#

"""
    Base.getindex(cfm::ℱ{T}, tstp::I64)

Return contour-ordered Green's function at given time step `tstp`.

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
    memcpy!(cfm, cfv, tstp)

    # Return the desired struct
    return cfv
end


#=
### *PseudoGF* : *Indexing*
=#

"""
    Base.getindex(pgf::PseudoGF{T}, tstp::I64)

Return pseudo-particle Green's function at given time step `tstp`.

See also: [`PseudoGF`](@ref), [`PseudoGFStep`](@ref).
"""
function Base.getindex(pgf::PseudoGF{T}, tstp::I64) where {T}
    # Sanity check
    @assert getntime(pgf) ≥ tstp ≥ 0

    # Get key parameters
    sign = getsign(pgf)
    ntau = getntau(pgf)
    ndim1, ndim2 = getdims(pgf)

    # Construct an empty `PseudoGFStep` struct
    pgv = PseudoGFStep(tstp, ntau, ndim1, ndim2, sign)

    # Extract data at time step `tstp` from `PseudoGF` object, then copy
    # them to `PseudoGFStep` object.
    memcpy!(pgf, pgv, tstp)

    # Return the desired struct
    return pgv
end


#=
### *Indexing* : *FFT Array*
=#

"""
    Base.getindex(A::FFTArray{T}, n::I64, dom::FFTDomain)

Return a copy of the matrix at the 1-based storage index `n` in the domain
`dom` (`FFT_TIME` or `FFT_FREQ`).

See also: [`FFTArray`](@ref).
"""
function Base.getindex(A::FFTArray{T}, n::I64, dom::FFTDomain) where {T}
    1 ≤ n ≤ getnft(A) || throw(BoundsError(1:getnft(A), n))
    return _fft_data(A, dom)[:, :, n]
end


#=
### *ℱᵐᵒᵛ* : *Relative Indexing*
=#

"""
    Base.getindex(A::Union{Gʳᵉᵗᵐᵒᵛ{T},Gˡᵉˢˢᵐᵒᵛ{T}}, delti::I64, deltj::I64)

Return a copy of the block `(delti, deltj)` (relative NESSi indices).
"""
function Base.getindex(A::Union{Gʳᵉᵗᵐᵒᵛ{T},Gˡᵉˢˢᵐᵒᵛ{T}},
                       delti::I64, deltj::I64) where {T}
    (0 ≤ delti ≤ A.tc && 0 ≤ deltj ≤ A.tc) || throw(BoundsError(A, (delti, deltj)))
    return Matrix(_mblock(A, delti, deltj))
end


"""
    Base.getindex(A::Union{gʳᵉᵗᵐᵒᵛ{S},gˡᵉˢˢᵐᵒᵛ{S}}, deltj::I64)

Return a copy of the block `deltj` of a moving time step.
"""
function Base.getindex(A::Union{gʳᵉᵗᵐᵒᵛ{S},gˡᵉˢˢᵐᵒᵛ{S}}, deltj::I64) where {S}
    0 ≤ deltj ≤ A.tc || throw(BoundsError(A, deltj))
    return Matrix(_mblock(A, deltj))
end


"""
    Base.getindex(G::ℱᵐᵒᵛ{T}, c::Symbol, n::I64, m::I64)

Return a copy of the component `c` (`:ret`, `:less`, or `:gtr`) at the
absolute KB time steps `(n, m)`. The blocks with `n < m` are
reconstructed by the hermitian relations. The greater component is
``G^{>} = G^{R} + G^{<}`` for `n ≥ m` and ``-G^{>}(t_m, t_n)^{\\dagger}``
otherwise.
"""
function Base.getindex(G::ℱᵐᵒᵛ{T}, c::Symbol, n::I64, m::I64) where {T}
    _check_initialized(G)
    c in (:ret, :less, :gtr) ||
        throw(ArgumentError("component must be :ret, :less, or :gtr, got $c"))
    if n ≥ m
        delti, deltj = _mabs(G, n, m)
        R = Matrix(_mblock(G.ret, delti, deltj))
        L = Matrix(_mblock(G.less, delti, deltj))
        c == :ret && return R
        c == :less && return L
        return R + L
    else
        X = G[c, m, n]
        return -Matrix(X')
    end
end


"""
    Base.getindex(v::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ}, c::Symbol, deltj::I64)

Return a copy of the block `deltj` of the component `c` (`:ret` or
`:less`) of a moving time step or view.
"""
function Base.getindex(v::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ}, c::Symbol, deltj::I64)
    c in (:ret, :less) ||
        throw(ArgumentError("component must be :ret or :less, got $c"))
    0 ≤ deltj ≤ getsize(v) || throw(BoundsError(v, deltj))
    return Matrix(_vblock(v, c, deltj))
end


#=
### *Cfᵐᵒᵛ* : *Indexing And Window Operations*
=#

"""
    Base.getindex(f::Cfᵐᵒᵛ{T}, delti::I64)

Return a copy of the slot `delti`, i.e. ``f(t_0 - \\delta_i)``.
"""
function Base.getindex(f::Cfᵐᵒᵛ{T}, delti::I64) where {T}
    0 ≤ delti ≤ f.tc || throw(BoundsError(f, delti))
    return Matrix(view(f.data, :, :, _mrow(f.head, f.tc, delti)))
end


#=
### *Public Service Functions*
=#

"""
    Base.getindex(AW::AbstractWeights, inds::Vararg{I64})

Provide a fast access to the weights. `AW` denotes the type of weights,
and `inds` is for the indices. This function works for the following
structs only:

* `PolynomialInterpolationWeights`
* `PolynomialDifferentiationWeights`
* `PolynomialIntegrationWeights`
* `BackwardDifferentiationWeights`
* `BoundaryConvolutionWeights`
"""
function Base.getindex(AW::AbstractWeights, inds::Vararg{I64})
    @assert length(inds) == ndims(AW.W)
    indsp1 = inds .+ 1
    return AW.W[indsp1...]
end


"""
    Base.getindex(GW::GregoryIntegrationWeights, inds::Vararg{I64})

Provide a fast access to the weights. `GW` denotes the type of weights,
and `inds` is for the indices. This function works for the following
structs only:

* `GregoryIntegrationWeights`

*References* :

See *Remarks* : *Gregory Integration*.
"""
function Base.getindex(GW::GregoryIntegrationWeights, inds::Vararg{I64})
    # Check the indices
    # 𝑛 and 𝑗 are row indices and column indices, respectively.
    @assert length(inds) == 2
    n, j = inds

    # Access the weights
    #
    # Defalut weights, zero.
    val = 0.0
    #
    # 0 ≤ 𝑛, 𝑗 ≤ k
    if n ≤ GW.k
        if j ≤ GW.k
            val = GW.σ[n + 1, j + 1]
        end
    end
    #
    # 𝑘 + 1 ≤ 𝑛 ≤ 2𝑘 + 1
    if ( GW.k + 1 ) ≤ n ≤ ( 2 * GW.k + 1 )
        # 0 ≤ 𝑗 ≤ 𝑘
        if j ≤ GW.k
            val = GW.Σ[n - GW.k, j + 1]
        # 𝑘 + 1 ≤ 𝑗 ≤ 𝑛
        else
            if j ≤ n
                val = GW.ω[n + 1 - j]
            end
        end
    end
    #
    # 𝑛 ≥ 2𝑘 + 2
    if n ≥ 2 * GW.k + 2
        # 0 ≤ 𝑗 ≤ 𝑘
        if j ≤ GW.k
            val = GW.ω[j + 1]
        # 𝑘 + 1 ≤ 𝑗 ≤ 𝑛 - 𝑘 - 1
        elseif  GW.k + 1 ≤ j ≤ n - GW.k - 1
            val = 1.0
        # 𝑛 - 𝑘 ≤ 𝑗 ≤ 𝑛
        elseif n - GW.k ≤ j ≤ n
            val = GW.ω[n + 1 - j]
        end
    end

    # Return the desired weights
    return val
end


