


#=
### *gᵐᵃᵗᵐ* : *Struct*
=#

"""
    gᵐᵃᵗᵐ{S}

Matsubara component (``G^M``) of contour Green's function at given time
step `tstp = 0`. It is designed for ``\tau < 0`` case. It is not an
independent component. It can be constructed from the `gᵐᵃᵗ{T}` struct.

See also: [`gʳᵉᵗ`](@ref), [`gˡᵐⁱˣ`](@ref), [`gˡᵉˢˢ`](@ref).
"""
mutable struct gᵐᵃᵗᵐ{S} <: CnAbstractVector{S}
    type  :: String
    sign  :: I64 # Used to distinguish fermions and bosons
    ntau  :: I64
    ndim1 :: I64
    ndim2 :: I64
    dataV :: Ref{gᵐᵃᵗ{S}}
end

#=
### *gᵐᵃᵗᵐ* : *Constructors*
=#

"""
    gᵐᵃᵗᵐ(sign::I64, mat::gᵐᵃᵗ{S})

Constructor. Note that the `matm` component is not independent. We use
the `mat` component to initialize it.
"""
function gᵐᵃᵗᵐ(sign::I64, mat::gᵐᵃᵗ{S}) where {S}
    # Sanity check
    @assert sign in (BOSE, FERMI)

    # Setup properties
    # Extract parameters from `mat`
    ntau = mat.ntau
    ndim1 = mat.ndim1
    ndim2 = mat.ndim2
    #
    # We don't allocate memory for `dataV` directly, but let it point to
    # the `mat` object.
    dataV = Ref(mat)

    # Call the default constructor
    gᵐᵃᵗᵐ("matm", sign, ntau, ndim1, ndim2, dataV)
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
### *gᵃᵈᵛ* : *Struct*
=#

"""
    gᵃᵈᵛ{S}

Advanced component (``G^{A}``) of contour Green's function at given
time step `tstp`.

Note that currently we do not need this component explicitly. However,
for the sake of completeness, we still define an empty struct for it.

See also: [`gᵐᵃᵗ`](@ref), [`gˡᵐⁱˣ`](@ref), [`gˡᵉˢˢ`](@ref).
"""
mutable struct gᵃᵈᵛ{S} <: CnAbstractVector{S}
    type  :: String
end

#=
### *gᵃᵈᵛ* : *Constructors*
=#

"""
    gᵃᵈᵛ()

Constructor. Note that the `adv` component is not independent. We use
the `ret` component to initialize it.
"""
function gᵃᵈᵛ()
    # Call the default constructor
    gᵃᵈᵛ("adv")
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
### *gʳᵐⁱˣ* : *Struct*
=#

"""
    gʳᵐⁱˣ{S}

Right-mixing component (``G^{⌈}``) of contour Green's function at given
time step `tstp`. Actually, it denotes ``G^{⌈}(τᵢ, tⱼ ≡ tstp)``

See also: [`gᵐᵃᵗ`](@ref), [`gʳᵉᵗ`](@ref), [`gˡᵉˢˢ`](@ref).
"""
mutable struct gʳᵐⁱˣ{S} <: CnAbstractVector{S}
    type  :: String
    sign  :: I64 # Used to distinguish fermions and bosons
    ntau  :: I64
    ndim1 :: I64
    ndim2 :: I64
    dataL :: Ref{gˡᵐⁱˣ{S}}
end

#=
### *gʳᵐⁱˣ* : *Constructors*
=#

"""
    gʳᵐⁱˣ(sign::I64, lmix::gˡᵐⁱˣ{S})

Constructor. Note that the `rmix` component is not independent. We use
the `lmix` component to initialize it.
"""
function gʳᵐⁱˣ(sign::I64, lmix::gˡᵐⁱˣ{S}) where {S}
    # Sanity check
    @assert sign in (BOSE, FERMI)

    # Setup properties
    # Extract parameters from `lmix`
    ntau  = lmix.ntau
    ndim1 = lmix.ndim1
    ndim2 = lmix.ndim2
    #
    # We don't allocate memory for `dataL` directly, but let it point to
    # the `lmix` object.
    dataL = Ref(lmix)

    # Call the default constructor
    gʳᵐⁱˣ("rmix", sign, ntau, ndim1, ndim2, dataL)
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
### *gᵍᵗʳ* : *Struct*
=#

"""
    gᵍᵗʳ{S}

Greater component (``G^{>}``) of contour Green's function at given
time step `tstp`.

See also: [`gʳᵉᵗ`](@ref), [`gˡᵐⁱˣ`](@ref), [`gˡᵉˢˢ`](@ref).
"""
mutable struct gᵍᵗʳ{S} <: CnAbstractVector{S}
    type  :: String
    tstp  :: I64
    ndim1 :: I64
    ndim2 :: I64
    dataL :: Ref{gˡᵉˢˢ{S}}
    dataR :: Ref{gʳᵉᵗ{S}}
end

#=
### *gᵍᵗʳ* : *Constructors*
=#

"""
    gᵍᵗʳ(less::gˡᵉˢˢ{S}, ret::gʳᵉᵗ{S})

Constructor. Note that the `gtr` component is not independent. We use
the `less` and `ret` components to initialize it.
"""
function gᵍᵗʳ(less::gˡᵉˢˢ{S}, ret::gʳᵉᵗ{S}) where {S}
    # Setup properties
    # Extract parameters from `less`
    tstp  = less.tstp
    ndim1 = less.ndim1
    ndim2 = less.ndim2
    #
    # We don't allocate memory for `dataL` and `dataR` directly, but
    # let them point to  `less` and `ret` objects, respectively.
    dataL = Ref(less)
    dataR = Ref(ret)

    # Call the default constructor
    gᵍᵗʳ("gtr", tstp, ndim1, ndim2, dataL, dataR)
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
