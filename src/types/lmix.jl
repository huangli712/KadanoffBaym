#
# Project : Lavender
# Source  : lmix.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks : Left-Mixing Green's Function*

The left-mixing component of contour-ordered Green's function reads

```math
\begin{equation}
G^{\rceil}(t,\tau') = \mp i \langle c^{\dagger}(\tau') c(t) \rangle,
\end{equation}
```

where ``t \in \mathcal{C}_1 \cup \mathcal{C}_2`` and
``\tau' \in \mathcal{C}_3``. We choose the upper
(lower) sign if the operators ``c`` and ``c^{\dagger}`` are bosonic
(fermionic). Its hermitian conjugate yields

```math
\begin{equation}
G^{\rceil}(t,\tau)^{*} = \mp G^{\lceil}(\beta - \tau,t),
\end{equation}
```

or equivalently

```math
\begin{equation}
G^{\rceil}(t,\tau) = \mp G^{\lceil}(\beta - \tau,t)^{*},
\end{equation}
```

and

```math
\begin{equation}
G^{\lceil}(\tau,t) = \mp G^{\rceil}(t,\beta - \tau)^{*},
\end{equation}
```

where ``G^{\lceil}(\tau,t')`` is the right-mixing Green's function.

*Reference* :

Please see [`NESSi`] Session `3` [especially Eq. (18c) and (18d)] for
more details.
=#

#=
### *Gˡᵐⁱˣ* : *Struct*
=#

"""
    Gˡᵐⁱˣ{T}

Left-mixing component (``G^{⌉}``) of contour-ordered Green's function.
We usually call this component `lmix`.

See also: [`Gᵐᵃᵗ`](@ref), [`Gʳᵉᵗ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
mutable struct Gˡᵐⁱˣ{T} <: CnAbstractMatrix{T}
    type  :: String
    ntime :: I64
    ntau  :: I64
    ndim1 :: I64
    ndim2 :: I64
    # Contiguous storage, data[:,:,i,m] is G^{⌉}(tᵢ,τₘ).
    data  :: Array{T,4}
end

#=
### *Gˡᵐⁱˣ* : *Constructors*
=#

"""
    Gˡᵐⁱˣ(ntime::I64, ntau::I64, ndim1::I64, ndim2::I64, v::T)

Constructor. All the matrix elements are set to be `v`.
"""
function Gˡᵐⁱˣ(ntime::I64, ntau::I64, ndim1::I64, ndim2::I64, v::T) where {T}
    # Sanity check
    @assert ntime ≥ 2
    @assert ntau  ≥ 2
    @assert ndim1 ≥ 1
    @assert ndim2 ≥ 1

    # Create contiguous storage, whose size is (ndim1, ndim2, ntime, ntau).
    data = fill(v, ndim1, ndim2, ntime, ntau)

    # Call the default constructor
    Gˡᵐⁱˣ("lmix", ntime, ntau, ndim1, ndim2, data)
end

"""
    Gˡᵐⁱˣ(ntime::I64, ntau::I64, ndim1::I64, ndim2::I64)

Constructor. All the matrix elements are set to be complex zero.
"""
function Gˡᵐⁱˣ(ntime::I64, ntau::I64, ndim1::I64, ndim2::I64)
    Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, zero(C64))
end

"""
    Gˡᵐⁱˣ(ntime::I64, ntau::I64, ndim1::I64)

Constructor. All the matrix elements are set to be complex zero.
"""
function Gˡᵐⁱˣ(ntime::I64, ntau::I64, ndim1::I64)
    Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim1, zero(C64))
end

"""
    Gˡᵐⁱˣ(ntime::I64, ntau::I64, x::Element{T})

Constructor. The matrix is initialized by `x`.
"""
function Gˡᵐⁱˣ(ntime::I64, ntau::I64, x::Element{T}) where {T}
    # Sanity check
    @assert ntime ≥ 2
    @assert ntau  ≥ 2

    ndim1, ndim2 = size(x)

    # Create contiguous storage, whose size is (ndim1, ndim2, ntime, ntau).
    data = Array{T,4}(undef, ndim1, ndim2, ntime, ntau)
    for i = 1:ntau
        for j = 1:ntime
            data[:,:,j,i] .= x
        end
    end

    # Call the default constructor
    Gˡᵐⁱˣ("lmix", ntime, ntau, ndim1, ndim2, data)
end

"""
    Gˡᵐⁱˣ(C::Cn, x::Element{T})

Constructor. The matrix is initialized by `x`.
"""
function Gˡᵐⁱˣ(C::Cn, x::Element{T}) where {T}
    # Sanity check
    @assert getdims(C) == size(x)

    # Create contiguous storage, whose size is (ndim1, ndim2, ntime, ntau).
    data = Array{T,4}(undef, C.ndim1, C.ndim2, C.ntime, C.ntau)
    for i = 1:C.ntau
        for j = 1:C.ntime
            data[:,:,j,i] .= x
        end
    end

    # Call the default constructor
    Gˡᵐⁱˣ("lmix", C.ntime, C.ntau, C.ndim1, C.ndim2, data)
end

"""
    Gˡᵐⁱˣ(C::Cn, v::T)

Constructor. All the matrix elements are set to be `v`.
"""
function Gˡᵐⁱˣ(C::Cn, v::T) where {T}
    Gˡᵐⁱˣ(C.ntime, C.ntau, C.ndim1, C.ndim2, v)
end

"""
    Gˡᵐⁱˣ(C::Cn)

Constructor. All the matrix elements are set to be complex zero.
"""
function Gˡᵐⁱˣ(C::Cn)
    Gˡᵐⁱˣ(C.ntime, C.ntau, C.ndim1, C.ndim2, zero(C64))
end

"""
    Gˡᵐⁱˣ()

Constructor. All the matrix elements are set to be complex zero. Notice
that all of the parameters are extracted from the `PCONTOUR` dict.

See also: [`PCONTOUR`](@ref), [`get_c`](@ref).
"""
function Gˡᵐⁱˣ()
    ntime = get_c("ntime")
    ntau = get_c("ntau")
    ndim1 = get_c("ndim1")
    ndim2 = get_c("ndim2")
    Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, zero(C64))
end

#=
*Remarks : Right-Mixing Green's Function*

The right-mixing component of contour-ordered Green's function reads

```math
\begin{equation}
G^{\lceil}(\tau,t') =  -i \langle c(\tau) c^{\dagger}(t')  \rangle,
\end{equation}
```

where ``t' \in \mathcal{C}_1 \cup \mathcal{C}_2`` and
``\tau \in \mathcal{C}_3``.
=#

#=
### *Gʳᵐⁱˣ* : *Struct*
=#

"""
    Gʳᵐⁱˣ{T}

Right-mixing component (``G^{⌈}``) of contour-ordered Green's function.
We usually call this component `rmix`.

See also: [`Gᵐᵃᵗ`](@ref), [`Gʳᵉᵗ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
mutable struct Gʳᵐⁱˣ{T} <: CnAbstractMatrix{T}
    type  :: String
    sign  :: I64 # Used to distinguish fermions and bosons
    ntime :: I64
    ntau  :: I64
    ndim1 :: I64
    ndim2 :: I64
    dataL :: Ref{Gˡᵐⁱˣ{T}}
end

#=
### *Gʳᵐⁱˣ* : *Constructors*
=#

"""
    Gʳᵐⁱˣ(sign::I64, lmix::Gˡᵐⁱˣ{T})

Constructor. Note that the `rmix` component is not independent. We use
the `lmix` component to initialize it.
"""
function Gʳᵐⁱˣ(sign::I64, lmix::Gˡᵐⁱˣ{T}) where {T}
    # Sanity check
    @assert sign in (BOSE, FERMI)

    # Setup properties
    # Extract parameters from `lmix`
    ntime = lmix.ntime
    ntau  = lmix.ntau
    ndim1 = lmix.ndim1
    ndim2 = lmix.ndim2
    #
    # We don't allocate memory for `dataL` directly, but let it point to
    # the `lmix` object.
    dataL = Ref(lmix)

    # Call the default constructor
    Gʳᵐⁱˣ("rmix", sign, ntime, ntau, ndim1, ndim2, dataL)
end

#=
### *gˡᵐⁱˣ* : *Struct*
=#

"""
    gˡᵐⁱˣ{S}

Left-mixing component (``G^{⌉}``) of contour-ordered Green's function at
given time step `tstp`. Actually, it denotes ``G^{⌉}(tᵢ ≡ tstp, τⱼ)``.

See also: [`gᵐᵃᵗ`](@ref), [`gʳᵉᵗ`](@ref), [`gˡᵉˢˢ`](@ref).
"""
mutable struct gˡᵐⁱˣ{S} <: CnAbstractVector{S}
    type  :: String
    ntau  :: I64
    ndim1 :: I64
    ndim2 :: I64
    data  :: VecArray{S}
end

#=
### *gˡᵐⁱˣ* : *Constructors*
=#

"""
    gˡᵐⁱˣ(ntau::I64, ndim1::I64, ndim2::I64, v::S)

Constructor. All the matrix elements are set to be `v`.
"""
function gˡᵐⁱˣ(ntau::I64, ndim1::I64, ndim2::I64, v::S) where {S}
    # Sanity check
    @assert ntau  ≥ 2
    @assert ndim1 ≥ 1
    @assert ndim2 ≥ 1

    # Create Element{S}
    element = fill(v, ndim1, ndim2)

    # Create VecArray{S}, whose size is indeed (ntau,).
    data = VecArray{S}(undef, ntau)
    for i = 1:ntau
        data[i] = copy(element)
    end

    # Call the default constructor
    gˡᵐⁱˣ("lmix", ntau, ndim1, ndim2, data)
end

"""
    gˡᵐⁱˣ(ntau::I64, ndim1::I64, ndim2::I64)

Constructor. All the matrix elements are set to be complex zero.
"""
function gˡᵐⁱˣ(ntau::I64, ndim1::I64, ndim2::I64)
    gˡᵐⁱˣ(ntau, ndim1, ndim2, zero(C64))
end

"""
    gˡᵐⁱˣ(ntau::I64, ndim1::I64)

Constructor. All the matrix elements are set to be complex.
"""
function gˡᵐⁱˣ(ntau::I64, ndim1::I64)
    gˡᵐⁱˣ(ntau, ndim1, ndim1, zero(C64))
end

"""
    gˡᵐⁱˣ(ntau::I64, x::Element{S})

Constructor. The matrix is initialized by `x`.
"""
function gˡᵐⁱˣ(ntau::I64, x::Element{S}) where {S}
    # Sanity check
    @assert ntau ≥ 2

    ndim1, ndim2 = size(x)

    # Create VecArray{S}, whose size is indeed (ntau,)
    data = VecArray{S}(undef, ntau)
    for i = 1:ntau
        data[i] = copy(x)
    end

    # Call the default constructor
    gˡᵐⁱˣ("lmix", ntau, ndim1, ndim2, data)
end

#=
### *gʳᵐⁱˣ* : *Struct*
=#

"""
    gʳᵐⁱˣ{S}

Right-mixing component (``G^{⌈}``) of contour-ordered Green's function at
given time step `tstp`. Actually, it denotes ``G^{⌈}(τᵢ, tⱼ ≡ tstp)``

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
