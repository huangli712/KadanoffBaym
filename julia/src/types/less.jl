#
# Project : Lavender
# Source  : less.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks : Lesser Green's Function*

The lesser component of contour-ordered Green's function reads

```math
\begin{equation}
G^{<}(t,t') = \mp i \langle c^{\dagger}(t') c(t) \rangle,
\end{equation}
```

where ``t,\ t' \in \mathcal{C}_1 \cup \mathcal{C}_2``. We choose the
upper (lower) sign if the operators ``c`` and ``c^{\dagger}`` are
bosonic (fermionic). Its hermitian conjugate yields

```math
\begin{equation}
G^{<}(t,t')^{*} = -G^{<}(t',t),
\end{equation}
```

or equivalently

```math
\begin{equation}
G^{<}(t,t') = -G^{<}(t',t)^{*},
\end{equation}
```

and

```math
\begin{equation}
G^{>}(t,t') = -G^{>}(t',t)^{*},
\end{equation}
```

The lesser component is related to the retarded, advanced, and Keldysh
Green's functions via

```math
\begin{equation}
G^{<} = \frac{1}{2}(G^{K} - G^{R} + G^{A}).
\end{equation}
```

*Reference* :

Please see [`NESSi`] Session `3` [especially Eq. (18a)] for more details.
=#

#=
### *Gˡᵉˢˢ* : *Struct*
=#

"""
    Gˡᵉˢˢ{T}

Lesser component (``G^{<}``) of contour-ordered Green's function. We
usually call this component `less`.

See also: [`Gᵐᵃᵗ`](@ref), [`Gʳᵉᵗ`](@ref), [`Gˡᵐⁱˣ`](@ref).
"""
mutable struct Gˡᵉˢˢ{T} <: CnAbstractMatrix{T}
    type  :: String
    ntime :: I64
    ndim1 :: I64
    ndim2 :: I64
    # Contiguous storage for the upper triangle (i ≤ j) only,
    # data[:,:,k] is G^{<}(tᵢ,tⱼ), where k = (j-1)j/2 + i.
    data  :: Array{T,3}
end

#=
### *Gˡᵉˢˢ* : *Constructors*
=#

"""
    Gˡᵉˢˢ(ntime::I64, ndim1::I64, ndim2::I64, v::T)

Constructor. All the matrix elements are set to be `v`.
"""
function Gˡᵉˢˢ(ntime::I64, ndim1::I64, ndim2::I64, v::T) where {T}
    # Sanity check
    @assert ntime ≥ 2
    @assert ndim1 ≥ 1
    @assert ndim2 ≥ 1

    # Create contiguous storage, whose size is (ndim1, ndim2, ntri).
    data = fill(v, ndim1, ndim2, _ntri(ntime))

    # Call the default constructor
    Gˡᵉˢˢ("less", ntime, ndim1, ndim2, data)
end

"""
    Gˡᵉˢˢ(ntime::I64, ndim1::I64, ndim2::I64)

Constructor. All the matrix elements are set to be complex zero.
"""
function Gˡᵉˢˢ(ntime::I64, ndim1::I64, ndim2::I64)
    Gˡᵉˢˢ(ntime, ndim1, ndim2, zero(C64))
end

"""
    Gˡᵉˢˢ(ntime::I64, ndim1::I64)

Constructor. All the matrix elements are set to be complex zero.
"""
function Gˡᵉˢˢ(ntime::I64, ndim1::I64)
    Gˡᵉˢˢ(ntime, ndim1, ndim1, zero(C64))
end

"""
    Gˡᵉˢˢ(ntime::I64, x::Element{T})

Constructor. The matrix is initialized by `x`.
"""
function Gˡᵉˢˢ(ntime::I64, x::Element{T}) where {T}
    # Sanity check
    @assert ntime ≥ 2

    ndim1, ndim2 = size(x)

    # Create contiguous storage, whose size is (ndim1, ndim2, ntri).
    data = Array{T,3}(undef, ndim1, ndim2, _ntri(ntime))
    for k = 1:_ntri(ntime)
        data[:,:,k] .= x
    end

    # Call the default constructor
    Gˡᵉˢˢ("less", ntime, ndim1, ndim2, data)
end

"""
    Gˡᵉˢˢ(C::Cn, x::Element{T})

Constructor. The matrix is initialized by `x`.
"""
function Gˡᵉˢˢ(C::Cn, x::Element{T}) where {T}
    # Sanity check
    @assert getdims(C) == size(x)

    # Create contiguous storage, whose size is (ndim1, ndim2, ntri).
    data = Array{T,3}(undef, C.ndim1, C.ndim2, _ntri(C.ntime))
    for k = 1:_ntri(C.ntime)
        data[:,:,k] .= x
    end

    # Call the default constructor
    Gˡᵉˢˢ("less", C.ntime, C.ndim1, C.ndim2, data)
end

"""
    Gˡᵉˢˢ(C::Cn, v::T)

Constructor. All the matrix elements are set to be `v`.
"""
function Gˡᵉˢˢ(C::Cn, v::T) where {T}
    Gˡᵉˢˢ(C.ntime, C.ndim1, C.ndim2, v)
end

"""
    Gˡᵉˢˢ(C::Cn)

Constructor. All the matrix elements are set to be complex zero.
"""
function Gˡᵉˢˢ(C::Cn)
    Gˡᵉˢˢ(C.ntime, C.ndim1, C.ndim2, zero(C64))
end

"""
    Gˡᵉˢˢ()

Constructor. All the matrix elements are set to be complex zero. Notice
that all of the parameters are extracted from the `PCONTOUR` dict.

See also: [`PCONTOUR`](@ref), [`get_c`](@ref).
"""
function Gˡᵉˢˢ()
    ntime = get_c("ntime")
    ndim1 = get_c("ndim1")
    ndim2 = get_c("ndim2")
    Gˡᵉˢˢ(ntime, ndim1, ndim2, zero(C64))
end

#=
*Remarks : Greater Green's Function*

The greater component of contour-ordered Green's function reads

```math
\begin{equation}
G^{>}(t,t') = - i \langle c(t) c^{\dagger}(t') \rangle,
\end{equation}
```

where ``t,\ t' \in \mathcal{C}_1 \cup \mathcal{C}_2``. Its hermitian
conjugate yields

```math
\begin{equation}
G^{>}(t,t')^{*} = -G^{>}(t',t).
\end{equation}
```

The greater component is related to the retarded, advanced, and Keldysh
Green's functions via

```math
\begin{equation}
G^{>} = \frac{1}{2}(G^{K} + G^{R} - G^{A}).
\end{equation}
```
=#

#=
### *Gᵍᵗʳ* : *Struct*
=#

"""
    Gᵍᵗʳ{T}

Greater component (``G^{>}``) of contour-ordered Green's function. We
usually call this component `gtr`.

See also: [`Gʳᵉᵗ`](@ref), [`Gˡᵐⁱˣ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
mutable struct Gᵍᵗʳ{T} <: CnAbstractMatrix{T}
    type  :: String
    ntime :: I64
    ndim1 :: I64
    ndim2 :: I64
    dataL :: Ref{Gˡᵉˢˢ{T}}
    dataR :: Ref{Gʳᵉᵗ{T}}
end

#=
### *Gᵍᵗʳ* : *Constructors*
=#

"""
    Gᵍᵗʳ(less::Gˡᵉˢˢ{T}, ret::Gʳᵉᵗ{T})

Constructor. Note that the `gtr` component is not independent. We use
the `less` and `ret` components to initialize it.
"""
function Gᵍᵗʳ(less::Gˡᵉˢˢ{T}, ret::Gʳᵉᵗ{T}) where {T}
    # Setup properties
    # Extract parameters from `less`
    ntime = less.ntime
    ndim1 = less.ndim1
    ndim2 = less.ndim2
    #
    # We don't allocate memory for `dataL` and `dataR` directly, but
    # let them point to  `less` and `ret` objects, respectively.
    dataL = Ref(less)
    dataR = Ref(ret)

    # Call the default constructor
    Gᵍᵗʳ("gtr", ntime, ndim1, ndim2, dataL, dataR)
end

#=
### *gˡᵉˢˢ* : *Struct*
=#

"""
    gˡᵉˢˢ{S}

Lesser component (``G^{<}``) of contour-ordered Green's function at given
time step `tstp`. Actually, it denotes ``G^{<}(tᵢ, tⱼ ≡ tstp)``.

See also: [`gᵐᵃᵗ`](@ref), [`gʳᵉᵗ`](@ref), [`gˡᵐⁱˣ`](@ref).
"""
mutable struct gˡᵉˢˢ{S} <: CnAbstractVector{S}
    type  :: String
    tstp  :: I64
    ndim1 :: I64
    ndim2 :: I64
    data  :: VecArray{S}
end

#=
### *gˡᵉˢˢ* : *Constructors*
=#

"""
    gˡᵉˢˢ(tstp::I64, ndim1::I64, ndim2::I64, v::S)

Constructor. All the matrix elements are set to be `v`.
"""
function gˡᵉˢˢ(tstp::I64, ndim1::I64, ndim2::I64, v::S) where {S}
    # Sanity check
    @assert tstp  ≥ 1
    @assert ndim1 ≥ 1
    @assert ndim2 ≥ 1

    # Create Element{S}
    element = fill(v, ndim1, ndim2)

    # Create VecArray{S}, whose size is indeed (tstp,).
    data = VecArray{S}(undef, tstp)
    for i = 1:tstp
        data[i] = copy(element)
    end

    # Call the default constructor
    gˡᵉˢˢ("less", tstp, ndim1, ndim2, data)
end

"""
    gˡᵉˢˢ(tstp::I64, ndim1::I64, ndim2::I64)

Constructor. All the matrix elements are set to be complex zero.
"""
function gˡᵉˢˢ(tstp::I64, ndim1::I64, ndim2::I64)
    gˡᵉˢˢ(tstp, ndim1, ndim2, zero(C64))
end

"""
    gˡᵉˢˢ(tstp::I64, ndim1::I64)

Constructor. All the matrix elements are set to be complex zero.
"""
function gˡᵉˢˢ(tstp::I64, ndim1::I64)
    gˡᵉˢˢ(tstp, ndim1, ndim1, zero(C64))
end

"""
    gˡᵉˢˢ(tstp::I64, x::Element{S})

Constructor. The matrix is initialized by `x`.
"""
function gˡᵉˢˢ(tstp::I64, x::Element{S}) where {S}
    # Sanity check
    @assert tstp ≥ 1

    ndim1, ndim2 = size(x)

    # Create VecArray{S}, whose size is indeed (tstp,)
    data = VecArray{S}(undef, tstp)
    for i = 1:tstp
        data[i] = copy(x)
    end

    # Call the default constructor
    gˡᵉˢˢ("less", tstp, ndim1, ndim2, data)
end

#=
### *gᵍᵗʳ* : *Struct*
=#

"""
    gᵍᵗʳ{S}

Greater component (``G^{>}``) of contour-ordered Green's function at given
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
