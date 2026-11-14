#
# Project : Lavender
# Source  : ret.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks : Retarded Green's Function*

The retarded component of contour-ordered Green's function reads

```math
\begin{equation}
G^{R}(t,t') =
    -i \theta(t-t') \langle [c(t), c^{\dagger}(t')]_{\mp} \rangle,
\end{equation}
```

Here, ``t``, ``t'`` belong to ``\mathcal{C}_1 ∪ \mathcal{C}_2``,
``\theta(t)`` is a step function, ``[,]_{-(+)}`` denotes an
(anti-)commutator. We choose the -(+) sign if the operators ``c``
and ``c^{\dagger}`` are bosonic (fermionic).

The retarded component is related to the advanced component by
hermitian conjugate:

```math
\begin{equation}
G^{R}(t,t') = G^{A}(t',t)^{*},
\end{equation}
```

and

```math
\begin{equation}
G^{R}(t,t')^{*} = G^{A}(t',t).
\end{equation}
```

The retarded component can be calculated with the lesser and greater
components:

```math
\begin{equation}
G^{R}(t,t') = \theta(t-t')[G^{>}(t,t') - G^{<}(t,t')].
\end{equation}
```

Note that ``G^{R}(t,t') = 0`` if ``t' > t``, which expresses the causality
of the retarded component. However, for the implementation of numerical
algorithms, it can be more convenient to drop the Heaviside function in
the above equation. Therefore, we define a modified retarded component by

```math
\begin{equation}
\tilde{G}^{R}(t,t') = G^{>}(t,t') - G^{<}(t,t').
\end{equation}
```

Its hermitian conjugate is as follows:

```math
\begin{equation}
\tilde{G}^{R}(t,t') = -\tilde{G}^{R}(t',t)^{*}.
\end{equation}
```

*Reference* :

Please see [`NESSi`] Session `3` [especially Eq. (20)] for more details.
=#

#=
### *Gʳᵉᵗ* : *Struct*
=#

"""
    Gʳᵉᵗ{T}

Retarded component (``G^R``) of contour-ordered Green's function. We
usually call this component `ret`.

See also: [`Gᵐᵃᵗ`](@ref), [`Gˡᵐⁱˣ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
mutable struct Gʳᵉᵗ{T} <: CnAbstractMatrix{T}
    type  :: String
    ntime :: I64
    ndim1 :: I64
    ndim2 :: I64
    # Contiguous storage for the lower triangle (i ≥ j) only,
    # data[:,:,k] is G^{R}(tᵢ,tⱼ), where k = (i-1)i/2 + j.
    data  :: Array{T,3}
end

#=
### *Gʳᵉᵗ* : *Constructors*
=#

"""
    Gʳᵉᵗ(ntime::I64, ndim1::I64, ndim2::I64, v::T)

Constructor. All the matrix elements are set to be `v`.
"""
function Gʳᵉᵗ(ntime::I64, ndim1::I64, ndim2::I64, v::T) where {T}
    # Sanity check
    @assert ntime ≥ 2
    @assert ndim1 ≥ 1
    @assert ndim2 ≥ 1

    # Create contiguous storage, whose size is (ndim1, ndim2, ntri).
    data = fill(v, ndim1, ndim2, _ntri(ntime))

    # Call the default constructor
    Gʳᵉᵗ("ret", ntime, ndim1, ndim2, data)
end

"""
    Gʳᵉᵗ(ntime::I64, ndim1::I64, ndim2::I64)

Constructor. All the matrix elements are set to be complex zero.
"""
function Gʳᵉᵗ(ntime::I64, ndim1::I64, ndim2::I64)
    Gʳᵉᵗ(ntime, ndim1, ndim2, zero(C64))
end

"""
    Gʳᵉᵗ(ntime::I64, ndim1::I64)

Constructor. All the matrix elements are set to be complex zero.
"""
function Gʳᵉᵗ(ntime::I64, ndim1::I64)
    Gʳᵉᵗ(ntime, ndim1, ndim1, zero(C64))
end

"""
    Gʳᵉᵗ(ntime::I64, x::Element{T})

Constructor. The matrix is initialized by `x`.
"""
function Gʳᵉᵗ(ntime::I64, x::Element{T}) where {T}
    # Sanity check
    @assert ntime ≥ 2

    ndim1, ndim2 = size(x)

    # Create contiguous storage, whose size is (ndim1, ndim2, ntri).
    data = Array{T,3}(undef, ndim1, ndim2, _ntri(ntime))
    for k = 1:_ntri(ntime)
        data[:,:,k] .= x
    end

    # Call the default constructor
    Gʳᵉᵗ("ret", ntime, ndim1, ndim2, data)
end

"""
    Gʳᵉᵗ(C::Cn, x::Element{T})

Constructor. The matrix is initialized by `x`.
"""
function Gʳᵉᵗ(C::Cn, x::Element{T}) where {T}
    # Sanity check
    @assert getdims(C) == size(x)

    # Create contiguous storage, whose size is (ndim1, ndim2, ntri).
    data = Array{T,3}(undef, C.ndim1, C.ndim2, _ntri(C.ntime))
    for k = 1:_ntri(C.ntime)
        data[:,:,k] .= x
    end

    # Call the default constructor
    Gʳᵉᵗ("ret", C.ntime, C.ndim1, C.ndim2, data)
end

"""
    Gʳᵉᵗ(C::Cn, v::T)

Constructor. All the matrix elements are set to be `v`.
"""
function Gʳᵉᵗ(C::Cn, v::T) where {T}
    Gʳᵉᵗ(C.ntime, C.ndim1, C.ndim2, v)
end

"""
    Gʳᵉᵗ(C::Cn)

Constructor. All the matrix elements are set to be complex zero.
"""
function Gʳᵉᵗ(C::Cn)
    Gʳᵉᵗ(C.ntime, C.ndim1, C.ndim2, zero(C64))
end

"""
    Gʳᵉᵗ()

Constructor. All the matrix elements are set to be complex zero. Notice
that all of the parameters are extracted from the `PCONTOUR` dict.

See also: [`PCONTOUR`](@ref), [`get_c`](@ref).
"""
function Gʳᵉᵗ()
    ntime = get_c("ntime")
    ndim1 = get_c("ndim1")
    ndim2 = get_c("ndim2")
    Gʳᵉᵗ(ntime, ndim1, ndim2, zero(C64))
end

#=
*Remarks : Advanced Green's Function*

The advanced component of contour-ordered Green's function reads

```math
\begin{equation}
G^{A}(t,t') =
    i \theta(t'-t) \langle [c(t), c^{\dagger}(t')]_{\mp} \rangle,
\end{equation}
```

Here, ``t``, ``t'`` belong to ``\mathcal{C}_1 ∪ \mathcal{C}_2``,
``\theta(t)`` is a step function, ``[,]_{-(+)}`` denotes an
(anti-)commutator. We choose the -(+) sign if the operators ``c``
and ``c^{\dagger}`` are bosonic (fermionic).
=#

#=
### *Gᵃᵈᵛ* : *Struct*
=#

"""
    Gᵃᵈᵛ{T}

Advanced component (``G^{A}``) of contour-ordered Green's function. We
usually call this component `adv`.

Note that currently we do not need this component explicitly. However,
for the sake of completeness, we still provide an implementation for it.

See also: [`Gᵐᵃᵗ`](@ref), [`Gˡᵐⁱˣ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
mutable struct Gᵃᵈᵛ{T} <: CnAbstractMatrix{T}
    type  :: String
    ntime :: I64
    ndim1 :: I64
    ndim2 :: I64
    dataR :: Ref{Gʳᵉᵗ{T}}
end

#=
### *Gᵃᵈᵛ* : *Constructors*
=#

"""
    Gᵃᵈᵛ(ret::Gʳᵉᵗ{T})

Constructor. Note that the `adv` component is not independent. We use
the `ret` component to initialize it.
"""
function Gᵃᵈᵛ(ret::Gʳᵉᵗ{T}) where {T}
    # Setup properties
    # Extract parameters from `ret`
    ntime = ret.ntime
    ndim1 = ret.ndim1
    ndim2 = ret.ndim2
    #
    # We don't allocate memory for `dataR` directly, but let it point to
    # the `ret` object.
    dataR = Ref(ret)

    # Call the default constructor
    Gᵃᵈᵛ("adv", ntime, ndim1, ndim2, dataR)
end

#=
### *gʳᵉᵗ* : *Struct*
=#

"""
    gʳᵉᵗ{S}

Retarded component (``G^{R}``) of contour-ordered Green's function at
given time step `tstp`. Actually, it denotes ``G^{R}(tᵢ = tstp, tⱼ)``.

See also: [`gᵐᵃᵗ`](@ref), [`gˡᵐⁱˣ`](@ref), [`gˡᵉˢˢ`](@ref).
"""
mutable struct gʳᵉᵗ{S} <: CnAbstractVector{S}
    type  :: String
    tstp  :: I64
    ndim1 :: I64
    ndim2 :: I64
    data  :: VecArray{S}
end

#=
### *gʳᵉᵗ* : *Constructors*
=#

"""
    gʳᵉᵗ(tstp::I64, ndim1::I64, ndim2::I64, v::S)

Constructor. All the vector elements are set to be `v`.
"""
function gʳᵉᵗ(tstp::I64, ndim1::I64, ndim2::I64, v::S) where {S}
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
    gʳᵉᵗ("ret", tstp, ndim1, ndim2, data)
end

"""
    gʳᵉᵗ(tstp::I64, ndim1::I64, ndim2::I64)

Constructor. All the vector elements are set to be complex zero.
"""
function gʳᵉᵗ(tstp::I64, ndim1::I64, ndim2::I64)
    gʳᵉᵗ(tstp, ndim1, ndim2, zero(C64))
end

"""
    gʳᵉᵗ(tstp::I64, ndim1::I64)

Constructor. All the vector elements are set to be complex zero.
"""
function gʳᵉᵗ(tstp::I64, ndim1::I64)
    gʳᵉᵗ(tstp, ndim1, ndim1, zero(C64))
end

"""
    gʳᵉᵗ(tstp::I64, x::Element{S})

Constructor. The vector is initialized by `x`.
"""
function gʳᵉᵗ(tstp::I64, x::Element{S}) where {S}
    # Sanity check
    @assert tstp ≥ 1

    ndim1, ndim2 = size(x)

    # Create VecArray{S}, whose size is indeed (tstp,)
    data = VecArray{S}(undef, tstp)
    for i = 1:tstp
        data[i] = copy(x)
    end

    # Call the default constructor
    gʳᵉᵗ("ret", tstp, ndim1, ndim2, data)
end

#=
### *gᵃᵈᵛ* : *Struct*
=#

"""
    gᵃᵈᵛ{S}

Advanced component (``G^{A}``) of contour-ordered Green's function at
given time step `tstp`.

Note that currently we do not need this component explicitly. However,
for the sake of completeness, we still provide an implementation for it.

See also: [`gᵐᵃᵗ`](@ref), [`gˡᵐⁱˣ`](@ref), [`gˡᵉˢˢ`](@ref).
"""
mutable struct gᵃᵈᵛ{S} <: CnAbstractVector{S}
    type  :: String
    tstp  :: I64
    ndim1 :: I64
    ndim2 :: I64
    dataR :: Ref{gʳᵉᵗ{S}}
end

#=
### *gᵃᵈᵛ* : *Constructors*
=#

"""
    gᵃᵈᵛ(g)

Constructor. Note that the `adv` component is not independent. We use
the `ret` component to initialize it.
"""
function gᵃᵈᵛ(ret::gʳᵉᵗ{S}) where {S}
    # Setup properties
    # Extract parameters from `ret`
    tstp  = ret.tstp
    ndim1 = ret.ndim1
    ndim2 = ret.ndim2
    #
    # We don't allocate memory for `dataR` directly, but let it point to
    # the `ret` object.
    dataR = Ref(ret)

    # Call the default constructor
    gᵃᵈᵛ("adv", tstp, ndim1, ndim2, dataR)
end
