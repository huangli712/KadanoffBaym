


#=
*Remarks : Lesser Green's Function*

The lesser component of contour Green's function reads

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
G^{<}(t,t')^{*} = -G^{<}(t',t).
\end{equation}
```

The lesser component is related to the retarded, advanced, and Keldysh
Green's functions via

```math
\begin{equation}
G^{<} = \frac{1}{2}(G^{K} - G^{R} + G^{A}).
\end{equation}
```
=#

#=
### *Gˡᵉˢˢ* : *Struct*
=#

"""
    Gˡᵉˢˢ{T}

Lesser component (``G^{<}``) of contour Green's function. We usually
call this component `less`.

See also: [`Gᵐᵃᵗ`](@ref), [`Gʳᵉᵗ`](@ref), [`Gˡᵐⁱˣ`](@ref).
"""
mutable struct Gˡᵉˢˢ{T} <: CnAbstractMatrix{T}
    type  :: String
    ntime :: I64
    ndim1 :: I64
    ndim2 :: I64
    data  :: MatArray{T}
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

    # Create Element{T}
    element = fill(v, ndim1, ndim2)

    # Create MatArray{T}, whose size is indeed (ntime, ntime).
    data = MatArray{T}(undef, ntime, ntime)
    for i = 1:ntime
        for j = 1:ntime
            data[j,i] = copy(element)
        end
    end

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
    data = MatArray{T}(undef, ntime, ntime)
    for i = 1:ntime
        for j = 1:ntime
            data[j,i] = copy(x)
        end
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

    # Create MatArray{T}, whose size is indeed (ntime, ntime).
    data = MatArray{T}(undef, C.ntime, C.ntime)
    for i = 1:C.ntime
        for j = 1:C.ntime
            data[j,i] = copy(x)
        end
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

#=
### *Gˡᵉˢˢ* : *Properties*
=#

"""
    getdims(less::Gˡᵉˢˢ{T})

Return the dimensional parameters of contour function.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function getdims(less::Gˡᵉˢˢ{T}) where {T}
    return (less.ndim1, less.ndim2)
end

"""
    getsize(less::Gˡᵉˢˢ{T})

Return the size of contour function.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function getsize(less::Gˡᵉˢˢ{T}) where {T}
    return less.ntime
end

"""
    equaldims(less::Gˡᵉˢˢ{T})

Return whether the dimensional parameters are equal.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function equaldims(less::Gˡᵉˢˢ{T}) where {T}
    return less.ndim1 == less.ndim2
end

"""
    iscompatible(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T})

Judge whether two `Gˡᵉˢˢ` objects are compatible.
"""
function iscompatible(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}) where {T}
    getsize(less1) == getsize(less2) &&
    getdims(less1) == getdims(less2)
end

"""
    iscompatible(C::Cn, less::Gˡᵉˢˢ{T})

Judge whether `C` (which is a `Cn` object) is compatible with `less`
(which is a `Gˡᵉˢˢ{T}` object).
"""
function iscompatible(C::Cn, less::Gˡᵉˢˢ{T}) where {T}
    C.ntime == getsize(less) &&
    getdims(C) == getdims(less)
end

"""
    iscompatible(less::Gˡᵉˢˢ{T}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `less`
(which is a `Gˡᵉˢˢ{T}` object).
"""
iscompatible(less::Gˡᵉˢˢ{T}, C::Cn) where {T} = iscompatible(C, less)

"""
    distance(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}, tstp::I64)

Calculate distance between two `Gˡᵉˢˢ` objects at given time step `tstp`.
"""
function distance(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}, tstp::I64) where {T}
    # Sanity check
    @assert 1 ≤ tstp ≤ less1.ntime

    err = 0
    #
    for i = 1:tstp
        err = err + abs(sum(less1.data[i,tstp] - less2.data[i,tstp]))
    end
    #
    return err
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
### *Gˡᵉˢˢ* : *Operations*
=#

"""
    memset!(less::Gˡᵉˢˢ{T}, x)

Reset all the matrix elements of `less` to `x`. `x` should be a
scalar number.
"""
function memset!(less::Gˡᵉˢˢ{T}, x) where {T}
    cx = convert(T, x)
    for i=1:less.ntime
        for j=1:less.ntime
            fill!(less.data[j,i], cx)
        end
    end
end

"""
    memset!(less::Gˡᵉˢˢ{T}, tstp::I64, x)

Reset the matrix elements of `less` at given time step `tstp` (and at all
`t` where `t < tstp`) to `x`. `x` should be a scalar number.
"""
function memset!(less::Gˡᵉˢˢ{T}, tstp::I64, x) where {T}
    @assert 1 ≤ tstp ≤ less.ntime
    cx = convert(T, x)
    for i=1:tstp
        fill!(less.data[i,tstp], cx)
    end
end

"""
    zeros!(less::Gˡᵉˢˢ{T})

Reset all the matrix elements of `less` to `zero`.
"""
zeros!(less::Gˡᵉˢˢ{T}) where {T} = memset!(less, zero(T))

"""
    zeros!(less::Gˡᵉˢˢ{T}, tstp::I64)

Reset the matrix elements of `less` at given time step `tstp` (and at all
`t` where `t < tstp`) to `zero`.
"""
zeros!(less::Gˡᵉˢˢ{T}, tstp::I64) where {T} = memset!(less, tstp, zero(T))

"""
    memcpy!(src::Gˡᵉˢˢ{T}, dst::Gˡᵉˢˢ{T})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::Gˡᵉˢˢ{T}, dst::Gˡᵉˢˢ{T}) where {T}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memcpy!(src::Gˡᵉˢˢ{T}, dst::Gˡᵉˢˢ{T}, tstp::I64)

Copy some matrix elements from `src` to `dst`. Only the matrix elements
at given time step `tstp` (and at all `t` where `t < tstp`) are copied.
"""
function memcpy!(src::Gˡᵉˢˢ{T}, dst::Gˡᵉˢˢ{T}, tstp::I64) where {T}
    @assert iscompatible(src, dst)
    @assert 1 ≤ tstp ≤ src.ntime
    for i=1:tstp
        dst.data[i,tstp] = copy(src.data[i,tstp])
    end
end

"""
    incr!(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}, tstp::I64, α::T)

Add a `Gˡᵉˢˢ` with given weight (`α`) at given time step `tstp` (and at
all `t` where `t < tstp`) to another `Gˡᵉˢˢ`.
"""
function incr!(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}, tstp::I64, α::T) where {T}
    @assert iscompatible(less1, less2)
    @assert 1 ≤ tstp ≤ less2.ntime
    for i = 1:tstp
        @. less1.data[i,tstp] = less1.data[i,tstp] + less2.data[i,tstp] * α
    end
end

"""
    smul!(less::Gˡᵉˢˢ{T}, tstp::I64, α::T)

Multiply a `Gˡᵉˢˢ` with given weight (`α`) at given time step `tstp` (and
at all `t` where `t < tstp`).
"""
function smul!(less::Gˡᵉˢˢ{T}, tstp::I64, α::T) where {T}
    @assert 1 ≤ tstp ≤ less.ntime
    for i = 1:tstp
        @. less.data[i,tstp] = less.data[i,tstp] * α
    end
end

"""
    smul!(x::Cf{T}, less::Gˡᵉˢˢ{T}, tstp::I64)

Left multiply a `Gˡᵉˢˢ` with given weight (`x`) at given time step `tstp`
(and at all `t` where `t < tstp`).
"""
function smul!(x::Cf{T}, less::Gˡᵉˢˢ{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ less.ntime
    for i = 1:tstp
        less.data[i,tstp] = x[i] * less.data[i,tstp]
    end
end

"""
    smul!(less::Gˡᵉˢˢ{T}, x::Element{T}, tstp::I64)

Right multiply a `Gˡᵉˢˢ` with given weight (`x`) at given time step `tstp`
(and at all `t` where `t < tstp`).
"""
function smul!(less::Gˡᵉˢˢ{T}, x::Element{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ less.ntime
    for i = 1:tstp
        less.data[i,tstp] = less.data[i,tstp] * x
    end
end

#=
### *Gˡᵉˢˢ* : *Traits*
=#

"""
    Base.:+(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T})

Operation `+` for two `Gˡᵉˢˢ` objects.
"""
function Base.:+(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}) where {T}
    # Sanity check
    @assert getsize(less1) == getsize(less2)
    @assert getdims(less1) == getdims(less2)

    Gˡᵉˢˢ(less1.type, less1.ntime, less1.ndim1, less1.ndim2, less1.data + less2.data)
end

"""
    Base.:-(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T})

Operation `-` for two `Gˡᵉˢˢ` objects.
"""
function Base.:-(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}) where {T}
    # Sanity check
    @assert getsize(less1) == getsize(less2)
    @assert getdims(less1) == getdims(less2)

    Gˡᵉˢˢ(less1.type, less1.ntime, less1.ndim1, less1.ndim2, less1.data - less2.data)
end

"""
    Base.:*(less::Gˡᵉˢˢ{T}, x)

Operation `*` for a `Gˡᵉˢˢ` object and a scalar value.
"""
function Base.:*(less::Gˡᵉˢˢ{T}, x) where {T}
    cx = convert(T, x)
    Gˡᵉˢˢ(less.type, less.ntime, less.ndim1, less.ndim2, less.data * cx)
end

"""
    Base.:*(x, less::Gˡᵉˢˢ{T})

Operation `*` for a scalar value and a `Gˡᵉˢˢ` object.
"""
Base.:*(x, less::Gˡᵉˢˢ{T}) where {T} = Base.:*(less, x)

#=
### *Gᵐᵃᵗᵐ* : *Struct*
=#

"""
    Gᵐᵃᵗᵐ{T}

Matsubara component (``G^M``) of contour Green's function. It is designed
for ``\tau < 0`` case. It is not an independent component. It can be
inferred or deduced from the `Gᵐᵃᵗ{T}` struct. We usually call this
component `matm`.

See also: [`Gʳᵉᵗ`](@ref), [`Gˡᵐⁱˣ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
mutable struct Gᵐᵃᵗᵐ{T} <: CnAbstractMatrix{T}
    type  :: String
    sign  :: I64 # Used to distinguish fermions and bosons
    ntau  :: I64
    ndim1 :: I64
    ndim2 :: I64
    dataM :: Ref{Gᵐᵃᵗ{T}}
end

#=
### *Gᵐᵃᵗᵐ* : *Constructors*
=#

"""
    Gᵐᵃᵗᵐ(sign::I64, mat::Gᵐᵃᵗ{T})

Constructor. Note that the `matm` component is not independent. We use
the `mat` component to initialize it.
"""
function Gᵐᵃᵗᵐ(sign::I64, mat::Gᵐᵃᵗ{T}) where {T}
    # Sanity check
    @assert sign in (BOSE, FERMI)

    # Setup properties
    # Extract parameters from `mat`
    ntau = mat.ntau
    ndim1 = mat.ndim1
    ndim2 = mat.ndim2
    #
    # We don't allocate memory for `dataM` directly, but let it point to
    # the `mat` object.
    dataM = Ref(mat)

    # Call the default constructor
    Gᵐᵃᵗᵐ("matm", sign, ntau, ndim1, ndim2, dataM)
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
*Remarks : Advanced Green's Function*

The advanced component of contour Green's function reads

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

Advanced component (``G^{A}``) of contour Green's function. We usually
call this component `adv`.

Note that currently we do not need this component explicitly. However,
for the sake of completeness, we still define an empty struct for it.

See also: [`Gᵐᵃᵗ`](@ref), [`Gˡᵐⁱˣ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
mutable struct Gᵃᵈᵛ{T} <: CnAbstractMatrix{T}
    type  :: String
end

#=
### *Gᵃᵈᵛ* : *Constructors*
=#

"""
    Gᵃᵈᵛ()

Constructor. Note that the `adv` component is not independent. We use
the `ret` component to initialize it.
"""
function Gᵃᵈᵛ()
    # Call the default constructor
    Gᵃᵈᵛ("adv")
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
*Remarks : Right-mixing Green's Function*

The right-mixing component of contour Green's function reads

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

Right-mixing component (``G^{⌈}``) of contour Green's function. We usually
call this component `rmix`.

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
*Remarks : Greater Green's Function*

The greater component of contour Green's function reads

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

Greater component (``G^{>}``) of contour Green's function. We usually
call this component `gtr`.

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
