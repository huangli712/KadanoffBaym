
#=
### *Cn* : *Constructors*
=#

"""
    Cn(ntime::I64, ntau::I64, ndim1::I64, ndim2::I64, tmax::F64, beta::F64)

Constructor. Create a general 𝐿-shape `Kadanoff-Baym` contour.
"""
function Cn(ntime::I64, ntau::I64,
            ndim1::I64, ndim2::I64,
            tmax::F64, beta::F64)
    # Sanity check
    @assert ntime ≥ 2
    @assert ntau ≥ 2
    @assert ndim1 ≥ 1
    @assert ndim2 ≥ 1
    @assert tmax > 0.0
    @assert beta > 0.0

    # Evaluate `dt` and `dtau`
    dt = tmax / ( ntime - 1 )
    dtau = beta / ( ntau - 1 )

    # Call the default constructor
    Cn(ntime, ntau, ndim1, ndim2, tmax, beta, dt, dtau)
end

"""
    Cn(ndim1::I64, ndim2::I64, tmax::F64, beta::F64)

Constructor. With default `ntime` (= 201) and `ntau` (= 1001). For
general matrix system.
"""
function Cn(ndim1::I64, ndim2::I64, tmax::F64, beta::F64)
    ntime = 201
    ntau = 1001
    Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
end

"""
    Cn(ndim1::I64, tmax::F64, beta::F64)

Constructor. With default `ntime` (= 201) and `ntau` (= 1001). For
special square matrix system.
"""
function Cn(ndim1::I64, tmax::F64, beta::F64)
    ndim2 = ndim1
    Cn(ndim1, ndim2, tmax, beta)
end

"""
    Cn(tmax::F64, beta::F64)

Constructor. With default `ntime` (= 201) and `ntau` (= 1001). For
special single band system.
"""
function Cn(tmax::F64, beta::F64)
    ndim1 = 1
    ndim2 = 1
    Cn(ndim1, ndim2, tmax, beta)
end

#=
### *Cn* : *Properties*
=#

"""
    getdims(C::Cn)

Return the dimensional parameters of contour.

See also: [`Cn`](@ref).
"""
function getdims(C::Cn)
    return (C.ndim1, C.ndim2)
end

"""
    equaldims(C::Cn)

Return whether the dimensional parameters are equal.

See also: [`Cn`](@ref).
"""
function equaldims(C::Cn)
    return C.ndim1 == C.ndim2
end

#=
### *Cn* : *Operations*
=#

"""
    refresh!(C::Cn)

Update the `dt` and `dtau` parameters of contour.

See also: [`Cn`](@ref).
"""
function refresh!(C::Cn)
    # Sanity check
    @assert C.ntime ≥ 2
    @assert C.ntau ≥ 2

    # Evaluate `dt` and `dtau` again
    C.dt = C.tmax / ( C.ntime - 1 )
    C.dtau = C.beta / ( C.ntau - 1 )
end

#=
### *Cn* : *Traits*
=#

"""
    Base.show(io::IO, C::Cn)

Display `Cn` struct on the terminal.

See also: [`Cn`](@ref).
"""
function Base.show(io::IO, C::Cn)
    println("ntime : ", C.ntime)
    println("ntau  : ", C.ntau )
    println("ndim1 : ", C.ndim1)
    println("ndim2 : ", C.ndim2)
    println("tmax  : ", C.tmax )
    println("beta  : ", C.beta )
    println("dt    : ", C.dt   )
    println("dtau  : ", C.dtau )
end

#=
*Contour-based Functions* :

It is a general matrix-valued function defined at the `Kadanoff-Baym`
contour:

```math
\begin{equation}
\mathcal{F} = f(t),
\end{equation}
```

where ``t \in \mathcal{C}_1 \cup \mathcal{C}_2 \cup \mathcal{C}_3``.
=#

#=
### *Cf* : *Struct*
=#

"""
    Cf{T}

It is a square-matrix-valued or rectangle-matrix-valued function of time.

See also: [`ℱ`](@ref), [`𝒻`](@ref).
"""
mutable struct Cf{T} <: CnAbstractFunction{T}
    ntime :: I64
    ndim1 :: I64
    ndim2 :: I64
    data  :: VecArray{T}
end

#=
### *Cf* : *Constructors*
=#

"""
    Cf(ntime::I64, ndim1::I64, ndim2::I64, v::T)

Constructor. All the matrix elements are set to be `v`.
"""
function Cf(ntime::I64, ndim1::I64, ndim2::I64, v::T) where {T}
    # Sanity check
    #
    # If `ntime = 0`, it means that the system stays at the equilibrium
    # state and this function is defined at the Matsubara axis only.
    @assert ntime ≥ 0
    @assert ndim1 ≥ 1
    @assert ndim2 ≥ 1

    # Create Element{T}
    element = fill(v, ndim1, ndim2)

    # Create VecArray{T}, whose size is indeed (ntime + 1,).
    #
    # Be careful, data[end] is the value of the function on the
    # Matsubara axis (initial state), while data[1:ntime] is for
    # the time evolution part.
    data = VecArray{T}(undef, ntime + 1)
    for i = 1:ntime + 1
        data[i] = copy(element)
    end

    # Call the default constructor
    Cf(ntime, ndim1, ndim2, data)
end

"""
    Cf(ntime::I64, ndim1::I64, ndim2::I64)

Constructor. All the matrix elements are set to be `CZERO`.
"""
function Cf(ntime::I64, ndim1::I64, ndim2::I64)
    Cf(ntime, ndim1, ndim2, CZERO)
end

"""
    Cf(ntime::I64, ndim1::I64)

Constructor. All the matrix elements are set to be `CZERO`.
"""
function Cf(ntime::I64, ndim1::I64)
    Cf(ntime, ndim1, ndim1, CZERO)
end

"""
    Cf(ntime::I64, x::Element{T})

Constructor. The matrix is initialized by `x`.
"""
function Cf(ntime::I64, x::Element{T}) where {T}
    # Sanity check
    @assert ntime ≥ 0

    ndim1, ndim2 = size(x)
    data = VecArray{T}(undef, ntime + 1)
    for i = 1:ntime + 1
        data[i] = copy(x)
    end

    # Call the default constructor
    Cf(ntime, ndim1, ndim2, data)
end

"""
    Cf(C::Cn, x::Element{T})

Constructor. The matrix is initialized by `x`.
"""
function Cf(C::Cn, x::Element{T}) where {T}
    # Sanity check
    @assert getdims(C) == size(x)

    # Create VecArray{T}, whose size is indeed (ntime + 1,).
    data = VecArray{T}(undef, C.ntime + 1)
    for i = 1:C.ntime + 1
        data[i] = copy(x)
    end

    # Call the default constructor
    Cf(C.ntime, C.ndim1, C.ndim2, data)
end

"""
    Cf(C::Cn, v::T)

Constructor. All the matrix elements are set to be `v`.
"""
function Cf(C::Cn, v::T) where {T}
    Cf(C.ntime, C.ndim1, C.ndim2, v)
end

"""
    Cf(C::Cn)

Constructor. All the matrix elements are set to be `CZERO`.
"""
function Cf(C::Cn)
    Cf(C.ntime, C.ndim1, C.ndim2, CZERO)
end

#=
### *Cf* : *Properties*
=#

"""
    getdims(cff::Cf{T})

Return the dimensional parameters of contour function.

See also: [`Cf`](@ref).
"""
function getdims(cff::Cf{T}) where {T}
    return (cff.ndim1, cff.ndim2)
end

"""
    getsize(cff::Cf{T})

Return the nominal size of contour function, i.e `ntime`. Actually, the
real size of contour function should be `ntime + 1`.

See also: [`Cf`](@ref).
"""
function getsize(cff::Cf{T}) where {T}
    return cff.ntime
end

"""
    equaldims(cff::Cf{T})

Return whether the dimensional parameters are equal.

See also: [`Cf`](@ref).
"""
function equaldims(cff::Cf{T}) where {T}
    return cff.ndim1 == cff.ndim2
end

"""
    iscompatible(cff1::Cf{T}, cff2::Cf{T})

Judge whether two `Cf` objects are compatible.
"""
function iscompatible(cff1::Cf{T}, cff2::Cf{T}) where {T}
    getsize(cff1) == getsize(cff2) &&
    getdims(cff1) == getdims(cff2)
end

"""
    iscompatible(C::Cn, cff::Cf{T})

Judge whether `C` (which is a `Cn` object) is compatible with `cff`
(which is a `Cf{T}` object).
"""
function iscompatible(C::Cn, cff::Cf{T}) where {T}
    C.ntime == getsize(cff) &&
    getdims(C) == getdims(cff)
end

"""
    iscompatible(cff::Cf{T}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `cff`
(which is a `Cf{T}` object).
"""
iscompatible(cff::Cf{T}, C::Cn) where {T} = iscompatible(C, cff)

#=
### *Cf* : *Indexing*
=#

"""
    Base.getindex(cff::Cf{T}, i::I64)

Visit the element stored in `Cf` object. If `i = 0`, it returns
the element at Matsubara axis. On the other hand, if `it > 0`, it will
return elements at real time axis.
"""
function Base.getindex(cff::Cf{T}, i::I64) where {T}
    # Sanity check
    @assert 0 ≤ i ≤ cff.ntime

    # Return 𝑓(𝑡ᵢ)
    if i == 0 # Matsubara axis
        cff.data[end]
    else # Real time axis
        cff.data[i]
    end
end

"""
    Base.setindex!(cff::Cf{T}, x::Element{T}, i::I64)

Setup the element in `Cf` object. If `i = 0`, it will setup the
element at Matsubara axis to `x`. On the other hand, if `it > 0`, it
will setup elements at real time axis.
"""
function Base.setindex!(cff::Cf{T}, x::Element{T}, i::I64) where {T}
    # Sanity check
    @assert size(x) == getdims(cff)
    @assert 0 ≤ i ≤ cff.ntime

    # 𝑓(𝑡ᵢ) = x
    if i == 0 # Matsubara axis
        cff.data[end] = copy(x)
    else # Real time axis
        cff.data[i] = copy(x)
    end
end

"""
    Base.setindex!(cff::Cf{T}, v::T, i::I64)

Setup the element in `Cf` object. If `i = 0`, it will setup the
element at Matsubara axis to `v`. On the other hand, if `it > 0`, it
will setup elements at real time axis. Here, `v` should be a scalar
number.
"""
function Base.setindex!(cff::Cf{T}, v::T, i::I64) where {T}
    # Sanity check
    @assert 0 ≤ i ≤ cff.ntime

    # 𝑓(𝑡ᵢ) .= v
    if i == 0 # Matsubara axis
        fill!(cff.data[end], v)
    else # Real time axis
        fill!(cff.data[i], v)
    end
end

#=
### *Cf* : *Operations*
=#

"""
    memset!(cff::Cf{T}, x)

Reset all the matrix elements of `cff` to `x`. `x` should be a
scalar number.
"""
function memset!(cff::Cf{T}, x) where {T}
    cx = convert(T, x)
    for i = 1:cff.ntime + 1
        fill!(cff.data[i], cx)
    end
end

"""
    zeros!(cff::Cf{T})

Reset all the matrix elements of `cff` to `ZERO`.
"""
zeros!(cff::Cf{T}) where {T} = memset!(cff, zero(T))

"""
    memcpy!(src::Cf{T}, dst::Cf{T})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::Cf{T}, dst::Cf{T}) where {T}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    incr!(cff1::Cf{T}, cff2::Cf{T}, alpha::T)

Add a `Cf` with given weight (`alpha`) to another `Cf`. Finally,
`cff1` will be changed.
"""
function incr!(cff1::Cf{T}, cff2::Cf{T}, alpha::T) where {T}
    @assert iscompatible(cff1, cff2)
    for i = 1:cff1.ntime + 1
        @. cff1.data[i] = cff1.data[i] + cff2.data[i] * alpha
    end
end

"""
    smul!(cff::Cf{T}, alpha::T)

Multiply a `Cf` with given weight (`alpha`).
"""
function smul!(cff::Cf{T}, alpha::T) where {T}
    for i = 1:cff.ntime + 1
        @. cff.data[i] = cff.data[i] * alpha
    end
end

"""
    smul!(x::Element{T}, cff::Cf{T})

Left multiply a `Cf` with given weight (`x`).
"""
function smul!(x::Element{T}, cff::Cf{T}) where {T}
    for i = 1:cff.ntime + 1
        cff.data[i] = x * cff.data[i]
    end
end

"""
    smul!(cff::Cf{T}, x::Element{T})

Right multiply a `Cf` with given weight (`x`).
"""
function smul!(cff::Cf{T}, x::Element{T}) where {T}
    for i = 1:cff.ntime + 1
        cff.data[i] = cff.data[i] * x
    end
end

#=
### *Cf* : *Traits*
=#

"""
    Base.:+(cff1::Cf{T}, cff2::Cf{T})

Operation `+` for two `Cf` objects.
"""
function Base.:+(cff1::Cf{T}, cff2::Cf{T}) where {T}
    # Sanity check
    @assert getsize(cff1) == getsize(cff2)
    @assert getdims(cff1) == getdims(cff2)

    Cf(cff1.ntime, cff1.ndim1, cff1.ndim2, cff1.data + cff2.data)
end

"""
    Base.:-(cff1::Cf{T}, cff2::Cf{T})

Operation `-` for two `Cf` objects.
"""
function Base.:-(cff1::Cf{T}, cff2::Cf{T}) where {T}
    # Sanity check
    @assert getsize(cff1) == getsize(cff2)
    @assert getdims(cff1) == getdims(cff2)

    Cf(cff1.ntime, cff1.ndim1, cff1.ndim2, cff1.data - cff2.data)
end

"""
    Base.:*(cff::Cf{T}, x)

Operation `*` for a `Cf` object and a scalar value.
"""
function Base.:*(cff::Cf{T}, x) where {T}
    cx = convert(T, x)
    Cf(cff.ntime, cff.ndim1, cff.ndim2, cff.data * cx)
end

"""
    Base.:*(x, cff::Cf{T})

Operation `*` for a scalar value and a `Cf` object.
"""
Base.:*(x, cff::Cf{T}) where {T} = Base.:*(cff, x)

#=
*Matsubara Green's Function* :

The Matsubara component of contour Green's function reads

```math
\begin{equation}
G^{M}(\tau, \tau') =
    -\langle \mathcal{T}_{\tau} c(\tau) c^{\dagger}(\tau') \rangle,
\end{equation}
```

where ``\tau`` and ``\tau'`` lie in ``\mathcal{C}_3``.

Note that the Matsubara component ``G^{M}`` plays a somewhat special
role, since it is always translationally invariant (``\mathcal{H}``
does not depend on imaginary time):

```math
\begin{equation}
G^{M}(\tau,\tau') \equiv G^{M}(\tau - \tau').
\end{equation}
```

Furthermore, it is real (Hermitian),

```math
\begin{equation}
G^{M}(\tau)^{*} = G^{M}(\tau),
\end{equation}
```

and it is also periodic (antiperiodic) for bosons (fermions),

```math
\begin{equation}
G^{M}(\tau) = \pm G^{M}(\tau + \beta).
\end{equation}
```

One can thus use its Fourier decomposition in terms of Matsubara
frequencies

```math
\begin{equation}
G^{M}(\tau,\tau') =
    T \sum_n e^{-i\omega_n(\tau-\tau')} G^{M}(i\omega_n),
\end{equation}
```

and

```math
\begin{equation}
G^{M}(i\omega_n) = \int^{\beta}_0 d\tau e^{i\omega_n}G^{M}(\tau).
\end{equation}
```
=#

#=
### *Gᵐᵃᵗ* : *Struct*
=#

"""
    Gᵐᵃᵗ{T}

Matsubara component (``G^M``) of contour Green's function. We usually
call this component `mat`. Here we just assume ``\tau ≥ 0``. While for
``\tau < 0``, please turn to the `Gᵐᵃᵗᵐ{T}` struct.

See also: [`Gʳᵉᵗ`](@ref), [`Gˡᵐⁱˣ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
mutable struct Gᵐᵃᵗ{T} <: CnAbstractMatrix{T}
    ntau  :: I64
    ndim1 :: I64
    ndim2 :: I64
    data  :: MatArray{T}
end

#=
### *Gᵐᵃᵗ* : *Constructors*
=#

"""
    Gᵐᵃᵗ(ntau::I64, ndim1::I64, ndim2::I64, v::T)

Constructor. All the matrix elements are set to be `v`.
"""
function Gᵐᵃᵗ(ntau::I64, ndim1::I64, ndim2::I64, v::T) where {T}
    # Sanity check
    @assert ntau ≥ 2
    @assert ndim1 ≥ 1
    @assert ndim2 ≥ 1

    # Create Element{T}
    element = fill(v, ndim1, ndim2)

    # Create MatArray{T}, whose size is indeed (ntau, 1).
    data = MatArray{T}(undef, ntau, 1)
    for i=1:ntau
        data[i,1] = copy(element)
    end

    # Call the default constructor
    Gᵐᵃᵗ(ntau, ndim1, ndim2, data)
end

"""
    Gᵐᵃᵗ(ntau::I64, ndim1::I64, ndim2::I64)

Constructor. All the matrix elements are set to be `CZERO`.
"""
function Gᵐᵃᵗ(ntau::I64, ndim1::I64, ndim2::I64)
    Gᵐᵃᵗ(ntau, ndim1, ndim2, CZERO)
end

"""
    Gᵐᵃᵗ(ntau::I64, ndim1::I64)

Constructor. All the matrix elements are set to be `CZERO`.
"""
function Gᵐᵃᵗ(ntau::I64, ndim1::I64)
    Gᵐᵃᵗ(ntau, ndim1, ndim1, CZERO)
end

"""
    Gᵐᵃᵗ(ntau::I64, x::Element{T})

Constructor. The matrix is initialized by `x`.
"""
function Gᵐᵃᵗ(ntau::I64, x::Element{T}) where {T}
    # Sanity check
    @assert ntau ≥ 2

    ndim1, ndim2 = size(x)
    data = MatArray{T}(undef, ntau, 1)
    for i=1:ntau
        data[i,1] = copy(x)
    end

    # Call the default constructor
    Gᵐᵃᵗ(ntau, ndim1, ndim2, data)
end

"""
    Gᵐᵃᵗ(C::Cn, x::Element{T})

Constructor. The matrix is initialized by `x`.
"""
function Gᵐᵃᵗ(C::Cn, x::Element{T}) where {T}
    # Sanity check
    @assert getdims(C) == size(x)

    # Create MatArray{T}, whose size is indeed (ntau, 1).
    data = MatArray{T}(undef, C.ntau, 1)
    for i=1:C.ntau
        data[i,1] = copy(x)
    end

    # Call the default constructor
    Gᵐᵃᵗ(C.ntau, C.ndim1, C.ndim2, data)
end

"""
    Gᵐᵃᵗ(C::Cn, v::T)

Constructor. All the matrix elements are set to be `v`.
"""
function Gᵐᵃᵗ(C::Cn, v::T) where {T}
    Gᵐᵃᵗ(C.ntau, C.ndim1, C.ndim2, v)
end

"""
    Gᵐᵃᵗ(C::Cn)

Constructor. All the matrix elements are set to be `CZERO`.
"""
function Gᵐᵃᵗ(C::Cn)
    Gᵐᵃᵗ(C.ntau, C.ndim1, C.ndim2, CZERO)
end

#=
### *Gᵐᵃᵗ* : *Properties*
=#

"""
    getdims(mat::Gᵐᵃᵗ{T})

Return the dimensional parameters of contour function.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function getdims(mat::Gᵐᵃᵗ{T}) where {T}
    return (mat.ndim1, mat.ndim2)
end

"""
    getsize(mat::Gᵐᵃᵗ{T})

Return the size of contour function. Here, it should be `ntau`.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function getsize(mat::Gᵐᵃᵗ{T}) where {T}
    return mat.ntau
end

"""
    equaldims(mat::Gᵐᵃᵗ{T})

Return whether the dimensional parameters are equal.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function equaldims(mat::Gᵐᵃᵗ{T}) where {T}
    return mat.ndim1 == mat.ndim2
end

"""
    iscompatible(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T})

Judge whether two `Gᵐᵃᵗ` objects are compatible.
"""
function iscompatible(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T}) where {T}
    getsize(mat1) == getsize(mat2) &&
    getdims(mat1) == getdims(mat2)
end

"""
    iscompatible(C::Cn, mat::Gᵐᵃᵗ{T})

Judge whether `C` (which is a `Cn` object) is compatible with `mat`
(which is a `Gᵐᵃᵗ{T}` object).
"""
function iscompatible(C::Cn, mat::Gᵐᵃᵗ{T}) where {T}
    C.ntau == getsize(mat) &&
    getdims(C) == getdims(mat)
end

"""
    iscompatible(mat::Gᵐᵃᵗ{T}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `mat`
(which is a `Gᵐᵃᵗ{T}` object).
"""
iscompatible(mat::Gᵐᵃᵗ{T}, C::Cn) where {T} = iscompatible(C, mat)

"""
    distance(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T})

Calculate distance between two `Gᵐᵃᵗ` objects.
"""
function distance(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T}) where {T}
    @assert iscompatible(mat1, mat2)

    err = 0.0
    #
    for m = 1:mat1.ntau
        err = err + abs(sum(mat1.data[m,1] - mat2.data[m,1]))
    end
    #
    return err
end

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
### *Gᵐᵃᵗ* : *Operations*
=#

"""
    memset!(mat::Gᵐᵃᵗ{T}, x)

Reset all the matrix elements of `mat` to `x`. `x` should be a
scalar number.
"""
function memset!(mat::Gᵐᵃᵗ{T}, x) where {T}
    cx = convert(T, x)
    for i = 1:mat.ntau
        fill!(mat.data[i,1], cx)
    end
end

"""
    zeros!(mat::Gᵐᵃᵗ{T})

Reset all the matrix elements of `mat` to `ZERO`.
"""
zeros!(mat::Gᵐᵃᵗ{T}) where {T} = memset!(mat, zero(T))

"""
    memcpy!(src::Gᵐᵃᵗ{T}, dst::Gᵐᵃᵗ{T})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::Gᵐᵃᵗ{T}, dst::Gᵐᵃᵗ{T}) where {T}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    incr!(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T}, alpha::T)

Add a `Gᵐᵃᵗ` with given weight (`alpha`) to another `Gᵐᵃᵗ`.

```math
G^M_1 ⟶ G^M_1 + α * G^M_2.
```
"""
function incr!(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T}, alpha::T) where {T}
    @assert iscompatible(mat1, mat2)
    for i = 1:mat2.ntau
        @. mat1.data[i,1] = mat1.data[i,1] + mat2.data[i,1] * alpha
    end
end

"""
    smul!(mat::Gᵐᵃᵗ{T}, alpha::T)

Multiply a `Gᵐᵃᵗ` with given weight (`alpha`).

```math
G^M ⟶ α * G^M.
```
"""
function smul!(mat::Gᵐᵃᵗ{T}, alpha::T) where {T}
    for i = 1:mat.ntau
        @. mat.data[i,1] = mat.data[i,1] * alpha
    end
end

"""
    smul!(x::Element{T}, mat::Gᵐᵃᵗ{T})

Left multiply a `Gᵐᵃᵗ` with given weight (`x`), which is actually a
matrix.
"""
function smul!(x::Element{T}, mat::Gᵐᵃᵗ{T}) where {T}
    for i = 1:mat.ntau
        mat.data[i,1] = x * mat.data[i,1]
    end
end

"""
    smul!(mat::Gᵐᵃᵗ{T}, x::Element{T})

Right multiply a `Gᵐᵃᵗ` with given weight (`x`), which is actually a
matrix.
"""
function smul!(mat::Gᵐᵃᵗ{T}, x::Element{T}) where {T}
    for i = 1:mat.ntau
        mat.data[i,1] = mat.data[i,1] * x
    end
end

#=
### *Gᵐᵃᵗ* : *Traits*
=#

"""
    Base.:+(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T})

Operation `+` for two `Gᵐᵃᵗ` objects.
"""
function Base.:+(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T}) where {T}
    # Sanity check
    @assert getsize(mat1) == getsize(mat2)
    @assert getdims(mat1) == getdims(mat2)

    Gᵐᵃᵗ(mat1.ntau, mat1.ndim1, mat1.ndim2, mat1.data + mat2.data)
end

"""
    Base.:-(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T})

Operation `-` for two `Gᵐᵃᵗ` objects.
"""
function Base.:-(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T}) where {T}
    # Sanity check
    @assert getsize(mat1) == getsize(mat2)
    @assert getdims(mat1) == getdims(mat2)

    Gᵐᵃᵗ(mat1.ntau, mat1.ndim1, mat1.ndim2, mat1.data - mat2.data)
end

"""
    Base.:*(mat::Gᵐᵃᵗ{T}, x)

Operation `*` for a `Gᵐᵃᵗ` object and a scalar value.
"""
function Base.:*(mat::Gᵐᵃᵗ{T}, x) where {T}
    cx = convert(T, x)
    Gᵐᵃᵗ(mat.ntau, mat.ndim1, mat.ndim2, mat.data * cx)
end

"""
    Base.:*(x, mat::Gᵐᵃᵗ{T})

Operation `*` for a scalar value and a `Gᵐᵃᵗ` object.
"""
Base.:*(x, mat::Gᵐᵃᵗ{T}) where {T} = Base.:*(mat, x)

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
    Gᵐᵃᵗᵐ(sign, ntau, ndim1, ndim2, dataM)
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
*Retarded Green's Function* :

The retarded component of contour Green's function reads

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
=#

#=
### *Gʳᵉᵗ* : *Struct*
=#

"""
    Gʳᵉᵗ{T}

Retarded component (``G^R``) of contour Green's function.

See also: [`Gᵐᵃᵗ`](@ref), [`Gˡᵐⁱˣ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
mutable struct Gʳᵉᵗ{T} <: CnAbstractMatrix{T}
    ntime :: I64
    ndim1 :: I64
    ndim2 :: I64
    data  :: MatArray{T}
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
    Gʳᵉᵗ(ntime, ndim1, ndim2, data)
end

"""
    Gʳᵉᵗ(ntime::I64, ndim1::I64, ndim2::I64)

Constructor. All the matrix elements are set to be `CZERO`.
"""
function Gʳᵉᵗ(ntime::I64, ndim1::I64, ndim2::I64)
    Gʳᵉᵗ(ntime, ndim1, ndim2, CZERO)
end

"""
    Gʳᵉᵗ(ntime::I64, ndim1::I64)

Constructor. All the matrix elements are set to be `CZERO`.
"""
function Gʳᵉᵗ(ntime::I64, ndim1::I64)
    Gʳᵉᵗ(ntime, ndim1, ndim1, CZERO)
end

"""
    Gʳᵉᵗ(ntime::I64, x::Element{T})

Constructor. The matrix is initialized by `x`.
"""
function Gʳᵉᵗ(ntime::I64, x::Element{T}) where {T}
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
    Gʳᵉᵗ(ntime, ndim1, ndim2, data)
end

"""
    Gʳᵉᵗ(C::Cn, x::Element{T})

Constructor. The matrix is initialized by `x`.
"""
function Gʳᵉᵗ(C::Cn, x::Element{T}) where {T}
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
    Gʳᵉᵗ(C.ntime, C.ndim1, C.ndim2, data)
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

Constructor. All the matrix elements are set to be `CZERO`.
"""
function Gʳᵉᵗ(C::Cn)
    Gʳᵉᵗ(C.ntime, C.ndim1, C.ndim2, CZERO)
end

#=
### *Gʳᵉᵗ* : *Properties*
=#

"""
    getdims(ret::Gʳᵉᵗ{T})

Return the dimensional parameters of contour function.

See also: [`Gʳᵉᵗ`](@ref).
"""
function getdims(ret::Gʳᵉᵗ{T}) where {T}
    return (ret.ndim1, ret.ndim2)
end

"""
    getsize(ret::Gʳᵉᵗ{T})

Return the size of contour function.

See also: [`Gʳᵉᵗ`](@ref).
"""
function getsize(ret::Gʳᵉᵗ{T}) where {T}
    return ret.ntime
end

"""
    equaldims(ret::Gʳᵉᵗ{T})

Return whether the dimensional parameters are equal.

See also: [`Gʳᵉᵗ`](@ref).
"""
function equaldims(ret::Gʳᵉᵗ{T}) where {T}
    return ret.ndim1 == ret.ndim2
end

"""
    iscompatible(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T})

Judge whether two `Gʳᵉᵗ` objects are compatible.
"""
function iscompatible(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}) where {T}
    getsize(ret1) == getsize(ret2) &&
    getdims(ret1) == getdims(ret2)
end

"""
    iscompatible(C::Cn, ret::Gʳᵉᵗ{T})

Judge whether `C` (which is a `Cn` object) is compatible with `ret`
(which is a `Gʳᵉᵗ{T}` object).
"""
function iscompatible(C::Cn, ret::Gʳᵉᵗ{T}) where {T}
    C.ntime == getsize(ret) &&
    getdims(C) == getdims(ret)
end

"""
    iscompatible(ret::Gʳᵉᵗ{T}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `ret`
(which is a `Gʳᵉᵗ{T}` object).
"""
iscompatible(ret::Gʳᵉᵗ{T}, C::Cn) where {T} = iscompatible(C, ret)

"""
    distance(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}, tstp::I64)

Calculate distance between two `Gʳᵉᵗ` objects at given time step `tstp`.
"""
function distance(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}, tstp::I64) where {T}
    # Sanity check
    @assert 1 ≤ tstp ≤ ret1.ntime

    err = 0
    #
    for i = 1:tstp
        err = err + abs(sum(ret1.data[tstp,i] - ret2.data[tstp,i]))
    end
    #
    return err
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
### *Gʳᵉᵗ* : *Operations*
=#

"""
    memset!(ret::Gʳᵉᵗ{T}, x)

Reset all the matrix elements of `ret` to `x`. `x` should be a
scalar number.
"""
function memset!(ret::Gʳᵉᵗ{T}, x) where {T}
    cx = convert(T, x)
    for i=1:ret.ntime
        for j=1:ret.ntime
            fill!(ret.data[j,i], cx)
        end
    end
end

"""
    memset!(ret::Gʳᵉᵗ{T}, tstp::I64, x)

Reset the matrix elements of `ret` at given time step `tstp` to `x`. `x`
should be a scalar number.
"""
function memset!(ret::Gʳᵉᵗ{T}, tstp::I64, x) where {T}
    @assert 1 ≤ tstp ≤ ret.ntime
    cx = convert(T, x)
    for i=1:tstp
        fill!(ret.data[tstp,i], cx)
    end
end

"""
    zeros!(ret::Gʳᵉᵗ{T})

Reset all the matrix elements of `ret` to `ZERO`.
"""
zeros!(ret::Gʳᵉᵗ{T}) where {T} = memset!(ret, zero(T))

"""
    zeros!(ret::Gʳᵉᵗ{T}, tstp::I64)

Reset the matrix elements of `ret` at given time step `tstp` to `ZERO`.
"""
zeros!(ret::Gʳᵉᵗ{T}, tstp::I64) where {T} = memset!(ret, tstp, zero(T))

"""
    memcpy!(src::Gʳᵉᵗ{T}, dst::Gʳᵉᵗ{T})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::Gʳᵉᵗ{T}, dst::Gʳᵉᵗ{T}) where {T}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memcpy!(src::Gʳᵉᵗ{T}, dst::Gʳᵉᵗ{T}, tstp::I64)

Copy some matrix elements from `src` to `dst`. Only the matrix elements
at given time step `tstp` are copied.
"""
function memcpy!(src::Gʳᵉᵗ{T}, dst::Gʳᵉᵗ{T}, tstp::I64) where {T}
    @assert iscompatible(src, dst)
    @assert 1 ≤ tstp ≤ src.ntime
    for i=1:tstp
        dst.data[tstp,i] = copy(src.data[tstp,i])
    end
end

"""
    incr!(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}, tstp::I64, alpha::T)

Add a `Gʳᵉᵗ` with given weight (`alpha`) at given time step `tstp` to
another `Gʳᵉᵗ`.
"""
function incr!(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}, tstp::I64, alpha::T) where {T}
    @assert iscompatible(ret1, ret2)
    @assert 1 ≤ tstp ≤ ret2.ntime
    for i = 1:tstp
        @. ret1.data[tstp,i] = ret1.data[tstp,i] + ret2.data[tstp,i] * alpha
    end
end

"""
    smul!(ret::Gʳᵉᵗ{T}, tstp::I64, alpha::T)

Multiply a `Gʳᵉᵗ` with given weight (`alpha`) at given time step `tstp`.
"""
function smul!(ret::Gʳᵉᵗ{T}, tstp::I64, alpha::T) where {T}
    @assert 1 ≤ tstp ≤ ret.ntime
    for i = 1:tstp
        @. ret.data[tstp,i] = ret.data[tstp,i] * alpha
    end
end

"""
    smul!(x::Element{T}, ret::Gʳᵉᵗ{T}, tstp::I64)

Left multiply a `Gʳᵉᵗ` with given weight (`x`) at given time step `tstp`.
"""
function smul!(x::Element{T}, ret::Gʳᵉᵗ{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ ret.ntime
    for i = 1:tstp
        ret.data[tstp,i] = x * ret.data[tstp,i]
    end
end

"""
    smul!(ret::Gʳᵉᵗ{T}, x::Cf{T}, tstp::I64)

Right multiply a `Gʳᵉᵗ` with given weight (`x`) at given time step `tstp`.
"""
function smul!(ret::Gʳᵉᵗ{T}, x::Cf{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ ret.ntime
    for i = 1:tstp
        ret.data[tstp,i] = ret.data[tstp,i] * x[i]
    end
end

#=
### *Gʳᵉᵗ* : *Traits*
=#

"""
    Base.:+(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T})

Operation `+` for two `Gʳᵉᵗ` objects.
"""
function Base.:+(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}) where {T}
    # Sanity check
    @assert getsize(ret1) == getsize(ret2)
    @assert getdims(ret1) == getdims(ret2)

    Gʳᵉᵗ(ret1.ntime, ret1.ndim1, ret1.ndim2, ret1.data + ret2.data)
end

"""
    Base.:-(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T})

Operation `-` for two `Gʳᵉᵗ` objects.
"""
function Base.:-(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}) where {T}
    # Sanity check
    @assert getsize(ret1) == getsize(ret2)
    @assert getdims(ret1) == getdims(ret2)

    Gʳᵉᵗ(ret1.ntime, ret1.ndim1, ret1.ndim2, ret1.data - ret2.data)
end

"""
    Base.:*(ret::Gʳᵉᵗ{T}, x)

Operation `*` for a `Gʳᵉᵗ` object and a scalar value.
"""
function Base.:*(ret::Gʳᵉᵗ{T}, x) where {T}
    cx = convert(T, x)
    Gʳᵉᵗ(ret.ntime, ret.ndim1, ret.ndim2, ret.data * cx)
end

"""
    Base.:*(x, ret::Gʳᵉᵗ{T})

Operation `*` for a scalar value and a `Gʳᵉᵗ` object.
"""
Base.:*(x, ret::Gʳᵉᵗ{T}) where {T} = Base.:*(ret, x)

#=
*Advanced Green's Function* :

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

Advanced component (``G^{A}``) of contour Green's function.

Note: currently we do not need this component explicitly. However, for
the sake of completeness, we still define an empty struct for it.

See also: [`Gᵐᵃᵗ`](@ref), [`Gˡᵐⁱˣ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
mutable struct Gᵃᵈᵛ{T} <: CnAbstractMatrix{T} end

#=
*Left-mixing Green's Function* :

The left-mixing component of contour Green's function reads

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

where ``G^{\lceil}(\tau,t')`` is the right-mixing Green's function.
=#

#=
### *Gˡᵐⁱˣ* : *Struct*
=#

"""
    Gˡᵐⁱˣ{T}

Left-mixing component (``G^{⌉}``) of contour Green's function.

See also: [`Gᵐᵃᵗ`](@ref), [`Gʳᵉᵗ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
mutable struct Gˡᵐⁱˣ{T} <: CnAbstractMatrix{T}
    ntime :: I64
    ntau  :: I64
    ndim1 :: I64
    ndim2 :: I64
    data  :: MatArray{T}
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
    @assert ntau ≥ 2
    @assert ndim1 ≥ 1
    @assert ndim2 ≥ 1

    # Create Element{T}
    element = fill(v, ndim1, ndim2)

    # Create MatArray{T}, whose size is indeed (ntime, ntau).
    data = MatArray{T}(undef, ntime, ntau)
    for i = 1:ntau
        for j = 1:ntime
            data[j,i] = copy(element)
        end
    end

    # Call the default constructor
    Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, data)
end

"""
    Gˡᵐⁱˣ(ntime::I64, ntau::I64, ndim1::I64, ndim2::I64)

Constructor. All the matrix elements are set to be `CZERO`.
"""
function Gˡᵐⁱˣ(ntime::I64, ntau::I64, ndim1::I64, ndim2::I64)
    Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, CZERO)
end

"""
    Gˡᵐⁱˣ(ntime::I64, ntau::I64, ndim1::I64)

Constructor. All the matrix elements are set to be `CZERO`.
"""
function Gˡᵐⁱˣ(ntime::I64, ntau::I64, ndim1::I64)
    Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim1, CZERO)
end

"""
    Gˡᵐⁱˣ(ntime::I64, ntau::I64, x::Element{T})

Constructor. The matrix is initialized by `x`.
"""
function Gˡᵐⁱˣ(ntime::I64, ntau::I64, x::Element{T}) where {T}
    # Sanity check
    @assert ntime ≥ 2
    @assert ntau ≥ 2

    ndim1, ndim2 = size(x)
    data = MatArray{T}(undef, ntime, ntau)
    for i = 1:ntau
        for j = 1:ntime
            data[j,i] = copy(x)
        end
    end

    # Call the default constructor
    Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, data)
end

"""
    Gˡᵐⁱˣ(C::Cn, x::Element{T})

Constructor. The matrix is initialized by `x`.
"""
function Gˡᵐⁱˣ(C::Cn, x::Element{T}) where {T}
    # Sanity check
    @assert getdims(C) == size(x)

    # Create MatArray{T}, whose size is indeed (ntime, ntau)
    data = MatArray{T}(undef, C.ntime, C.ntau)
    for i = 1:C.ntau
        for j = 1:C.ntime
            data[j,i] = copy(x)
        end
    end

    # Call the default constructor
    Gˡᵐⁱˣ(C.ntime, C.ntau, C.ndim1, C.ndim2, data)
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

Constructor. All the matrix elements are set to be `CZERO`.
"""
function Gˡᵐⁱˣ(C::Cn)
    Gˡᵐⁱˣ(C.ntime, C.ntau, C.ndim1, C.ndim2, CZERO)
end

#=
### *Gˡᵐⁱˣ* : *Properties*
=#

"""
    getdims(lmix::Gˡᵐⁱˣ{T})

Return the dimensional parameters of contour function.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function getdims(lmix::Gˡᵐⁱˣ{T}) where {T}
    return (lmix.ndim1, lmix.ndim2)
end

"""
    getsize(lmix::Gˡᵐⁱˣ{T})

Return the size of contour function.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function getsize(lmix::Gˡᵐⁱˣ{T}) where {T}
    return (lmix.ntime, lmix.ntau)
end

"""
    equaldims(lmix::Gˡᵐⁱˣ{T})

Return whether the dimensional parameters are equal.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function equaldims(lmix::Gˡᵐⁱˣ{T}) where {T}
    return lmix.ndim1 == lmix.ndim2
end

"""
    iscompatible(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T})

Judge whether two `Gˡᵐⁱˣ` objects are compatible.
"""
function iscompatible(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}) where {T}
    getsize(lmix1) == getsize(lmix2) &&
    getdims(lmix1) == getdims(lmix2)
end

"""
    iscompatible(C::Cn, lmix::Gˡᵐⁱˣ{T})

Judge whether `C` (which is a `Cn` object) is compatible with `lmix`
(which is a `Gˡᵐⁱˣ{T}` object).
"""
function iscompatible(C::Cn, lmix::Gˡᵐⁱˣ{T}) where {T}
    C.ntime, C.ntau == getsize(lmix) &&
    getdims(C) == getdims(lmix)
end

"""
    iscompatible(lmix::Gˡᵐⁱˣ{T}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `lmix`
(which is a `Gˡᵐⁱˣ{T}` object).
"""
iscompatible(lmix::Gˡᵐⁱˣ{T}, C::Cn) where {T} = iscompatible(C, lmix)

"""
    distance(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}, tstp::I64)

Calculate distance between two `Gˡᵐⁱˣ` objects at given time step `tstp`.
"""
function distance(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}, tstp::I64) where {T}
    # Sanity check
    @assert 1 ≤ tstp ≤ lmix1.ntime

    err = 0
    #
    for i = 1:lmix1.ntau
        err = err + abs(sum(lmix1.data[tstp,i] - lmix2.data[tstp,i]))
    end
    #
    return err
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
### *Gˡᵐⁱˣ* : *Operations*
=#

"""
    memset!(lmix::Gˡᵐⁱˣ{T}, x)

Reset all the matrix elements of `lmix` to `x`. `x` should be a
scalar number.
"""
function memset!(lmix::Gˡᵐⁱˣ{T}, x) where {T}
    cx = convert(T, x)
    for i=1:lmix.ntau
        for j=1:lmix.ntime
            fill!(lmix.data[j,i], cx)
        end
    end
end

"""
    memset!(lmix::Gˡᵐⁱˣ{T}, tstp::I64, x)

Reset the matrix elements of `lmix` at given time step `tstp` to `x`. `x`
should be a scalar number.
"""
function memset!(lmix::Gˡᵐⁱˣ{T}, tstp::I64, x) where {T}
    @assert 1 ≤ tstp ≤ lmix.ntime
    cx = convert(T, x)
    for i=1:lmix.ntau
        fill!(lmix.data[tstp,i], cx)
    end
end

"""
    zeros!(lmix::Gˡᵐⁱˣ{T})

Reset all the matrix elements of `lmix` to `ZERO`.
"""
zeros!(lmix::Gˡᵐⁱˣ{T}) where {T} = memset!(lmix, zero(T))

"""
    zeros!(lmix::Gˡᵐⁱˣ{T}, tstp::I64)

Reset the matrix elements of `lmix` at given time step `tstp` to `ZERO`.
"""
zeros!(lmix::Gˡᵐⁱˣ{T}, tstp::I64) where {T} = memset!(lmix, tstp, zero(T))

"""
    memcpy!(src::Gˡᵐⁱˣ{T}, dst::Gˡᵐⁱˣ{T})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::Gˡᵐⁱˣ{T}, dst::Gˡᵐⁱˣ{T}) where {T}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memcpy!(src::Gˡᵐⁱˣ{T}, dst::Gˡᵐⁱˣ{T}, tstp::I64)

Copy some matrix elements from `src` to `dst`. Only the matrix elements
at given time step `tstp` are copied.
"""
function memcpy!(src::Gˡᵐⁱˣ{T}, dst::Gˡᵐⁱˣ{T}, tstp::I64) where {T}
    @assert iscompatible(src, dst)
    @assert 1 ≤ tstp ≤ src.ntime
    for i=1:src.ntau
        dst.data[tstp,i] = copy(src.data[tstp,i])
    end
end

"""
    incr!(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}, tstp::I64, alpha::T)

Add a `Gˡᵐⁱˣ` with given weight (`alpha`) at given time step `tstp` to
another `Gˡᵐⁱˣ`.
"""
function incr!(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}, tstp::I64, alpha::T) where {T}
    @assert iscompatible(lmix1, lmix2)
    @assert 1 ≤ tstp ≤ lmix2.ntime
    for i = 1:lmix2.ntau
        @. lmix1.data[tstp,i] = lmix1.data[tstp,i] + lmix2.data[tstp,i] * alpha
    end
end

"""
    smul!(lmix::Gˡᵐⁱˣ{T}, tstp::I64, alpha::T)

Multiply a `Gˡᵐⁱˣ` with given weight (`alpha`) at given time
step `tstp`.
"""
function smul!(lmix::Gˡᵐⁱˣ{T}, tstp::I64, alpha::T) where {T}
    @assert 1 ≤ tstp ≤ lmix.ntime
    for i = 1:lmix.ntau
        @. lmix.data[tstp,i] = lmix.data[tstp,i] * alpha
    end
end

"""
    smul!(x::Element{T}, lmix::Gˡᵐⁱˣ{T}, tstp::I64)

Left multiply a `Gˡᵐⁱˣ` with given weight (`x`) at given time
step `tstp`.
"""
function smul!(x::Element{T}, lmix::Gˡᵐⁱˣ{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ lmix.ntime
    for i = 1:lmix.ntau
        lmix.data[tstp,i] = x * lmix.data[tstp,i]
    end
end

"""
    smul!(lmix::Gˡᵐⁱˣ{T}, x::Element{T}, tstp::I64)

Right multiply a `Gˡᵐⁱˣ` with given weight (`x`) at given time
step `tstp`.
"""
function smul!(lmix::Gˡᵐⁱˣ{T}, x::Element{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ lmix.ntime
    for i = 1:lmix.ntau
        lmix.data[tstp,i] = lmix.data[tstp,i] * x
    end
end

#=
### *Gˡᵐⁱˣ* : *Traits*
=#

"""
    Base.:+(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T})

Operation `+` for two `Gˡᵐⁱˣ` objects.
"""
function Base.:+(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}) where {T}
    # Sanity check
    @assert getsize(lmix1) == getsize(lmix2)
    @assert getdims(lmix1) == getdims(lmix2)

    Gˡᵐⁱˣ(lmix1.ntime, lmix1.ntau, lmix1.ndim1, lmix1.ndim2, lmix1.data + lmix2.data)
end

"""
    Base.:-(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T})

Operation `-` for two `Gˡᵐⁱˣ` objects.
"""
function Base.:-(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}) where {T}
    # Sanity check
    @assert getsize(lmix1) == getsize(lmix2)
    @assert getdims(lmix1) == getdims(lmix2)

    Gˡᵐⁱˣ(lmix1.ntime, lmix1.ntau, lmix1.ndim1, lmix1.ndim2, lmix1.data - lmix2.data)
end

"""
    Base.:*(lmix::Gˡᵐⁱˣ{T}, x)

Operation `*` for a `Gˡᵐⁱˣ` object and a scalar value.
"""
function Base.:*(lmix::Gˡᵐⁱˣ{T}, x) where {T}
    cx = convert(T, x)
    Gˡᵐⁱˣ(lmix.ntime, lmix.ntau, lmix.ndim1, lmix.ndim2, lmix.data * cx)
end

"""
    Base.:*(x, lmix::Gˡᵐⁱˣ{T})

Operation `*` for a scalar value and a `Gˡᵐⁱˣ` object.
"""
Base.:*(x, lmix::Gˡᵐⁱˣ{T}) where {T} = Base.:*(lmix, x)

#=
*Right-mixing Green's Function* :

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

Right-mixing component (``G^{⌈}``) of contour Green's function.

See also: [`Gᵐᵃᵗ`](@ref), [`Gʳᵉᵗ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
mutable struct Gʳᵐⁱˣ{T} <: CnAbstractMatrix{T}
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
    Gʳᵐⁱˣ(sign, ntime, ntau, ndim1, ndim2, dataL)
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
*Lesser Green's Function* :

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

Lesser component (``G^{<}``) of contour Green's function.

See also: [`Gᵐᵃᵗ`](@ref), [`Gʳᵉᵗ`](@ref), [`Gˡᵐⁱˣ`](@ref).
"""
mutable struct Gˡᵉˢˢ{T} <: CnAbstractMatrix{T}
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
    Gˡᵉˢˢ(ntime, ndim1, ndim2, data)
end

"""
    Gˡᵉˢˢ(ntime::I64, ndim1::I64, ndim2::I64)

Constructor. All the matrix elements are set to be `CZERO`.
"""
function Gˡᵉˢˢ(ntime::I64, ndim1::I64, ndim2::I64)
    Gˡᵉˢˢ(ntime, ndim1, ndim2, CZERO)
end

"""
    Gˡᵉˢˢ(ntime::I64, ndim1::I64)

Constructor. All the matrix elements are set to be `CZERO`.
"""
function Gˡᵉˢˢ(ntime::I64, ndim1::I64)
    Gˡᵉˢˢ(ntime, ndim1, ndim1, CZERO)
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
    Gˡᵉˢˢ(ntime, ndim1, ndim2, data)
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
    Gˡᵉˢˢ(C.ntime, C.ndim1, C.ndim2, data)
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

Constructor. All the matrix elements are set to be `CZERO`.
"""
function Gˡᵉˢˢ(C::Cn)
    Gˡᵉˢˢ(C.ntime, C.ndim1, C.ndim2, CZERO)
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

Reset the matrix elements of `less` at given time step `tstp` to `x`. `x`
should be a scalar number.
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

Reset all the matrix elements of `less` to `ZERO`.
"""
zeros!(less::Gˡᵉˢˢ{T}) where {T} = memset!(less, zero(T))

"""
    zeros!(less::Gˡᵉˢˢ{T}, tstp::I64)

Reset the matrix elements of `less` at given time step `tstp` to `ZERO`.
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
at given time step `tstp` are copied.
"""
function memcpy!(src::Gˡᵉˢˢ{T}, dst::Gˡᵉˢˢ{T}, tstp::I64) where {T}
    @assert iscompatible(src, dst)
    @assert 1 ≤ tstp ≤ src.ntime
    for i=1:tstp
        dst.data[i,tstp] = copy(src.data[i,tstp])
    end
end

"""
    incr!(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}, tstp::I64, alpha::T)

Add a `Gˡᵉˢˢ` with given weight (`alpha`) at given time step `tstp` to
another `Gˡᵉˢˢ`.
"""
function incr!(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}, tstp::I64, alpha::T) where {T}
    @assert iscompatible(less1, less2)
    @assert 1 ≤ tstp ≤ less2.ntime
    for i = 1:tstp
        @. less1.data[i,tstp] = less1.data[i,tstp] + less2.data[i,tstp] * alpha
    end
end

"""
    smul!(less::Gˡᵉˢˢ{T}, tstp::I64, alpha::T)

Multiply a `Gˡᵉˢˢ` with given weight (`alpha`) at given time
step `tstp`.
"""
function smul!(less::Gˡᵉˢˢ{T}, tstp::I64, alpha::T) where {T}
    @assert 1 ≤ tstp ≤ less.ntime
    for i = 1:tstp
        @. less.data[i,tstp] = less.data[i,tstp] * alpha
    end
end

"""
    smul!(x::Cf{T}, less::Gˡᵉˢˢ{T}, tstp::I64)

Left multiply a `Gˡᵉˢˢ` with given weight (`x`) at given time
step `tstp`.
"""
function smul!(x::Cf{T}, less::Gˡᵉˢˢ{T}, tstp::I64) where {T}
    @assert 1 ≤ tstp ≤ less.ntime
    for i = 1:tstp
        less.data[i,tstp] = x[i] * less.data[i,tstp]
    end
end

"""
    smul!(less::Gˡᵉˢˢ{T}, x::Element{T}, tstp::I64)

Right multiply a `Gˡᵉˢˢ` with given weight (`x`) at given time
step `tstp`.
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

    Gˡᵉˢˢ(less1.ntime, less1.ndim1, less1.ndim2, less1.data + less2.data)
end

"""
    Base.:-(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T})

Operation `-` for two `Gˡᵉˢˢ` objects.
"""
function Base.:-(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}) where {T}
    # Sanity check
    @assert getsize(less1) == getsize(less2)
    @assert getdims(less1) == getdims(less2)

    Gˡᵉˢˢ(less1.ntime, less1.ndim1, less1.ndim2, less1.data - less2.data)
end

"""
    Base.:*(less::Gˡᵉˢˢ{T}, x)

Operation `*` for a `Gˡᵉˢˢ` object and a scalar value.
"""
function Base.:*(less::Gˡᵉˢˢ{T}, x) where {T}
    cx = convert(T, x)
    Gˡᵉˢˢ(less.ntime, less.ndim1, less.ndim2, less.data * cx)
end

"""
    Base.:*(x, less::Gˡᵉˢˢ{T})

Operation `*` for a scalar value and a `Gˡᵉˢˢ` object.
"""
Base.:*(x, less::Gˡᵉˢˢ{T}) where {T} = Base.:*(less, x)

#=
*Greater Green's Function* :

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

Greater component (``G^{>}``) of contour Green's function.

See also: [`Gʳᵉᵗ`](@ref), [`Gˡᵐⁱˣ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
mutable struct Gᵍᵗʳ{T} <: CnAbstractMatrix{T}
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
    Gᵍᵗʳ(ntime, ndim1, ndim2, dataL, dataR)
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
*Full Contour Green's Functions* :

As mentioned before, there are six linearly independent ''physical''
Green's functions. Assuming the hermitian symmetry, the number of
independent components is limited to four. Hence, in this package,
we just use ``{G^{M},\ G^{R},\ G^{\rceil},\ G^{<}}`` as the minimal
set of independent contour-ordered Green's functions. We call them
as `mat`, `ret`, `lmix`, and `less` components throughout the package.
=#

#=
### *ℱ* : *Struct*
=#

"""
    ℱ{T}

Standard contour-ordered Green's function. It includes four independent
components, namely `mat`, `ret`, `lmix`, and `less`.
"""
mutable struct ℱ{T} <: CnAbstractFunction{T}
    sign :: I64 # Used to distinguish fermions and bosons
    mat  :: Gᵐᵃᵗ{T}
    ret  :: Gʳᵉᵗ{T}
    lmix :: Gˡᵐⁱˣ{T}
    less :: Gˡᵉˢˢ{T}
end

#=
### *ℱ* : *Constructors*
=#

"""
    ℱ(C::Cn, v::T, sign::I64)

Standard constructor. This function is initialized by `v`.
"""
function ℱ(C::Cn, v::T, sign::I64) where {T}
    # Sanity check
    @assert sign in (BOSE, FERMI)

    # Create mat, ret, lmix, and less.
    mat  = Gᵐᵃᵗ(C, v)
    ret  = Gʳᵉᵗ(C, v)
    lmix = Gˡᵐⁱˣ(C, v)
    less = Gˡᵉˢˢ(C, v)

    # Call the default constructor
    ℱ(sign, mat, ret, lmix, less)
end

"""
    ℱ(C::Cn, sign::I64 = FERMI)

Constructor. Create a contour Green's function with zero initial values.
"""
function ℱ(C::Cn, sign::I64 = FERMI)
    # Setup sign
    @assert sign in (BOSE, FERMI)

    # Create mat, ret, lmix, and less.
    mat  = Gᵐᵃᵗ(C)
    ret  = Gʳᵉᵗ(C)
    lmix = Gˡᵐⁱˣ(C)
    less = Gˡᵉˢˢ(C)

    # Call the default constructor
    ℱ(sign, mat, ret, lmix, less)
end

#=
### *ℱ* : *Properties*
=#

"""
    getdims(cfm::ℱ{T})

Return the dimensional parameters of contour Green's function.

See also: [`ℱ`](@ref).
"""
function getdims(cfm::ℱ{T}) where {T}
    return getdims(cfm.less)
end

"""
    getntime(cfm::ℱ{T})

Return the `ntime` parameter of contour Green's function.
"""
function getntime(cfm::ℱ{T}) where {T}
    return getsize(cfm.less)
end

"""
    getntau(cfm::ℱ{T})

Return the `ntau` parameter of contour Green's function.
"""
function getntau(cfm::ℱ{T}) where {T}
    return getsize(cfm.mat)
end

"""
    getsign(cfm::ℱ{T})

Return the `sign` parameter of contour Green's function.
"""
function getsign(cfm::ℱ{T}) where {T}
    return cfm.sign
end

"""
    equaldims(cfm::ℱ{T})

Return whether the dimensional parameters are equal.

See also: [`ℱ`](@ref).
"""
function equaldims(cfm::ℱ{T}) where {T}
    return equaldims(cfm.less)
end

"""
    density(cfm::ℱ{T}, tstp::I64)

Returns the density matrix at given time step `tstp`. If `tstp = 0`,
it denotes the equilibrium state. However, when `tstp > 0`, it means
the nonequilibrium state.

See also: [`Gᵐᵃᵗ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
function density(cfm::ℱ{T}, tstp::I64) where {T}
    # Sanity check
    @assert 0 ≤ tstp ≤ getntime(cfm)

    if tstp == 0
        return -cfm.mat[getntime(cfm)]
    else
        return cfm.less[tstp, tstp] * getsign(cfm) * CZI
    end
end

"""
    distance(cfm1::ℱ{T}, cfm2::ℱ{T}, tstp::I64)

Calculate distance between two `ℱ` objects at given time step `tstp`.
"""
function distance(cfm1::ℱ{T}, cfm2::ℱ{T}, tstp::I64) where {T}
    # Sanity check
    @assert 0 ≤ tstp ≤ getntime(cfm1)

    err = 0.0
    #
    if tstp == 0
        err = err + distance(cfm1.mat, cfm2.mat)
    else
        err = err + distance(cfm1.ret, cfm2.ret, tstp)
        err = err + distance(cfm1.lmix, cfm2.lmix, tstp)
        err = err + distance(cfm1.less, cfm2.less, tstp)
    end
    #
    return err
end

#=
### *ℱ* : *Operations*
=#

"""
    memset!(cfm::ℱ{T}, x)

Reset all the matrix elements of `cfm` to `x`. `x` should be a
scalar number.
"""
function memset!(cfm::ℱ{T}, x) where {T}
    memset!(cfm.mat, x)
    memset!(cfm.ret, x)
    memset!(cfm.lmix, x)
    memset!(cfm.less, x)
end

"""
    memset!(cfm::ℱ{T}, tstp::I64, x)

Reset the matrix elements of `cfm` at given time step `tstp` to `x`. `x`
should be a scalar number. Note that `tstp = 0` means the equilibrium
state, at this time this function will reset the Matsubara component
only. However, when `tstp > 0`, the `ret`, `lmix`, and `less` components
will be changed.
"""
function memset!(cfm::ℱ{T}, tstp::I64, x) where {T}
    @assert 0 ≤ tstp ≤ getntime(cfm)
    if tstp > 0
        memset!(cfm.ret, tstp, x)
        memset!(cfm.lmix, tstp, x)
        memset!(cfm.less, tstp, x)
    else
        @assert tstp == 0
        memset!(cfm.mat, x)
    end
end

"""
    zeros!(cfm::ℱ{T})

Reset all the matrix elements of `cfm` to `ZERO`.
"""
zeros!(cfm::ℱ{T}) where {T} = memset!(cfm, zero(T))

"""
    zeros!(cfm::ℱ{T}, tstp::I64)

Reset the matrix elements of `cfm` at given time step `tstp` to `ZERO`.
"""
zeros!(cfm::ℱ{T}, tstp::I64) where {T} = memset!(cfm, tstp, zero(T))

"""
    memcpy!(src::ℱ{T}, dst::ℱ{T}, tstp::I64)

Copy contour Green's function at given time step `tstp`. Note that
`tstp = 0` means the equilibrium state, at this time this function
will copy the Matsubara component only. However, when `tstp > 0`,
the `ret`, `lmix`, and `less` components will be copied.
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

"""
    incr!(cfm1::ℱ{T}, cfm2::ℱ{T}, tstp::I64, alpha)

Adds a `ℱ` with given weight (`alpha`) to another `ℱ` (at given
time step `tstp`).
"""
function incr!(cfm1::ℱ{T}, cfm2::ℱ{T}, tstp::I64, alpha) where {T}
    @assert 0 ≤ tstp ≤ getntime(cfm2)
    α = convert(T, alpha)
    if tstp > 0
        incr!(cfm1.ret, cfm2.ret, tstp, α)
        incr!(cfm1.lmix, cfm2.lmix, tstp, α)
        incr!(cfm1.less, cfm2.less, tstp, α)
    else
        @assert tstp == 0
        incr!(cfm1.mat, cfm2.mat, α)
    end
end

"""
    incr!(cfm1::ℱ{T}, cfm2::ℱ{T}, alpha)

Adds a `ℱ` with given weight (`alpha`) to another `ℱ` (at all
possible time step `tstp`).
"""
function incr!(cfm1::ℱ{T}, cfm2::ℱ{T}, alpha) where {T}
    for tstp = 0:getntime(cfm2)
        incr!(cfm1, cfm2, tstp, alpha)
    end
end

"""
    smul!(cfm::ℱ{T}, tstp::I64, alpha)

Multiply a `ℱ` with given weight (`alpha`) at given time
step `tstp`.
"""
function smul!(cfm::ℱ{T}, tstp::I64, alpha) where {T}
    @assert 0 ≤ tstp ≤ getntime(cfm)
    α = convert(T, alpha)
    if tstp > 0
        smul!(cfm.ret, tstp, α)
        smul!(cfm.lmix, tstp, α)
        smul!(cfm.less, tstp, α)
    else
        @assert tstp == 0
        smul!(cfm.mat, α)
    end
end

"""
    smul!(cff::Cf{T}, cfm::ℱ{T}, tstp::I64)

Left multiply a `ℱ` with given weight (`Cf`) at given time
step `tstp`.
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

Right multiply a `ℱ` with given weight (`Cf`) at given time
step `tstp`.
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

#=
### *ℱ* : *I/O*
=#

"""
    read!(fname::AbstractString, cfm::ℱ{T})

Read the contour Green's functions from given file.
"""
function read!(fname::AbstractString, cfm::ℱ{T}) where {T}
end

"""
    write(fname::AbstractString, cfm::ℱ{T})

Write the contour Green's functions to given file.
"""
function write(fname::AbstractString, cfm::ℱ{T}) where {T}
end

#=
### *ℱ* : *Traits*
=#

"""
    Base.getproperty(cfm::ℱ{T}, symbol::Symbol)

Visit the properties stored in `ℱ` object. It provides access to
the Matsubara (minus, `matm`), advanced (`adv`), right-mixing (`rmix`),
and greater (`gtr`) components of the contour-ordered Green's function.
"""
function Base.getproperty(cfm::ℱ{T}, symbol::Symbol) where {T}
    if symbol === :matm
        return Gᵐᵃᵗᵐ(cfm.sign, cfm.mat)
    #
    elseif symbol === :adv
        error("Sorry, this feature has not been implemented")
    #
    elseif symbol === :rmix
        return Gʳᵐⁱˣ(cfm.sign, cfm.lmix)
    #
    elseif symbol === :gtr
        return Gᵍᵗʳ(cfm.less, cfm.ret)
    #
    else # Fallback to getfield()
        return getfield(cfm, symbol)
    end
end

#=
### *gᵐᵃᵗ* : *Struct*
=#

"""
    gᵐᵃᵗ{S}

Matsubara component (``G^{M}``) of contour Green's function at given
time step `tstp`. Actually, `gᵐᵃᵗ{S}` is equivalent to `Gᵐᵃᵗ{T}`.

See also: [`gʳᵉᵗ`](@ref), [`gˡᵐⁱˣ`](@ref), [`gˡᵉˢˢ`](@ref).
"""
mutable struct gᵐᵃᵗ{S} <: CnAbstractVector{S}
    ntau  :: I64
    ndim1 :: I64
    ndim2 :: I64
    data  :: VecArray{S}
end

#=
### *gᵐᵃᵗ* : *Constructors*
=#

"""
    gᵐᵃᵗ(ntau::I64, ndim1::I64, ndim2::I64, v::S)

Constructor. All the vector elements are set to be `v`.
"""
function gᵐᵃᵗ(ntau::I64, ndim1::I64, ndim2::I64, v::S) where {S}
    # Sanity check
    @assert ntau ≥ 2
    @assert ndim1 ≥ 1
    @assert ndim2 ≥ 1

    # Create Element{S}
    element = fill(v, ndim1, ndim2)

    # Create VecArray{S}, whose size is indeed (ntau,)
    data = VecArray{S}(undef, ntau)
    for i = 1:ntau
        data[i] = copy(element)
    end

    # Call the default constructor
    gᵐᵃᵗ(ntau, ndim1, ndim2, data)
end

"""
    gᵐᵃᵗ(ntau::I64, ndim1::I64, ndim2::I64)

Constructor. All the vector elements are set to be `CZERO`.
"""
function gᵐᵃᵗ(ntau::I64, ndim1::I64, ndim2::I64)
    gᵐᵃᵗ(ntau, ndim1, ndim2, CZERO)
end

"""
    gᵐᵃᵗ(ntau::I64, ndim1::I64)

Constructor. All the vector elements are set to be `CZERO`.
"""
function gᵐᵃᵗ(ntau::I64, ndim1::I64)
    gᵐᵃᵗ(ntau, ndim1, ndim1, CZERO)
end

"""
    gᵐᵃᵗ(ntau::I64, x::Element{S})

Constructor. The vector is initialized by `x`.
"""
function gᵐᵃᵗ(ntau::I64, x::Element{S}) where {S}
    # Sanity check
    @assert ntau ≥ 2

    ndim1, ndim2 = size(x)
    data = VecArray{S}(undef, ntau)
    for i = 1:ntau
        data[i] = copy(x)
    end

    # Call the default constructor
    gᵐᵃᵗ(ntau, ndim1, ndim2, data)
end

#=
### *gᵐᵃᵗ* : *Properties*
=#

"""
    getdims(mat::gᵐᵃᵗ{S})

Return the dimensional parameters of contour function.

See also: [`gᵐᵃᵗ`](@ref).
"""
function getdims(mat::gᵐᵃᵗ{S}) where {S}
    return (mat.ndim1, mat.ndim2)
end

"""
    getsize(mat::gᵐᵃᵗ{S})

Return the size of contour function.

See also: [`gᵐᵃᵗ`](@ref).
"""
function getsize(mat::gᵐᵃᵗ{S}) where {S}
    return mat.ntau
end

"""
    equaldims(mat::gᵐᵃᵗ{S})

Return whether the dimensional parameters are equal.

See also: [`gᵐᵃᵗ`](@ref).
"""
function equaldims(mat::gᵐᵃᵗ{S}) where {S}
    return mat.ndim1 == mat.ndim2
end

"""
    iscompatible(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S})

Judge whether two `gᵐᵃᵗ` objects are compatible.
"""
function iscompatible(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}) where {S}
    getsize(mat1) == getsize(mat2) &&
    getdims(mat1) == getdims(mat2)
end

"""
    iscompatible(mat1::gᵐᵃᵗ{S}, mat2::Gᵐᵃᵗ{S})

Judge whether the `gᵐᵃᵗ` and `Gᵐᵃᵗ` objects are compatible.
"""
function iscompatible(mat1::gᵐᵃᵗ{S}, mat2::Gᵐᵃᵗ{S}) where {S}
    getsize(mat1) == getsize(mat2) &&
    getdims(mat1) == getdims(mat2)
end

"""
    iscompatible(mat1::Gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S})

Judge whether the `gᵐᵃᵗ` and `Gᵐᵃᵗ` objects are compatible.
"""
iscompatible(mat1::Gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}) where {S} = iscompatible(mat2, mat1)

"""
    iscompatible(C::Cn, mat::gᵐᵃᵗ{S})

Judge whether `C` (which is a `Cn` object) is compatible with `mat`
(which is a `gᵐᵃᵗ{S}` object).
"""
function iscompatible(C::Cn, mat::gᵐᵃᵗ{S}) where {S}
    C.ntau == getsize(mat) &&
    getdims(C) == getdims(mat)
end

"""
    iscompatible(mat::gᵐᵃᵗ{S}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `mat`
(which is a `gᵐᵃᵗ{S}` object).
"""
iscompatible(mat::gᵐᵃᵗ{S}, C::Cn) where {S} = iscompatible(C, mat)

"""
    distance(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S})

Calculate distance between two `gᵐᵃᵗ` objects.
"""
function distance(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}) where {S}
    @assert iscompatible(mat1, mat2)

    err = 0.0
    #
    for m = 1:mat1.ntau
        err = err + abs(sum(mat1.data[m] - mat2.data[m]))
    end
    #
    return err
end

"""
    distance(mat1::gᵐᵃᵗ{S}, mat2::Gᵐᵃᵗ{S})

Calculate distance between a `gᵐᵃᵗ` object and a `Gᵐᵃᵗ` object.
"""
function distance(mat1::gᵐᵃᵗ{S}, mat2::Gᵐᵃᵗ{S}) where {S}
    @assert iscompatible(mat1, mat2)

    err = 0.0
    #
    for m = 1:mat1.ntau
        err = err + abs(sum(mat1.data[m] - mat2.data[m,1]))
    end
    #
    return err
end

"""
    distance(mat1::Gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S})

Calculate distance between a `gᵐᵃᵗ` object and a `Gᵐᵃᵗ` object.
"""
distance(mat1::Gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}) where {S} = distance(mat2, mat1)

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
### *gᵐᵃᵗ* : *Operations*
=#

"""
    memset!(mat::gᵐᵃᵗ{S}, x)

Reset all the vector elements of `mat` to `x`. `x` should be a
scalar number.
"""
function memset!(mat::gᵐᵃᵗ{S}, x) where {S}
    cx = convert(S, x)
    for i = 1:mat.ntau
        fill!(mat.data[i], cx)
    end
end

"""
    zeros!(mat::gᵐᵃᵗ{S})

Reset all the vector elements of `mat` to `ZERO`.
"""
zeros!(mat::gᵐᵃᵗ{S}) where {S} = memset!(mat, zero(S))

"""
    memcpy!(src::gᵐᵃᵗ{S}, dst::gᵐᵃᵗ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::gᵐᵃᵗ{S}, dst::gᵐᵃᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memcpy!(src::Gᵐᵃᵗ{S}, dst::gᵐᵃᵗ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::Gᵐᵃᵗ{S}, dst::gᵐᵃᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data[:,1])
end

"""
    memcpy!(src::gᵐᵃᵗ{S}, dst::Gᵐᵃᵗ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::gᵐᵃᵗ{S}, dst::Gᵐᵃᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    @. dst.data[:,1] = copy(src.data)
end

"""
    incr!(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}, alpha::S)

Add a `gᵐᵃᵗ` with given weight (`alpha`) to another `gᵐᵃᵗ`.
"""
function incr!(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}, alpha::S) where {S}
    @assert iscompatible(mat1, mat2)
    for i = 1:mat2.ntau
        @. mat1.data[i] = mat1.data[i] + mat2.data[i] * alpha
    end
end

"""
    incr!(mat1::Gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}, alpha::S)

Add a `gᵐᵃᵗ` with given weight (`alpha`) to a `Gᵐᵃᵗ`.
"""
function incr!(mat1::Gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}, alpha::S) where {S}
    @assert iscompatible(mat1, mat2)
    for i = 1:mat2.ntau
        @. mat1.data[i,1] = mat1.data[i,1] + mat2.data[i] * alpha
    end
end

"""
    incr!(mat1::gᵐᵃᵗ{S}, mat2::Gᵐᵃᵗ{S}, alpha::S)

Add a `Gᵐᵃᵗ` with given weight (`alpha`) to a `gᵐᵃᵗ`.
"""
function incr!(mat1::gᵐᵃᵗ{S}, mat2::Gᵐᵃᵗ{S}, alpha::S) where {S}
    @assert iscompatible(mat1, mat2)
    for i = 1:mat1.ntau
        @. mat1.data[i] = mat1.data[i] + mat2.data[i,1] * alpha
    end
end

"""
    smul!(mat::gᵐᵃᵗ{S}, alpha::S)

Multiply a `gᵐᵃᵗ` with given weight (`alpha`).
"""
function smul!(mat::gᵐᵃᵗ{S}, alpha::S) where {S}
    for i = 1:mat.ntau
        @. mat.data[i] = mat.data[i] * alpha
    end
end

"""
    smul!(x::Element{S}, mat::gᵐᵃᵗ{S})

Left multiply a `gᵐᵃᵗ` with given weight (`x`).
"""
function smul!(x::Element{S}, mat::gᵐᵃᵗ{S}) where {S}
    for i = 1:mat.ntau
        mat.data[i] = x * mat.data[i]
    end
end

"""
    smul!(mat::gᵐᵃᵗ{S}, x::Element{S})

Right multiply a `gᵐᵃᵗ` with given weight (`x`).
"""
function smul!(mat::gᵐᵃᵗ{S}, x::Element{S}) where {S}
    for i = 1:mat.ntau
        mat.data[i] = mat.data[i] * x
    end
end

#=
### *gᵐᵃᵗ* : *Traits*
=#

"""
    Base.:+(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S})

Operation `+` for two `gᵐᵃᵗ` objects.
"""
function Base.:+(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}) where {S}
    # Sanity check
    @assert getsize(mat1) == getsize(mat2)
    @assert getdims(mat1) == getdims(mat2)

    gᵐᵃᵗ(mat1.ntau, mat1.ndim1, mat1.ndim2, mat1.data + mat2.data)
end

"""
    Base.:-(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S})

Operation `-` for two `gᵐᵃᵗ` objects.
"""
function Base.:-(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}) where {S}
    # Sanity check
    @assert getsize(mat1) == getsize(mat2)
    @assert getdims(mat1) == getdims(mat2)

    gᵐᵃᵗ(mat1.ntau, mat1.ndim1, mat1.ndim2, mat1.data - mat2.data)
end

"""
    Base.:*(mat::gᵐᵃᵗ{S}, x)

Operation `*` for a `gᵐᵃᵗ` object and a scalar value.
"""
function Base.:*(mat::gᵐᵃᵗ{S}, x) where {S}
    cx = convert(S, x)
    gᵐᵃᵗ(mat.ntau, mat.ndim1, mat.ndim2, mat.data * cx)
end

"""
    Base.:*(x, mat::gᵐᵃᵗ{S})

Operation `*` for a scalar value and a `gᵐᵃᵗ` object.
"""
Base.:*(x, mat::gᵐᵃᵗ{S}) where {S} = Base.:*(mat, x)

#=
### *gᵐᵃᵗᵐ* : *Struct*
=#

"""
    gᵐᵃᵗᵐ{S}

Matsubara component (``G^M``) of contour Green's function at given time
step `tstp = 0`. It is designed for ``\tau < 0`` case. It is not an
independent component. It can be constructed from the `gᵐᵃᵗ{T}` struct.

See also: [`Gʳᵉᵗ`](@ref), [`Gˡᵐⁱˣ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
mutable struct gᵐᵃᵗᵐ{S} <: CnAbstractVector{S}
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
    gᵐᵃᵗᵐ(sign, ntau, ndim1, ndim2, dataV)
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
### *gʳᵉᵗ* : *Struct*
=#

"""
    gʳᵉᵗ{S}

Retarded component (``G^{R}``) of contour Green's function at given
time step `tstp`. Actually, it denotes ``G^{R}(tᵢ = tstp, tⱼ)``.

See also: [`gᵐᵃᵗ`](@ref), [`gˡᵐⁱˣ`](@ref), [`gˡᵉˢˢ`](@ref).
"""
mutable struct gʳᵉᵗ{S} <: CnAbstractVector{S}
    tstp  :: I64
    ndim1 :: I64
    ndim2 :: I64
    data  :: VecArray{S}
end

#=
### *gʳᵉᵗ* : *Constructors*
=#

"""
    gʳᵉᵗ(tstp::I64, ndim1::I64, ndim2::I64, v::S) where {S}

Constructor. All the vector elements are set to be `v`.
"""
function gʳᵉᵗ(tstp::I64, ndim1::I64, ndim2::I64, v::S) where {S}
    # Sanity check
    @assert tstp ≥ 1
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
    gʳᵉᵗ(tstp, ndim1, ndim2, data)
end

"""
    gʳᵉᵗ(tstp::I64, ndim1::I64, ndim2::I64)

Constructor. All the vector elements are set to be `CZERO`.
"""
function gʳᵉᵗ(tstp::I64, ndim1::I64, ndim2::I64)
    gʳᵉᵗ(tstp, ndim1, ndim2, CZERO)
end

"""
    gʳᵉᵗ(tstp::I64, ndim1::I64)

Constructor. All the vector elements are set to be `CZERO`.
"""
function gʳᵉᵗ(tstp::I64, ndim1::I64)
    gʳᵉᵗ(tstp, ndim1, ndim1, CZERO)
end

"""
    gʳᵉᵗ(tstp::I64, x::Element{S})

Constructor. The vector is initialized by `x`.
"""
function gʳᵉᵗ(tstp::I64, x::Element{S}) where {S}
    # Sanity check
    @assert tstp ≥ 1

    ndim1, ndim2 = size(x)
    data = VecArray{S}(undef, tstp)
    for i = 1:tstp
        data[i] = copy(x)
    end

    # Call the default constructor
    gʳᵉᵗ(tstp, ndim1, ndim2, data)
end

#=
### *gʳᵉᵗ* : *Properties*
=#

"""
    getdims(ret::gʳᵉᵗ{S})

Return the dimensional parameters of contour function.

See also: [`gʳᵉᵗ`](@ref).
"""
function getdims(ret::gʳᵉᵗ{S}) where {S}
    return (ret.ndim1, ret.ndim2)
end

"""
    getsize(ret::gʳᵉᵗ{S})

Return the size of contour function.

See also: [`gʳᵉᵗ`](@ref).
"""
function getsize(ret::gʳᵉᵗ{S}) where {S}
    return ret.tstp
end

"""
    equaldims(ret::gʳᵉᵗ{S})

Return whether the dimensional parameters are equal.

See also: [`gʳᵉᵗ`](@ref).
"""
function equaldims(ret::gʳᵉᵗ{S}) where {S}
    return ret.ndim1 == ret.ndim2
end

"""
    iscompatible(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S})

Judge whether two `gʳᵉᵗ` objects are compatible.
"""
function iscompatible(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}) where {S}
    getsize(ret1) == getsize(ret2) &&
    getdims(ret1) == getdims(ret2)
end

"""
    iscompatible(ret1::gʳᵉᵗ{S}, ret2::Gʳᵉᵗ{S})

Judge whether the `gʳᵉᵗ` and `Gʳᵉᵗ` objects are compatible.
"""
function iscompatible(ret1::gʳᵉᵗ{S}, ret2::Gʳᵉᵗ{S}) where {S}
    getsize(ret1) ≤ getsize(ret2) &&
    getdims(ret1) == getdims(ret2)
end

"""
    iscompatible(ret1::Gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S})

Judge whether the `gʳᵉᵗ` and `Gʳᵉᵗ` objects are compatible.
"""
iscompatible(ret1::Gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}) where {S} = iscompatible(ret2, ret1)

"""
    iscompatible(C::Cn, ret::gʳᵉᵗ{S})

Judge whether `C` (which is a `Cn` object) is compatible with `ret`
(which is a `gʳᵉᵗ{S}` object).
"""
function iscompatible(C::Cn, ret::gʳᵉᵗ{S}) where {S}
    C.ntime ≥ getsize(ret) &&
    getdims(C) == getdims(ret)
end

"""
    iscompatible(ret::gʳᵉᵗ{S}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `ret`
(which is a `gʳᵉᵗ{S}` object).
"""
iscompatible(ret::gʳᵉᵗ{S}, C::Cn) where {S} = iscompatible(C, ret)

"""
    distance(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S})

Calculate distance between two `gʳᵉᵗ` objects.
"""
function distance(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}) where {S}
    @assert iscompatible(ret1, ret2)

    err = 0.0
    #
    for m = 1:ret1.tstp
        err = err + abs(sum(ret1.data[m] - ret2.data[m]))
    end
    #
    return err
end

"""
    distance(ret1::gʳᵉᵗ{S}, ret2::Gʳᵉᵗ{S}, tstp::I64)

Calculate distance between a `gʳᵉᵗ` object and a `Gʳᵉᵗ` object at
given time step `tstp`.
"""
function distance(ret1::gʳᵉᵗ{S}, ret2::Gʳᵉᵗ{S}, tstp::I64) where {S}
    @assert iscompatible(ret1, ret2)
    @assert ret1.tstp == tstp

    err = 0.0
    #
    for m = 1:ret1.tstp
        err = err + abs(sum(ret1.data[m] - ret2.data[tstp,m]))
    end
    #
    return err
end

"""
    distance(ret1::Gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}, tstp::I64)

Calculate distance between a `gʳᵉᵗ` object and a `Gʳᵉᵗ` object at
given time step `tstp`.
"""
distance(ret1::Gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}, tstp::I64) where {S} = distance(ret2, ret1, tstp)

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
### *gʳᵉᵗ* : *Operations*
=#

"""
    memset!(ret::gʳᵉᵗ{S}, x)

Reset all the vector elements of `ret` to `x`. `x` should be a
scalar number.
"""
function memset!(ret::gʳᵉᵗ{S}, x) where {S}
    cx = convert(T, x)
    for i=1:ret.tstp
        fill!(ret.data[i], cx)
    end
end

"""
    zeros!(ret::gʳᵉᵗ{S})

Reset all the vector elements of `ret` to `ZERO`.
"""
zeros!(ret::gʳᵉᵗ{S}) where {S} = memset!(ret, zero(S))

"""
    memcpy!(src::gʳᵉᵗ{S}, dst::gʳᵉᵗ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::gʳᵉᵗ{S}, dst::gʳᵉᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memcpy!(src::Gʳᵉᵗ{S}, dst::gʳᵉᵗ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::Gʳᵉᵗ{S}, dst::gʳᵉᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    tstp = dst.tstp
    @. dst.data = copy(src.data[tstp,1:tstp])
end

"""
    memcpy!(src::gʳᵉᵗ{S}, dst::Gʳᵉᵗ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::gʳᵉᵗ{S}, dst::Gʳᵉᵗ{S}) where {S}
    @assert iscompatible(src, dst)
    tstp = src.tstp
    @. dst.data[tstp,1:tstp] = copy(src.data)
end

"""
    incr!(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}, alpha::S)

Add a `gʳᵉᵗ` with given weight (`alpha`) to another `gʳᵉᵗ`.
"""
function incr!(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}, alpha::S) where {S}
    @assert iscompatible(ret1, ret2)
    tstp = ret2.tstp
    for i = 1:tstp
        @. ret1.data[i] = ret1.data[i] + ret2.data[i] * alpha
    end
end

"""
    incr!(ret1::Gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}, alpha::S)

Add a `gʳᵉᵗ` with given weight (`alpha`) to a `Gʳᵉᵗ`.
"""
function incr!(ret1::Gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}, alpha::S) where {S}
    @assert iscompatible(ret1, ret2)
    tstp = ret2.tstp
    for i = 1:tstp
        @. ret1.data[tstp,i] = ret1.data[tstp,i] + ret2.data[i] * alpha
    end
end

"""
    incr!(ret1::gʳᵉᵗ{S}, ret2::Gʳᵉᵗ{S}, alpha::S)

Add a `Gʳᵉᵗ` with given weight (`alpha`) to a `gʳᵉᵗ`.
"""
function incr!(ret1::gʳᵉᵗ{S}, ret2::Gʳᵉᵗ{S}, alpha::S) where {S}
    @assert iscompatible(ret1, ret2)
    tstp = ret1.tstp
    for i = 1:tstp
        @. ret1.data[i] = ret1.data[i] + ret2.data[tstp,i] * alpha
    end
end

"""
    smul!(ret::gʳᵉᵗ{S}, alpha::S)

Multiply a `gʳᵉᵗ` with given weight (`alpha`).
"""
function smul!(ret::gʳᵉᵗ{S}, alpha::S) where {S}
    for i = 1:ret.tstp
        @. ret.data[i] = ret.data[i] * alpha
    end
end

"""
    smul!(x::Element{S}, ret::gʳᵉᵗ{S})

Left multiply a `gʳᵉᵗ` with given weight (`x`).
"""
function smul!(x::Element{S}, ret::gʳᵉᵗ{S}) where {S}
    for i = 1:ret.tstp
        ret.data[i] = x * ret.data[i]
    end
end

"""
    smul!(ret::gʳᵉᵗ{S}, x::Cf{S})

Right multiply a `gʳᵉᵗ` with given weight (`x`).
"""
function smul!(ret::gʳᵉᵗ{S}, x::Cf{S}) where {S}
    for i = 1:ret.tstp
        ret.data[i] = ret.data[i] * x[i]
    end
end

#=
### *gʳᵉᵗ* : *Traits*
=#

"""
    Base.:+(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S})

Operation `+` for two `gʳᵉᵗ` objects.
"""
function Base.:+(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}) where {S}
    # Sanity check
    @assert getsize(ret1) == getsize(ret2)
    @assert getdims(ret1) == getdims(ret2)

    gʳᵉᵗ(ret1.tstp, ret1.ndim1, ret1.ndim2, ret1.data + ret2.data)
end

"""
    Base.:-(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S})

Operation `-` for two `gʳᵉᵗ` objects.
"""
function Base.:-(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}) where {S}
    # Sanity check
    @assert getsize(ret1) == getsize(ret2)
    @assert getdims(ret1) == getdims(ret2)

    gʳᵉᵗ(ret1.tstp, ret1.ndim1, ret1.ndim2, ret1.data - ret2.data)
end

"""
    Base.:*(ret::gʳᵉᵗ{S}, x)

Operation `*` for a `gʳᵉᵗ` object and a scalar value.
"""
function Base.:*(ret::gʳᵉᵗ{S}, x) where {S}
    cx = convert(S, x)
    gʳᵉᵗ(ret.tstp, ret.ndim1, ret.ndim2, ret.data * cx)
end

"""
    Base.:*(x, ret::gʳᵉᵗ{S})

Operation `*` for a scalar value and a `gʳᵉᵗ` object.
"""
Base.:*(x, ret::gʳᵉᵗ{S}) where {S} = Base.:*(ret, x)

#=
### *gᵃᵈᵛ* : *Struct*
=#

mutable struct gᵃᵈᵛ{S} <: CnAbstractVector{S} end

#=
### *gˡᵐⁱˣ* : *Struct*
=#

"""
    gˡᵐⁱˣ{S}

Left-mixing component (``G^{⌉}``) of contour Green's function at given
time step `tstp`. Actually, it denotes ``G^{⌉}(tᵢ ≡ tstp, τⱼ)``.

See also: [`gᵐᵃᵗ`](@ref), [`gʳᵉᵗ`](@ref), [`gˡᵉˢˢ`](@ref).
"""
mutable struct gˡᵐⁱˣ{S} <: CnAbstractVector{S}
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
    @assert ntau ≥ 2
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
    gˡᵐⁱˣ(ntau, ndim1, ndim2, data)
end

"""
    gˡᵐⁱˣ(ntau::I64, ndim1::I64, ndim2::I64)

Constructor. All the matrix elements are set to be `CZERO`.
"""
function gˡᵐⁱˣ(ntau::I64, ndim1::I64, ndim2::I64)
    gˡᵐⁱˣ(ntau, ndim1, ndim2, CZERO)
end

"""
    gˡᵐⁱˣ(ntau::I64, ndim1::I64)

Constructor. All the matrix elements are set to be `CZERO`.
"""
function gˡᵐⁱˣ(ntau::I64, ndim1::I64)
    gˡᵐⁱˣ(ntau, ndim1, ndim1, CZERO)
end

"""
    gˡᵐⁱˣ(ntau::I64, x::Element{S})

Constructor. The matrix is initialized by `x`.
"""
function gˡᵐⁱˣ(ntau::I64, x::Element{S}) where {S}
    # Sanity check
    @assert ntau ≥ 2

    ndim1, ndim2 = size(x)
    data = VecArray{S}(undef, ntau)
    for i = 1:ntau
        data[i] = copy(x)
    end

    # Call the default constructor
    gˡᵐⁱˣ(ntau, ndim1, ndim2, data)
end

#=
### *gˡᵐⁱˣ* : *Properties*
=#

"""
    getdims(lmix::gˡᵐⁱˣ{S})

Return the dimensional parameters of contour function.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function getdims(lmix::gˡᵐⁱˣ{S}) where {S}
    return (lmix.ndim1, lmix.ndim2)
end

"""
    getsize(lmix::gˡᵐⁱˣ{S})

Return the size of contour function.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function getsize(lmix::gˡᵐⁱˣ{S}) where {S}
    return lmix.ntau
end

"""
    equaldims(lmix::gˡᵐⁱˣ{S})

Return whether the dimensional parameters are equal.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function equaldims(lmix::gˡᵐⁱˣ{S}) where {S}
    return lmix.ndim1 == lmix.ndim2
end

"""
    iscompatible(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S})

Judge whether two `gˡᵐⁱˣ` objects are compatible.
"""
function iscompatible(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}) where {S}
    getsize(lmix1) == getsize(lmix2) &&
    getdims(lmix1) == getdims(lmix2)
end

"""
    iscompatible(lmix1::gˡᵐⁱˣ{S}, lmix2::Gˡᵐⁱˣ{S})

Judge whether the `gˡᵐⁱˣ` and `Gˡᵐⁱˣ` objects are compatible.
"""
function iscompatible(lmix1::gˡᵐⁱˣ{S}, lmix2::Gˡᵐⁱˣ{S}) where {S}
    getsize(lmix1) == lmix2.ntau &&
    getdims(lmix1) == getdims(lmix2)
end

"""
    iscompatible(lmix1::Gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S})

Judge whether the `gˡᵐⁱˣ` and `Gˡᵐⁱˣ` objects are compatible.
"""
iscompatible(lmix1::Gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}) where {S} = iscompatible(lmix2, lmix1)

"""
    iscompatible(C::Cn, lmix::gˡᵐⁱˣ{S})

Judge whether `C` (which is a `Cn` object) is compatible with `lmix`
(which is a `gˡᵐⁱˣ{S}` object).
"""
function iscompatible(C::Cn, lmix::gˡᵐⁱˣ{S}) where {S}
    C.ntau == getsize(lmix) &&
    getdims(C) == getdims(lmix)
end

"""
    iscompatible(lmix::gˡᵐⁱˣ{S}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `lmix`
(which is a `gˡᵐⁱˣ{S}` object).
"""
iscompatible(lmix::gˡᵐⁱˣ{S}, C::Cn) where {S} = iscompatible(C, lmix)

"""
    distance(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S})

Calculate distance between two `gˡᵐⁱˣ` objects.
"""
function distance(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}) where {S}
    @assert iscompatible(lmix1, lmix2)

    err = 0.0
    #
    for m = 1:lmix1.ntau
        err = err + abs(sum(lmix1.data[m] - lmix2.data[m]))
    end
    #
    return err
end

"""
    distance(lmix1::gˡᵐⁱˣ{S}, lmix2::Gˡᵐⁱˣ{S}, tstp::I64)

Calculate distance between a `gˡᵐⁱˣ` object and a `Gˡᵐⁱˣ` object at
given time step `tstp`.
"""
function distance(lmix1::gˡᵐⁱˣ{S}, lmix2::Gˡᵐⁱˣ{S}, tstp::I64) where {S}
    @assert iscompatible(lmix1, lmix2)

    err = 0.0
    #
    for m = 1:lmix1.ntau
        err = err + abs(sum(lmix1.data[m] - lmix2.data[tstp,m]))
    end
    #
    return err
end

"""
    distance(lmix1::Gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}, tstp::I64)

Calculate distance between a `gˡᵐⁱˣ` object and a `Gˡᵐⁱˣ` object at
given time step `tstp`.
"""
distance(lmix1::Gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}, tstp::I64) where {S} = distance(lmix2, lmix1, tstp)

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
### *gˡᵐⁱˣ* : *Operations*
=#

"""
    memset!(lmix::gˡᵐⁱˣ{S}, x)

Reset all the matrix elements of `lmix` to `x`. `x` should be a
scalar number.
"""
function memset!(lmix::gˡᵐⁱˣ{S}, x) where {S}
    cx = convert(S, x)
    for i=1:lmix.ntau
        fill!(lmix.data[i], cx)
    end
end

"""
    zeros!(lmix::gˡᵐⁱˣ{S})

Reset all the matrix elements of `lmix` to `ZERO`.
"""
zeros!(lmix::gˡᵐⁱˣ{S}) where {S} = memset!(lmix, zero(S))

"""
    memcpy!(src::gˡᵐⁱˣ{S}, dst::gˡᵐⁱˣ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::gˡᵐⁱˣ{S}, dst::gˡᵐⁱˣ{S}) where {S}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memcpy!(src::Gˡᵐⁱˣ{S}, dst::gˡᵐⁱˣ{S}, tstp::I64)

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::Gˡᵐⁱˣ{S}, dst::gˡᵐⁱˣ{S}, tstp::I64) where {S}
    @assert iscompatible(src, dst)
    @assert 1 ≤ tstp ≤ src.ntime
    @. dst.data = copy(src.data[tstp,:])
end

"""
    memcpy!(src::gˡᵐⁱˣ{S}, dst::Gˡᵐⁱˣ{S}, tstp::I64)

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::gˡᵐⁱˣ{S}, dst::Gˡᵐⁱˣ{S}, tstp::I64) where {S}
    @assert iscompatible(src, dst)
    @assert 1 ≤ tstp ≤ dst.ntime
    @. dst.data[tstp,:] = copy(src.data)
end

"""
    incr!(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}, alpha::S)

Add a `gˡᵐⁱˣ` with given weight (`alpha`) to another `gˡᵐⁱˣ`.
"""
function incr!(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}, alpha::S) where {S}
    @assert iscompatible(lmix1, lmix2)
    for i = 1:lmix2.ntau
        @. lmix1.data[i] = lmix1.data[i] + lmix2.data[i] * alpha
    end
end

"""
    incr!(lmix1::Gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}, tstp::I64, alpha::S)

Add a `gˡᵐⁱˣ` with given weight (`alpha`) to a `Gˡᵐⁱˣ`.
"""
function incr!(lmix1::Gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}, tstp::I64, alpha::S) where {S}
    @assert iscompatible(lmix1, lmix2)
    @assert 1 ≤ tstp ≤ lmix1.ntime
    for i = 1:lmix2.ntau
        @. lmix1.data[tstp,i] = lmix1.data[tstp,i] + lmix2.data[i] * alpha
    end
end

"""
    incr!(lmix1::gˡᵐⁱˣ{S}, lmix2::Gˡᵐⁱˣ{S}, tstp::I64, alpha::S)

Add a `Gˡᵐⁱˣ` with given weight (`alpha`) to a `gˡᵐⁱˣ`.
"""
function incr!(lmix1::gˡᵐⁱˣ{S}, lmix2::Gˡᵐⁱˣ{S}, tstp::I64, alpha::S) where {S}
    @assert iscompatible(lmix1, lmix2)
    @assert 1 ≤ tstp ≤ lmix2.ntime
    for i = 1:lmix1.ntau
        @. lmix1.data[i] = lmix1.data[i] + lmix2.data[tstp,i] * alpha
    end
end

"""
    smul!(lmix::gˡᵐⁱˣ{S}, alpha::S)

Multiply a `gˡᵐⁱˣ` with given weight (`alpha`).
"""
function smul!(lmix::gˡᵐⁱˣ{S}, alpha::S) where {S}
    for i = 1:lmix.ntau
        @. lmix.data[i] = lmix.data[i] * alpha
    end
end

"""
    smul!(x::Element{S}, lmix::gˡᵐⁱˣ{S})

Left multiply a `gˡᵐⁱˣ` with given weight (`x`).
"""
function smul!(x::Element{S}, lmix::gˡᵐⁱˣ{S}) where {S}
    for i = 1:lmix.ntau
        lmix.data[i] = x * lmix.data[i]
    end
end

"""
    smul!(lmix::gˡᵐⁱˣ{S}, x::Element{S})

Right multiply a `gˡᵐⁱˣ` with given weight (`x`).
"""
function smul!(lmix::gˡᵐⁱˣ{S}, x::Element{S}) where {S}
    for i = 1:lmix.ntau
        lmix.data[i] = lmix.data[i] * x
    end
end

#=
### *gˡᵐⁱˣ* : *Traits*
=#

"""
    Base.:+(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S})

Operation `+` for two `gˡᵐⁱˣ` objects.
"""
function Base.:+(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}) where {S}
    # Sanity check
    @assert getsize(lmix1) == getsize(lmix2)
    @assert getdims(lmix1) == getdims(lmix2)

    gˡᵐⁱˣ(lmix1.ntau, lmix1.ndim1, lmix1.ndim2, lmix1.data + lmix2.data)
end

"""
    Base.:-(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S})

Operation `-` for two `gˡᵐⁱˣ` objects.
"""
function Base.:-(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}) where {S}
    # Sanity check
    @assert getsize(lmix1) == getsize(lmix2)
    @assert getdims(lmix1) == getdims(lmix2)

    gˡᵐⁱˣ(lmix1.ntau, lmix1.ndim1, lmix1.ndim2, lmix1.data - lmix2.data)
end

"""
    Base.:*(lmix::gˡᵐⁱˣ{S}, x)

Operation `*` for a `gˡᵐⁱˣ` object and a scalar value.
"""
function Base.:*(lmix::gˡᵐⁱˣ{S}, x) where {S}
    cx = convert(S, x)
    gˡᵐⁱˣ(lmix.ntau, lmix.ndim1, lmix.ndim2, lmix.data * cx)
end

"""
    Base.:*(x, lmix::gˡᵐⁱˣ{S})

Operation `*` for a scalar value and a `gˡᵐⁱˣ` object.
"""
Base.:*(x, lmix::gˡᵐⁱˣ{S}) where {S} = Base.:*(lmix, x)

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
    gʳᵐⁱˣ(sign, ntau, ndim1, ndim2, dataL)
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
### *gˡᵉˢˢ* : *Struct*
=#

"""
    gˡᵉˢˢ{S}

Lesser component (``G^{<}``) of contour Green's function at given
time step `tstp`. Actually, it denotes ``G^{<}(tᵢ, tⱼ ≡ tstp)``.
"""
mutable struct gˡᵉˢˢ{S} <: CnAbstractVector{S}
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
    @assert tstp ≥ 1
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
    gˡᵉˢˢ(tstp, ndim1, ndim2, data)
end

"""
    gˡᵉˢˢ(tstp::I64, ndim1::I64, ndim2::I64)

Constructor. All the matrix elements are set to be `CZERO`.
"""
function gˡᵉˢˢ(tstp::I64, ndim1::I64, ndim2::I64)
    gˡᵉˢˢ(tstp, ndim1, ndim2, CZERO)
end

"""
    gˡᵉˢˢ(tstp::I64, ndim1::I64)

Constructor. All the matrix elements are set to be `CZERO`.
"""
function gˡᵉˢˢ(tstp::I64, ndim1::I64)
    gˡᵉˢˢ(tstp, ndim1, ndim1, CZERO)
end

"""
    gˡᵉˢˢ(tstp::I64, x::Element{S})

Constructor. The matrix is initialized by `x`.
"""
function gˡᵉˢˢ(tstp::I64, x::Element{S}) where {S}
    # Sanity check
    @assert tstp ≥ 1

    ndim1, ndim2 = size(x)
    data = VecArray{S}(undef, tstp)
    for i = 1:tstp
        data[i] = copy(x)
    end

    # Call the default constructor
    gˡᵉˢˢ(tstp, ndim1, ndim2, data)
end

#=
### *gˡᵉˢˢ* : *Properties*
=#

"""
    getdims(less::gˡᵉˢˢ{S})

Return the dimensional parameters of contour function.

See also: [`gˡᵉˢˢ`](@ref).
"""
function getdims(less::gˡᵉˢˢ{S}) where {S}
    return (less.ndim1, less.ndim2)
end

"""
    getsize(less::gˡᵉˢˢ{S})

Return the size of contour function.

See also: [`gˡᵉˢˢ`](@ref).
"""
function getsize(less::gˡᵉˢˢ{S}) where {S}
    return less.tstp
end

"""
    equaldims(less::gˡᵉˢˢ{S})

Return whether the dimensional parameters are equal.

See also: [`gˡᵉˢˢ`](@ref).
"""
function equaldims(less::gˡᵉˢˢ{S}) where {S}
    return less.ndim1 == less.ndim2
end

"""
    iscompatible(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S})

Judge whether two `gˡᵉˢˢ` objects are compatible.
"""
function iscompatible(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}) where {S}
    getsize(less1) == getsize(less2) &&
    getdims(less1) == getdims(less2)
end

"""
    iscompatible(less1::gˡᵉˢˢ{S}, less2::Gˡᵉˢˢ{S})

Judge whether the `gˡᵉˢˢ` and `Gˡᵉˢˢ` objects are compatible.
"""
function iscompatible(less1::gˡᵉˢˢ{S}, less2::Gˡᵉˢˢ{S}) where {S}
    getsize(less1) ≤ getsize(less2) &&
    getdims(less1) == getdims(less2)
end

"""
    iscompatible(less1::Gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S})

Judge whether the `gˡᵉˢˢ` and `Gˡᵉˢˢ` objects are compatible.
"""
iscompatible(less1::Gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}) where {S} = iscompatible(less2, less1)

"""
    iscompatible(C::Cn, less::gˡᵉˢˢ{S})

Judge whether `C` (which is a `Cn` object) is compatible with `less`
(which is a `gˡᵉˢˢ{S}` object).
"""
function iscompatible(C::Cn, less::gˡᵉˢˢ{S}) where {S}
    C.ntime ≥ getsize(less) &&
    getdims(C) == getdims(less)
end

"""
    iscompatible(less::gˡᵉˢˢ{S}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `less`
(which is a `gˡᵉˢˢ{S}` object).
"""
iscompatible(less::gˡᵉˢˢ{S}, C::Cn) where {S} = iscompatible(C, less)

"""
    distance(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S})

Calculate distance between two `gˡᵉˢˢ` objects.
"""
function distance(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}) where {S}
    @assert iscompatible(less1, less2)

    err = 0.0
    #
    for m = 1:less1.tstp
        err = err + abs(sum(less1.data[m] - less2.data[m]))
    end
    #
    return err
end

"""
    distance(less1::gˡᵉˢˢ{S}, less2::Gˡᵉˢˢ{S}, tstp::I64)

Calculate distance between a `gˡᵉˢˢ` object and a `Gˡᵉˢˢ` object at
given time step `tstp`.
"""
function distance(less1::gˡᵉˢˢ{S}, less2::Gˡᵉˢˢ{S}, tstp::I64) where {S}
    @assert iscompatible(less1, less2)
    @assert tstp == less1.tstp

    err = 0.0
    #
    for m = 1:less1.tstp
        err = err + abs(sum(less1.data[m] - less2.data[m,tstp]))
    end
    #
    return err
end

"""
    distance(less1::Gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}, tstp::I64)

Calculate distance between a `gˡᵉˢˢ` object and a `Gˡᵉˢˢ` object at
given time step `tstp`.
"""
distance(less1::Gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}, tstp::I64) where {S} = distance(less2, less1, tstp)

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
### *gˡᵉˢˢ* : *Operations*
=#

"""
    memset!(less::gˡᵉˢˢ{S}, x)

Reset all the matrix elements of `less` to `x`. `x` should be a
scalar number.
"""
function memset!(less::gˡᵉˢˢ{S}, x) where {S}
    cx = convert(S, x)
    for i=1:less.tstp
        fill!(less.data[i], cx)
    end
end

"""
    zeros!(less::gˡᵉˢˢ{S})

Reset all the matrix elements of `less` to `ZERO`.
"""
zeros!(less::gˡᵉˢˢ{S}) where {S} = memset!(less, zero(S))

"""
    memcpy!(src::gˡᵉˢˢ{S}, dst::gˡᵉˢˢ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::gˡᵉˢˢ{S}, dst::gˡᵉˢˢ{S}) where {S}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    memcpy!(src::Gˡᵉˢˢ{S}, dst::gˡᵉˢˢ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::Gˡᵉˢˢ{S}, dst::gˡᵉˢˢ{S}) where {S}
    @assert iscompatible(src, dst)
    tstp = dst.tstp
    @. dst.data = copy(src.data[1:tstp,tstp])
end

"""
    memcpy!(src::gˡᵉˢˢ{S}, dst::Gˡᵉˢˢ{S})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::gˡᵉˢˢ{S}, dst::Gˡᵉˢˢ{S}) where {S}
    @assert iscompatible(src, dst)
    tstp = src.tstp
    @. dst.data[1:tstp,tstp] = copy(src.data)
end

"""
    incr!(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}, alpha::S)

Add a `gˡᵉˢˢ` with given weight (`alpha`) to another `gˡᵉˢˢ`.
"""
function incr!(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}, alpha::S) where {S}
    @assert iscompatible(less1, less2)
    tstp = less2.tstp
    for i = 1:tstp
        @. less1.data[i] = less1.data[i] + less2.data[i] * alpha
    end
end

"""
    incr!(less1::Gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}, alpha::S)

Add a `gˡᵉˢˢ` with given weight (`alpha`) to a `Gˡᵉˢˢ`.
"""
function incr!(less1::Gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}, alpha::S) where {S}
    @assert iscompatible(less1, less2)
    tstp = less2.tstp
    for i = 1:tstp
        @. less1.data[i,tstp] = less1.data[i,tstp] + less2.data[i] * alpha
    end
end

"""
    incr!(less1::gˡᵉˢˢ{S}, less2::Gˡᵉˢˢ{S}, alpha::S)

Add a `Gˡᵉˢˢ` with given weight (`alpha`) to a `gˡᵉˢˢ`.
"""
function incr!(less1::gˡᵉˢˢ{S}, less2::Gˡᵉˢˢ{S}, alpha::S) where {S}
    @assert iscompatible(less1, less2)
    tstp = less1.tstp
    for i = 1:tstp
        @. less1.data[i] = less1.data[i] + less2.data[i,tstp] * alpha
    end
end

"""
    smul!(less::gˡᵉˢˢ{S}, alpha::S)

Multiply a `gˡᵉˢˢ` with given weight (`alpha`).
"""
function smul!(less::gˡᵉˢˢ{S}, alpha::S) where {S}
    for i = 1:less.tstp
        @. less.data[i] = less.data[i] * alpha
    end
end

"""
    smul!(x::Cf{S}, less::gˡᵉˢˢ{S})

Left multiply a `gˡᵉˢˢ` with given weight (`x`).
"""
function smul!(x::Cf{S}, less::gˡᵉˢˢ{S}) where {S}
    for i = 1:less.tstp
        less.data[i] = x[i] * less.data[i]
    end
end

"""
    smul!(less::gˡᵉˢˢ{S}, x::Element{S})

Right multiply a `gˡᵉˢˢ` with given weight (`x`).
"""
function smul!(less::gˡᵉˢˢ{S}, x::Element{S}) where {S}
    for i = 1:less.tstp
        less.data[i] = less.data[i] * x
    end
end

#=
### *gˡᵉˢˢ* : *Traits*
=#

"""
    Base.:+(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S})

Operation `+` for two `gˡᵉˢˢ` objects.
"""
function Base.:+(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}) where {S}
    # Sanity check
    @assert getsize(less1) == getsize(less2)
    @assert getdims(less1) == getdims(less2)

    gˡᵉˢˢ(less1.tstp, less1.ndim1, less1.ndim2, less1.data + less2.data)
end

"""
    Base.:-(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S})

Operation `-` for two `gˡᵉˢˢ` objects.
"""
function Base.:-(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}) where {S}
    # Sanity check
    @assert getsize(less1) == getsize(less2)
    @assert getdims(less1) == getdims(less2)

    gˡᵉˢˢ(less1.tstp, less1.ndim1, less1.ndim2, less1.data - less2.data)
end

"""
    Base.:*(less::gˡᵉˢˢ{S}, x)

Operation `*` for a `gˡᵉˢˢ` object and a scalar value.
"""
function Base.:*(less::gˡᵉˢˢ{S}, x) where {S}
    cx = convert(S, x)
    gˡᵉˢˢ(less.tstp, less.ndim1, less.ndim2, less.data * cx)
end

"""
    Base.:*(x, less::gˡᵉˢˢ{S})

Operation `*` for a scalar value and a `gˡᵉˢˢ` object.
"""
Base.:*(x, less::gˡᵉˢˢ{S}) where {S} = Base.:*(less, x)

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
    gᵍᵗʳ(tstp, ndim1, ndim2, dataL, dataR)
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

#=
*Full Contour Green's Functions at Given Time Step `tstp`* :

In general, it can be viewed as a slice of the contour Green's function
at time axis. It includes four independent components.

* ``G^{M}(\tau)``
* ``G^{R}(t_i \equiv tstp, t_j)``, where ``t_j \le tstp``
* ``G^{⌉}(t_i \equiv tstp, \tau_j)``
* ``G^{<}(t_i, t_j \equiv tstp)``, where ``t_i \le tstp``

We also name them as `mat`, `ret`, `lmix`, and `less`, respectively.
=#

#=
### *𝒻* : *Struct*
=#

"""
    𝒻{S}

Standard contour-ordered Green's function at given time step `tstp`. It
includes four independent components, namely `mat`, `ret`, `lmix`, and
`less`. If `tstp = 0`, it denotes the equilibrium state (only the `mat`
component is valid). On the other hand, `tstp > 0` means nonequilibrium
state.
"""
mutable struct 𝒻{S} <: CnAbstractFunction{S}
    sign :: I64 # Used to distinguish fermions and bosons
    tstp :: I64
    mat  :: gᵐᵃᵗ{S}
    ret  :: gʳᵉᵗ{S}
    lmix :: gˡᵐⁱˣ{S}
    less :: gˡᵉˢˢ{S}
end

#=
### *𝒻* : *Constructors*
=#

"""
    𝒻(C::Cn, tstp::I64, v::S, sign::I64 = FERMI)

Standard constructor. This function is initialized by `v`.
"""
function 𝒻(C::Cn, tstp::I64, v::S, sign::I64 = FERMI) where {S}
    # Sanity check
    @assert sign in (BOSE, FERMI)
    @assert C.ntime ≥ tstp ≥ 0

    # Create mat, ret, lmix, and less.
    mat = gᵐᵃᵗ(C.ntau, C.ndim1, C.ndim2, v)
    #
    if tstp == 0
        # Actually, at this time this component should not be accessed.
        ret = gʳᵉᵗ(tstp + 1, C.ndim1, C.ndim2, v)
    else
        ret = gʳᵉᵗ(tstp, C.ndim1, C.ndim2, v)
    end
    #
    lmix = gˡᵐⁱˣ(C.ntau, C.ndim1, C.ndim2, v)
    #
    if tstp == 0
        # Actually, at this time this component should not be accessed.
        less = gˡᵉˢˢ(tstp + 1, C.ndim1, C.ndim2, v)
    else
        less = gˡᵉˢˢ(tstp, C.ndim1, C.ndim2, v)
    end

    # Call the default constructor
    𝒻(sign, tstp, mat, ret, lmix, less)
end

"""
    𝒻(C::Cn, tstp::I64, sign::I64 = FERMI)

Constructor. Create a fermionic contour function with zero initial values.
"""
function 𝒻(C::Cn, tstp::I64, sign::I64 = FERMI)
    # Sanity check
    @assert sign in (BOSE, FERMI)
    @assert C.ntime ≥ tstp ≥ 0

    # Create mat, ret, lmix, and less.
    mat = gᵐᵃᵗ(C.ntau, C.ndim1, C.ndim2)
    #
    if tstp == 0
        # Actually, at this time this component should not be accessed.
        ret = gʳᵉᵗ(tstp + 1, C.ndim1, C.ndim2)
    else
        ret = gʳᵉᵗ(tstp, C.ndim1, C.ndim2)
    end
    #
    lmix = gˡᵐⁱˣ(C.ntau, C.ndim1, C.ndim2)
    #
    if tstp == 0
        # Actually, at this time this component should not be accessed.
        less = gˡᵉˢˢ(tstp + 1, C.ndim1, C.ndim2)
    else
        less = gˡᵉˢˢ(tstp, C.ndim1, C.ndim2)
    end

    # Call the default constructor
    𝒻(sign, tstp, mat, ret, lmix, less)
end

"""
    𝒻(tstp::I64, ntau::I64, ndim1::I64, ndim2::I64, sign::I64 = FERMI)

Constructor. Create a fermionic contour function with zero initial values.
"""
function 𝒻(tstp::I64, ntau::I64, ndim1::I64, ndim2::I64, sign::I64 = FERMI)
    # Sanity check
    @assert sign in (BOSE, FERMI)
    @assert tstp ≥ 0
    @assert ntau ≥ 2
    @assert ndim1 ≥ 1
    @assert ndim2 ≥ 1

    # Create mat, ret, lmix, and less.
    mat = gᵐᵃᵗ(ntau, ndim1, ndim2)
    #
    if tstp == 0
        # Actually, at this time this component should not be accessed.
        ret = gʳᵉᵗ(tstp + 1, ndim1, ndim2)
    else
        ret = gʳᵉᵗ(tstp, ndim1, ndim2)
    end
    #
    lmix = gˡᵐⁱˣ(ntau, ndim1, ndim2)
    #
    if tstp == 0
        # Actually, at this time this component should not be accessed.
        less = gˡᵉˢˢ(tstp + 1, ndim1, ndim2)
    else
        less = gˡᵉˢˢ(tstp, ndim1, ndim2)
    end

    # Call the default constructor
    𝒻(sign, tstp, mat, ret, lmix, less)
end

#=
### *𝒻* : *Properties*
=#

"""
    getdims(cfv::𝒻{S})

Return the dimensional parameters of contour Green's function.

See also: [`𝒻`](@ref).
"""
function getdims(cfv::𝒻{S}) where {S}
    return getdims(cfv.less)
end

"""
    getntau(cfv::𝒻{S})

Return the `ntau` parameter of contour Green's function.
"""
function getntau(cfv::𝒻{S}) where {S}
    return getsize(cfv.mat)
end

"""
    gettstp(cfv::𝒻{S})

Return the `tstp` parameter of contour Green's function.
"""
function gettstp(cfv::𝒻{S}) where {S}
    return cfv.tstp # getsize(cfv.less) is wrong when cfv.tstp = 0!
end

"""
    getsign(cfv::𝒻{S})

Return the `sign` parameter of contour Green's function.
"""
function getsign(cfv::𝒻{S}) where {S}
    return cfv.sign
end

"""
    equaldims(cfv::𝒻{S})

Return whether the dimensional parameters are equal.

See also: [`𝒻`](@ref).
"""
function equaldims(cfv::𝒻{S}) where {S}
    return equaldims(cfv.less)
end

"""
    density(cfv::𝒻{S}, tstp::I64)

Returns the density matrix at given time step `tstp`. If `tstp = 0`,
it denotes the equilibrium state. However, when `tstp > 0`, it means
the nonequilibrium state.

See also: [`gᵐᵃᵗ`](@ref), [`gˡᵉˢˢ`](@ref).
"""
function density(cfv::𝒻{S}, tstp::I64) where {S}
    # Sanity check
    @assert tstp == gettstp(cfv)

    if tstp == 0
        return -cfv.mat[getntau(cfv)]
    else
        return cfv.less[tstp] * getsign(cfv) * CZI
    end
end

"""
    distance(cfv1::𝒻{S}, cfv2::𝒻{S}, tstp::I64)

Calculate distance between two `𝒻` objects at given time step `tstp`.
"""
function distance(cfv1::𝒻{S}, cfv2::𝒻{S}, tstp::I64) where {S}
    # Sanity check
    @assert tstp == gettstp(cfv1)

    err = 0.0
    #
    if tstp == 0
        err = err + distance(cfv1.mat, cfv2.mat)
    else
        err = err + distance(cfv1.ret, cfv2.ret)
        err = err + distance(cfv1.lmix, cfv2.lmix)
        err = err + distance(cfv1.less, cfv2.less)
    end
    #
    return err
end

"""
    distance(cfv1::𝒻{S}, cfm2::ℱ{S}, tstp::I64)

Calculate distance between a `𝒻` object and a `ℱ` object at
given time step `tstp`.
"""
function distance(cfv1::𝒻{S}, cfm2::ℱ{S}, tstp::I64) where {S}
    # Sanity check
    @assert tstp == gettstp(cfv1)

    err = 0.0
    #
    if tstp == 0
        err = err + distance(cfv1.mat, cfm2.mat)
    else
        err = err + distance(cfv1.ret, cfm2.ret, tstp)
        err = err + distance(cfv1.lmix, cfm2.lmix, tstp)
        err = err + distance(cfv1.less, cfm2.less, tstp)
    end
    #
    return err
end

"""
    distance(cfm1::ℱ{S}, cfv2::𝒻{S}, tstp::I64)

Calculate distance between a `𝒻` object and a `ℱ` object at
given time step `tstp`.
"""
distance(cfm1::ℱ{S}, cfv2::𝒻{S}, tstp::I64) where {S} = distance(cfv2, cfm1, tstp)

#=
### *𝒻* : *Indexing*
=#

"""
    Base.getindex(cfm::ℱ{T}, tstp::I64)

Return contour Green's function at given time step `tstp`.

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
    memcpy!(cfm, cfv)

    # Return the desired struct
    return cfv
end

"""
    Base.setindex!(cfm::ℱ{S}, cfv::𝒻{S}, tstp::I64)

Setup contout Green's function at given time step `tstp`.

See also: [`ℱ`](@ref), [`𝒻`](@ref).
"""
function Base.setindex!(cfm::ℱ{S}, cfv::𝒻{S}, tstp::I64) where {S}
    # Sanity check
    @assert tstp == gettstp(cfv)
    @assert 0 ≤ tstp ≤ getntime(cfm)

    # Copy data from `𝒻` object to `ℱ` object
    memcpy!(cfv, cfm)
end

#=
### *𝒻* : *Operations*
=#

"""
    memset!(cfv::𝒻{S}, x)

Reset all the matrix elements of `cfv` to `x`. `x` should be a
scalar number.
"""
function memset!(cfv::𝒻{S}, x) where {S}
    memset!(cfv.mat, x)
    memset!(cfv.ret, x)
    memset!(cfv.lmix, x)
    memset!(cfv.less, x)
end

"""
    memset!(cfv::𝒻{S}, tstp::I64, x)

Reset all the matrix elements of `cfv` to `x`. `x` should be a
scalar number. If `tstp = 0`, only the `mat` component is updated.
On the other hand, if `tstp > 0`, the `ret`, `lmix`, and `less`
components will be updated.
"""
function memset!(cfv::𝒻{S}, tstp::I64, x) where {S}
    @assert tstp == gettstp(cfv)
    if tstp > 0
        memset!(cfv.ret, x)
        memset!(cfv.lmix, x)
        memset!(cfv.less, x)
    else
        memset!(cfv.mat, x)
    end
end

"""
    zeros!(cfv::𝒻{S})

Reset all the matrix elements of `cfv` to `ZERO`.
"""
zeros!(cfv::𝒻{S}) where {S} = memset!(cfv, zero(S))

"""
    zeros!(cfv::𝒻{S}, tstp::I64)

Reset all the matrix elements of `cfv` to `ZERO` at given time step `tstp`.
"""
zeros!(cfv::𝒻{S}, tstp::I64) where {S} = memset!(cfv, tstp, zero(S))

"""
    memcpy!(src::𝒻{S}, dst::𝒻{S}, tstp::I64)

Extract data from a `𝒻` object (at given time step `tstp`), then
copy them to another `𝒻` object.

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

Extract data from a `ℱ` object (at given time step `tstp`), then
copy them to a `𝒻` object.

See also: [`ℱ`](@ref), [`𝒻`](@ref).
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

Extract data from a `𝒻` object, then copy them to a `ℱ` object
(at given time step `tstp`).

See also: [`ℱ`](@ref), [`𝒻`](@ref).
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

"""
    incr!(cfv1::𝒻{S}, cfv2::𝒻{S}, tstp::I64, alpha)

Adds a `𝒻` with given weight (`alpha`) to another `𝒻` (at given
time step `tstp`).
"""
function incr!(cfv1::𝒻{S}, cfv2::𝒻{S}, tstp::I64, alpha) where {S}
    @assert gettstp(cfv1) == gettstp(cfv2) == tstp
    α = convert(S, alpha)
    if tstp > 0
        incr!(cfv1.ret, cfv2.ret, α)
        incr!(cfv1.lmix, cfv2.lmix, α)
        incr!(cfv1.less, cfv2.less, α)
    else
        incr!(cfv1.mat, cfv2.mat, α)
    end
end

"""
    incr!(cfm::ℱ{S}, cfv::𝒻{S}, tstp::I64, alpha)

Adds a `𝒻` with given weight (`alpha`) to a `ℱ` (at given
time step `tstp`).
"""
function incr!(cfm::ℱ{S}, cfv::𝒻{S}, tstp::I64, alpha) where {S}
    @assert 0 ≤ tstp ≤ getntime(cfm)
    @assert tstp == gettstp(cfv)
    α = convert(S, alpha)
    if tstp > 0
        incr!(cfm.ret, cfv.ret, α)
        incr!(cfm.lmix, cfv.lmix, tstp, α)
        incr!(cfm.less, cfv.less, α)
    else
        incr!(cfm.mat, cfv.mat, α)
    end
end

"""
    incr!(cfv::𝒻{S}, cfm::ℱ{S}, tstp::I64, alpha)

Adds a `ℱ` with given weight (`alpha`) to a `𝒻` (at given
time step `tstp`).
"""
function incr!(cfv::𝒻{S}, cfm::ℱ{S}, tstp::I64, alpha) where {S}
    @assert 0 ≤ tstp ≤ getntime(cfm)
    @assert tstp == gettstp(cfv)
    α = convert(S, alpha)
    if tstp > 0
        incr!(cfv.ret, cfm.ret, α)
        incr!(cfv.lmix, cfm.lmix, tstp, α)
        incr!(cfv.less, cfm.less, α)
    else
        incr!(cfv.mat, cfm.mat, α)
    end
end

"""
    smul!(cfv::𝒻{S}, tstp::I64, alpha)

Multiply a `𝒻` with given weight (`alpha`) at given time
step `tstp`.
"""
function smul!(cfv::𝒻{S}, tstp::I64, alpha) where {S}
    @assert tstp == gettstp(cfv)
    α = convert(S, alpha)
    if tstp > 0
        smul!(cfv.ret, α)
        smul!(cfv.lmix, α)
        smul!(cfv.less, α)
    else
        smul!(cfv.mat, α)
    end
end

"""
    smul!(cff::Cf{S}, cfv::𝒻{S}, tstp::I64)

Left multiply a `𝒻` with given weight (`Cf`) at given time
step `tstp`.
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

Right multiply a `𝒻` with given weight (`Cf`) at given time
step `tstp`.
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

#=
### *𝒻* : *I/O*
=#

"""
    read!(fname::AbstractString, cfv::𝒻{S})

Read the contour Green's functions from given file.
"""
function read!(fname::AbstractString, cfv::𝒻{S}) where {S}
end

"""
    write(fname::AbstractString, cfv::𝒻{S})

Write the contour Green's functions to given file.
"""
function write(fname::AbstractString, cfv::𝒻{S}) where {S}
end

#=
### *𝒻* : *Traits*
=#
