#
# Project : Lavender
# Source  : contour.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/10/15
#


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
*Remarks : Contour-based Functions*

It is a general matrix-valued function defined at the `Kadanoff-Baym`
contour:

```math
\begin{equation}
f_{\mathcal{C}} = f(t),
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

Constructor. All the matrix elements are set to be complex zero.
"""
function Cf(ntime::I64, ndim1::I64, ndim2::I64)
    Cf(ntime, ndim1, ndim2, zero(C64))
end

"""
    Cf(ntime::I64, ndim1::I64)

Constructor. All the matrix elements are set to be complex zero.
"""
function Cf(ntime::I64, ndim1::I64)
    Cf(ntime, ndim1, ndim1, zero(C64))
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

Constructor. All the matrix elements are set to be complex zero.
"""
function Cf(C::Cn)
    Cf(C.ntime, C.ndim1, C.ndim2, zero(C64))
end

#=
### *Cf* : *Properties*
=#

"""
    getdims(cf::Cf{T})

Return the dimensional parameters of contour function.

See also: [`Cf`](@ref).
"""
function getdims(cf::Cf{T}) where {T}
    return (cf.ndim1, cf.ndim2)
end

"""
    getsize(cf::Cf{T})

Return the nominal size of contour function, i.e `ntime`. Actually, the
real size of contour function should be `ntime + 1`.

See also: [`Cf`](@ref).
"""
function getsize(cf::Cf{T}) where {T}
    return cf.ntime
end

"""
    equaldims(cf::Cf{T})

Return whether the dimensional parameters are equal.

See also: [`Cf`](@ref).
"""
function equaldims(cf::Cf{T}) where {T}
    return cf.ndim1 == cf.ndim2
end

"""
    iscompatible(cf1::Cf{T}, cf2::Cf{T})

Judge whether two `Cf` objects are compatible.
"""
function iscompatible(cf1::Cf{T}, cf2::Cf{T}) where {T}
    getsize(cf1) == getsize(cf2) &&
    getdims(cf1) == getdims(cf2)
end

"""
    iscompatible(C::Cn, cf::Cf{T})

Judge whether `C` (which is a `Cn` object) is compatible with `cf`
(which is a `Cf{T}` object).
"""
function iscompatible(C::Cn, cf::Cf{T}) where {T}
    C.ntime == getsize(cf) &&
    getdims(C) == getdims(cf)
end

"""
    iscompatible(cf::Cf{T}, C::Cn)

Judge whether `C` (which is a `Cn` object) is compatible with `cf`
(which is a `Cf{T}` object).
"""
iscompatible(cf::Cf{T}, C::Cn) where {T} = iscompatible(C, cf)

#=
### *Cf* : *Indexing*
=#

"""
    Base.getindex(cf::Cf{T}, i::I64)

Visit the element stored in `Cf` object. If `i = 0`, it returns
the element at Matsubara axis. On the other hand, if `i > 0`, it will
return elements at real time axis.
"""
function Base.getindex(cf::Cf{T}, i::I64) where {T}
    # Sanity check
    @assert 0 ≤ i ≤ cf.ntime

    # Return 𝑓(𝑡ᵢ)
    if i == 0 # Matsubara axis
        cf.data[end]
    else # Real time axis
        cf.data[i]
    end
end

"""
    Base.setindex!(cf::Cf{T}, x::Element{T}, i::I64)

Setup the element in `Cf` object. If `i = 0`, it will setup the
element at Matsubara axis to `x`. On the other hand, if `i > 0`, it
will setup elements at real time axis.
"""
function Base.setindex!(cf::Cf{T}, x::Element{T}, i::I64) where {T}
    # Sanity check
    @assert size(x) == getdims(cf)
    @assert 0 ≤ i ≤ cf.ntime

    # 𝑓(𝑡ᵢ) = x
    if i == 0 # Matsubara axis
        cf.data[end] = copy(x)
    else # Real time axis
        cf.data[i] = copy(x)
    end
end

"""
    Base.setindex!(cf::Cf{T}, v::T, i::I64)

Setup the element in `Cf` object. If `i = 0`, it will setup the
element at Matsubara axis to `v`. On the other hand, if `i > 0`, it
will setup elements at real time axis. Here, `v` should be a scalar
number.
"""
function Base.setindex!(cf::Cf{T}, v::T, i::I64) where {T}
    # Sanity check
    @assert 0 ≤ i ≤ cf.ntime

    # 𝑓(𝑡ᵢ) .= v
    if i == 0 # Matsubara axis
        fill!(cf.data[end], v)
    else # Real time axis
        fill!(cf.data[i], v)
    end
end

#=
### *Cf* : *Operations*
=#

"""
    memset!(cf::Cf{T}, x)

Reset all the matrix elements of `cf` to `x`. `x` should be a
scalar number.
"""
function memset!(cf::Cf{T}, x) where {T}
    cx = convert(T, x)
    for i = 1:cf.ntime + 1
        fill!(cf.data[i], cx)
    end
end

"""
    zeros!(cf::Cf{T})

Reset all the matrix elements of `cf` to `zero`.
"""
zeros!(cf::Cf{T}) where {T} = memset!(cf, zero(T))

"""
    memcpy!(src::Cf{T}, dst::Cf{T})

Copy all the matrix elements from `src` to `dst`.
"""
function memcpy!(src::Cf{T}, dst::Cf{T}) where {T}
    @assert iscompatible(src, dst)
    @. dst.data = copy(src.data)
end

"""
    incr!(cf1::Cf{T}, cf2::Cf{T}, α::T)

Add a `Cf` with given weight (`α`) to another `Cf`. Finally,
`cf1` will be changed.
"""
function incr!(cf1::Cf{T}, cf2::Cf{T}, α::T) where {T}
    @assert iscompatible(cf1, cf2)
    for i = 1:cf1.ntime + 1
        @. cf1.data[i] = cf1.data[i] + cf2.data[i] * α
    end
end

"""
    smul!(cf::Cf{T}, α::T)

Multiply a `Cf` with given weight (`α`).
"""
function smul!(cf::Cf{T}, α::T) where {T}
    for i = 1:cf.ntime + 1
        @. cf.data[i] = cf.data[i] * α
    end
end

"""
    smul!(x::Element{T}, cf::Cf{T})

Left multiply a `Cf` with given weight (`x`).
"""
function smul!(x::Element{T}, cf::Cf{T}) where {T}
    for i = 1:cf.ntime + 1
        cf.data[i] = x * cf.data[i]
    end
end

"""
    smul!(cf::Cf{T}, x::Element{T})

Right multiply a `Cf` with given weight (`x`).
"""
function smul!(cf::Cf{T}, x::Element{T}) where {T}
    for i = 1:cf.ntime + 1
        cf.data[i] = cf.data[i] * x
    end
end

#=
### *Cf* : *Traits*
=#

"""
    Base.:+(cf1::Cf{T}, cf2::Cf{T})

Operation `+` for two `Cf` objects.
"""
function Base.:+(cf1::Cf{T}, cf2::Cf{T}) where {T}
    # Sanity check
    @assert getsize(cf1) == getsize(cf2)
    @assert getdims(cf1) == getdims(cf2)

    Cf(cf1.ntime, cf1.ndim1, cf1.ndim2, cf1.data + cf2.data)
end

"""
    Base.:-(cf1::Cf{T}, cf2::Cf{T})

Operation `-` for two `Cf` objects.
"""
function Base.:-(cf1::Cf{T}, cf2::Cf{T}) where {T}
    # Sanity check
    @assert getsize(cf1) == getsize(cf2)
    @assert getdims(cf1) == getdims(cf2)

    Cf(cf1.ntime, cf1.ndim1, cf1.ndim2, cf1.data - cf2.data)
end

"""
    Base.:*(cf::Cf{T}, x)

Operation `*` for a `Cf` object and a scalar value.
"""
function Base.:*(cf::Cf{T}, x) where {T}
    cx = convert(T, x)
    Cf(cf.ntime, cf.ndim1, cf.ndim2, cf.data * cx)
end

"""
    Base.:*(x, cf::Cf{T})

Operation `*` for a scalar value and a `Cf` object.
"""
Base.:*(x, cf::Cf{T}) where {T} = Base.:*(cf, x)
