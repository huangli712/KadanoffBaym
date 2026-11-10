#
# Project : Lavender
# Source  : cf.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks : Kadanoff-Baym Contour*

We adopt an 𝐿-shaped `Kadanoff-Baym` contour ``\mathcal{C}`` with
three branches:

* ``\mathcal{C}_1``: ``0 \longrightarrow t_{\text{max}}``
* ``\mathcal{C}_2``: ``t_{\text{max}} \longrightarrow 0``
* ``\mathcal{C}_3``: ``0 \longrightarrow -i\beta``

where ``t_{\text{max}}`` is the maximal time up to which one wants to
let the system evolve and ``\beta`` is the inverse temperature
(``\beta \equiv 1/T``).

The expectation value of an observable ``\mathcal{O}`` measured at time
``t`` is given by

```math
\begin{equation}
\langle \mathcal{O}(t) \rangle =
    \frac{\text{Tr}\left[
                   \mathcal{T}_{\mathcal{C}}
                   e^{-i\int_{\mathcal{C}} d\bar{t} \mathcal{H}(\bar{t})}
                   \mathcal{O}(t)
                   \right]}
         {\text{Tr}\left[
                   \mathcal{T}_{\mathcal{C}}
                   e^{-i\int_{\mathcal{C}} d\bar{t} \mathcal{H}(\bar{t})}
         \right]},
\end{equation}
```

where ``\mathcal{T}_{\mathcal{C}}`` is a contour-ordering operator
that arranges operators on the contour ``\mathcal{C}`` in the order
``0 \to t_{\text{max}} \to 0 \to -i\beta``, while ``\mathcal{T}_{\tau}``
is the time-ordering operator only on the imaginary-time axis.

The contour-ordered formalism reveals its full power when it is applied
to higher-order correlation functions,

```math
\begin{equation}
\langle \mathcal{T}_{\mathcal{C}} \mathcal{A}(t) \mathcal{B}(t') \rangle
    =
    \frac{1}{Z} \text{Tr}
    \left[
        \mathcal{T}_{\mathcal{C}}
        e^{-i\int_{\mathcal{C}} d\bar{t} \mathcal{H}(\bar{t})}
        \mathcal{A}(t) \mathcal{B}(t')
    \right].
\end{equation}
```

Here ``\mathcal{A}`` and ``\mathcal{B}`` are combinations of particle's
creation and annihilation operators. We call them ''fermionic'' if they
contain odd number of fermion's creation or annihilation operators, and
''bosonic'' otherwise. In this expression, ``t`` and ``t'`` can lie
anywhere on ``\mathcal{C}``, and the contour-ordered product of two
operators ``\mathcal{A}`` and ``\mathcal{B}`` is defined as

```math
\begin{equation}
\mathcal{T}_{\mathcal{C}} \mathcal{A}(t) \mathcal{B}(t')
    = \theta_{\mathcal{C}}(t,t') \mathcal{A}(t) \mathcal{B}(t')
    \pm \theta_{\mathcal{C}}(t',t) \mathcal{B}(t') \mathcal{A}(t),
\end{equation}
```

where ``\theta_{\mathcal{C}}(t,t') = 1`` when ``t'`` comes earlier than
``t`` in the contour ordering (denoted by ``t \succ t'``) and 0 otherwise
(``t \prec t'``). The sign ``\pm`` is taken to be minus when the operators
``\mathcal{A}`` and ``\mathcal{B}`` are both fermionic and plus otherwise.

=#

#=
### *Cn* : *Struct*
=#

"""
    Cn

𝐿-shape `Kadanoff-Baym` contour. It includes the following members:

* ntime -> Number of time slices in real time axis [0, 𝑡max].
* ntau -> Number of time slices in imaginary time axis [0, β].
* ndim1 -> Size of operators that stored in the contour.
* ndim2 -> Size of operators that stored in the contour.
* tmax -> Maximum 𝑡.
* beta -> β, inverse temperature.
* dt -> δ𝑡, time step in real time axis.
* dtau -> δτ, time step in imaginary time axis.

See also: [`CnAbstractContour`](@ref).
"""
mutable struct Cn <: CnAbstractContour
    ntime :: I64
    ntau  :: I64
    ndim1 :: I64
    ndim2 :: I64
    tmax  :: F64
    beta  :: F64
    dt    :: F64
    dtau  :: F64
end

#=
### *Cn* : *Constructors*
=#

"""
    Cn(ntime::I64, ntau::I64, ndim1::I64,
       ndim2::I64, tmax::F64, beta::F64)

Constructor. Create a general 𝐿-shape `Kadanoff-Baym` contour.
"""
function Cn(ntime::I64, ntau::I64, ndim1::I64,
            ndim2::I64, tmax::F64, beta::F64)
    # Sanity check
    @assert ntime ≥ 2
    @assert ntau  ≥ 2
    @assert ndim1 ≥ 1
    @assert ndim2 ≥ 1
    @assert tmax  > 0.0
    @assert beta  > 0.0

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

"""
    Cn()

Constructor. The parameters should be extracted from the `PCONTOUR` dict.

See also: [`PCONTOUR`](@ref), [`get_c`](@ref).
"""
function Cn()
    ntime = get_c("ntime")
    ntau = get_c("ntau")
    ndim1 = get_c("ndim1")
    ndim2 = get_c("ndim2")
    tmax = get_c("tmax")
    beta = get_c("beta")
    Cn(ntime, ntau, ndim1, ndim2, tmax, beta)
end


#=
*Remarks : Contour-Based Functions*

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

"""
    Cf()

Constructor. All the matrix elements are set to be complex zero. Notice
that all of the parameters are extracted from the `PCONTOUR` dict.

See also: [`PCONTOUR`](@ref), [`get_c`](@ref).
"""
function Cf()
    ntime = get_c("ntime")
    ndim1 = get_c("ndim1")
    ndim2 = get_c("ndim2")
    Cf(ntime, ndim1, ndim2, zero(C64))
end


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
