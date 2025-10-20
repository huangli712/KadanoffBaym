#
# Project : Lavender
# Source  : structs.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/10/20
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

---

*Remarks : Contour-ordered Green's Functions*

In the many-body theories, single-particle Green's functions are the
fundamental objects. They describe single-particle excitations as well
as statistical distributions of particles, and play a central role in
the formulation of nonequilibrium dynamical mean-field theory. We define
the nonequilibrium Green's function as the contour-ordered expectation
value,

```math
\begin{equation}
G(t,t') \equiv
    -i \langle \mathcal{T}_{\mathcal{C}} c(t) c^{\dagger}(t') \rangle,
\end{equation}
```

where ``c^{\dagger}(c)`` is a creation (annihilation) operators of
particles, and ``t,\ t' \in \mathcal{C}``. For simplicity, spin and
orbital indices associated with the operators are not shown. Because
of the three branches, on which the time arguments ``t`` and ``t'``
can lie, the Green's function has ``3 \times 3 = 9`` components:

```math
\begin{equation}
G(t,t') \equiv G_{ij}(t,t'),
\end{equation}
```

where ``t \in \mathcal{C}_i``, ``t' \in \mathcal{C}_j``, and
``i,\ j = 1,\ 2,\ 3``. Conventionally, we can express them in a
``3 \times 3`` matrix form

```math
\begin{equation}
\hat{G} =
\begin{pmatrix}
G_{11} & G_{12} & G_{13} \\
G_{21} & G_{22} & G_{23} \\
G_{31} & G_{32} & G_{33} \\
\end{pmatrix}.
\end{equation}
```

In general, one can shift the operator with the largest real-time
argument from ``\mathcal{C}_1`` to ``\mathcal{C}_2`` (and vice versa),
because the time evolution along ``\mathcal{C}_1`` and ``\mathcal{C}_2``
to the right of that operator cancels. This kind of redundancy implies
the following relations among the components of the above matrix:

```math
\begin{equation}
G_{11}(t,t') = G_{12}(t,t'),\ \text{for}\ t \le t',
\end{equation}
```

```math
\begin{equation}
G_{11}(t,t') = G_{21}(t,t'),\ \text{for}\ t > t',
\end{equation}
```

```math
\begin{equation}
G_{22}(t,t') = G_{21}(t,t'),\ \text{for}\ t < t',
\end{equation}
```

```math
\begin{equation}
G_{22}(t,t') = G_{12}(t,t'),\ \text{for}\ t \ge t',
\end{equation}
```

```math
\begin{equation}
G_{13}(t,\tau') = G_{23}(t,\tau'),
\end{equation}
```

```math
\begin{equation}
G_{31}(\tau,t') = G_{32}(\tau,t').
\end{equation}
```

The above equations can be summarized as

```math
\begin{equation}
G_{11} + G_{22} = G_{12} + G_{21}.
\end{equation}
```

These equations thus allow one to eliminate three components out of
nine in the nonequilibrium Green's function. To this end, we introduce
six linearly independent physical Green's functions, namely the retarded
(``G^{R}``), advanced (``G^{A}``), Keldysh (``G^{K}``), left-mixing
(``G^{\rceil}``), right-mixing (``G^{\lceil}``), and Matsubara Green's
functions (``G^{M}``). Their definitions and relevant properties will
be given in the following remarks if needed.
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
*Remarks : Matsubara Green's Function*

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
    type  :: String
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
    @assert ntau  ≥ 2
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
    Gᵐᵃᵗ("mat", ntau, ndim1, ndim2, data)
end

"""
    Gᵐᵃᵗ(ntau::I64, ndim1::I64, ndim2::I64)

Constructor. All the matrix elements are set to be complex zero.
"""
function Gᵐᵃᵗ(ntau::I64, ndim1::I64, ndim2::I64)
    Gᵐᵃᵗ(ntau, ndim1, ndim2, zero(C64))
end

"""
    Gᵐᵃᵗ(ntau::I64, ndim1::I64)

Constructor. All the matrix elements are set to be complex zero.
"""
function Gᵐᵃᵗ(ntau::I64, ndim1::I64)
    Gᵐᵃᵗ(ntau, ndim1, ndim1, zero(C64))
end

"""
    Gᵐᵃᵗ(ntau::I64, x::Element{T})

Constructor. The matrix is initialized by `x`.
"""
function Gᵐᵃᵗ(ntau::I64, x::Element{T}) where {T}
    # Sanity check
    @assert ntau ≥ 2

    ndim1, ndim2 = size(x)

    # Create MatArray{T}, whose size is indeed (ntau, 1).
    data = MatArray{T}(undef, ntau, 1)
    for i=1:ntau
        data[i,1] = copy(x)
    end

    # Call the default constructor
    Gᵐᵃᵗ("mat", ntau, ndim1, ndim2, data)
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
    Gᵐᵃᵗ("mat", C.ntau, C.ndim1, C.ndim2, data)
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

Constructor. All the matrix elements are set to be complex zero.
"""
function Gᵐᵃᵗ(C::Cn)
    Gᵐᵃᵗ(C.ntau, C.ndim1, C.ndim2, zero(C64))
end

"""
    Gᵐᵃᵗ()

Constructor. All the matrix elements are set to be complex zero. Notice
that all of the parameters are extracted from the `PCONTOUR` dict.

See also: [`PCONTOUR`](@ref), [`get_c`](@ref).
"""
function Gᵐᵃᵗ()
    ntau = get_c("ntau")
    ndim1 = get_c("ndim1")
    ndim2 = get_c("ndim2")
    Gᵐᵃᵗ(ntau, ndim1, ndim2, zero(C64))
end

#=
*Remarks : Retarded Green's Function*

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

Retarded component (``G^R``) of contour Green's function. We usually
call this component `ret`.

See also: [`Gᵐᵃᵗ`](@ref), [`Gˡᵐⁱˣ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
mutable struct Gʳᵉᵗ{T} <: CnAbstractMatrix{T}
    type  :: String
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

    # Create MatArray{T}, whose size is indeed (ntime, ntime).
    data = MatArray{T}(undef, ntime, ntime)
    for i = 1:ntime
        for j = 1:ntime
            data[j,i] = copy(x)
        end
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

    # Create MatArray{T}, whose size is indeed (ntime, ntime).
    data = MatArray{T}(undef, C.ntime, C.ntime)
    for i = 1:C.ntime
        for j = 1:C.ntime
            data[j,i] = copy(x)
        end
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
*Remarks : Left-mixing Green's Function*

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

Left-mixing component (``G^{⌉}``) of contour Green's function. We usually
call this component `lmix`.

See also: [`Gᵐᵃᵗ`](@ref), [`Gʳᵉᵗ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
mutable struct Gˡᵐⁱˣ{T} <: CnAbstractMatrix{T}
    type  :: String
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
    @assert ntau  ≥ 2
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

    # Create MatArray{T}, whose size is indeed (ntime, ntau).
    data = MatArray{T}(undef, ntime, ntau)
    for i = 1:ntau
        for j = 1:ntime
            data[j,i] = copy(x)
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

    # Create MatArray{T}, whose size is indeed (ntime, ntau)
    data = MatArray{T}(undef, C.ntime, C.ntau)
    for i = 1:C.ntau
        for j = 1:C.ntime
            data[j,i] = copy(x)
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
### *gᵐᵃᵗ* : *Struct*
=#

"""
    gᵐᵃᵗ{S}

Matsubara component (``G^{M}``) of contour Green's function at given
time step `tstp`. Actually, `gᵐᵃᵗ{S}` is equivalent to `Gᵐᵃᵗ{T}`.

See also: [`gʳᵉᵗ`](@ref), [`gˡᵐⁱˣ`](@ref), [`gˡᵉˢˢ`](@ref).
"""
mutable struct gᵐᵃᵗ{S} <: CnAbstractVector{S}
    type  :: String
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
    @assert ntau  ≥ 2
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
    gᵐᵃᵗ("mat", ntau, ndim1, ndim2, data)
end

"""
    gᵐᵃᵗ(ntau::I64, ndim1::I64, ndim2::I64)

Constructor. All the vector elements are set to be complex zero.
"""
function gᵐᵃᵗ(ntau::I64, ndim1::I64, ndim2::I64)
    gᵐᵃᵗ(ntau, ndim1, ndim2, zero(C64))
end

"""
    gᵐᵃᵗ(ntau::I64, ndim1::I64)

Constructor. All the vector elements are set to be complex zero.
"""
function gᵐᵃᵗ(ntau::I64, ndim1::I64)
    gᵐᵃᵗ(ntau, ndim1, ndim1, zero(C64))
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
    gᵐᵃᵗ("mat", ntau, ndim1, ndim2, data)
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
    gʳᵉᵗ(tstp::I64, ndim1::I64, ndim2::I64, v::S) where {S}

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
    data = VecArray{S}(undef, tstp)
    for i = 1:tstp
        data[i] = copy(x)
    end

    # Call the default constructor
    gʳᵉᵗ("ret", tstp, ndim1, ndim2, data)
end

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
    data = VecArray{S}(undef, ntau)
    for i = 1:ntau
        data[i] = copy(x)
    end

    # Call the default constructor
    gˡᵐⁱˣ("lmix", ntau, ndim1, ndim2, data)
end

#=
### *gˡᵉˢˢ* : *Struct*
=#

"""
    gˡᵉˢˢ{S}

Lesser component (``G^{<}``) of contour Green's function at given
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
    data = VecArray{S}(undef, tstp)
    for i = 1:tstp
        data[i] = copy(x)
    end

    # Call the default constructor
    gˡᵉˢˢ("less", tstp, ndim1, ndim2, data)
end

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
*Remarks : Full Contour Green's Functions*

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
*Remarks : Full Contour Green's Functions at Given Time Step `tstp`*

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
    @assert tstp  ≥ 0
    @assert ntau  ≥ 2
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
