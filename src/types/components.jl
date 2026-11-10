#
# Project : Lavender
# Source  : components.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks : Contour-Ordered Green's Functions*

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
*Remarks : Matsubara Green's Function*

The Matsubara component of contour-ordered Green's function reads

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

Matsubara component (``G^M``) of contour-ordered Green's function. We
usually call this component `mat`. Here we just assume ``\tau ≥ 0``.
While for ``\tau < 0``, please turn to the `Gᵐᵃᵗᵐ{T}` struct.

See also: [`Gʳᵉᵗ`](@ref), [`Gˡᵐⁱˣ`](@ref), [`Gˡᵉˢˢ`](@ref).
"""
mutable struct Gᵐᵃᵗ{T} <: CnAbstractMatrix{T}
    type  :: String
    ntau  :: I64
    ndim1 :: I64
    ndim2 :: I64
    # Contiguous storage, data[:,:,m] is G^{M}(τₘ).
    data  :: Array{T,3}
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

    # Create contiguous storage, whose size is (ndim1, ndim2, ntau).
    data = fill(v, ndim1, ndim2, ntau)

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

    # Create contiguous storage, whose size is (ndim1, ndim2, ntau).
    data = Array{T,3}(undef, ndim1, ndim2, ntau)
    for i=1:ntau
        data[:,:,i] .= x
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

    # Create contiguous storage, whose size is (ndim1, ndim2, ntau).
    data = Array{T,3}(undef, C.ndim1, C.ndim2, C.ntau)
    for i=1:C.ntau
        data[:,:,i] .= x
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
*Remarks* :

In the KadanoffBaym library, the four independent components of contour
Green's function are as follows:

* *Gᵐᵃᵗ*
* *Gʳᵉᵗ*
* *Gˡᵐⁱˣ*
* *Gˡᵉˢˢ*

They have been defined above. Next, we would like to define some auxiliary
components of contour-ordered Green's function. They are:

* *Gᵐᵃᵗᵐ*
* *Gᵃᵈᵛ*
* *Gʳᵐⁱˣ*
* *Gᵍᵗʳ*

Note that these components are related with each other.

=#

#=
### *Gᵐᵃᵗᵐ* : *Struct*
=#

"""
    Gᵐᵃᵗᵐ{T}

Matsubara component (``G^M``) of contour-ordered Green's function. It is
designed for ``\tau < 0`` case. It is not an independent component. It
can be inferred or deduced from the `Gᵐᵃᵗ{T}` struct. We usually call this
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


