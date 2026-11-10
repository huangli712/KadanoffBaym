#
# Project : Lavender
# Source  : green.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
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

*Remarks : Full Contour-Ordered Green's Functions*

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
    ℱ(C::Cn, x::Element{T}, sign::I64)

Standard constructor. This function is initialized by `x`.
"""
function ℱ(C::Cn, x::Element{T}, sign::I64) where {T}
    # Sanity check
    @assert sign in (BOSE, FERMI)

    # Create mat, ret, lmix, and less.
    mat  = Gᵐᵃᵗ(C, x)
    ret  = Gʳᵉᵗ(C, x)
    lmix = Gˡᵐⁱˣ(C, x)
    less = Gˡᵉˢˢ(C, x)

    # Call the default constructor
    ℱ(sign, mat, ret, lmix, less)
end

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

Constructor. Create a contour-ordered Green's function with zero initial
values.
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

"""
    ℱ(sign::I64 = FERMI)

Constructor. Create a contour-ordered Green's function with zero initial
values.
"""
function ℱ(sign::I64 = FERMI)
    # Setup sign
    @assert sign in (BOSE, FERMI)

    # Create mat, ret, lmix, and less.
    mat  = Gᵐᵃᵗ()
    ret  = Gʳᵉᵗ()
    lmix = Gˡᵐⁱˣ()
    less = Gˡᵉˢˢ()

    # Call the default constructor
    ℱ(sign, mat, ret, lmix, less)
end

#=
*Remarks : Full Contour-Ordered Green's Functions at Given Time Step `tstp`*

In general, it can be viewed as a slice of the contour-ordered Green's
function at time axis. It includes four independent components.

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
    𝒻(C::Cn, tstp::I64, x::Element{S}, sign::I64 = FERMI)

Standard constructor. This function is initialized by `x`.
"""
function 𝒻(C::Cn, tstp::I64, x::Element{S}, sign::I64 = FERMI) where {S}
    # Sanity check
    @assert sign in (BOSE, FERMI)
    @assert C.ntime ≥ tstp ≥ 0

    # Create mat, ret, lmix, and less.
    mat = gᵐᵃᵗ(C.ntau, x)
    #
    if tstp == 0
        # Actually, at this time this component should not be accessed.
        ret = gʳᵉᵗ(tstp + 1, x)
    else
        ret = gʳᵉᵗ(tstp, x)
    end
    #
    lmix = gˡᵐⁱˣ(C.ntau, x)
    #
    if tstp == 0
        # Actually, at this time this component should not be accessed.
        less = gˡᵉˢˢ(tstp + 1, x)
    else
        less = gˡᵉˢˢ(tstp, x)
    end

    # Call the default constructor
    𝒻(sign, tstp, mat, ret, lmix, less)
end

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

