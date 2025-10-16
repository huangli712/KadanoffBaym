#
# Project : Lavender
# Source  : gfull.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/10/15
#

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
