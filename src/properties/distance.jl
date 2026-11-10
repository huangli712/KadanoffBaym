#
# Project : Lavender
# Source  : distance.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    distance(cf1::Cf{T}, cf2::Cf{T})

Calculate distance between two `Cf` objects.
"""
function distance(cf1::Cf{T}, cf2::Cf{T}) where {T}
    @assert iscompatible(cf1, cf2)

    err = 0.0
    #
    for m = 1:getsize(cf1)+1
        err = err + abs(sum(cf1.data[m] - cf2.data[m]))
    end
    #
    return err
end

"""
    distance(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T})

Calculate distance between two `Gᵐᵃᵗ` objects.
"""
function distance(mat1::Gᵐᵃᵗ{T}, mat2::Gᵐᵃᵗ{T}) where {T}
    @assert iscompatible(mat1, mat2)

    err = 0.0
    #
    for m = 1:getntau(mat1)
        err = err + abs(sum(_block(mat1, m) - _block(mat2, m)))
    end
    #
    return err
end

"""
    distance(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}, tstp::I64)

Calculate distance between two `Gʳᵉᵗ` objects at given time step `tstp`.
"""
function distance(ret1::Gʳᵉᵗ{T}, ret2::Gʳᵉᵗ{T}, tstp::I64) where {T}
    # Sanity check
    @assert iscompatible(ret1, ret2)
    @assert 1 ≤ tstp ≤ getsize(ret1)
    @assert 1 ≤ tstp ≤ getsize(ret2)

    err = 0
    #
    for i = 1:tstp
        err = err + abs(sum(_block(ret1, tstp, i) - _block(ret2, tstp, i)))
    end
    #
    return err
end

"""
    distance(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}, tstp::I64)

Calculate distance between two `Gˡᵐⁱˣ` objects at given time step `tstp`.
"""
function distance(lmix1::Gˡᵐⁱˣ{T}, lmix2::Gˡᵐⁱˣ{T}, tstp::I64) where {T}
    # Sanity check
    @assert iscompatible(lmix1, lmix2)
    @assert 1 ≤ tstp ≤ getntime(lmix1)
    @assert 1 ≤ tstp ≤ getntime(lmix2)

    err = 0
    #
    for i = 1:getntau(lmix1)
        err = err + abs(sum(_block(lmix1, tstp, i) - _block(lmix2, tstp, i)))
    end
    #
    return err
end

"""
    distance(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}, tstp::I64)

Calculate distance between two `Gˡᵉˢˢ` objects at given time step `tstp`.
"""
function distance(less1::Gˡᵉˢˢ{T}, less2::Gˡᵉˢˢ{T}, tstp::I64) where {T}
    # Sanity check
    @assert iscompatible(less1, less2)
    @assert 1 ≤ tstp ≤ getsize(less1)
    @assert 1 ≤ tstp ≤ getsize(less2)

    err = 0
    #
    for i = 1:tstp
        err = err + abs(sum(_block(less1, i, tstp) - _block(less2, i, tstp)))
    end
    #
    return err
end

"""
    distance(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S})

Calculate distance between two `gᵐᵃᵗ` objects.
"""
function distance(mat1::gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}) where {S}
    @assert iscompatible(mat1, mat2)

    err = 0.0
    #
    for m = 1:getntau(mat1)
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
    for m = 1:getntau(mat1)
        err = err + abs(sum(mat1.data[m] - _block(mat2, m)))
    end
    #
    return err
end

"""
    distance(mat1::Gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S})

Calculate distance between a `gᵐᵃᵗ` object and a `Gᵐᵃᵗ` object.
"""
distance(mat1::Gᵐᵃᵗ{S}, mat2::gᵐᵃᵗ{S}) where {S} = distance(mat2, mat1)

"""
    distance(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S})

Calculate distance between two `gʳᵉᵗ` objects.
"""
function distance(ret1::gʳᵉᵗ{S}, ret2::gʳᵉᵗ{S}) where {S}
    @assert iscompatible(ret1, ret2)

    err = 0.0
    #
    for m = 1:gettstp(ret1)
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
    @assert gettstp(ret1) == tstp
    @assert 1 ≤ tstp ≤ getntime(ret2)

    err = 0.0
    #
    for m = 1:gettstp(ret1)
        err = err + abs(sum(ret1.data[m] - _block(ret2, tstp, m)))
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

"""
    distance(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S})

Calculate distance between two `gˡᵐⁱˣ` objects.
"""
function distance(lmix1::gˡᵐⁱˣ{S}, lmix2::gˡᵐⁱˣ{S}) where {S}
    @assert iscompatible(lmix1, lmix2)

    err = 0.0
    #
    for m = 1:getntau(lmix1)
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
    @assert 1 ≤ tstp ≤ getntime(lmix2)

    err = 0.0
    #
    for m = 1:getntau(lmix1)
        err = err + abs(sum(lmix1.data[m] - _block(lmix2, tstp, m)))
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

"""
    distance(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S})

Calculate distance between two `gˡᵉˢˢ` objects.
"""
function distance(less1::gˡᵉˢˢ{S}, less2::gˡᵉˢˢ{S}) where {S}
    @assert iscompatible(less1, less2)

    err = 0.0
    #
    for m = 1:gettstp(less1)
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
    @assert tstp == gettstp(less1)
    @assert 1 ≤ tstp ≤ getntime(less2)

    err = 0.0
    #
    for m = 1:gettstp(less1)
        err = err + abs(sum(less1.data[m] - _block(less2, m, tstp)))
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
### *ℱ* and *𝒻* : *Distance*
=#

#=
*Remarks* : *Distance Between Contour-Ordered Green's Functions*

The distance between two contour-ordered Green's functions `A` and `B`
at given time step `tstp` is defined as the sum of the Frobenius norms
of the differences of the matrix blocks:

```math
\begin{equation}
d^{R}(t) = \sum_{j} \| A^{R}(t,t_j) - B^{R}(t,t_j) \|,
\end{equation}
```

```math
\begin{equation}
d^{\rceil}(t) = \sum_{m} \| A^{\rceil}(t,\tau_m) - B^{\rceil}(t,\tau_m) \|,
\end{equation}
```

```math
\begin{equation}
d^{<}(t) = \sum_{i} \| A^{<}(t_i,t) - B^{<}(t_i,t) \|.
\end{equation}
```

Here ``t_j \le t`` and ``t_i \le t``. The total distance is
``d^{R} + d^{\rceil} + d^{<}`` at the real time axis, and
``\sum_m \| A^{M}(\tau_m) - B^{M}(\tau_m) \|`` at the Matsubara axis.
Note that it is not normalized.

*References* :

Please see [`NESSi`] `cntr_utilities_impl.hpp`, `distance_norm2`.
=#

"""
    distance(cfa::Union{ℱ{T},𝒻{T}}, cfb::Union{ℱ{T},𝒻{T}}, tstp::I64, ::Val{:ret})

Calculate distance between the retarded components of two contour-ordered
Green's functions at given time step `tstp`, i.e., the sum of the
Frobenius norms of ``A^{R}(t,t_j) - B^{R}(t,t_j)``. It returns zero if
`tstp = 0`.

### Arguments
* cfa  -> A `ℱ` or `𝒻` object.
* cfb  -> A `ℱ` or `𝒻` object.
* tstp -> Index of time step.

### Returns
* err -> Distance, a F64 number.
"""
function distance(cfa::Union{ℱ{T},𝒻{T}}, cfb::Union{ℱ{T},𝒻{T}},
                  tstp::I64, ::Val{:ret}) where {T}
    # Sanity check
    _check_tstp(cfa, tstp)
    _check_tstp(cfb, tstp)
    @assert getdims(cfa) == getdims(cfb)

    err = 0.0
    #
    for j = 1:tstp
        err = err + _blkdist(_ret_at(cfa, tstp, j), _ret_at(cfb, tstp, j))
    end
    #
    return err
end

"""
    distance(cfa::Union{ℱ{T},𝒻{T}}, cfb::Union{ℱ{T},𝒻{T}}, tstp::I64, ::Val{:tv})

Calculate distance between the left-mixing components of two
contour-ordered Green's functions at given time step `tstp`, i.e., the
sum of the Frobenius norms of ``A^{\\rceil}(t,\\tau_m) - B^{\\rceil}(t,\\tau_m)``.
It returns zero if `tstp = 0`.

### Arguments
* cfa  -> A `ℱ` or `𝒻` object.
* cfb  -> A `ℱ` or `𝒻` object.
* tstp -> Index of time step.

### Returns
* err -> Distance, a F64 number.
"""
function distance(cfa::Union{ℱ{T},𝒻{T}}, cfb::Union{ℱ{T},𝒻{T}},
                  tstp::I64, ::Val{:tv}) where {T}
    # Sanity check
    _check_tstp(cfa, tstp)
    _check_tstp(cfb, tstp)
    @assert getdims(cfa) == getdims(cfb)
    @assert getntau(cfa) == getntau(cfb)

    err = 0.0
    #
    if tstp > 0
        for m = 1:getntau(cfa)
            err = err + _blkdist(_lmix_at(cfa, tstp, m), _lmix_at(cfb, tstp, m))
        end
    end
    #
    return err
end

"""
    distance(cfa::Union{ℱ{T},𝒻{T}}, cfb::Union{ℱ{T},𝒻{T}}, tstp::I64, ::Val{:les})

Calculate distance between the lesser components of two contour-ordered
Green's functions at given time step `tstp`, i.e., the sum of the
Frobenius norms of ``A^{<}(t_i,t) - B^{<}(t_i,t)``. It returns zero if
`tstp = 0`.

### Arguments
* cfa  -> A `ℱ` or `𝒻` object.
* cfb  -> A `ℱ` or `𝒻` object.
* tstp -> Index of time step.

### Returns
* err -> Distance, a F64 number.
"""
function distance(cfa::Union{ℱ{T},𝒻{T}}, cfb::Union{ℱ{T},𝒻{T}},
                  tstp::I64, ::Val{:les}) where {T}
    # Sanity check
    _check_tstp(cfa, tstp)
    _check_tstp(cfb, tstp)
    @assert getdims(cfa) == getdims(cfb)

    err = 0.0
    #
    for i = 1:tstp
        err = err + _blkdist(_less_at(cfa, tstp, i), _less_at(cfb, tstp, i))
    end
    #
    return err
end

"""
    distance(cfa::Union{ℱ{T},𝒻{T}}, cfb::Union{ℱ{T},𝒻{T}}, tstp::I64)

Calculate distance between two contour-ordered Green's functions at
given time step `tstp`. If `tstp = 0`, only the Matsubara component is
considered. Otherwise, it is the sum of the distances of the `ret`,
`lmix`, and `less` components.

### Arguments
* cfa  -> A `ℱ` or `𝒻` object.
* cfb  -> A `ℱ` or `𝒻` object.
* tstp -> Index of time step.

### Returns
* err -> Distance, a F64 number.
"""
function distance(cfa::Union{ℱ{T},𝒻{T}}, cfb::Union{ℱ{T},𝒻{T}},
                  tstp::I64) where {T}
    # Sanity check
    _check_tstp(cfa, tstp)
    _check_tstp(cfb, tstp)
    @assert getdims(cfa) == getdims(cfb)
    @assert getntau(cfa) == getntau(cfb)

    err = 0.0
    #
    if tstp == 0
        for m = 1:getntau(cfa)
            err = err + _blkdist(_mat_at(cfa, m), _mat_at(cfb, m))
        end
    else
        err = err + distance(cfa, cfb, tstp, Val(:ret))
        err = err + distance(cfa, cfb, tstp, Val(:tv))
        err = err + distance(cfa, cfb, tstp, Val(:les))
    end
    #
    return err
end

"""
    distance(pga::Union{PseudoGF{T},PseudoGFStep{T}},
             pgb::Union{PseudoGF{T},PseudoGFStep{T}}, tstp::I64)

Calculate distance between two pseudo-particle Green's functions at
given time step `tstp`. The definition is the same as that of `ℱ`. If
`tstp = 0`, only the Matsubara component is considered. Otherwise, it
is the sum of the distances of the `ret` (greater), `lmix`, and `less`
components.

### Arguments
* pga  -> A `PseudoGF` or `PseudoGFStep` object.
* pgb  -> A `PseudoGF` or `PseudoGFStep` object.
* tstp -> Index of time step.

### Returns
* err -> Distance, a F64 number.
"""
function distance(pga::Union{PseudoGF{T},PseudoGFStep{T}},
                  pgb::Union{PseudoGF{T},PseudoGFStep{T}},
                  tstp::I64) where {T}
    # Sanity check
    _check_tstp(pga, tstp)
    _check_tstp(pgb, tstp)
    @assert getdims(pga) == getdims(pgb)
    @assert getntau(pga) == getntau(pgb)

    err = 0.0
    #
    if tstp == 0
        for m = 1:getntau(pga)
            err = err + _blkdist(_mat_at(pga, m), _mat_at(pgb, m))
        end
    else
        for j = 1:tstp
            err = err + _blkdist(_ret_at(pga, tstp, j), _ret_at(pgb, tstp, j))
        end
        for m = 1:getntau(pga)
            err = err + _blkdist(_lmix_at(pga, tstp, m), _lmix_at(pgb, tstp, m))
        end
        for i = 1:tstp
            err = err + _blkdist(_less_at(pga, tstp, i), _less_at(pgb, tstp, i))
        end
    end
    #
    return err
end

"""
    distance(A::FFTArray{T}, B::FFTArray{T}, dom::FFTDomain)

Euclidean (Frobenius) distance between the data of two FFTArray structs
in the domain `dom`.

*References* :

Please see [`NESSi`] `ness2_utils_impl.hpp`, `distance_norm2`.
"""
function distance(A::FFTArray{T}, B::FFTArray{T}, dom::FFTDomain) where {T}
    size(A.time) == size(B.time) ||
        throw(DimensionMismatch("A and B must have the same size"))
    X = _fft_data(A, dom)
    Y = _fft_data(B, dom)
    s = 0.0
    for i in eachindex(X)
        s += abs2(X[i] - Y[i])
    end
    return sqrt(s)
end

"""
    distance(A::HermMatrixNESS{T}, B::HermMatrixNESS{T}, dom::FFTDomain)

Euclidean distance between two HermMatrixNESS structs in the domain
`dom`, i.e., ``\\sqrt{d_R^2 + d_<^2}``.

*References* :

Please see [`NESSi`] `ness2_utils_impl.hpp`, `distance_norm2`.
"""
function distance(A::HermMatrixNESS{T}, B::HermMatrixNESS{T}, dom::FFTDomain) where {T}
    x = distance(A.ret, B.ret, dom)
    y = distance(A.les, B.les, dom)
    return sqrt(x * x + y * y)
end

"""
    distance(x::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ}, y::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ})

Distance between two moving time steps: the sum of the Frobenius norms
of the differences of the blocks, first of the retarded and then of the
lesser component, `deltj = 0, ⋯, tc`. It has the same definition as the
distance of the full contour functions.

*References* :

Please see [`NESSi`] `cntr_utilities_impl.hpp`, `distance_norm2` for
`herm_matrix_timestep_moving_view`.
"""
function distance(x::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ}, y::Union{𝒻ᵐᵒᵛ,𝒻ᵛⁱᵉʷᵐᵒᵛ})
    _check_mrow(x, y)
    err = 0.0
    for c in (:ret, :less)
        X = _mrows(x, c)
        Y = _mrows(y, c)
        for j = 1:size(X, 3)
            err += _blkdist(view(X, :, :, j), view(Y, :, :, j))
        end
    end
    return err
end

"""
    distance(G₁::ℱᵐᵒᵛ, G₂::ℱᵐᵒᵛ, tstp::I64)

Distance between the rows of the absolute KB time step `tstp` of two
moving Green's functions, see `distance(x::𝒻ᵐᵒᵛ, y::𝒻ᵐᵒᵛ)`.
"""
function distance(G₁::ℱᵐᵒᵛ, G₂::ℱᵐᵒᵛ, tstp::I64)
    _check_initialized(G₁)
    _check_initialized(G₂)
    d₁ = G₁.tstp - tstp
    d₂ = G₂.tstp - tstp
    (0 ≤ d₁ ≤ G₁.tc && 0 ≤ d₂ ≤ G₂.tc) || throw(BoundsError(G₁, tstp))
    return distance(timestep_view(G₁, d₁), timestep_view(G₂, d₂))
end

"""
    _blkdist(a::AbstractMatrix, b::AbstractMatrix)

Return the Frobenius norm of `a - b`. It is a private function.
"""
function _blkdist(a::AbstractMatrix, b::AbstractMatrix)
    @assert size(a) == size(b)
    s = 0.0
    for j in axes(a, 2)
        for i in axes(a, 1)
            s = s + abs2(a[i,j] - b[i,j])
        end
    end
    return sqrt(s)
end
