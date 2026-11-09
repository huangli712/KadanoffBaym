#
# Project : Lavender
# Source  : differentiation.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/09
#

#=
*Remarks* : *Differentiation of Contour-Ordered Green's Functions*

The left and right derivatives of a two-time contour function are
defined as

```math
\begin{equation}
[\partial_1 A](t,t') = +i \partial_t A(t,t'),
\end{equation}
```

```math
\begin{equation}
[\partial_2 A](t,t') = -i \partial_{t'} A(t,t').
\end{equation}
```

For the Matsubara component, which is translationally invariant, the
two derivatives are identical,

```math
\begin{equation}
[\partial_1 A]^{M}(\tau) = [\partial_2 A]^{M}(\tau) = -\partial_\tau A^{M}(\tau).
\end{equation}
```

At the real time axis, the derivative at the time step ``n`` is
evaluated by the ``k``-th order polynomial differentiation weights
``D^{(k)}_{nl}`` if ``n \le k`` (the values at ``0, \cdots, k`` are
needed), and by the backward differentiation formula of order ``k + 1``
otherwise. For the mixing component, the derivative with respect to the
imaginary time argument is evaluated in the same way, and it carries no
factor ``i``.

The elements outside the stored triangles of the `ret` and `less`
components are reconstructed from `Acc` via the hermitian conjugate
relations, ``A^{R}(t,t') = -[A_{cc}^{R}(t',t)]^{\dagger}`` and
``A^{<}(t,t') = -[A_{cc}^{<}(t',t)]^{\dagger}``. If `A` is hermitian,
simply provide `Acc = A`.

*References* :

Please see [`NESSi`] `cntr_differentiation_impl.hpp`.
=#

#=
### *Public API* : *Matsubara Component*
=#

"""
    deriv1_matsubara!(dA::ℱ{T}, A::ℱ{T}, I::Integrator, beta::F64)

Calculate the derivative ``-\\partial_\\tau A^{M}(\\tau)`` and store it in
the Matsubara component of `dA`. Forward (backward) differentiation is
used near ``\\tau = 0`` (``\\tau = \\beta``), and backward differentiation
is used for the remaining points.

### Arguments
* dA   -> Result, a contour-ordered Green's function.
* A    -> A contour-ordered Green's function.
* I    -> An Integrator struct.
* beta -> Inverse temperature.

### Returns
N/A
"""
function deriv1_matsubara!(dA::ℱ{T}, A::ℱ{T}, I::Integrator, beta::F64) where {T}
    # Get key parameters
    k = I.k
    k₁ = k + 1
    ntau = getntau(A)
    nN = ntau - 1 # Index of the last imaginary time point, starts from 0.

    # Sanity check
    #
    # The largest index of imaginary time point accessed is 2k+1.
    @assert dA !== A
    @assert beta > 0.0
    @assert getntau(dA) == ntau
    @assert getdims(dA) == getdims(A)
    @assert ntau ≥ 2 * k + 2

    # Imaginary time step
    hv = beta / nN

    c = zeros(T, getdims(A))
    #
    # Forward differentiation near τ = 0, and backward differentiation
    # near τ = β.
    for t = 0:k
        fill!(c, zero(T))
        for l = 0:k₁
            c .= c .- I.BDW[l] .* _block(A.mat, t + l + 1)
        end
        c .= c ./ (-hv)
        copyto!(_block(dA.mat, t + 1), c)
        #
        fill!(c, zero(T))
        for l = 0:k₁
            c .= c .+ I.BDW[l] .* _block(A.mat, nN - t - l + 1)
        end
        c .= c ./ (-hv)
        copyto!(_block(dA.mat, nN - t + 1), c)
    end
    #
    # Backward differentiation for the remaining points
    for t = k₁:nN-k₁
        fill!(c, zero(T))
        for l = 0:k₁
            c .= c .+ I.BDW[l] .* _block(A.mat, t - l + 1)
        end
        c .= c ./ (-hv)
        copyto!(_block(dA.mat, t + 1), c)
    end
end

#=
### *Public API* : *Retarded And Lesser Components*
=#

"""
    deriv1_element!(i::I64, j::I64, dA::ℱ{T}, A::ℱ{T}, Acc::ℱ{T},
                    I::Integrator, h::F64)

Calculate the left derivative ``i \\partial_t A(t_i,t_j)``. If `j ≤ i`,
the `ret` component of `dA` at `(i,j)` is set. If `j ≥ i`, the `less`
component of `dA` at `(i,j)` is set. If `i ≤ k + 1`, `A` must be known at
the time steps `1, ..., k + 1`.

### Arguments
* i   -> Index of the first real time argument.
* j   -> Index of the second real time argument.
* dA  -> Result, a contour-ordered Green's function.
* A   -> A contour-ordered Green's function.
* Acc -> Hermitian conjugate partner of `A`.
* I   -> An Integrator struct.
* h   -> Time step at real time axis.

### Returns
N/A
"""
function deriv1_element!(i::I64, j::I64, dA::ℱ{T}, A::ℱ{T}, Acc::ℱ{T},
                         I::Integrator, h::F64) where {T}
    # Get key parameters
    k = I.k
    k₁ = k + 1
    ntime = getntime(A)

    # Sanity check
    @assert dA !== A
    @assert dA !== Acc
    @assert h > 0.0
    @assert 1 ≤ i ≤ ntime
    @assert 1 ≤ j ≤ ntime
    @assert ntime ≥ k₁
    @assert equaldims(A)
    @assert getdims(A) == getdims(Acc) == getdims(dA)

    c = zeros(T, getdims(A))

    # Retarded component
    if j ≤ i
        fill!(c, zero(T))
        if i - 1 < k₁
            for l = 0:k
                w = I.PDW[i - 1, l]
                if l + 1 < j
                    c .= c .- w .* adjoint(_block(Acc.ret, j, l + 1))
                else
                    c .= c .+ w .* _block(A.ret, l + 1, j)
                end
            end
        else
            for l = 0:k₁
                w = I.BDW[l]
                if i - l < j
                    c .= c .- w .* adjoint(_block(Acc.ret, j, i - l))
                else
                    c .= c .+ w .* _block(A.ret, i - l, j)
                end
            end
        end
        _block(dA.ret, i, j) .= c .* (im / h)
    end

    # Lesser component
    if j ≥ i
        fill!(c, zero(T))
        if i - 1 < k₁
            for l = 0:k
                w = I.PDW[i - 1, l]
                if l + 1 > j
                    c .= c .- w .* adjoint(_block(Acc.less, j, l + 1))
                else
                    c .= c .+ w .* _block(A.less, l + 1, j)
                end
            end
        else
            # Here i - l ≤ j always holds.
            for l = 0:k₁
                c .= c .+ I.BDW[l] .* _block(A.less, i - l, j)
            end
        end
        _block(dA.less, i, j) .= c .* (im / h)
    end
end

"""
    deriv2_element!(i::I64, j::I64, dA::ℱ{T}, A::ℱ{T}, Acc::ℱ{T},
                    I::Integrator, h::F64)

Calculate the right derivative ``-i \\partial_{t'} A(t_i,t_j)``. If
`j ≤ i`, the `ret` component of `dA` at `(i,j)` is set. If `j ≥ i`, the
`less` component of `dA` at `(i,j)` is set. If `j ≤ k + 1`, `A` must be
known at the time steps `1, ..., k + 1`.

### Arguments
* i   -> Index of the first real time argument.
* j   -> Index of the second real time argument.
* dA  -> Result, a contour-ordered Green's function.
* A   -> A contour-ordered Green's function.
* Acc -> Hermitian conjugate partner of `A`.
* I   -> An Integrator struct.
* h   -> Time step at real time axis.

### Returns
N/A
"""
function deriv2_element!(i::I64, j::I64, dA::ℱ{T}, A::ℱ{T}, Acc::ℱ{T},
                         I::Integrator, h::F64) where {T}
    # Get key parameters
    k = I.k
    k₁ = k + 1
    ntime = getntime(A)

    # Sanity check
    @assert dA !== A
    @assert dA !== Acc
    @assert h > 0.0
    @assert 1 ≤ i ≤ ntime
    @assert 1 ≤ j ≤ ntime
    @assert ntime ≥ k₁
    @assert equaldims(A)
    @assert getdims(A) == getdims(Acc) == getdims(dA)

    c = zeros(T, getdims(A))

    # Retarded component
    if j ≤ i
        fill!(c, zero(T))
        if j - 1 < k₁
            for l = 0:k
                w = I.PDW[j - 1, l]
                if i < l + 1
                    c .= c .- w .* adjoint(_block(Acc.ret, l + 1, i))
                else
                    c .= c .+ w .* _block(A.ret, i, l + 1)
                end
            end
        else
            # Here i ≥ j - l always holds.
            for l = 0:k₁
                c .= c .+ I.BDW[l] .* _block(A.ret, i, j - l)
            end
        end
        _block(dA.ret, i, j) .= c .* (-im / h)
    end

    # Lesser component
    if j ≥ i
        fill!(c, zero(T))
        if j - 1 < k₁
            for l = 0:k
                w = I.PDW[j - 1, l]
                if i > l + 1
                    c .= c .- w .* adjoint(_block(Acc.less, l + 1, i))
                else
                    c .= c .+ w .* _block(A.less, i, l + 1)
                end
            end
        else
            for l = 0:k₁
                w = I.BDW[l]
                if i > j - l
                    c .= c .- w .* adjoint(_block(Acc.less, j - l, i))
                else
                    c .= c .+ w .* _block(A.less, i, j - l)
                end
            end
        end
        _block(dA.less, i, j) .= c .* (-im / h)
    end
end

#=
### *Public API* : *Left-Mixing Component*
=#

"""
    deriv1_tv!(n::I64, dA::ℱ{T}, A::ℱ{T}, I::Integrator, h::F64)

Calculate the derivative ``i \\partial_t A^{\\rceil}(t_n,\\tau)`` and store
it in the `lmix` component of `dA` at time step `n`.

### Arguments
* n  -> Index of time step.
* dA -> Result, a contour-ordered Green's function.
* A  -> A contour-ordered Green's function.
* I  -> An Integrator struct.
* h  -> Time step at real time axis.

### Returns
N/A
"""
function deriv1_tv!(n::I64, dA::ℱ{T}, A::ℱ{T}, I::Integrator, h::F64) where {T}
    # Get key parameters
    k = I.k
    k₁ = k + 1
    ntau = getntau(A)

    # Sanity check
    @assert dA !== A
    @assert h > 0.0
    @assert 1 ≤ n ≤ getntime(A)
    @assert getntime(A) ≥ k₁
    @assert getdims(A) == getdims(dA)

    c = zeros(T, getdims(A))
    #
    for m = 1:ntau
        fill!(c, zero(T))
        if n - 1 < k₁
            for l = 0:k
                c .= c .+ I.PDW[n - 1, l] .* _block(A.lmix, l + 1, m)
            end
        else
            for l = 0:k₁
                c .= c .+ I.BDW[l] .* _block(A.lmix, n - l, m)
            end
        end
        _block(dA.lmix, n, m) .= c .* (im / h)
    end
end

"""
    deriv2_tv!(n::I64, dA::ℱ{T}, A::ℱ{T}, I::Integrator, beta::F64)

Calculate the derivative ``\\partial_\\tau A^{\\rceil}(t_n,\\tau)`` and store
it in the `lmix` component of `dA` at time step `n`. Note that there is
no factor ``i``.

### Arguments
* n    -> Index of time step.
* dA   -> Result, a contour-ordered Green's function.
* A    -> A contour-ordered Green's function.
* I    -> An Integrator struct.
* beta -> Inverse temperature.

### Returns
N/A
"""
function deriv2_tv!(n::I64, dA::ℱ{T}, A::ℱ{T}, I::Integrator, beta::F64) where {T}
    # Get key parameters
    k = I.k
    k₁ = k + 1
    ntau = getntau(A)

    # Sanity check
    @assert dA !== A
    @assert beta > 0.0
    @assert 1 ≤ n ≤ getntime(A)
    @assert ntau ≥ k₁
    @assert getdims(A) == getdims(dA)

    # Imaginary time step
    hv = beta / (ntau - 1)

    c = zeros(T, getdims(A))
    #
    for m = 1:ntau
        fill!(c, zero(T))
        if m - 1 < k₁
            for l = 0:k
                c .= c .+ I.PDW[m - 1, l] .* _block(A.lmix, n, l + 1)
            end
        else
            for l = 0:k₁
                c .= c .+ I.BDW[l] .* _block(A.lmix, n, m - l)
            end
        end
        _block(dA.lmix, n, m) .= c ./ hv
    end
end

#=
### *Public API* : *Time Step*
=#

"""
    deriv1_timestep!(tstp::I64, dA::ℱ{T}, A::ℱ{T}, Acc::ℱ{T},
                     I::Integrator, beta::F64, h::F64)

Calculate the left derivative of `A` at time step `tstp`. If `tstp = 0`,
the Matsubara component is calculated. Otherwise, the `ret` component at
`(tstp, j ≤ tstp)`, the `less` component at `(tstp, j ≥ tstp)`, and the
`lmix` component at `tstp` are calculated.

### Arguments
* tstp -> Index of time step.
* dA   -> Result, a contour-ordered Green's function.
* A    -> A contour-ordered Green's function.
* Acc  -> Hermitian conjugate partner of `A`.
* I    -> An Integrator struct.
* beta -> Inverse temperature.
* h    -> Time step at real time axis.

### Returns
N/A
"""
function deriv1_timestep!(tstp::I64, dA::ℱ{T}, A::ℱ{T}, Acc::ℱ{T},
                          I::Integrator, beta::F64, h::F64) where {T}
    @assert 0 ≤ tstp ≤ getntime(A)
    if tstp == 0
        deriv1_matsubara!(dA, A, I, beta)
    else
        for j = 1:getntime(A)
            deriv1_element!(tstp, j, dA, A, Acc, I, h)
        end
        deriv1_tv!(tstp, dA, A, I, h)
    end
end

"""
    deriv1_timestep!(tstp::I64, dA::ℱ{T}, A::ℱ{T}, Acc::ℱ{T},
                     beta::F64, h::F64, order::I64)

Calculate the left derivative of `A` at time step `tstp`. Here `order`
is the integration order (1 ≤ order ≤ 5).
"""
function deriv1_timestep!(tstp::I64, dA::ℱ{T}, A::ℱ{T}, Acc::ℱ{T},
                          beta::F64, h::F64, order::I64) where {T}
    @assert 1 ≤ order ≤ 5
    deriv1_timestep!(tstp, dA, A, Acc, Integrator(order), beta, h)
end

"""
    deriv1_timestep!(tstp::I64, dA::ℱ{T}, A::ℱ{T},
                     I::Integrator, beta::F64, h::F64)

Calculate the left derivative of a hermitian `A` (`Acc = A`) at time step
`tstp`.
"""
function deriv1_timestep!(tstp::I64, dA::ℱ{T}, A::ℱ{T},
                          I::Integrator, beta::F64, h::F64) where {T}
    deriv1_timestep!(tstp, dA, A, A, I, beta, h)
end

"""
    deriv1_timestep!(tstp::I64, dA::ℱ{T}, A::ℱ{T},
                     beta::F64, h::F64, order::I64)

Calculate the left derivative of a hermitian `A` (`Acc = A`) at time step
`tstp`. Here `order` is the integration order (1 ≤ order ≤ 5).
"""
function deriv1_timestep!(tstp::I64, dA::ℱ{T}, A::ℱ{T},
                          beta::F64, h::F64, order::I64) where {T}
    deriv1_timestep!(tstp, dA, A, A, beta, h, order)
end

"""
    deriv2_timestep!(tstp::I64, dA::ℱ{T}, A::ℱ{T}, Acc::ℱ{T},
                     I::Integrator, beta::F64, h::F64)

Calculate the right derivative of `A` at time step `tstp`. If `tstp = 0`,
the Matsubara component is calculated (it is identical to the left
derivative). Otherwise, the `ret` component at `(tstp, j ≤ tstp)`, the
`less` component at `(tstp, j ≥ tstp)`, and the `lmix` component at
`tstp` are calculated.

### Arguments
* tstp -> Index of time step.
* dA   -> Result, a contour-ordered Green's function.
* A    -> A contour-ordered Green's function.
* Acc  -> Hermitian conjugate partner of `A`.
* I    -> An Integrator struct.
* beta -> Inverse temperature.
* h    -> Time step at real time axis.

### Returns
N/A
"""
function deriv2_timestep!(tstp::I64, dA::ℱ{T}, A::ℱ{T}, Acc::ℱ{T},
                          I::Integrator, beta::F64, h::F64) where {T}
    @assert 0 ≤ tstp ≤ getntime(A)
    if tstp == 0
        deriv1_matsubara!(dA, A, I, beta)
    else
        for j = 1:getntime(A)
            deriv2_element!(tstp, j, dA, A, Acc, I, h)
        end
        deriv2_tv!(tstp, dA, A, I, beta)
    end
end

"""
    deriv2_timestep!(tstp::I64, dA::ℱ{T}, A::ℱ{T}, Acc::ℱ{T},
                     beta::F64, h::F64, order::I64)

Calculate the right derivative of `A` at time step `tstp`. Here `order`
is the integration order (1 ≤ order ≤ 5).
"""
function deriv2_timestep!(tstp::I64, dA::ℱ{T}, A::ℱ{T}, Acc::ℱ{T},
                          beta::F64, h::F64, order::I64) where {T}
    @assert 1 ≤ order ≤ 5
    deriv2_timestep!(tstp, dA, A, Acc, Integrator(order), beta, h)
end

"""
    deriv2_timestep!(tstp::I64, dA::ℱ{T}, A::ℱ{T},
                     I::Integrator, beta::F64, h::F64)

Calculate the right derivative of a hermitian `A` (`Acc = A`) at time
step `tstp`.
"""
function deriv2_timestep!(tstp::I64, dA::ℱ{T}, A::ℱ{T},
                          I::Integrator, beta::F64, h::F64) where {T}
    deriv2_timestep!(tstp, dA, A, A, I, beta, h)
end

"""
    deriv2_timestep!(tstp::I64, dA::ℱ{T}, A::ℱ{T},
                     beta::F64, h::F64, order::I64)

Calculate the right derivative of a hermitian `A` (`Acc = A`) at time
step `tstp`. Here `order` is the integration order (1 ≤ order ≤ 5).
"""
function deriv2_timestep!(tstp::I64, dA::ℱ{T}, A::ℱ{T},
                          beta::F64, h::F64, order::I64) where {T}
    deriv2_timestep!(tstp, dA, A, A, beta, h, order)
end
