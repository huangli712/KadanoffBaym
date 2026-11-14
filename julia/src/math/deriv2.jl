#
# Project : Lavender
# Source  : deriv2.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

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

"""
    deriv2_lmix!(n::I64, dA::ℱ{T}, A::ℱ{T}, I::Integrator, beta::F64)

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
function deriv2_lmix!(n::I64, dA::ℱ{T}, A::ℱ{T}, I::Integrator, beta::F64) where {T}
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
        deriv2_lmix!(tstp, dA, A, I, beta)
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

