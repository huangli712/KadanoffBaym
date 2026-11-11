#
# Project : Lavender
# Source  : timestep.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Public Convolution API 1*
=#

"""
    convolution_time_step(
        n::I64,
        C::ℱ{T},
        A::ℱ{T}, Acc::ℱ{T},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

To calculate convolution of two contour-ordered Green's functions, i.e.,
`C(t,t') = A(t,t'') ∗ B(t'',t)` at a given time step `t = nh`.

### Arguments
* n -> Index of given time step.
* A -> Contour-ordered Green's function, A(t,t'').
* Acc -> Hermitian conjugate partner of A.
* B -> Contour-ordered Green's function, B(t'',t').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
The Matsubara component of C will not be modified in this function.

See also: [`conv_mat`](@ref).
"""
function convolution_time_step(
    n::I64,
    C::ℱ{T},
    A::ℱ{T}, Acc::ℱ{T},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64;
    threaded::Bool = false
) where {T}
    # Extract parameters
    ntime = getntime(A)

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert beta > 0.0
    @assert h > 0.0

    # Perform the convolution
    ws = _ConvWorkspace(A, B)
    _convolution_time_step!(n, C, A, Acc, nothing, nothing, B, Bcc, I, beta, h, ws; threaded = threaded)
end

"""
    convolution_time_step(
        n::I64,
        C::ℱ{T},
        A::ℱ{T}, Acc::ℱ{T},
        B::ℱ{T}, Bcc::ℱ{T},
        order::I64,
        beta::F64,
        h::F64
    ) where {T}

To calculate convolution of two contour-ordered Green's functions, i.e.,
`C(t,t') = A(t,t'') ∗ B(t'',t)` at a given time step `t = nh`.

### Arguments
* n -> Index of given time step.
* A -> Contour-ordered Green's function, A(t,t'').
* Acc -> Hermitian conjugate partner of A.
* B -> Contour-ordered Green's function, B(t'',t').
* Bcc -> Hermitian conjugate partner of B.
* order -> Order for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
The Matsubara component of C will not be modified in this function.

See also: [`conv_mat`](@ref).
"""
function convolution_time_step(
    n::I64,
    C::ℱ{T},
    A::ℱ{T}, Acc::ℱ{T},
    B::ℱ{T}, Bcc::ℱ{T},
    order::I64,
    beta::F64,
    h::F64;
    threaded::Bool = false
) where {T}
    # Sanity check
    @assert 10 ≥ order ≥ 1

    # Create numerical integrator
    I = Integrator(order)

    # Perrform time convolution
    convolution_time_step(n, C, A, Acc, B, Bcc, I, beta, h; threaded = threaded)
end

"""
    convolution_time_step(
        n::I64,
        C::ℱ{T}, A::ℱ{T}, B::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

To calculate convolution of two contour-ordered Green's functions, i.e.,
`C(t,t') = A(t,t'') ∗ B(t'',t)` at a given time step `t = nh`.

### Arguments
* n -> Index of given time step.
* A -> Contour-ordered Green's function, A(t,t'').
* B -> Contour-ordered Green's function, B(t'',t').
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
The Matsubara component of C will not be modified in this function.

See also: [`conv_mat`](@ref).
"""
function convolution_time_step(
    n::I64,
    C::ℱ{T}, A::ℱ{T}, B::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64;
    threaded::Bool = false
) where {T}
    # Perform time convolution
    convolution_time_step(n, C, A, A, B, B, I, beta, h; threaded = threaded)
end

"""
    convolution_time_step(
        n::I64,
        C::ℱ{T}, A::ℱ{T}, B::ℱ{T},
        order::I64,
        beta::F64,
        h::F64
    ) where {T}

To calculate convolution of two contour-ordered Green's functions, i.e.,
`C(t,t') = A(t,t'') ∗ B(t'',t)` at a given time step `t = nh`.

### Arguments
* n -> Index of given time step.
* A -> Contour-ordered Green's function, A(t,t'').
* B -> Contour-ordered Green's function, B(t'',t').
* order -> Order for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
The Matsubara component of C will not be modified in this function.

See also: [`conv_mat`](@ref).
"""
function convolution_time_step(
    n::I64,
    C::ℱ{T}, A::ℱ{T}, B::ℱ{T},
    order::I64,
    beta::F64,
    h::F64;
    threaded::Bool = false
) where {T}
    # Sanity check
    @assert 10 ≥ order ≥ 1

    # Create numerical integrator
    I = Integrator(order)

    # Perform time convolution
    convolution_time_step(n, C, A, A, B, B, I, beta, h; threaded = threaded)
end

#=
### *Public Convolution API 2*
=#

"""
    convolution_time_step(
        n::I64,
        C::ℱ{T},
        A::ℱ{T}, Acc::ℱ{T},
        fₜ::VecArray{T},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64
    )

Try to calculate convolution of two contour-ordered Green's functions
(A and B) and a time-dependent function fₜ at a given time step `t = nh`.
In other words, it implements `C(t,t') = A(t,t'') ∗ f(t'') ∗ B(t'',t)`.

### Arguments
* n -> Index of given time step.
* A -> Contour-ordered Green's function, A(t,t'').
* Acc -> Hermitian conjugate partner of A.
* fₜ -> Values of time-dependent function, f(t'').
* B -> Contour-ordered Green's function, B(t'',t').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
The Matsubara component of C will not be modified in this function.

See also: [`conv_mat`](@ref).
"""
function convolution_time_step(
    n::I64,
    C::ℱ{T},
    A::ℱ{T}, Acc::ℱ{T},
    fₜ::VecArray{T},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64;
    threaded::Bool = false
) where {T}
    # Extract parameters
    ntime = getntime(A)

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert length(fₜ) ≥ n
    @assert ntime ≥ n ≥ 1
    @assert beta > 0.0
    @assert h > 0.0

    # Perform the convolution. fₜ[end] is the value of f at the
    # Matsubara branch.
    ws = _ConvWorkspace(A, B)
    _convolution_time_step!(n, C, A, Acc, fₜ, fₜ[end], B, Bcc, I, beta, h, ws; threaded = threaded)
end

"""
    convolution_time_step(
        n::I64,
        C::ℱ{T},
        A::ℱ{T}, Acc::ℱ{T},
        fₜ::VecArray{T},
        B::ℱ{T}, Bcc::ℱ{T},
        order::I64,
        beta::F64,
        h::F64
    )

Try to calculate convolution of two contour-ordered Green's functions
(A and B) and a time-dependent function fₜ at a given time step `t = nh`.
In other words, it implements `C(t,t') = A(t,t'') ∗ f(t'') ∗ B(t'',t)`.

### Arguments
* n -> Index of given time step.
* A -> Contour-ordered Green's function, A(t,t'').
* Acc -> Hermitian conjugate partner of A.
* fₜ -> Values of time-dependent function, f(t'').
* B -> Contour-ordered Green's function, B(t'',t').
* Bcc -> Hermitian conjugate partner of B.
* order -> Order for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
The Matsubara component of C will not be modified in this function.

See also: [`conv_mat`](@ref).
"""
function convolution_time_step(
    n::I64,
    C::ℱ{T},
    A::ℱ{T}, Acc::ℱ{T},
    fₜ::VecArray{T},
    B::ℱ{T}, Bcc::ℱ{T},
    order::I64,
    beta::F64,
    h::F64;
    threaded::Bool = false
) where {T}
    # Sanity check
    @assert 10 ≥ order ≥ 1

    # Create numerical integrator
    I = Integrator(order)

    # Perrform time convolution
    convolution_time_step(n, C, A, Acc, fₜ, B, Bcc, I, beta, h; threaded = threaded)
end

"""
    convolution_time_step(
        n::I64,
        C::ℱ{T}, A::ℱ{T}, fₜ::VecArray{T}, B::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64
    )

Try to calculate convolution of two contour-ordered Green's functions
(A and B) and a time-dependent function fₜ at a given time step `t = nh`.
In other words, it implements `C(t,t') = A(t,t'') ∗ f(t'') ∗ B(t'',t)`.

### Arguments
* n -> Index of given time step.
* A -> Contour-ordered Green's function, A(t,t'').
* fₜ -> Values of time-dependent function, f(t'').
* B -> Contour-ordered Green's function, B(t'',t').
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
The Matsubara component of C will not be modified in this function.

See also: [`conv_mat`](@ref).
"""
function convolution_time_step(
    n::I64,
    C::ℱ{T}, A::ℱ{T}, fₜ::VecArray{T}, B::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64;
    threaded::Bool = false
) where {T}
    # Perform time convolution
    convolution_time_step(n, C, A, A, fₜ, B, B, I, beta, h; threaded = threaded)
end

"""
    convolution_time_step(
        n::I64,
        C::ℱ{T}, A::ℱ{T}, fₜ::VecArray{T}, B::ℱ{T},
        order::I64,
        beta::F64,
        h::F64
    ) where {T}

Try to calculate convolution of two contour-ordered Green's functions
(A and B) and a time-dependent function fₜ at a given time step `t = nh`.
In other words, it implements `C(t,t') = A(t,t'') ∗ f(t'') ∗ B(t'',t)`.

### Arguments
* n -> Index of given time step.
* A -> Contour-ordered Green's function, A(t,t'').
* fₜ -> Values of time-dependent function, f(t'').
* B -> Contour-ordered Green's function, B(t'',t').
* order -> Order for numerical integration.
* beta -> Inverse temperature, β.
* h -> Time step interval.

### Returns
* C -> Contour-ordered Green's function, C(t,t').

### Notes
The Matsubara component of C will not be modified in this function.

See also: [`conv_mat`](@ref).
"""
function convolution_time_step(
    n::I64,
    C::ℱ{T}, A::ℱ{T}, fₜ::VecArray{T}, B::ℱ{T},
    order::I64,
    beta::F64,
    h::F64;
    threaded::Bool = false
) where {T}
    # Sanity check
    @assert 10 ≥ order ≥ 1

    # Create numerical integrator
    I = Integrator(order)

    # Perform time convolution
    convolution_time_step(n, C, A, A, fₜ, B, B, I, beta, h; threaded = threaded)
end

"""
    _convolution_time_step!(n, C, A, Acc, fₜ, f₀, B, Bcc, I, beta, h, ws)

Evaluate the retarded, left-mixing, and lesser components of
`C = A ∗ f ∗ B` at time step `n`. If `fₜ` and `f₀` are `nothing`, then
`f = 1`. The retarded component is overwritten, while the left-mixing
and lesser components are accumulated, which is the same as the public
[`convolution_time_step`](@ref). It is a private function.
"""
function _convolution_time_step!(
    n::I64,
    C::ℱ{T},
    A::ℱ{T}, Acc::ℱ{T},
    fₜ, f₀,
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64,
    ws::_ConvWorkspace{T};
    threaded::Bool = false
) where {T}
    _check_noalias(C, A, Acc, B, Bcc, (:ret, :lmix, :less))
    sign = A.sign
    δτ = beta / (getntau(A) - 1)

    if _use_threads(threaded)
        _convolution_time_step_threaded!(n, C, A, Acc, fₜ, f₀, B, Bcc, I, h, δτ, ws)
        return
    end

    # For retarded component
    #
    # Cᴿ = Aᴿ ∗ fₜ ∗ Bᴿ
    _conv_ret!(n, C.ret, A.ret, Acc.ret, fₜ, B.ret, Bcc.ret, I, h, ws)

    # For left-mixing component
    #
    # C^⌉ = Aᴿ ∗ fₜ ∗ B^⌉ + A^⌉ ∗ f₀ ∗ Bᴹ
    row = _LmixRow(C.lmix, n)
    _conv_ret_lmix_kernel!(row, n, A.ret, Acc.ret, fₜ, B.lmix, I, h, ws)
    _conv_lmix_mat_kernel!(row, n, A.lmix, f₀, B.mat, I, sign, δτ, ws)

    # For lesser component
    #
    # C^< = Aᴿ ∗ fₜ ∗ B^< + A^< ∗ fₜ ∗ Bᴬ + A^⌉ ∗ f₀ ∗ B^⌈
    col = _LessCol(C.less, n)
    _conv_ret_less_kernel!(col, 1:n, n, A.ret, Acc.ret, fₜ, B.less, Bcc.less, I, h, ws)
    _conv_less_adv_kernel!(col, 1:n, n, A.less, Acc.less, fₜ, B.ret, Bcc.ret, I, h, ws)
    _conv_lmix_rmix_kernel!(col, 1:n, n, A.lmix, f₀, Bcc.lmix, I, sign, -im * δτ, ws)
end

"""
    _convolution_time_step_threaded!(n, C, A, Acc, fₜ, f₀, B, Bcc, I, h, δτ, ws)

Threaded version of the body of `_convolution_time_step!`. See *Remarks*
: *Threads*. It is a private function.
"""
function _convolution_time_step_threaded!(n::I64, C::ℱ{T}, A::ℱ{T}, Acc::ℱ{T},
                                          fₜ, f₀, B::ℱ{T}, Bcc::ℱ{T},
                                          I::Integrator, h::F64, δτ::F64,
                                          ws::_ConvWorkspace{T}) where {T}
    sign = A.sign

    # Retarded component, owned columns m
    tgt = _RetRow(C.ret, n)
    _tforeach(1:n, ws) do ms, wsₜ
        for m in ms
            fill!(tgt[m], zero(T))
        end
        _conv_ret_kernel!(tgt, n, A.ret, Acc.ret, fₜ, B.ret, Bcc.ret, I, h, wsₜ; ms = ms)
    end

    # Left-mixing component, owned imaginary time points m. The blocks
    # f₀ Bᴹ are evaluated once before the parallel region.
    row = _LmixRow(C.lmix, n)
    Bt = _fbmat!(ws, f₀, B.mat)
    _tforeach(1:getntau(A), ws) do ms, wsₜ
        _conv_ret_lmix_kernel!(row, n, A.ret, Acc.ret, fₜ, B.lmix, I, h, wsₜ; ms = ms)
        _conv_lmix_mat_loop!(row, n, A.lmix, Bt, I, sign, δτ, ms)
    end

    # Lesser component, owned first times j
    col = _LessCol(C.less, n)
    _tforeach(1:n, ws) do ms, wsₜ
        _conv_ret_less_kernel!(col, ms, n, A.ret, Acc.ret, fₜ, B.less, Bcc.less, I, h, wsₜ)
        _conv_less_adv_kernel!(col, ms, n, A.less, Acc.less, fₜ, B.ret, Bcc.ret, I, h, wsₜ)
        _conv_lmix_rmix_kernel!(col, ms, n, A.lmix, f₀, Bcc.lmix, I, sign, -im * δτ, wsₜ)
    end

    return nothing
end
