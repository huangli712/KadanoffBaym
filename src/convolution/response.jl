#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks* : *Response Convolution*

The response of an observable to a perturbation `f(t)` reads

```math
\begin{equation}
c(t) = \int_{\mathcal{C}} d\bar{t}~W(t,\bar{t}) f(\bar{t}),
\end{equation}
```

where `W` is a hermitian response function. Only the selected matrix
elements ``W_{a_1 a_2}`` and ``f_{b_1 b_2}`` enter the integral, so that
the result is a scalar. At the real time axis (time step ``n``), it is
evaluated by

```math
\begin{equation}
c_n = h \sum_{j} \Omega^{(k)}_{n-1,j-1} W^{R}(t_n,t_j) f(t_j)
    - i \delta\tau \sum_{m} \Omega^{(k)}_{N_{\tau},m-1}
      W^{\rceil}(t_n,\tau_m) f(0^-).
\end{equation}
```

At the Matsubara axis, it is

```math
\begin{equation}
c_0 = \delta\tau \sum_{m} \Omega^{(k)}_{N_{\tau},m-1} W^{M}(\tau_m) f(0^-).
\end{equation}
```

Here ``\Omega^{(k)}`` are the Gregory weights, and ``N_{\tau} + 1`` is the
number of imaginary time points. If ``n - 1 < k``, the Gregory weights
need the values at the time steps ``1, \cdots, k + 1``. The missing
elements ``W^{R}(t_n,t_j)`` (``j > n``) are reconstructed by the hermitian
relation ``W^{R}(t_n,t_j) = -[W^{R}(t_j,t_n)]^{\dagger}``. Thus, the `𝒻`
version is restricted to `n = 0` or `n - 1 ≥ k`.

*References* :

Please see [`NESSi`] `cntr_response_convolution_impl.hpp`.
=#

"""
    _response_integrate(n₀::I64, I::Integrator, g::Function)

Evaluate ``\\sum_i \\Omega^{(k)}_{n₀,i} g(i)``, i.e., the integral of `g`
from `0` to `n₀` in units of the grid spacing. If `n₀ ≤ k`, the start-up
weights for `i = 0, ⋯, k` are used. It is a private function.
"""
function _response_integrate(n₀::I64, I::Integrator, g::Function)
    res = zero(C64)
    if n₀ > 0
        n₁ = max(n₀, I.k)
        for i = 0:n₁
            res = res + I.GIW[n₀,i] * g(i)
        end
    end
    return res
end


"""
    _check_response(W, a1, a2, f, b1, b2, I)

Sanity check for the response convolution. It is a private function.
"""
function _check_response(W, a1::I64, a2::I64, f::Cf, b1::I64, b2::I64, I::Integrator)
    @assert equaldims(W)
    @assert all(x -> 1 ≤ x ≤ getdims(W)[1], (a1, a2))
    @assert all(x -> 1 ≤ x ≤ getdims(f)[1], (b1, b2))
    @assert getdims(f)[1] == getdims(f)[2]
    @assert I.k ≤ getntau(W) - 1
end


"""
    response_convolution(
        tstp::I64,
        W::ℱ{T},
        a1::I64, a2::I64,
        f::Cf{T},
        b1::I64, b2::I64,
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Calculate the response `c(t) = ∫ W(t,t') f(t')` at time step `tstp` for
the matrix elements `W[a1,a2]` and `f[b1,b2]`. If `tstp = 0`, the
Matsubara branch is evaluated. `W` is assumed to be hermitian.

### Arguments
* tstp -> Index of time step.
* W    -> Response function, a contour-ordered Green's function.
* a1   -> Row index of `W`.
* a2   -> Column index of `W`.
* f    -> Contour function.
* b1   -> Row index of `f`.
* b2   -> Column index of `f`.
* I    -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* h    -> Time step interval.

### Returns
* c -> Response, a C64 number.
"""
function response_convolution(
    tstp::I64,
    W::ℱ{T},
    a1::I64, a2::I64,
    f::Cf{T},
    b1::I64, b2::I64,
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    # Sanity check
    _check_response(W, a1, a2, f, b1, b2, I)
    @assert 0 ≤ tstp ≤ getntime(W)
    @assert f.ntime ≥ tstp
    #
    # The start-up weights need the time steps 1, ⋯, k + 1
    if 1 ≤ tstp ≤ I.k
        @assert getntime(W) ≥ I.k + 1
        @assert f.ntime ≥ I.k + 1
    end
    @assert beta > 0.0
    @assert h > 0.0

    ntau = getntau(W)
    δτ = beta / (ntau - 1)
    f₀ = f[0][b1,b2]

    if tstp == 0
        res = _response_integrate(ntau - 1, I, i -> W.mat[i+1][a1,a2])
        return δτ * res * f₀
    end

    # Left-mixing part
    res = _response_integrate(ntau - 1, I, i -> W.lmix[tstp,i+1][a1,a2])
    c = δτ * res * (-im * f₀)

    # Retarded part. For tstp - 1 < k, the future points are reconstructed
    # by the hermitian relation.
    g = function (i)
        j = i + 1
        if j ≤ tstp
            w = W.ret[tstp,j][a1,a2]
        else
            @assert j ≤ getntime(W)
            w = -conj(W.ret[j,tstp][a2,a1])
        end
        return w * f[j][b1,b2]
    end
    c = c + h * _response_integrate(tstp - 1, I, g)

    return c
end


"""
    response_convolution(
        tstp::I64,
        W::𝒻{T},
        a1::I64, a2::I64,
        f::Cf{T},
        b1::I64, b2::I64,
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Calculate the response `c(t) = ∫ W(t,t') f(t')` at time step `tstp` for
the matrix elements `W[a1,a2]` and `f[b1,b2]`, where `W` is given at time
step `tstp` only. Thus, only `tstp = 0` or `tstp - 1 ≥ k` is allowed.

### Arguments
See above.

### Returns
* c -> Response, a C64 number.
"""
function response_convolution(
    tstp::I64,
    W::𝒻{T},
    a1::I64, a2::I64,
    f::Cf{T},
    b1::I64, b2::I64,
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    # Sanity check
    _check_response(W, a1, a2, f, b1, b2, I)
    _check_tstp(W, tstp)
    @assert tstp == 0 || tstp - 1 ≥ I.k
    @assert f.ntime ≥ tstp
    @assert beta > 0.0
    @assert h > 0.0

    ntau = getntau(W)
    δτ = beta / (ntau - 1)
    f₀ = f[0][b1,b2]

    if tstp == 0
        res = _response_integrate(ntau - 1, I, i -> W.mat.data[i+1][a1,a2])
        return δτ * res * f₀
    end

    res = _response_integrate(ntau - 1, I, i -> W.lmix.data[i+1][a1,a2])
    c = δτ * res * (-im * f₀)
    g = i -> W.ret.data[i+1][a1,a2] * f[i+1][b1,b2]
    c = c + h * _response_integrate(tstp - 1, I, g)

    return c
end


"""
    response_convolution(
        tstp::I64,
        W::Union{ℱ{T},𝒻{T}},
        a1::I64, a2::I64,
        f::Cf{T},
        b1::I64, b2::I64,
        order::I64,
        beta::F64,
        h::F64
    ) where {T}

Calculate the response `c(t) = ∫ W(t,t') f(t')` at time step `tstp`.
Here `order` is the integration order (1 ≤ order ≤ 5).
"""
function response_convolution(
    tstp::I64,
    W::Union{ℱ{T},𝒻{T}},
    a1::I64, a2::I64,
    f::Cf{T},
    b1::I64, b2::I64,
    order::I64,
    beta::F64,
    h::F64
) where {T}
    @assert 1 ≤ order ≤ 5
    response_convolution(tstp, W, a1, a2, f, b1, b2, Integrator(order), beta, h)
end

