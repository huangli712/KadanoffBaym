#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Convolution* : ``G^{R}`` *Component*

*Remarks* :

The evaluation of ``C^{R}(t,t')`` at given time slice ``n`` is implemented
as follows:

```math
\begin{equation}
C^{R}(nh,mh) = C^{R}_1[A,f,B](n,m),
\end{equation}
```

where ``m = 0,\ \cdots,\ n`` and ``h`` means the interval in the real
time axis (``\equiv \delta t``).

```math
\begin{equation}
C^{R}_1[A,f,B](n,m) = \int^{nh}_{mh} d\bar{t} \
    A^{R}(nh,\bar{t}) f(\bar{t}) B^{R}(\bar{t},mh).
\end{equation}
```

Actually, we implement the following equations:

```math
\begin{equation}
C^{R}_1[A,f,B](n,m) = h \sum^{n}_{j = m}
    w^{(k)}_{n-m,j-m} A^{R}_{n,j} f_j B^{R}_{j,m},
    \quad n > k,\ n - m > k.
\end{equation}
```

```math
\begin{equation}
C^{R}_1[A,f,B](n,m) = h \sum^{k}_{j = 0}
    w^{(k)}_{n-m,j} A^{R}_{n,n-j} f_{n-j} \tilde{B}^{R}_{n-j,m},
    \quad n > k,\ n - m \le k.
\end{equation}
```

```math
\begin{equation}
C^{R}_1[A,f,B](n,m) = h \sum^{k}_{j = 0}
    I^{(k)}_{m,n;j} \tilde{A}^{R}_{n,j} f_j \tilde{B}^{R}_{j,m},
    \quad n \le k.
\end{equation}
```

As mentioned in [`NESSI`] Session `9`, the ``\tilde{B}^{R}_{j,m}`` in the
above equations indicates that ``B^{R}_{j,m}`` is also evaluated outside
the domain ``j \ge m``, and thus needs to be reconstructed from
``B^{\ddagger}``, i.e.,

```math
\begin{equation}
\tilde{B}^{R}_{j,m} = B^{R}_{j,m} = -(B^{\ddagger})^{R}_{j,m}.
\end{equation}
```

Analogous definitions also hold for ``\tilde{A}^{R}_{n,j}`` and
``\tilde{B}^{R}_{n-j,m}`` that appear in the above equations.

*References* :

Please see [`NESSi`] Sections `9` and `11` for more details.
=#

"""
    conv_ret(
        n::I64,
        C::Gʳᵉᵗ{T},
        A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
        B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
        I::Integrator,
        h::F64
    ) where {T}

Try to calculate the retarded component (Cᴿ) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B). Actually, it implements `Cᴿ(t,t') = Aᴿ(t,t'') ∗ Bᴿ(t'',t')`
at time step `t = nh` for all `t'` where `t' ≤ t`.

### Arguments
* n -> Index of given time step.
* A -> Retarded component of contour-ordered Green's function, Aᴿ(t,t').
* Acc -> Hermitian conjugate partner of A.
* B -> Retarded component of contour-ordered Green's function, Bᴿ(t,t').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* h -> Time step interval.

### Returns
* C -> Retarded component of contour-ordered Green's function, Cᴿ(t,t').

See also: [`convolution_time_step`](@ref).
"""
function conv_ret(
    n::I64,
    C::Gʳᵉᵗ{T},
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
    I::Integrator,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(A)

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert h > 0.0

    # Evaluate the retarded convolution at a given time step
    #
    # See [NESSi] Eq. (110a) - (110c)
    ws = _ConvWorkspace(A.data, B.data)
    _conv_ret!(n, C, A, Acc, nothing, B, Bcc, I, h, ws)
end


"""
    conv_ret(
        n::I64,
        C::Gʳᵉᵗ{T},
        A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
        fₜ::VecArray{T},
        B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
        I::Integrator,
        h::F64
    ) where {T}

Try to calculate the retarded component (Cᴿ) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B) and a time-dependent function fₜ. Actually, it just implements
`Cᴿ(t,t') = Aᴿ(t,t'') ∗ f(t'') ∗ Bᴿ(t'',t')` at time step `t = nh` for
all `t'` where `t' ≤ t`.

### Arguments
* n -> Index of given time step.
* A -> Retarded component of contour-ordered Green's function, Aᴿ(t,t').
* Acc -> Hermitian conjugate partner of A.
* fₜ -> Values of time-dependent function, f(t'').
* B -> Retarded component of contour-ordered Green's function, Bᴿ(t,t').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* h -> Time step interval.

### Returns
* C -> Retarded component of contour-ordered Green's function, Cᴿ(t,t').

See also: [`convolution_time_step`](@ref).
"""
function conv_ret(
    n::I64,
    C::Gʳᵉᵗ{T},
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    fₜ::VecArray{T},
    B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
    I::Integrator,
    h::F64
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
    @assert h > 0.0

    # Evaluate the retarded convolution at a given time step
    #
    # See [NESSi] Eq. (110a) - (110c)
    ws = _ConvWorkspace(A.data, B.data)
    _conv_ret!(n, C, A, Acc, fₜ, B, Bcc, I, h, ws)
end


"""
    _conv_ret_kernel!(tgt, n, A, Acc, f, B, Bcc, I, s, ws)

Accumulate `s * [Aᴿ ∗ f ∗ Bᴿ](tₙ,tₘ)` into `tgt[m]` for `m = 1, ⋯, n`.
See [NESSi] Eq. (110). It is a private function.
"""
function _conv_ret_kernel!(
    tgt, n::I64,
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    f,
    B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
    I::Integrator,
    s::Number,
    ws::_ConvWorkspace{T};
    ms::UnitRange{I64} = 1:n
) where {T}
    k = I.k
    lo, hi = first(ms), last(ms)

    if n ≥ k + 1
        # See [NESSi] Eq. (110a)
        for j = lo:n
            a = _af!(ws, _block(A, n, j), f, j)
            for m = lo:min(j, hi)
                w = I.GIW[n-m,n-j]
                mul!(tgt[m], a, _block(B, j, m), s * w, true)
            end
        end
        #
        # See [NESSi] Eq. (110b)
        for j = 1:k
            max(n - j + 1, lo) ≤ hi || continue
            a = _af!(ws, _block(A, n, n - j), f, n - j)
            for m = max(n-j+1, lo):hi
                w = I.GIW[n-m,j]
                mul!(tgt[m], a, adjoint(_block(Bcc, m, n - j)), -s * w, true)
            end
        end
    else
        # See [NESSi] Eq. (110c)
        for j = 1:k+1
            if n ≥ j
                _conv_ret_start!(tgt, n, j, _af!(ws, _block(A, n, j), f, j),
                                 s, B, Bcc, I, ms)
            else
                _conv_ret_start!(tgt, n, j, _af!(ws, adjoint(_block(Acc, j, n)), f, j),
                                 -s, B, Bcc, I, ms)
            end
        end
    end
end


"""
    _conv_ret_start!(tgt, n, j, a, s, B, Bcc, I)

Accumulate the `j`-th term of [NESSi] Eq. (110c). It is a private function.
"""
function _conv_ret_start!(tgt, n::I64, j::I64, a, s::Number,
                          B::Gʳᵉᵗ, Bcc::Gʳᵉᵗ, I::Integrator,
                          ms::UnitRange{I64} = 1:n)
    for m in ms
        w = I.XIW[m-1,n-1,j-1]
        if j ≥ m
            mul!(tgt[m], a, _block(B, j, m), s * w, true)
        else
            mul!(tgt[m], a, adjoint(_block(Bcc, m, j)), -s * w, true)
        end
    end
end


"""
    _conv_ret!(n, C, A, Acc, f, B, Bcc, I, h, ws)

Evaluate `Cᴿ(tₙ,tₘ) = [Aᴿ ∗ f ∗ Bᴿ](tₙ,tₘ)` for `m = 1, ⋯, n`. The old
values are overwritten. It is a private function.
"""
function _conv_ret!(
    n::I64,
    C::Gʳᵉᵗ{T},
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    f,
    B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
    I::Integrator,
    h::F64,
    ws::_ConvWorkspace{T}
) where {T}
    tgt = _RetRow(C, n)
    for m = 1:n
        fill!(tgt[m], zero(T))
    end
    _conv_ret_kernel!(tgt, n, A, Acc, f, B, Bcc, I, h, ws)
end


