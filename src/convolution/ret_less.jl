#
# Project : Lavender
# Source  : ret_less.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Convolution* : ``G^{<}`` *Component*

*Remarks* :

The evaluation of ``C^{<}(t,t')`` at given time slice ``n`` is
implemented as follows:

```math
\begin{equation}
C^{<}(mh,nh) = C^{<}_{1}[A,f,B](m,n) +
               C^{<}_{2}[A,f,B](m,n) +
               C^{<}_{3}[A,f,B](m,n).
\end{equation}
```

where ``m = 0,\ \cdots,\ n`` and ``h`` means the interval in the real
time axis (``\equiv \delta t``).

```math
\begin{equation}
C^{<}_{1}[A,f,B](m,n) = \int^{mh}_{0} d\bar{t}~
    A^{R}(mh,\bar{t}) f(\bar{t}) B^{<}(\bar{t},nh).
\end{equation}
```

```math
\begin{equation}
C^{<}_{2}[A,f,B](m,n) = \int^{nh}_{0} d\bar{t}~
    A^{<}(mh,\bar{t}) f(\bar{t}) B^{A}(\bar{t},nh).
\end{equation}
```

```math
\begin{equation}
C^{<}_{3}[A,f,B](m,n) = -i \int^{\beta}_{0} d\tau~
    A^{\rceil}(mh,\tau) f(0^{-}) B^{\lceil}(\tau,nh).
\end{equation}
```

Actually, we implement the following equations:

```math
\begin{equation}
C^{<}_{1}[A,f,B](m,n) = h\sum^{m}_{j=0}~
    w^{(k)}_{m,j} A^{R}_{m,j} f_{j} B^{<}_{j,n}, \quad m > k.
\end{equation}
```

```math
\begin{equation}
C^{<}_{1}[A,f,B](m,n) = h\sum^{k}_{j=0}~
    w^{(k)}_{m,j} \tilde{A}^{R}_{m,j} f_{j} B^{<}_{j,n}, \quad m \le k.
\end{equation}
```

```math
\begin{equation}
C^{<}_{2}[A,f,B](m,n) = h\sum^{n}_{j=0}~
    w^{(k)}_{n,j} A^{<}_{m,j} f_{j} B^{A}_{j,n}, \quad n > k.
\end{equation}
```

```math
\begin{equation}
C^{<}_{2}[A,f,B](m,n) = h\sum^{k}_{j=0}~
    w^{(k)}_{n,j} A^{<}_{m,j} f_{j} \tilde{B}^{A}_{j,n}, \quad n \le k.
\end{equation}
```

```math
\begin{equation}
C^{<}_{3}[A,f,B](m,n) = -i h_{\tau} \sum^{N_{\tau}}_{j=0}~
    w^{(k)}_{N_{\tau},j} A^{\rceil}_{m,j} f_{-1} B^{\lceil}_{j,n}.
\end{equation}
```

Because the advanced and the right-mixing components of contour-ordered
Green's functions are not really stored in ℱ., these quantities must be
reconstructed from the hermitian conjugate. For examples:

```math
\begin{equation}
C^{\lceil}(\tau,t) =
    -\xi \left([C^{\ddagger}]^{\rceil}(t,\beta-\tau) \right)^{\dagger},
\end{equation}
```

```math
\begin{equation}
C^{R}(t,t') = \left([C^{\ddagger}]^{A}(t',t)\right)^{\dagger}.
\end{equation}
```

*References* :

Please see [`NESSi`] Sections `3.1`, `9` and `11` for more details.
=#

"""
    conv_ret_less(
        n::I64,
        C::Gˡᵉˢˢ{T},
        A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
        B::Gˡᵉˢˢ{T}, Bcc::Gˡᵉˢˢ{T},
        I::Integrator,
        h::F64
    ) where {T}

Try to calculate the lesser component (C^<) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B). Actually, it implements `C^<(t,t') = Aᴿ(t,t'') ∗ B^<(t'',t')`
at time step `t' = nh` for all `t` where `t ≤ t'`. That is to say, only
the C₁ part of C^< is calculated.

### Arguments
* n -> Index for given time step.
* A -> Retarded component of contour-ordered Green's function, Aᴿ(t,t'').
* Acc -> Hermitian conjugate partner of A.
* B -> Lesser component of contour-ordered Green's function, B^<(t'',t').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* h -> Time step interval.

### Returns
* C -> Lesser component of contour-ordered Green's function, C^<(t,t').

See also: [`convolution_time_step`](@ref).
"""
function conv_ret_less(
    n::I64,
    C::Gˡᵉˢˢ{T},
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    B::Gˡᵉˢˢ{T}, Bcc::Gˡᵉˢˢ{T},
    I::Integrator,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(A)

    # Sanity check
    @assert getntime(A) == getntime(B)
    @assert getntime(B) == getntime(C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert h > 0.0

    #
    # Evaluate the lesser convolution at a given time step
    #
    # See [NESSi] Eq. (117) - (118)
    #
    # For the contributions from C₂ and C₃, see conv_less_adv() and
    # conv_lmix_rmix() please.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_ret_less_kernel!(_LessCol(C, n), 1:n, n, A, Acc, nothing, B, Bcc, I, h, ws)
end

"""
    conv_ret_less(
        n::I64,
        C::Gˡᵉˢˢ{T},
        A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
        fₜ::VecArray{T},
        B::Gˡᵉˢˢ{T}, Bcc::Gˡᵉˢˢ{T},
        I::Integrator,
        h::F64
    ) where {T}

Try to calculate the lesser component (C^<) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B) and a time-dependent function fₜ. Actually, it just implements
`C^<(t,t') = Aᴿ(t,t'') ∗ f(t'') ∗ B^<(t'',t')` at time step `t' = nh`
for all `t` where `t ≤ t'`. That is to say, only the C₁ part of C^< is
calculated.

### Arguments
* n -> Index for given time step.
* A -> Retarded component of contour-ordered Green's function, Aᴿ(t,t'').
* Acc -> Hermitian conjugate partner of A.
* fₜ -> Values of time-dependent function, f(t'').
* B -> Lesser component of contour-ordered Green's function, B^<(t'',t').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* h -> Time step interval.

### Returns
* C -> Lesser component of contour-ordered Green's function, C^<(t,t').

See also: [`convolution_time_step`](@ref).
"""
function conv_ret_less(
    n::I64,
    C::Gˡᵉˢˢ{T},
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    fₜ::VecArray{T},
    B::Gˡᵉˢˢ{T}, Bcc::Gˡᵉˢˢ{T},
    I::Integrator,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    k = I.k

    # Sanity check
    @assert getntime(A) == getntime(B)
    @assert getntime(B) == getntime(C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert length(fₜ) ≥ n
    @assert length(fₜ) ≥ k + 1
    @assert ntime ≥ n ≥ 1
    @assert h > 0.0

    #
    # Evaluate the lesser convolution at a given time step
    #
    # See [NESSi] Eq. (117) - (118)
    #
    # For the contributions from C₂ and C₃, see conv_less_adv() and
    # conv_lmix_rmix() please.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_ret_less_kernel!(_LessCol(C, n), 1:n, n, A, Acc, fₜ, B, Bcc, I, h, ws)
end

"""
    _conv_ret_less_kernel!(tgt, ms, n, A, Acc, f, B, Bcc, I, s, ws)

Accumulate `s * [Aᴿ ∗ f ∗ B^<](tₘ,tₙ)` into `tgt[m]` for `m ∈ ms`. Here
`ms ⊆ 1:max(n,k+1)`. See [NESSi] Eqs. (117) - (118). It is a private
function.
"""
function _conv_ret_less_kernel!(
    tgt, ms::UnitRange{I64}, n::I64,
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    f,
    B::Gˡᵉˢˢ{T}, Bcc::Gˡᵉˢˢ{T},
    I::Integrator,
    s::Number,
    ws::_ConvWorkspace{T}
) where {T}
    n₁ = max(n - 1, I.k) + 1
    @assert isempty(ms) || (first(ms) ≥ 1 && last(ms) ≤ n₁)

    for j = 1:n₁
        # The lesser component f(tⱼ) B^<(tⱼ,tₙ), see [NESSi] Eq. (18a)
        if j ≤ n
            _conv_ret_less_col!(tgt, ms, j, _fb!(ws, f, j, _block(B, j, n)),
                                s, A, Acc, I)
        else
            _conv_ret_less_col!(tgt, ms, j, _fb!(ws, f, j, adjoint(_block(Bcc, n, j))),
                                -s, A, Acc, I)
        end
    end
end

"""
    _conv_ret_less_col!(tgt, ms, j, b, s, A, Acc, I)

Accumulate the `j`-th term of [NESSi] Eq. (118). It is a private function.
"""
function _conv_ret_less_col!(tgt, ms::UnitRange{I64}, j::I64, b, s::Number,
                             A::Gʳᵉᵗ, Acc::Gʳᵉᵗ, I::Integrator)
    k = I.k
    for m in ms
        if m - 1 > k
            # See [NESSi] Eq. (118a)
            j ≤ m || continue
            mul!(tgt[m], _block(A, m, j), b, s * I.GIW[m-1,j-1], true)
        else
            # See [NESSi] Eq. (118b)
            j ≤ k + 1 || continue
            w = I.GIW[m-1,j-1]
            if j > m
                mul!(tgt[m], adjoint(_block(Acc, j, m)), b, -s * w, true)
            else
                mul!(tgt[m], _block(A, m, j), b, s * w, true)
            end
        end
    end
end

