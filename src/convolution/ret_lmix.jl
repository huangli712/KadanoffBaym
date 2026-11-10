#
# Project : Lavender
# Source  : ret_lmix.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Convolution* : ``G^{⌉}`` *Component*

*Remarks* :

The evaluation of ``C^{\rceil}(t,\tau)`` at given time slice ``n`` is
implemented as follows:

```math
\begin{equation}
C^{\rceil}(nh,mh_{\tau}) =
    C^{\rceil}_1[A,f,B](n,m) +
    C^{\rceil}_2[A,f,B](n,m) +
    C^{\rceil}_3[A,f,B](n,m),
\end{equation}
```

where ``m = 0,\ \cdots,\ N_{\tau}``. In the above equation, ``h`` means
the interval in the real time axis (``\equiv \delta t``) and ``h_{\tau}``
means the interval in the imaginary time axis (``\equiv \delta \tau``).

```math
\begin{equation}
C^{\rceil}_1[A,f,B](n,m) = \int^{nh}_{0} d\bar{t}~
    A^{R}(nh,\bar{t}) f(\bar{t}) B^{\rceil}(\bar{t},mh_{\tau}).
\end{equation}
```

```math
\begin{equation}
C^{\rceil}_2[A,f,B](n,m) = \int^{mh_{\tau}}_{0} d\tau~
    A^{\rceil}(nh,\tau') f(0^{-}) B^{M}(\tau'-mh_{\tau}).
\end{equation}
```

```math
\begin{equation}
C^{\rceil}_3[A,f,B](n,m) = \int^{\beta}_{mh_{\tau}} d\tau~
    A^{\rceil}(nh,\tau') f(0^{-}) B^{M}(\tau' - mh_{\tau}).
\end{equation}
```

Actually, we implement the following equations:

```math
\begin{equation}
C^{\rceil}_1[A,f,B](n,m) = h \sum^{n}_{j = 0}
    w^{(k)}_{n,j} A^{R}_{n,j} f_j B^{\rceil}_{j,m},
    \quad n > k.
\end{equation}
```

```math
\begin{equation}
C^{\rceil}_1[A,f,B](n,m) = h \sum^{k}_{j = 0}
    w^{(k)}_{n,j} \tilde{A}^{R}_{n,j} f_j B^{\rceil}_{j,m},
    \quad n \le k.
\end{equation}
```

```math
\begin{equation}
C^{\rceil}_2[A,f,B](n,m) = h_{\tau} \sum^{k}_{j,l = 0}
    R^{(k)}_{m;j,l} A^{\rceil}_l f_{-1} \xi B^{M}_{N_{\tau}-j},
    \quad m \le k.
\end{equation}
```

```math
\begin{equation}
C^{\rceil}_2[A,f,B](n,m) = h_{\tau} \sum^{m}_{l = 0}
    w^{(k)}_{m,l} A^{\rceil}_{m-l} f_{-1} \xi B^{M}_{N_{\tau}-l},
    \quad m > k.
\end{equation}
```

```math
\begin{equation}
C^{\rceil}_3[A,f,B](n,m) = h_{\tau} \sum^{k}_{j,l = 0}
    R^{(k)}_{N_{\tau}-m;j,l} A^{\rceil}_{N_{\tau}-l} f_{-1} B^{M}_{j},
    \quad m \ge N_{\tau} - k.
\end{equation}
```

```math
\begin{equation}
C^{\rceil}_3[A,f,B](n,m) = h_{\tau} \sum^{N_{\tau}-m}_{l = 0}
    w^{(k)}_{N_{\tau}-m,l} A^{\rceil}_{m+l} f_{-1} B^{M}_{l},
    \quad m < N_{\tau} - k.
\end{equation}
```

*References* :

Please see [`NESSi`] Sections `9` and `11` for more details.
=#

"""
    conv_ret_lmix(
        n::I64,
        C::Gˡᵐⁱˣ{T},
        A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
        B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
        I::Integrator,
        h::F64
    ) where {T}

Try to calculate the left-mixing component (C^⌉) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B). Actually, it implements `C^⌉(t,τ) = Aᴿ(t,t') ∗ B^⌉(t',τ)` at
time step `t = nh` for all `τ ∈ [0,β]`. That is to say, only the C₁ part
of C^⌉ is calculated.

### Arguments
* n -> Index of given time steo.
* A -> Retarded component of contour-ordered Green's function, Aᴿ(t,t').
* Acc -> Hermitian conjugate partner of A.
* B -> Left-mixing component of contour-ordered Green's function, B^⌉(t',τ).
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* h -> Time step interval.

### Returns
* C -> Left-mixing component of contour-ordered Green's function, C^⌉(t,τ).

See also: [`conv_lmix_mat`](@ref).
"""
function conv_ret_lmix(
    n::I64,
    C::Gˡᵐⁱˣ{T},
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
    I::Integrator,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(C)

    # Sanity check
    @assert getntime(A) == getntime(B)
    @assert getntime(A) == getntime(C)
    @assert iscompatible(B, C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert h > 0.0

    #
    # Evaluate the left-mixing convolution at a given time step
    #
    # See [NESSi] Eq. (111) - (112)
    #
    # For the contributions from the C₂ and C₃, see conv_lmix_mat().
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_ret_lmix_kernel!(_LmixRow(C, n), n, A, Acc, nothing, B, I, h, ws)
end


"""
    conv_ret_lmix(
        n::I64,
        C::Gˡᵐⁱˣ{T},
        A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
        fₜ::VecArray{T},
        B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
        I::Integrator,
        h::F64
    ) where {T}

Try to calculate the left-mixing component (C^⌉) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B) and a time-dependent function fₜ. Actually, it just implements
`C^⌉(t,τ) = Aᴿ(t,t') ∗ f(t') ∗ B^⌉(t',τ)` at time step `t = nh` for all
`τ ∈ [0,β]`. That is to say, only the C₁ part of C^⌉ is calculated.

### Arguments
* n -> Index of given time steo.
* A -> Retarded component of contour-ordered Green's function, Aᴿ(t,t').
* Acc -> Hermitian conjugate partner of A.
* fₜ -> Values of time-dependent function, f(t').
* B -> Left-mixing component of contour-ordered Green's function, B^⌉(t',τ).
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* h -> Time step interval.

### Returns
* C -> Left-mixing component of contour-ordered Green's function, C^⌉(t,τ).

See also: [`conv_lmix_mat`](@ref).
"""
function conv_ret_lmix(
    n::I64,
    C::Gˡᵐⁱˣ{T},
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    fₜ::VecArray{T},
    B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
    I::Integrator,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(C)

    # Sanity check
    @assert getntime(A) == getntime(B)
    @assert getntime(A) == getntime(C)
    @assert iscompatible(B, C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert length(fₜ) ≥ n
    @assert ntime ≥ n ≥ 1
    @assert h > 0.0

    #
    # Evaluate the left-mixing convolution at a given time step
    #
    # See [NESSi] Eq. (111) - (112)
    #
    # For the contributions from the C₂ and C₃, see conv_lmix_mat().
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_ret_lmix_kernel!(_LmixRow(C, n), n, A, Acc, fₜ, B, I, h, ws)
end


"""
    _conv_ret_lmix_kernel!(tgt, n, A, Acc, f, B, I, s, ws)

Accumulate `s * [Aᴿ ∗ f ∗ B^⌉](tₙ,τₘ)` into `tgt[m]` for all `m`. See
[NESSi] Eqs. (111) - (112). It is a private function.
"""
function _conv_ret_lmix_kernel!(
    tgt, n::I64,
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    f,
    B::Gˡᵐⁱˣ{T},
    I::Integrator,
    s::Number,
    ws::_ConvWorkspace{T};
    ms::UnitRange{I64} = 1:B.ntau
) where {T}
    n₁ = max(n - 1, I.k) + 1
    for j = 1:n₁
        w = I.GIW[n-1,j-1]
        if n < j
            _conv_lmix_row!(tgt, _af!(ws, adjoint(_block(Acc, j, n)), f, j),
                            -s * w, B, j, ms)
        else
            _conv_lmix_row!(tgt, _af!(ws, _block(A, n, j), f, j),
                            s * w, B, j, ms)
        end
    end
end


