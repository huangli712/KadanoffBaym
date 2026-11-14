#
# Project : Lavender
# Source  : less.jl
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

"""
    conv_less_adv(
        n::I64,
        C::Gˡᵉˢˢ{T},
        A::Gˡᵉˢˢ{T}, Acc::Gˡᵉˢˢ{T},
        B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
        I::Integrator,
        h::F64
    ) where {T}

Try to calculate the lesser component (C^<) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B). Actually, it implements `C^<(t,t') = A^<(t,t'') ∗ Bᴬ(t'',t')`
at time step `t' = nh` for all `t` where `t ≤ t'`. That is to say, only
the C₂ part of C^< is calculated.

### Arguments
* n -> Index for given time step.
* A -> Lesser component of contour-ordered Green's function, A^<(t,t'').
* Acc -> Hermitian conjugate partner of A.
* B -> Retarded component of contour-ordered Green's function, Bᴿ(t',t'').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* h -> Time step interval.

### Returns
* C -> Lesser component of contour-ordered Green's function, C^<(t,t').

### Notes
The advanced component of contour-ordered Green's function, Bᴬ(t'',t), is
not ready. It should be calculated from the corresponding lesser
component, Bᴿ(t',t'').

See also: [`convolution_time_step`](@ref).
"""
function conv_less_adv(
    n::I64,
    C::Gˡᵉˢˢ{T},
    A::Gˡᵉˢˢ{T}, Acc::Gˡᵉˢˢ{T},
    B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
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
    # See [NESSi] Eq. (119) - (120)
    #
    # For the contributions from C₁ and C₃, see conv_ret_less() and
    # conv_lmix_rmix() please.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_less_adv_kernel!(_LessCol(C, n), 1:n, n, A, Acc, nothing, B, Bcc, I, h, ws)
end

"""
    conv_less_adv(
        n::I64,
        C::Gˡᵉˢˢ{T},
        A::Gˡᵉˢˢ{T}, Acc::Gˡᵉˢˢ{T},
        fₜ::VecArray{T},
        B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
        I::Integrator,
        h::F64
    ) where {T}

Try to calculate the lesser component (C^<) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B) and a time-dependent function fₜ. Actually, it just implements
`C^<(t,t') = A^<(t,t'') ∗ f(t'') ∗ Bᴬ(t'',t')` at time step `t' = nh`
for all `t` where `t ≤ t'`. That is to say, only the C₂ part of C^< is
calculated.

### Arguments
* n -> Index for given time step.
* A -> Lesser component of contour-ordered Green's function, A^<(t,t'').
* Acc -> Hermitian conjugate partner of A.
* fₜ -> Values of time-dependent function, f(t'').
* B -> Retarded component of contour-ordered Green's function, Bᴿ(t',t'').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* h -> Time step interval.

### Returns
* C -> Lesser component of contour-ordered Green's function, C^<(t,t').

### Notes
The advanced component of contour-ordered Green's function, Bᴬ(t'',t), is
not ready. Thus, it should be calculated from the corresponding lesser
component, Bᴿ(t',t'').

See also: [`convolution_time_step`](@ref).
"""
function conv_less_adv(
    n::I64,
    C::Gˡᵉˢˢ{T},
    A::Gˡᵉˢˢ{T}, Acc::Gˡᵉˢˢ{T},
    fₜ::VecArray{T},
    B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
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
    # See [NESSi] Eq. (119) - (120)
    #
    # For the contributions from C₁ and C₃, see conv_ret_less() and
    # conv_lmix_rmix() please.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_less_adv_kernel!(_LessCol(C, n), 1:n, n, A, Acc, fₜ, B, Bcc, I, h, ws)
end

"""
    _conv_less_adv_kernel!(tgt, ms, n, A, Acc, f, B, Bcc, I, s, ws)

Accumulate `s * [A^< ∗ f ∗ Bᴬ](tₘ,tₙ)` into `tgt[m]` for `m ∈ ms`. Here
`ms ⊆ 1:max(n,k+1)`. See [NESSi] Eqs. (119) - (120). It is a private
function.
"""
function _conv_less_adv_kernel!(
    tgt, ms::UnitRange{I64}, n::I64,
    A::Gˡᵉˢˢ{T}, Acc::Gˡᵉˢˢ{T},
    f,
    B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
    I::Integrator,
    s::Number,
    ws::_ConvWorkspace{T};
    diagcc::Bool = false
) where {T}
    n₁ = max(n - 1, I.k) + 1
    @assert isempty(ms) || (first(ms) ≥ 1 && last(ms) ≤ n₁)

    for j = 1:n₁
        w = I.GIW[n-1,j-1]
        # The advanced component f(tⱼ) Bᴬ(tⱼ,tₙ), see [NESSi] Eq. (18a)
        if j ≤ n
            _conv_less_adv_col!(tgt, ms, j, _fb!(ws, f, j, adjoint(_block(Bcc, n, j))),
                                s * w, A, Acc, diagcc)
        else
            _conv_less_adv_col!(tgt, ms, j, _fb!(ws, f, j, _block(B, j, n)),
                                -s * w, A, Acc, diagcc)
        end
    end
end

"""
    _conv_less_adv_col!(tgt, ms, j, b, α, A, Acc, diagcc = false)

Accumulate the `j`-th term of [NESSi] Eq. (120). If `diagcc = true`, the
diagonal block ``A^{<}(t_m,t_m)`` is replaced by ``-Acc^{<}(t_m,t_m)^{\\dagger}``,
which follows `incr_convolution_les` of NESSi for the early time steps.
It is a private function.
"""
function _conv_less_adv_col!(tgt, ms::UnitRange{I64}, j::I64, b, α::Number,
                             A::Gˡᵉˢˢ, Acc::Gˡᵉˢˢ, diagcc::Bool = false)
    for m in ms
        if j < m || (diagcc && j == m)
            mul!(tgt[m], adjoint(_block(Acc, j, m)), b, -α, true)
        else
            mul!(tgt[m], _block(A, m, j), b, α, true)
        end
    end
end

"""
    conv_lmix_rmix(
        n::I64,
        C::Gˡᵉˢˢ{T},
        A::Gˡᵐⁱˣ{T}, Acc::Gˡᵐⁱˣ{T},
        B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
        I::Integrator,
        beta::F64,
        sign::I64
    ) where {T}

Try to calculate the lesser component (C^<) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B). Actually, it implements `C^<(t,t') = A^⌉(t,τ) ∗ B^⌈(τ,t')` at
time step `t' = nh` for all `t` where `t ≤ t'`. That is to say, only the
C₃ part of C^< is calculated.

### Arguments
* n -> Index for given time step.
* A -> Left-mixing component of contour-ordered Green's function, A^⌉(t,τ).
* Acc -> Hermitian conjugate partner of A.
* B -> Left-mixing component of contour-ordered Green's function, B^⌉(t',τ).
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Less component of contour-ordered Green's function, C^<(t,t').

### Notes
The right-mixing component of contour-ordered Green's function, B^⌈(τ,t'),
is not ready. It should be calculated from the corresponding left-mixing
component, B^⌉(t',τ).

See also: [`convolution_time_step`](@ref).
"""
function conv_lmix_rmix(
    n::I64,
    C::Gˡᵉˢˢ{T},
    A::Gˡᵐⁱˣ{T}, Acc::Gˡᵐⁱˣ{T},
    B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
    I::Integrator,
    beta::F64,
    sign::I64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    ntau = getntau(A)

    # Sanity check
    @assert getntime(A) == getntime(C)
    @assert iscompatible(A, B)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert beta > 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate δτ
    δτ = beta / (ntau - 1)

    #
    # Evaluate the lesser convolution at a given time step
    #
    # See [NESSi] Eq. (121) - (122)
    #
    # For the contributions from C₁ and C₂, see conv_ret_less() and
    # conv_less_adv() please.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_lmix_rmix_kernel!(_LessCol(C, n), 1:n, n, A, nothing, Bcc, I, sign, -im * δτ, ws)
end

"""
    conv_lmix_rmix(
        n::I64,
        C::Gˡᵉˢˢ{T},
        A::Gˡᵐⁱˣ{T}, Acc::Gˡᵐⁱˣ{T},
        f₀::Element{T},
        B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
        I::Integrator,
        beta::F64,
        sign::I64
    ) where {T}

Try to calculate the lesser component (C^<) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B) and a constant matrix (time-independent) f₀. Actually, it just
implements `C^<(t,t') = A^⌉(t,τ) ∗ f₀ ∗ B^⌈(τ,t')` at time step `t' = nh`
for all `t` where `t ≤ t'`. That is to say, only the C₃ part of C^< is
calculated.

### Arguments
* n -> Index for given time step.
* A -> Left-mixing component of contour-ordered Green's function, A^⌉(t,τ).
* Acc -> Hermitian conjugate partner of A.
* f₀ -> Value of function f at τ = -iβ.
* B -> Left-mixing component of contour-ordered Green's function, B^⌉(t',τ).
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Less component of contour-ordered Green's function, C^<(t,t').

### Notes
The right-mixing component of contour-ordered Green's function, B^⌈(τ,t'),
is not ready. It should be calculated from the corresponding left-mixing
component, B^⌉(t',τ).

The argument `f₀` is assumed to be constant matrix.

See also: [`convolution_time_step`](@ref).
"""
function conv_lmix_rmix(
    n::I64,
    C::Gˡᵉˢˢ{T},
    A::Gˡᵐⁱˣ{T}, Acc::Gˡᵐⁱˣ{T},
    f₀::Element{T},
    B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
    I::Integrator,
    beta::F64,
    sign::I64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    ntau = getntau(A)

    # Sanity check
    @assert getntime(A) == getntime(C)
    @assert iscompatible(A, B)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert beta > 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate δτ
    δτ = beta / (ntau - 1)

    #
    # Evaluate the lesser convolution at a given time step
    #
    # See [NESSi] Eq. (121) - (122)
    #
    # For the contributions from C₁ and C₂, see conv_ret_less() and
    # conv_less_adv() please.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_lmix_rmix_kernel!(_LessCol(C, n), 1:n, n, A, f₀, Bcc, I, sign, -im * δτ, ws)
end

"""
    _conv_lmix_rmix_kernel!(tgt, ms, n, A, f₀, Bcc, I, sign, s, ws)

Accumulate `s * [A^⌉ ∗ f₀ ∗ B^⌈](tₘ,tₙ) / (-i)` into `tgt[m]` for `m ∈ ms`,
i.e., the factor `-i δτ` should be included in `s`. Here
`ms ⊆ 1:max(n,k+1)`. See [NESSi] Eqs. (121) - (122). It is a private
function.
"""
function _conv_lmix_rmix_kernel!(
    tgt, ms::UnitRange{I64}, n::I64,
    A::Gˡᵐⁱˣ{T},
    f₀,
    Bcc::Gˡᵐⁱˣ{T},
    I::Integrator,
    sign::I64,
    s::Number,
    ws::_ConvWorkspace{T}
) where {T}
    ntau = A.ntau
    @assert isempty(ms) || (first(ms) ≥ 1 && last(ms) ≤ max(n - 1, I.k) + 1)

    for j = 1:ntau
        w = I.GIW[ntau-1,j-1]
        # The right-mixing component f₀ B^⌈(τⱼ,tₙ), see [NESSi] Eq. (18d)
        b = _f0b!(ws, f₀, adjoint(_block(Bcc, n, ntau - j + 1)))
        _conv_lmix_rmix_col!(tgt, ms, j, b, -sign * s * w, A)
    end
end

"""
    _conv_lmix_rmix_col!(tgt, ms, j, b, α, A)

Accumulate the `j`-th term of [NESSi] Eq. (122). It is a private function.
"""
function _conv_lmix_rmix_col!(tgt, ms::UnitRange{I64}, j::I64, b, α::Number,
                              A::Gˡᵐⁱˣ)
    for m in ms
        mul!(tgt[m], _block(A, m, j), b, α, true)
    end
end

#=
*Remarks* : *Partial Lesser Convolutions*

The lesser component of `A ∗ f ∗ B` consists of three terms:

```math
\begin{equation}
C^{<} = A^{R} \ast f \ast B^{<} + A^{<} \ast f \ast B^{A}
      + A^{\rceil} \ast f \ast B^{\lceil}.
\end{equation}
```

They are evaluated by `incr_convolution_ret_less!`,
`incr_convolution_less_adv!`, and `incr_convolution_lmix_rmix!`,
respectively. In the time stepping of the Dyson and VIE2 equations, only
the latter two terms enter the source term, while the first one belongs
to the implicit linear system. None of them reads the target `C`.

*References* :

Please see [`NESSi`] `cntr_convolution_impl.hpp`,
`convolution_timestep_les_retles`, `convolution_timestep_les_lesadv`, and
`convolution_timestep_les_tvvt`.
=#

"""
    incr_convolution_ret_less!(
        tstp::I64,
        C::_ConvTarget{T},
        A::_ConvGF{T}, Acc::_ConvGF{T},
        f::Union{Cf{T},Nothing},
        B::_ConvGF{T}, Bcc::_ConvGF{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment `C^<(t',t)` (`t' ≤ t`, `t` is given by `tstp`) by
`alpha * [Aᴿ ∗ f ∗ B^<](t',t)`. The argument `beta` is not used. It is
kept for an unified interface.

### Arguments
See [`incr_convolution_lmix!`](@ref).

### Returns
N/A
"""
function incr_convolution_ret_less!(
    tstp::I64,
    C::_ConvTarget{T},
    A::_ConvGF{T}, Acc::_ConvGF{T},
    f::Union{Cf{T},Nothing},
    B::_ConvGF{T}, Bcc::_ConvGF{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target = (:less,))
    @assert tstp ≥ 1
    @assert h > 0.0

    ws = _ConvWorkspace(A, B)
    _incr_ret_less!(_less_target(C, tstp), 1:tstp, tstp,
                    A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
end

"""
    incr_convolution_less_adv!(
        tstp::I64,
        C::_ConvTarget{T},
        A::_ConvGF{T}, Acc::_ConvGF{T},
        f::Union{Cf{T},Nothing},
        B::_ConvGF{T}, Bcc::_ConvGF{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment `C^<(t',t)` (`t' ≤ t`, `t` is given by `tstp`) by
`alpha * [A^< ∗ f ∗ Bᴬ](t',t)`. The argument `beta` is not used. It is
kept for an unified interface.

### Arguments
See [`incr_convolution_lmix!`](@ref).

### Returns
N/A
"""
function incr_convolution_less_adv!(
    tstp::I64,
    C::_ConvTarget{T},
    A::_ConvGF{T}, Acc::_ConvGF{T},
    f::Union{Cf{T},Nothing},
    B::_ConvGF{T}, Bcc::_ConvGF{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target = (:less,))
    @assert tstp ≥ 1
    @assert h > 0.0

    ws = _ConvWorkspace(A, B)
    _incr_less_adv!(_less_target(C, tstp), 1:tstp, tstp,
                    A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
end

"""
    incr_convolution_lmix_rmix!(
        tstp::I64,
        C::_ConvTarget{T},
        A::_ConvGF{T}, Acc::_ConvGF{T},
        f::Union{Cf{T},Nothing},
        B::_ConvGF{T}, Bcc::_ConvGF{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment `C^<(t',t)` (`t' ≤ t`, `t` is given by `tstp`) by
`alpha * [A^⌉ ∗ f₀ ∗ B^⌈](t',t)`, including the factor `-i`. Here `f[0]`
is used. The argument `h` is not used. It is kept for an unified
interface.

### Arguments
See [`incr_convolution_lmix!`](@ref).

### Returns
N/A
"""
function incr_convolution_lmix_rmix!(
    tstp::I64,
    C::_ConvTarget{T},
    A::_ConvGF{T}, Acc::_ConvGF{T},
    f::Union{Cf{T},Nothing},
    B::_ConvGF{T}, Bcc::_ConvGF{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target = (:less,))
    @assert tstp ≥ 1
    @assert beta > 0.0

    ws = _ConvWorkspace(A, B)
    _incr_lmix_rmix!(_less_target(C, tstp), 1:tstp, tstp,
                     A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
end

#
# Workspace-bearing versions of the three partial lesser convolutions.
# The target `tgt[m]` (m ∈ ms) is C^<(tₘ,tₙ). They are private functions.
#

function _incr_ret_less!(tgt, ms::UnitRange{I64}, n::I64,
                         A::_ConvGF{T}, Acc::_ConvGF{T}, f, B::_ConvGF{T}, Bcc::_ConvGF{T},
                         I::Integrator, beta::F64, h::F64,
                         alpha::Number, ws::_ConvWorkspace{T}) where {T}
    _conv_ret_less_kernel!(tgt, ms, n, A.ret, Acc.ret, f,
                           B.less, Bcc.less, I, alpha * h, ws)
end
#

function _incr_less_adv!(tgt, ms::UnitRange{I64}, n::I64,
                         A::_ConvGF{T}, Acc::_ConvGF{T}, f, B::_ConvGF{T}, Bcc::_ConvGF{T},
                         I::Integrator, beta::F64, h::F64,
                         alpha::Number, ws::_ConvWorkspace{T};
                         diagcc::Bool = false) where {T}
    _conv_less_adv_kernel!(tgt, ms, n, A.less, Acc.less, f,
                           B.ret, Bcc.ret, I, alpha * h, ws; diagcc = diagcc)
end
#
# In incr_convolution_les of NESSi, the diagonal block A^<(t,t) of the
# term A^< ∗ f ∗ Bᴬ is taken as -Acc^<(t,t)† if tstp ≤ 2k + 2 (NESSi
# index). The two choices agree if Acc is the hermitian conjugate partner
# of A. See [NESSi] cntr_convolution_impl.hpp, incr_convolution_les.
#

_incr_les_diagcc(n::I64, I::Integrator) = n - 1 ≤ 2 * I.k + 2
#

function _incr_lmix_rmix!(tgt, ms::UnitRange{I64}, n::I64,
                          A::_ConvGF{T}, Acc::_ConvGF{T}, f, B::_ConvGF{T}, Bcc::_ConvGF{T},
                          I::Integrator, beta::F64, h::F64,
                          alpha::Number, ws::_ConvWorkspace{T}) where {T}
    δτ = beta / (getntau(A) - 1)
    _conv_lmix_rmix_kernel!(tgt, ms, n, A.lmix, _f0(f), Bcc.lmix, I,
                            A.sign, alpha * (-im * δτ), ws)
end

"""
    _incr_convolution_less_adv!(
        buf::AbstractVector{<:AbstractMatrix},
        tstp::I64,
        A::_ConvGF{T}, Acc::_ConvGF{T},
        f::Union{Cf{T},Nothing},
        B::_ConvGF{T}, Bcc::_ConvGF{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number,
        ws::_ConvWorkspace{T}
    ) where {T}

Scratch-target version of [`incr_convolution_less_adv!`](@ref) for the
start-up time steps `1 ≤ tstp ≤ k`. Let `n₁ = max(tstp - 1, k)`, then
`buf[j]` (`j = 1, ⋯, n₁ + 1`) is incremented by
`alpha * [A^< ∗ f ∗ Bᴬ](tⱼ,t)`. Note that the blocks `j > tstp` are also
evaluated, which are needed in the start-up procedure. It is a private
function.

### Returns
* buf -> The same vector.
"""
function _incr_convolution_less_adv!(
    buf::AbstractVector{<:AbstractMatrix},
    tstp::I64,
    A::_ConvGF{T}, Acc::_ConvGF{T},
    f::Union{Cf{T},Nothing},
    B::_ConvGF{T}, Bcc::_ConvGF{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number,
    ws::_ConvWorkspace{T}
) where {T}
    # Sanity check
    n₁ = max(tstp - 1, I.k)
    @assert 1 ≤ tstp ≤ I.k
    @assert length(buf) == n₁ + 1
    _check_incr(tstp, A, A, Acc, f, B, Bcc; target = ())
    @assert getntime(A) ≥ n₁ + 1
    @assert h > 0.0

    _incr_less_adv!(buf, 1:n₁+1, tstp, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
    return buf
end

"""
    _incr_convolution_lmix_rmix!(
        buf::AbstractVector{<:AbstractMatrix},
        tstp::I64,
        A::_ConvGF{T}, Acc::_ConvGF{T},
        f::Union{Cf{T},Nothing},
        B::_ConvGF{T}, Bcc::_ConvGF{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number,
        ws::_ConvWorkspace{T}
    ) where {T}

Scratch-target version of [`incr_convolution_lmix_rmix!`](@ref) for the
start-up time steps `1 ≤ tstp ≤ k`. Let `n₁ = max(tstp - 1, k)`, then
`buf[j]` (`j = 1, ⋯, n₁ + 1`) is incremented by
`alpha * [A^⌉ ∗ f₀ ∗ B^⌈](tⱼ,t)`. It is a private function.

### Returns
* buf -> The same vector.
"""
function _incr_convolution_lmix_rmix!(
    buf::AbstractVector{<:AbstractMatrix},
    tstp::I64,
    A::_ConvGF{T}, Acc::_ConvGF{T},
    f::Union{Cf{T},Nothing},
    B::_ConvGF{T}, Bcc::_ConvGF{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number,
    ws::_ConvWorkspace{T}
) where {T}
    # Sanity check
    n₁ = max(tstp - 1, I.k)
    @assert 1 ≤ tstp ≤ I.k
    @assert length(buf) == n₁ + 1
    _check_incr(tstp, A, A, Acc, f, B, Bcc; target = ())
    @assert getntime(A) ≥ n₁ + 1
    @assert beta > 0.0

    _incr_lmix_rmix!(buf, 1:n₁+1, tstp, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
    return buf
end

"""
    incr_convolution_les!(
        tstp::I64,
        C::_ConvTarget{T},
        A::_ConvGF{T}, Acc::_ConvGF{T},
        f::Union{Cf{T},Nothing},
        B::_ConvGF{T}, Bcc::_ConvGF{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment the lesser component `C^<(t',t)` (`t' ≤ t`, `t` is given by
`tstp`) by `alpha * [A ∗ f ∗ B]^<(t',t)`. It is the sum of the three
partial lesser convolutions.

### Arguments
See [`incr_convolution_lmix!`](@ref).

### Returns
N/A

See also: [`incr_convolution_ret_less!`](@ref),
[`incr_convolution_less_adv!`](@ref), [`incr_convolution_lmix_rmix!`](@ref).
"""
function incr_convolution_les!(
    tstp::I64,
    C::_ConvTarget{T},
    A::_ConvGF{T}, Acc::_ConvGF{T},
    f::Union{Cf{T},Nothing},
    B::_ConvGF{T}, Bcc::_ConvGF{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target = (:less,))
    @assert tstp ≥ 1
    @assert beta > 0.0
    @assert h > 0.0

    ws = _ConvWorkspace(A, B)
    _incr_convolution_les!(tstp, C, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
end

#
# Workspace-bearing version of incr_convolution_les!(). It is a private
# function.
#

function _incr_convolution_les!(tstp::I64, C, A::_ConvGF{T}, Acc::_ConvGF{T}, f,
                                B::_ConvGF{T}, Bcc::_ConvGF{T}, I::Integrator,
                                beta::F64, h::F64,
                                alpha::Number, ws::_ConvWorkspace{T}) where {T}
    tgt = _less_target(C, tstp)
    _incr_ret_less!(tgt, 1:tstp, tstp, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
    _incr_less_adv!(tgt, 1:tstp, tstp, A, Acc, f, B, Bcc, I, beta, h, alpha, ws;
                    diagcc = _incr_les_diagcc(tstp, I))
    _incr_lmix_rmix!(tgt, 1:tstp, tstp, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
end

"""
    _incr_less_src_threaded!(tstp, C, A, Acc, B, Bcc, I, beta, h, alpha)

Threaded version of `incr_convolution_lmix_rmix!` followed by
`incr_convolution_less_adv!` with `f = 1`, the first times are
distributed over the threads. The accumulation order of every output
block is the same as in the serial code. It is a private function.
"""
function _incr_less_src_threaded!(tstp::I64, C, A::ℱ{T}, Acc::ℱ{T}, B::ℱ{T}, Bcc::ℱ{T},
                                    I::Integrator, beta::F64, h::F64,
                                    alpha::Number) where {T}
    _check_incr(tstp, C, A, Acc, nothing, B, Bcc; target = (:less,))
    ws = _ConvWorkspace(A, B)
    tgt = _less_target(C, tstp)
    _tforeach(1:tstp, ws) do ms, wsₜ
        _incr_lmix_rmix!(tgt, ms, tstp, A, Acc, nothing, B, Bcc, I, beta, h, alpha, wsₜ)
        _incr_less_adv!(tgt, ms, tstp, A, Acc, nothing, B, Bcc, I, beta, h, alpha, wsₜ)
    end
    return C
end
