#
# Project : Lavender
# Source  : lmix.jl
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

"""
    conv_lmix_mat(
        n::I64,
        C::Gˡᵐⁱˣ{T}, A::Gˡᵐⁱˣ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator,
        beta::F64,
        sign::I64
    ) where {T}

Try to calculate the left-mixing component (C^⌉) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B). Actually, it implements `C^⌉(t,τ) = A^⌉(t,τ') ∗ Bᴹ(τ'-τ)` at
time step `t = nh` for all `τ ∈ [0,β]`. That is to say, only the C₂ and
C₃ parts of C^⌉ are calculated.

### Arguments
* n -> Index for given time step.
* A -> Left-mixing component of contour-ordered Green's function, A^⌉(t,τ').
* B -> Retarded component of contour-ordered Green's function, Bᴹ(τ'-τ).
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Left-mixing component of contour-ordered Green's function, C^⌉(t,τ).

See also: [`conv_ret_lmix`](@ref).
"""
function conv_lmix_mat(
    n::I64,
    C::Gˡᵐⁱˣ{T}, A::Gˡᵐⁱˣ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    beta::F64,
    sign::I64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    ntau = getntau(A)

    # Sanity check
    @assert getntau(A) == getntau(B)
    @assert getntau(B) == getntau(C)
    @assert iscompatible(A, C)
    @assert ntime ≥ n ≥ 1
    @assert beta > 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate δτ
    δτ = beta / (ntau - 1)

    #
    # Evaluate the left-mixing convolution at a given time step
    #
    # It is very similar to computing the Matsubara convolution.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_lmix_mat_kernel!(_LmixRow(C, n), n, A, nothing, B, I, sign, δτ, ws)
end

"""
    conv_lmix_mat(
        n::I64,
        C::Gˡᵐⁱˣ{T}, A::Gˡᵐⁱˣ{T}, f₀::Element{T}, B::Gᵐᵃᵗ{T},
        I::Integrator,
        beta::F64,
        sign::I64
    ) where {T}

Try to calculate the left-mixing component (C^⌉) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B) and a constant matrix (time-independent) f₀. Actually, it just
implements `C^⌉(t,τ) = A^⌉(t,τ') ∗ f₀ ∗ Bᴹ(τ'-τ)` at time step `t = nh`
for all `τ ∈ [0,β]`. That is to say, only the C₂ and C₃ parts of C^⌉ are
calculated.

### Arguments
* n -> Index for given time step.
* A -> Left-mixing component of contour-ordered Green's function, A^⌉(t,τ').
* f₀ -> Value of function f at τ = -iβ.
* B -> Retarded component of contour-ordered Green's function, Bᴹ(τ'-τ).
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Left-mixing component of contour-ordered Green's function, C^⌉(t,τ).

### Notes
The argument `f₀` is assumed to be constant matrix.

See also: [`conv_ret_lmix`](@ref).
"""
function conv_lmix_mat(
    n::I64,
    C::Gˡᵐⁱˣ{T}, A::Gˡᵐⁱˣ{T}, f₀::Element{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    beta::F64,
    sign::I64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    ntau = getntau(A)

    # Sanity check
    @assert getntau(A) == getntau(B)
    @assert getntau(B) == getntau(C)
    @assert iscompatible(A, C)
    @assert ntime ≥ n ≥ 1
    @assert beta > 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate δτ
    δτ = beta / (ntau - 1)

    #
    # Evaluate the left-mixing convolution at a given time step
    #
    # f₀ = f(-iβ) is applied block by block, B is not copied.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_lmix_mat_kernel!(_LmixRow(C, n), n, A, f₀, B, I, sign, δτ, ws)
end

"""
    _conv_lmix_mat_kernel!(tgt, n, A, f₀, B, I, sign, s, ws)

Accumulate `s * [A^⌉ ∗ f₀ ∗ Bᴹ](tₙ,τₘ)` into `tgt[m]` for all `m`. See
[NESSi] Eqs. (113) - (116). It is a private function.
"""
function _conv_lmix_mat_kernel!(
    tgt, n::I64,
    A::Gˡᵐⁱˣ{T},
    f₀,
    B::Gᵐᵃᵗ{T},
    I::Integrator,
    sign::I64,
    s::Number,
    ws::_ConvWorkspace{T};
    ms::UnitRange{I64} = 1:A.ntau
) where {T}
    _conv_lmix_mat_loop!(tgt, n, A, _fbmat!(ws, f₀, B), I, sign, s, ms)
end

"""
    _conv_lmix_row!(tgt, a, α, B, j)

Accumulate `α * a * B^⌉(tⱼ,τₘ)` into `tgt[m]` for all `m`. It is a private
function.
"""
function _conv_lmix_row!(tgt, a, α::Number, B::Gˡᵐⁱˣ, j::I64,
                         ms::UnitRange{I64} = 1:B.ntau)
    for m in ms
        mul!(tgt[m], a, _block(B, j, m), α, true)
    end
end

"""
    _conv_lmix_mat_loop!(tgt, n, A, Bt, I, sign, s)

The integration loop of [`_conv_lmix_mat_kernel!`](@ref). Here `Bt[m]`
returns the block `f₀ Bᴹ(τₘ)`. It is a private function.
"""
function _conv_lmix_mat_loop!(tgt, n::I64, A::Gˡᵐⁱˣ, Bt, I::Integrator,
                              sign::I64, s::Number,
                              ms::UnitRange{I64} = 1:A.ntau)
    k = I.k
    ntau = A.ntau

    for m in ms
        c = tgt[m]

        # Try to calculate the contributions from 0 to τ
        #
        # See [NESSi] Eq. (113) - (114)
        #
        if m == 1
            # PASS
        elseif m < k + 1 # Strange boundary correction
            for j = 1:k+1
                indb = ntau
                for l = 1:k+1
                    w = I.BCW[m-2,l-1,j-1]
                    mul!(c, _block(A, n, j), Bt[indb], sign * s * w, true)
                    indb = indb - 1
                end
            end
        else # Usual Gregory integration
            inda = m
            indb = ntau
            for l = 1:m
                w = I.GIW[m-1,l-1]
                mul!(c, _block(A, n, inda), Bt[indb], sign * s * w, true)
                inda = inda - 1
                indb = indb - 1
            end
        end

        # Try to calculate the contributions from τ to β
        #
        # See [NESSi] Eq. (115) - (116)
        #
        if m == ntau
            # PASS
        elseif m > ntau - k # Strange boundary correction
            inda = ntau
            for l = 1:k+1
                for j = 1:k+1
                    w = I.BCW[ntau-m-1,l-1,j-1]
                    mul!(c, _block(A, n, inda), Bt[j], s * w, true)
                end
                inda = inda - 1
            end
        elseif m > ntau - 2*k - 1 # Usual Gregory integration
            inda = m
            for l = 1:ntau-m+1
                w = I.GIW[ntau-m,l-1]
                mul!(c, _block(A, n, inda), Bt[l], s * w, true)
                inda = inda + 1
            end
        else # Usual Gregory integration
            inda = m
            indb = 1
            for l = m:ntau
                w = I.GIW[ntau-m,ntau-l]
                mul!(c, _block(A, n, inda), Bt[indb], s * w, true)
                inda = inda + 1
                indb = indb + 1
            end
        end
    end
end

"""
    incr_convolution_lmix!(
        tstp::I64,
        C::Union{ℱ{T},𝒻{T}},
        A::ℱ{T}, Acc::ℱ{T},
        f::Union{Cf{T},Nothing},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment the left-mixing component of `C` at time step `tstp` by
`alpha * [A ∗ f ∗ B]^⌉(t,τ)`. The real-time part uses `f(t'')`, while the
imaginary-time part uses `f[0]`.

### Arguments
* tstp  -> Index of time step, tstp ≥ 1.
* C     -> Target, a `ℱ` or `𝒻` object.
* A     -> Contour-ordered Green's function, A(t,t'').
* Acc   -> Hermitian conjugate partner of A.
* f     -> Contour function, f(t''), or `nothing`.
* B     -> Contour-ordered Green's function, B(t'',t').
* Bcc   -> Hermitian conjugate partner of B.
* I     -> Struct for numerical integration.
* beta  -> Inverse temperature, β.
* h     -> Time step interval.
* alpha -> Weight.

### Returns
N/A
"""
function incr_convolution_lmix!(
    tstp::I64,
    C::Union{ℱ{T},𝒻{T}},
    A::ℱ{T}, Acc::ℱ{T},
    f::Union{Cf{T},Nothing},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target = (:lmix,))
    @assert tstp ≥ 1
    @assert beta > 0.0
    @assert h > 0.0

    ws = _ConvWorkspace(A, B)
    _incr_convolution_lmix!(tstp, C, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
end

#
# Workspace-bearing version of incr_convolution_lmix!(). It is a private
# function.
#

function _incr_convolution_lmix!(tstp::I64, C, A::ℱ{T}, Acc::ℱ{T}, f,
                               B::ℱ{T}, Bcc::ℱ{T}, I::Integrator,
                               beta::F64, h::F64,
                               alpha::Number, ws::_ConvWorkspace{T}) where {T}
    δτ = beta / (getntau(A) - 1)
    tgt = _lmix_target(C, tstp)
    _conv_ret_lmix_kernel!(tgt, tstp, A.ret, Acc.ret, f, B.lmix, I, alpha * h, ws)
    _conv_lmix_mat_kernel!(tgt, tstp, A.lmix, _f0(f), B.mat, I, A.sign, alpha * δτ, ws)
end

"""
    _incr_lmix_threaded!(tstp, C, A, Acc, B, Bcc, I, beta, h, alpha)

Threaded version of `incr_convolution_lmix!` with `f = 1`, the imaginary
time points are distributed over the threads. The accumulation order of
every output block is the same as in the serial code. It is a private
function.
"""
function _incr_lmix_threaded!(tstp::I64, C, A::ℱ{T}, Acc::ℱ{T}, B::ℱ{T}, Bcc::ℱ{T},
                            I::Integrator, beta::F64, h::F64, alpha::Number) where {T}
    _check_incr(tstp, C, A, Acc, nothing, B, Bcc; target = (:lmix,))
    ws = _ConvWorkspace(A, B)
    δτ = beta / (getntau(A) - 1)
    tgt = _lmix_target(C, tstp)
    Bt = _fbmat!(ws, nothing, B.mat)
    _tforeach(1:getntau(A), ws) do ms, wsₜ
        _conv_ret_lmix_kernel!(tgt, tstp, A.ret, Acc.ret, nothing, B.lmix, I, alpha * h, wsₜ;
                               ms = ms)
        _conv_lmix_mat_loop!(tgt, tstp, A.lmix, Bt, I, A.sign, alpha * δτ, ms)
    end
    return C
end
