#
# Project : Lavender
# Source  : convolution.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/01/06
#

#=
*Remarks* : *Convolution*

**Convolution Type 1 : C = A ∗ B**

The convolution of two correlators ``A(t,t')`` and ``B(t,t')`` reads

```math
\begin{equation}
C(t,t') = [A \ast B](t,t')
        = \int_{\mathcal{C}} d\bar{t} \
          A(t,\bar{t}) B(\bar{t},t').
\end{equation}
```

It is one of the most basic operations on the contour ``\mathcal{C}``.

**Convolution Type 2 : C = A ∗ f ∗ B**

The most general convolution of two contour-ordered Green's functions `A`
and `B` and a time-dependent function `f` is given by the integral:

```math
\begin{equation}
C(t,t') = [A \ast B](t,t')
        = \int_{\mathcal{C}} d\bar{t} \
          A(t,\bar{t}) f(\bar{t}) B(\bar{t},t').
\end{equation}
```

---

**Langreth Rules 1 : C = A ∗ B**

Using the Langreth rules, the convolution integral (`Convolution Type 1`)
is split into contributions from the Matsubara, retarded, left-mixing,
and lesser components:

```math
\begin{equation}
C^{M}(\tau) = \int^{\beta}_0 d\bar{\tau} \
    A^{M} (\tau - \bar{\tau}) B^{M}(\bar{\tau}).
\end{equation}
```

```math
\begin{equation}
C^{R}(t,t') = \int^{t}_{t'} d\bar{t} \
    A^{R}(t,\bar{t}) B^{R}(\bar{t},t').
\end{equation}
```

```math
\begin{equation}
C^{\rceil}(t,\tau) = \int^t_0 d\bar{t} \
    A^{R}(t,\bar{t}) B^{\rceil} (\bar{t},\tau)
                   + \int^{\beta}_0 d\bar{\tau} \
    A^{\rceil}(t,\bar{\tau}) B^{M}(\bar{\tau} - \tau).
\end{equation}
```

```math
\begin{equation}
C^{<}(t,t') = \int^t_0 d\bar{t} \
    A^{R}(t,\bar{t}) B^{<}(\bar{t},t')
            + \int^{t'}_0 d\bar{t}\
    A^{<}(t,\bar{t}) B^{A}(\bar{t},t')
            -i \int^{\beta}_0 d\bar{\tau} \
    A^{\rceil}(t,\bar{\tau}) B^{\lceil}(\bar{\tau},t').

\end{equation}
```

**Langreth Rules 2 : C = A ∗ f ∗ B**

Using the Langreth rules, the convolution integral (`Convolution Type 2`)
is split into contributions from the Matsubara, retarded, left-mixing,
and lesser components:

```math
\begin{equation}
C^{M}(\tau) = \int^{\beta}_0 d\bar{\tau}\
    A^{M} (\tau - \bar{\tau}) f(0^-) B^{M}(\bar{\tau}).
\end{equation}
```

```math
\begin{equation}
C^{R}(t,t') = \int^{t}_{t'} d\bar{t} \
    A^{R}(t,\bar{t}) f(\bar{t}) B^{R}(\bar{t},t').
\end{equation}
```

```math
\begin{equation}
C^{\rceil}(t,\tau) = \int^t_0 d\bar{t}\
    A^{R}(t,\bar{t}) f(\bar{t}) B^{\rceil} (\bar{t},\tau)
                   + \int^{\beta}_0 d\bar{\tau}\
    A^{\rceil}(t,\bar{\tau}) f(0^-) B^{M}(\bar{\tau} - \tau).
\end{equation}
```

```math
\begin{equation}
C^{<}(t,t') = \int^t_0 d\bar{t}\
    A^{R}(t,\bar{t}) f(\bar{t}) B^{<}(\bar{t},t')
            + \int^{t'}_0 d\bar{t}\
    A^{<}(t,\bar{t}) f(\bar{t}) B^{A}(\bar{t},t')
            -i \int^{\beta}_0 d\bar{\tau}\
    A^{\rceil}(t,\bar{\tau}) f(0^-) B^{\lceil}(\bar{\tau},t').

\end{equation}
```

---

**Assumption**

In the evaluation of the above integrals we make in general no assunption
on the hermitian properties of `A` and `B`. The integrals constitute
different contributions to the convolution, which we separate into the
Matsubara, retarded, left-mixing, and lesser components of a contour
function `C`. All the equations are obtained in a straightforward way from
the `Gregory integration` if the integration interval includes more than
``k + 1`` function values, and from the `polynomial integration` or the
`boundary convolution` otherwise.
=#

#=
### *Public Convolution API*
=#

"""
    convolution(C, A, B)

TO_BE_DONE
"""
function convolution(C, A, B)
    C = A * B
end

"""
    convolution(C, A, f, B)

TO_BE_DONE
"""
function convolution(C, A, f, B)
    C = A * f * B
end

"""
    convolution_time_step(C, A, B)

TO_BE_DONE
"""
function convolution_time_step(C, A, B)
    C = A * B
end

"""
    convolution_time_step(C, A, f, B)

TO_BE_DONE
"""
function convolution_time_step(C, A, f, B)
    C = A * f * B
end

#=
### *Convolution* : ``G^{M}`` *Component*
=#

#=
*Remarks* :

The evaluation of ``C^{M}(\tau)`` is implemented as follows:

```math
\begin{equation}
C^{M}(mh_\tau) = C^{M}_1[A,f,B](m) + C^{M}_2[A,f,B](m),
\end{equation}
```

where ``m = 0,\ \cdots,\ N_{\tau}`` (it means that the number of imaginary
time points is ``N_{\tau}+1``) and ``h_{\tau}`` means the interval in the
imaginary time axis (``\equiv \delta \tau``).

```math
\begin{equation}
C^{M}_1[A,f,B](m) = \int^{mh_{\tau}}_0 d\tau' \
    A^{M}(mh_{\tau} - \tau') f(0^-) B^{M}(\tau').
\end{equation}
```

```math
\begin{equation}
C^{M}_2[A,f,B](m) = \int^{\beta}_{mh_{\tau}} d\tau' \
    A^{M}(mh_{\tau} - \tau') f(0^-) B^{M}(\tau').
\end{equation}
```

Actually, we adopted the following equations:

```math
\begin{equation}
C^{M}_1[A,f,B](m) = h_{\tau} \sum^{k}_{j,l = 0}
    R^{(k)}_{m;j,l} A^{M}_j f_{-1} B^{M}_l, \quad m \le k,
\end{equation}
```

```math
\begin{equation}
C^{M}_1[A,f,B](m) = h_{\tau} \sum^{m}_{l = 0}
    w^{(k)}_{m,l} A^{M}_{m-l} f_{-1} B^{M}_l, \quad m > k.
\end{equation}
```

```math
\begin{equation}
C^{M}_2[A,f,B](m) = h_{\tau} \sum^{k}_{j,l = 0}
    R^{(k)}_{N_{\tau}-m; j,l} \xi
    A^{M}_{N_{\tau}-j} f_{-1} B^{M}_{N_{\tau} - l}, \quad m \ge N_{\tau} -k,
\end{equation}
```

```math
\begin{equation}
C^{M}_2[A,f,B](m) = h_{\tau} \sum^{N_{\tau} - m}_{l = 0}
    w^{(k)}_{N_{\tau}-m,l} \xi
    A^{M}_{N_{\tau}-l} f_{-1} B^{M}_{m+l}, \quad m < N_{\tau} - k.
\end{equation}
```

Note that ``A^{M}(\tau)`` at the values ``\tau \in [-\beta, 0]``
is obtained by using the periodicity property:

```math
\begin{equation}
A^{M}(\tau + \beta) = \xi A^{M}(\tau).
\end{equation}
```

We also associate fermions (bosons) with the negative (positive) sign
``\xi \equiv -1`` (``\xi \equiv 1``).

*References* :

Please see [`NESSi`] Sections `9` and `11` for more details.
=#

"""
    conv_mat(
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator,
        beta::F64,
        sign::I64
    ) where {T}

Try to calculate convolution between two Matsubara Green's functions,
i.e. `Cᴹ = Aᴹ ∗ Bᴹ`. Actually, Aᴹ, Bᴹ, and Cᴹ are defined at imaginary
time axis, instead of Matsubara axis.

### Arguments
* A -> Matsubara Green's function, Aᴹ(τ).
* B -> Matsubara Green's function, Bᴹ(τ).
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Matsubara Green's function, Cᴹ(τ).

See also: [`conv_mat_mat_1`](@ref).
"""
function conv_mat(
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    beta::F64,
    sign::I64
) where {T}
    # Extract parameters
    ntau = getntau(C)

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert beta ≥ 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate δτ
    δτ = convert(T, beta / (ntau - 1))

    # Evaluate the convolution
    for m = 1:ntau
        conv_mat_mat_1(m, C, A, B, I, sign)
    end

    # Multiplied by δτ
    smul!(C, δτ)
end

#=
*Remarks* : *Matsubara Integral 1*

The Matsubara integral 1 reads:

```math
\begin{equation}
C(\tau) = \int^{\beta}_0 d\tau'\ A(\tau - \tau') B(\tau').
\end{equation}
```

The objects `A`, `B`, and `C` are of type Matsubara Green's function,
i.e., Matsubara component of contour-ordered Green's function. They are
anti-periodic, i.e.,

```math
\begin{equation}
A(\beta - \tau) = -A(-\tau),
\end{equation}
```
for ``0 \le \tau \le \beta``.

The following `conv_mat_mat_1()` and `conv_mat_mat_1p()` functions will
try to calculate similar integral.
=#

"""
    conv_mat_mat_1(
        m::I64,
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator,
        sign::I64
    ) where {T}

Try to calculate the Matsubara integral 1, i.e., convolution of A(τ-τ')
and B(τ'). The integral lower and upper limits are 0 and β, respectively.

### Arguments
* m -> Index for imaginary time points [current τ is (m-1)δτ].
* A -> Matsubara Green's function, A(τ-τ').
* B -> Matsubara Green's function, B(τ').
* I -> Struct for numerical integration.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Matsubara Green's function, C ≡ A ∗ B.

See also: [`conv_mat_mat_1p`](@ref).
"""
function conv_mat_mat_1(
    m::I64,
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    sign::I64
) where {T}
    # Extract parameters
    ntau = getntau(A)
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 1 ≤ m ≤ ntau
    @assert sign in (FERMI, BOSE)

    # Try to calculate the contributions from 0 to τ
    # Please refer to Eq.(105)-(106) in [NESSi]
    c₁ = similar(C[1])
    fill!(c₁, zero(T))
    #
    if m ≥ k + 1 # Usual Gregory integration
        ind = m
        for l = 1:m
            @. c₁ = c₁ + I.GIW[m-1,l-1] * A[ind] * B[l]
            ind = ind - 1
        end
    elseif m > 1 # Strange boundary correction
        for j = 1:k+1
            for l = 1:k+1
                @. c₁ = c₁ + I.BCW[m-2,j-1,l-1] * A[j] * B[l]
            end
        end
    end

    # Try to calculate the contributions from τ to β
    # Please refer to Eq.(107)-(108) in [NESSi]
    c₂ = similar(C[2])
    fill!(c₂, zero(T))
    #
    if ntau - m ≥ k # Usual Gregory integration
        inda = ntau
        indb = m
        indg = 0
        for l = m:ntau
            @. c₂ = c₂ + I.GIW[ntau-m,indg] * A[inda] * B[indb]
            inda = inda - 1
            indb = indb + 1
            indg = indg + 1
        end
    elseif ntau - m > 0 # Strange boundary correction
        inda = ntau
        for j = 1:k+1
            indb = ntau
            for l = 1:k+1
                @. c₂ = c₂ + I.BCW[ntau-m-1,j-1,l-1] * A[inda] * B[indb]
                indb = indb - 1
            end
            inda = inda - 1
        end
    end

    # Assemble the final results
    @. C[m] = c₁ + sign * c₂
end

"""
    conv_mat_mat_1p(
        m::I64,
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator
    ) where {T}

Try to calculate the Matsubara integral 1, i.e., convolution of A(τ-τ')
and B(τ'). The integral lower and upper limits are 0 and τ, respectively.

### Arguments
* m -> Index for imaginary time points [current τ is (m-1)δτ].
* A -> Matsubara Green's function, A(τ-τ').
* B -> Matsubara Green's function, B(τ').
* I -> Struct for numerical integration.

### Returns
* C -> Matsubara Green's function, C ≡ A ∗ B.

See also: [`conv_mat_mat_1`](@ref).
"""
function conv_mat_mat_1p(
    m::I64,
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator
) where {T}
    # Extract parameters
    ntau = getntau(A)
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 1 ≤ m ≤ ntau

    # We only calculate the contributions from 0 to τ
    # Please refer to Eq.(105)-(106) in [NESSi]
    c₁ = similar(C[1])
    fill!(c₁, zero(T))
    #
    if m ≥ k + 1 # Usual Gregory integration
        ind = m
        for l = 1:m
            @. c₁ = c₁ + I.GIW[m-1,l-1] * A[ind] * B[l]
            ind = ind - 1
        end
    elseif m > 1 # Strange boundary correction
        for j = 1:k+1
            for l = 1:k+1
                @. c₁ = c₁ + I.BCW[m-2,j-1,l-1] * A[j] * B[l]
            end
        end
    end

    # The contributions from τ to β are discarded.

    # Assemble the final results
    @. C[m] = c₁
end

#=
*Remarks* : *Matsubara Integral 2*

The Matsubara integral 2 reads:

```math
\begin{equation}
C(\tau) = \int^{\beta}_0 d\tau'\ A(\tau') B(\tau' - \tau).
\end{equation}
```

The objects `A`, `B`, and `C` are of type Matsubara Green's function,
i.e., Matsubara component of contour-ordered Green's function. They are
anti-periodic, i.e.,

```math
\begin{equation}
A(\beta - \tau) = -A(-\tau),
\end{equation}
```
for ``0 \le \tau \le \beta``.

The following `conv_mat_mat_2()` and `conv_mat_mat_2p()` functions will
try to calculate similar integral.
=#

"""
    conv_mat_mat_2(
        m::I64,
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator,
        sign::I64
    ) where {T}

Try to calculate the Matsubara integral 2, i.e., convolution of A(τ') and
B(τ'-τ). The integral lower and upper limits are 0 and β, respectively.

### Arguments
* m -> Index for imaginary time points [current τ is (m-1)δτ].
* A -> Matsubara Green's function, A(τ').
* B -> Matsubara Green's function, B(τ'-τ).
* I -> Struct for numerical integration.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Matsubara Green's function, C ≡ A ∗ B.

See also: [`conv_mat_mat_2p`](@ref).
"""
function conv_mat_mat_2(
    m::I64,
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    sign::I64
) where {T}
    # Extract parameters
    ntau = getntau(A)
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 1 ≤ m ≤ ntau
    @assert sign in (FERMI, BOSE)

    # Try to calculate the contributions from 0 to τ
    c₁ = similar(C[1])
    fill!(c₁, zero(T))
    #
    if m == 1
        # PASS
    elseif m < k + 1 # Strange boundary correction
        inda = 1
        for j = 1:k+1
            indb = ntau
            for l = 1:k+1
                @. c₁ = c₁ + I.BCW[m-2,l-1,j-1] * A[inda] * B[indb]
                indb = indb - 1
            end
            inda = inda + 1
        end
    else # Usual Gregory integration
        inda = m
        indb = ntau
        for l = 1:m
            @. c₁ = c₁ + I.GIW[m-1,l-1] * A[inda] * B[indb]
            inda = inda - 1
            indb = indb - 1
        end
    end

    # Try to calculate the contributions from τ to β
    c₂ = similar(C[2])
    fill!(c₂, zero(T))
    #
    if m == ntau
        # PASS
    elseif m > ntau - k # Strange boundary correction
        inda = ntau
        for l = 1:k+1
            for j = 1:k+1
                @. c₂ = c₂ + I.BCW[ntau-m-1,l-1,j-1] * A[inda] * B[j]
            end
            inda = inda - 1
        end
    elseif m > ntau - 2*k - 1 # Usual Gregory integration
        inda = m
        for l = 1:ntau-m+1
            @. c₂ = c₂ + I.GIW[ntau-m,l-1] * A[inda] * B[l]
            inda = inda + 1
        end
    else # Usual Gregory integration
        inda = m
        indb = 1
        for l = m:ntau
            @. c₂ = c₂ + I.GIW[ntau-m,ntau-l] * A[inda] * B[indb]
            inda = inda + 1
            indb = indb + 1
        end
    end

    # Assemble the final results
    @. C[m] = c₁ + sign * c₂
end

"""
    conv_mat_mat_2p(
        m::I64,
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator
    ) where {T}

Try to calculate the Matsubara integral 2, i.e., convolution of A(τ') and
B(τ'-τ). The integral lower and upper limits are τ and β, respectively.

### Arguments
* m -> Index for imaginary time points [current τ is (m-1)δτ].
* A -> Matsubara Green's function, A(τ').
* B -> Matsubara Green's function, B(τ'-τ).
* I -> Struct for numerical integration.

### Returns
* C -> Matsubara Green's function, C ≡ A ∗ B.

See also: [`conv_mat_mat_2`](@ref).
"""
function conv_mat_mat_2p(
    m::I64,
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator
) where {T}
    # Extract parameters
    ntau = getntau(A)
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 1 ≤ m ≤ ntau

    # The contributions from 0 to τ are discarded.

    # Try to calculate the contributions from τ to β
    c₂ = similar(C[2])
    fill!(c₂, zero(T))
    #
    if m == ntau
        # PASS
    elseif m > ntau - k # Strange boundary correction
        inda = ntau
        for l = 1:k+1
            for j = 1:k+1
                @. c₂ = c₂ + I.BCW[ntau-m-1,l-1,j-1] * A[inda] * B[j]
            end
            inda = inda - 1
        end
    elseif m > ntau - 2*k - 1 # Usual Gregory integration
        inda = m
        for l = 1:ntau-m+1
            @. c₂ = c₂ + I.GIW[ntau-m,l-1] * A[inda] * B[l]
            inda = inda + 1
        end
    else # Usual Gregory integration
        inda = m
        indb = 1
        for l = m:ntau
            @. c₂ = c₂ + I.GIW[ntau-m,ntau-l] * A[inda] * B[indb]
            inda = inda + 1
            indb = indb + 1
        end
    end

    # Assemble the final results
    @. C[m] = c₂
end

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
* Acc -> Complex conjugate to A.
* B -> Retarded component of contour-ordered Green's function, Bᴿ(t,t').
* Bcc -> Complex conjugate to B.
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
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert h > 0.0

    # Create Element{T}, which is a matrix whose size is (ndim1,ndim2).
    elem = similar(C[1,1])
    fill!(elem, zero(T))

    # Create VecArray{T}, whose size is indeed (n,).
    # It is used to save the intermediate results.
    result = VecArray{T}(undef, n)
    for i = 1:n
        result[i] = copy(elem)
    end

    # Evaluate the retarded convolution at a given time step
    if n ≥ k + 1

    #
    # For n > k, n - m > k case
    #
    # See [NESSi] Eq. (110a)
    #

        for j = 1:n
            for m = 1:j
                weight = I.GIW[n-m,n-j] * h
                #
                atmp = A[n,j]
                btmp = B[j,m]
                #
                @. result[m] = result[m] + weight * atmp * btmp
            end
        end

    #
    # For n > k, n - m ≤ k case
    #
    # See [NESSi] Eq. (110b)
    #

        for j = 1:k
            for m = n-j+1:n
                weight = I.GIW[n-m,j] * h
                #
                # Special treatment for the \tilde{B}^{R}_{n-j,m} term
                atmp = A[n,n-j]
                btmp = -conj(Bcc[m,n-j])
                #
                @. result[m] = result[m] + weight * atmp * btmp
            end
        end

    #
    # For n ≤ k case
    #
    # See [NESSi] Eq. (110c)
    #

    else

        for m = 1:n
            for j = 1:k+1
                weight = I.XIW[m-1,n-1,j-1] * h
                #
                # Treat \tilde{A}^{R} term
                if n ≥ j
                    atmp = A[n,j]
                else
                    atmp = -conj(Acc[j,n])
                end
                #
                # Treat \tilde{B}^{R} term
                if j ≥ m
                    btmp = B[j,m]
                else
                    btmp = -conj(Bcc[m,j])
                end
                #
                @. result[m] = result[m] + weight * atmp * btmp
            end
        end

    end

    # Write the intermediate results into C
    for m = 1:n
        @. C[n,m] = result[m]
    end
end

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
* Acc -> Complex conjugate to A.
* B -> Left-mixing component of contour-ordered Green's function, B^⌉(t',τ).
* Bcc -> Complex conjugate to B.
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
    ntau = getntau(C)
    k = I.k

    # Sanity check
    @assert getntime(A) == getntime(B)
    @assert getntime(A) == getntime(C)
    @assert iscompatible(B, C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert h > 0.0

    # Evaluate the upper limit for summation
    n₁ = (n - 1) > k ? (n - 1) : k
    n₁ = n₁ + 1

    # Create Element{T}, which is a matrix whose size is (ndim1,ndim2).
    elem = similar(C[1,1])
    fill!(elem, zero(T))

    # Create VecArray{T}, whose size is indeed (ntau,).
    result = VecArray{T}(undef, ntau)
    for i = 1:ntau
        result[i] = copy(elem)
    end

    #
    # Evaluate the left-mixing convolution at a given time step
    #
    # See [NESSi] Eq. (111) - (112)
    #
    for j = 1:n₁
        weight = I.GIW[n-1,j-1] * h

        # Special treatment for the \tilde{A}^{R}_{n,j} term
        if n < j
            atmp = -conj(A[j,n])
        else
            atmp = A[n,j]
        end

        for m = 1:ntau
            btmp = B[j,m]
            @. result[m] = result[m] + weight * atmp * btmp
        end
    end

    # Write the intermediate results into C
    #
    # For the contributions from the C₂ and C₃, see conv_lmix_mat().
    for m = 1:ntau
        @. C[n,m] = C[n,m] + result[m]
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
    k = I.k

    # Sanity check
    @assert getntau(A) == getntau(B)
    @assert getntau(B) == getntau(C)
    @assert iscompatible(A, C)
    @assert ntime ≥ n ≥ 1
    @assert beta > 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate δτ
    δτ = convert(T, beta / (ntau - 1))
    
    # Allocate memory for c₂ and c₃.
    # Both of them are matrices, whose size is (ndim1,ndim2).
    c₂ = similar(C[n,1])
    c₃ = similar(C[n,2])

    #
    # Evaluate the left-mixing convolution at a given time step
    #
    # It is very similar to computing the Matsubara convolution.
    #
    # Please refer to conv_mat_mat_2()
    #
    for m = 1:ntau

        # Try to calculate the contributions from 0 to τ
        #
        # See [NESSi] Eq. (113) - (114)
        #
        # Reset the intermediate array
        fill!(c₂, zero(T))
        #
        if m == 1
            # PASS
        elseif m < k + 1 # Strange boundary correction
            inda = 1
            for j = 1:k+1
                indb = ntau
                for l = 1:k+1
                    @. c₂ = c₂ + I.BCW[m-2,l-1,j-1] * A[n,inda] * B[indb]
                    indb = indb - 1
                end
                inda = inda + 1
            end
        else # Usual Gregory integration
            inda = m
            indb = ntau
            for l = 1:m
                @. c₂ = c₂ + I.GIW[m-1,l-1] * A[n,inda] * B[indb]
                inda = inda - 1
                indb = indb - 1
            end
        end

        # Try to calculate the contributions from τ to β
        #
        # See [NESSi] Eq. (115) - (116)
        #
        # Reset the intermediate array
        fill!(c₃, zero(T))
        #
        if m == ntau
            # PASS
        elseif m > ntau - k # Strange boundary correction
            inda = ntau
            for l = 1:k+1
                for j = 1:k+1
                    @. c₃ = c₃ + I.BCW[ntau-m-1,l-1,j-1] * A[n,inda] * B[j]
                end
                inda = inda - 1
            end
        elseif m > ntau - 2*k - 1 # Usual Gregory integration
            inda = m
            for l = 1:ntau-m+1
                @. c₃ = c₃ + I.GIW[ntau-m,l-1] * A[n,inda] * B[l]
                inda = inda + 1
            end
        else # Usual Gregory integration
            inda = m
            indb = 1
            for l = m:ntau
                @. c₃ = c₃ + I.GIW[ntau-m,ntau-l] * A[n,inda] * B[indb]
                inda = inda + 1
                indb = indb + 1
            end
        end

        # Assemble the final results
        @. C[n,m] = C[n,m] + ( c₂ + sign * c₃ ) * δτ

    end
end

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
C^{<}_{1}[A,f,B](m,n) = \int^{nh}_{0} d\bar{t}~
    A^{R}(nh,\bar{t}) f(\bar{t}) B^{<}(\bar{t},mh).
\end{equation}
```

```math
\begin{equation}
C^{<}_{2}[A,f,B](m,n) = \int^{mh}_{0} d\bar{t}~
    A^{<}(nh,\bar{t}) f(\bar{t}) B^{A}(\bar{t},mh).
\end{equation}
```

```math
\begin{equation}
C^{<}_{3}[A,f,B](m,n) = -i \int^{\beta}_{0} d\tau~
    A^{\rceil}(nh,\tau) f(0^{-}) B^{\lceil}(\tau,mh).
\end{equation}
```

Actually, we implement the following equations:

```math
\begin{equation}
C^{<}_{1}[A,f,B](m,n) = h\sum^{n}_{j=0}~
    w^{(k)}_{n,j} A^{R}_{n,j} f_{j} B^{<}_{j,m}, \quad n > k.
\end{equation}
```

```math
\begin{equation}
C^{<}_{1}[A,f,B](m,n) = h\sum^{k}_{j=0}~
    w^{(k)}_{n,j} \tilde{A}^{R}_{n,j} f_{j} B^{<}_{j,m}, \quad n \le k.
\end{equation}
```

```math
\begin{equation}
C^{<}_{2}[A,f,B](m,n) = h\sum^{m}_{j=0}~
    w^{(k)}_{m,j} A^{<}_{n,j} f_{j} B^{A}_{j,m}, \quad m > k.
\end{equation}
```

```math
\begin{equation}
C^{<}_{2}[A,f,B](m,n) = h\sum^{k}_{j=0}~
    w^{(k)}_{m,j} A^{<}_{n,j} f_{j} \tilde{B}^{A}_{j,m}, \quad m \le k.
\end{equation}
```

```math
\begin{equation}
C^{<}_{3}[A,f,B](m,n) = -i h_{\tau} \sum^{N_{\tau}}_{j=0}~
    w^{(k)}_{N_{\tau},j} A^{\rceil}_{n,j} f_{-1} B^{\lceil}_{j,m}.
\end{equation}
```

*References* :

Please see [`NESSi`] Sections `9` and `11` for more details.
=#

"""
"""
function conv_ret_less(
    n::I64,
    C::Gˡᵉˢˢ{T}, A::Gʳᵉᵗ{T}, B::Gˡᵉˢˢ{T},
    I::Integrator,
    h::F64
) where {T}
    # Extract parameters
    k = I.k

    # Sanity check

    n₁ = (n - 1) > k ? (n - 1) : k
    n₁ = n₁ + 1

    #@show n, k, n₁

    # Create Element{T}
    elem = Element{T}(undef, getdims(C))
    fill!(elem, zero(T))

    # Create VecArray{T}, whose size is indeed (n₁,).
    btmp = VecArray{T}(undef, n₁)
    for i = 1:n₁
        btmp[i] = copy(elem)
    end

    result = VecArray{T}(undef, n)
    for i = 1:n
        result[i] = copy(elem)
    end

    for m = 1:n₁
        if m ≤ n
            @. btmp[m] = B[m,n]
        else
            @. btmp[m] = -conj(B[n,m])
        end
    end

    for j = 1:n
        for m = 1:j
            weight = I.GIW[j-1,m-1] * h
            atmp = A[j,m]
            @. result[j] = result[j] + weight * atmp * btmp[m]
        end

        if j - 1 < k
            for m = j+1:k+1
                weight = I.GIW[j-1,m-1] * h
                atmp = -conj(A[m,j])
                @. result[j] = result[j] + weight * atmp * btmp[m]
            end
        end

        @show j, result[j]
    end

end

"""
"""
function conv_less_adv(
    n::I64,
    C::Gˡᵉˢˢ{T}, A::Gˡᵉˢˢ{T}, B::Gʳᵉᵗ{T},
    I::Integrator,
    h::F64
) where {T}
    # Extract parameters
    k = I.k

    # Sanity

    n₁ = (n - 1) > k ? (n - 1) : k
    n₁ = n₁ + 1
    #@show n, k, n₁

    # Create Element{T}
    elem = Element{T}(undef, getdims(C))
    fill!(elem, zero(T))

    # Create VecArray{T}, whose size is indeed (n₁,).
    btmp = VecArray{T}(undef, n₁)
    for i = 1:n₁
        btmp[i] = copy(elem)
    end

    result = VecArray{T}(undef, n₁)
    for i = 1:n₁
        result[i] = copy(elem)
    end

    for m = 1:n₁
        weight = I.GIW[n-1,m-1] * h
        if m ≤ n
            @. btmp[m] = conj(B[n,m]) * weight
        else
            @. btmp[m] = -B[m,n] * weight
        end
        #@show m, weight, btmp[m]
    end

    for j = 1:n₁
        for m = 1:j-1
            #@show j, m
            atmp = -conj(A[m,j])
            #@show j, m, atmp
            @. result[j] = result[j] + atmp * btmp[m]
        end
        #@show j, result[j]
    end

    for m = 1:n₁
        jmax = min(n₁, m)
        #@show m, jmax
        for j = 1:jmax
            @. result[j] = result[j] + A[j,m] * btmp[m]
            #@show m, j, A[j,m], btmp[m]
        end
    end

    #@show n, n₁, result[1:n₁]
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
* Acc -> Complex conjugate to A.
* B -> Left-mixing component of contour-ordered Green's function, B^⌉(t',τ).
* Bcc -> Complex conjugate to B.
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
    k = I.k

    # Sanity check
    @assert getntime(A) == getntime(C)
    @assert iscompatible(A, B)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert beta > 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate the upper limit for summation
    n₁ = (n - 1) > k ? (n - 1) : k
    n₁ = n₁ + 1

    # Evaluate δτ
    δτ = convert(T, beta / (ntau - 1))

    # Create Element{T}, which is a matrix whose size is (ndim1,ndim2).
    elem = similar(C[1,1])
    fill!(elem, zero(T))

    # Create VecArray{T}, whose size is indeed (n₁,).
    # It is used to save intermediate results.
    result = VecArray{T}(undef, n₁)
    for i = 1:n₁
        result[i] = copy(elem)
    end

    #
    # Try to calculate the right-mixing component of contour-ordered
    # Green's function at first.
    #
    # See [NESSi] Eq. (18d).
    #
    btmp = VecArray{T}(undef, ntau)
    for m = 1:ntau
        btmp[m] = conj(B[n,ntau-m+1]) * (-sign)
    end

    for j = 1:n₁
        for m = 1:ntau
            weight = I.GIW[ntau - 1, m - 1]
            @. result[j] = result[j] + weight * A[j,m] * btmp[m]
        end
        @show j, result[j] * (-δτ * im)
    end
end
