#
# Project : Lavender
# Source  : langreth.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/12/27
#

#=
*Remarks* : *Convolution*

**Convolution Type 1 : C = A ⋆ B**

The convolution of two correlators ``A(t,t')`` and ``B(t,t')`` reads

```math
\begin{equation}
C(t,t') = [A \ast B](t,t')
        = \int_{\mathcal{C}} d\bar{t}\
          A(t,\bar{t}) B(\bar{t},t').
\end{equation}
```

It is one of the most basic operations on the contour ``\mathcal{C}``.

**Convolution Type 2 : C = A ⋆ f ⋆ B**

The most general convolution of two contour-ordered Green's functions `A`
and `B` and a time-dependent function `f` is given by the integral:

```math
\begin{equation}
C(t,t') = [A \ast B](t,t')
        = \int_{\mathcal{C}} d\bar{t}\
          A(t,\bar{t}) f(\bar{t}) B(\bar{t},t').
\end{equation}
```

---

**Langreth Rules 1 : C = A ⋆ B**

Using the Langreth rules, the convolution integral (`Convolution Type 1`)
is split into contributions from the Matsubara, retarded, left-mixing,
and lesser components:

```math
\begin{equation}
C^{M}(\tau) = \int^{\beta}_0 d\bar{\tau}\
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
C^{\rceil}(t,\tau) = \int^t_0 d\bar{t}\
    A^{R}(t,\bar{t}) B^{\rceil} (\bar{t},\tau)
                   + \int^{\beta}_0 d\bar{\tau}\
    A^{\rceil}(t,\bar{\tau}) B^{M}(\bar{\tau} - \tau).
\end{equation}
```

```math
\begin{equation}
C^{<}(t,t') = \int^t_0 d\bar{t}\
    A^{R}(t,\bar{t}) B^{<}(\bar{t},t')
            + \int^{t'}_0 d\bar{t}\
    A^{<}(t,\bar{t}) B^{A}(\bar{t},t')
            -i \int^{\beta}_0 d\bar{\tau}\
    A^{\rceil}(t,\bar{\tau}) B^{\lceil}(\bar{\tau},t').

\end{equation}
```

**Langreth Rules 2 : C = A ⋆ f ⋆ B**

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
the Gregory integration if the integration interval includes more than
``k + 1`` function values, and from the polynomial integration or the
boundary convolution otherwise.
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
time points is ``N_{\tau}+1``).

```math
\begin{equation}
C^{M}_1[A,f,B](m) = \int^{mh_{\tau}}_0 d\tau'\
    A^{M}(mh_{\tau} - \tau') f(0^-) B^{M}(\tau').
\end{equation}
```

```math
\begin{equation}
C^{M}_2[A,f,B](m) = \int^{\beta}_{mh_{\tau}} d\tau'\
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
    R^{(k)}_{N_{\tau}-m; j,l} \xi A^{M}_{N_{\tau}-j} f_{-1} B^{M}_{N_{\tau} - l}, \quad m \ge N_{\tau} -k,
\end{equation}
```

```math
\begin{equation}
C^{M}_2[A,f,B](m) = h_{\tau} \sum^{N_{\tau} - m}_{l = 0}
    w^{(k)}_{N_{\tau}-m,l} \xi A^{M}_{N_{\tau}-l} f_{-1} B^{M}_{m+l}, \quad m < N_{\tau} - k.
\end{equation}
```

Note that ``A^{M}(\tau)`` at the values ``\tau \in [-\beta, 0]``
is obtained by using the periodicity property

```math
\begin{equation}
A^{M}(\tau + \beta) = \xi A^{M}(\tau).
\end{equation}
```

*References* :

Please see [`NESSi`] Sections `9` and `11` for more details.
=#

"""
    conv_mat()

Try to calculate.

### Arguments
### Returns

See also: [`conv_mat_mat_1`](@ref).
"""
function conv_mat(
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    beta::F64,
    sig::I64
) where {T}
    ntau = getntau(C)
    δτ = C64(beta / (ntau - 1))
    for m = 1:ntau
        conv_mat_mat_1(m, C, A, B, I, sig)
    end
    smul!(C, δτ)
end

#=
*Remarks* : *Matsubara Integral 1*

The Matsubara integral 1 reads:

```math
\begin{equation}
C(\tau) = \int^{\beta}_0 d\tau'\ A(\tau - \tau') B(\tau') 
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
        sig::I64 
    )

Try to calculate the Matsubara integral 1, i.e., convolution of A(τ-τ')
and B(τ'). The integral lower and upper limits are 0 and β, respectively.

### Arguments
* m -> Index for imaginary time points [current τ is (m-1)δτ].
* A -> Matsubara Green's function, A(τ-τ').
* B -> Matsubara Green's function, B(τ').
* I -> A numerical integrator.
* sig -> Set `sig = -1` for fermions or `sig = +1` for bosons.

### Returns
* C -> Matsubara Green's function, C ≡ A ∗ B.

See also: [`conv_mat_mat_1p`](@ref).
"""
function conv_mat_mat_1(
    m::I64,
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    sig::I64
) where {T}
    # Extract parameters
    ntau = A.ntau
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 1 ≤ m ≤ ntau
    @assert sig in (FERMI, BOSE)

    # Try to calculate the contributions from 0 to τ
    # Please refer to Eq.(105)-(106) in [NESSi]
    c1 = similar(C[1])
    fill!(c1, zero(T))
    #
    if m ≥ k + 1 # Usual Gregory integration
        ind = m
        for l = 1:m
            @. c1 = c1 + I.GIW[m-1,l-1] * A[ind] * B[l]
            ind = ind - 1
        end
    elseif m > 1 # Strange boundary correction
        for j = 1:k+1
            for l = 1:k+1
                @. c1 = c1 + I.BCW[m-2,j-1,l-1] * A[j] * B[l]
            end
        end
    end

    # Try to calculate the contributions from τ to β
    # Please refer to Eq.(107)-(108) in [NESSi]
    c2 = similar(C[2])
    fill!(c2, zero(T))
    #
    if ntau - m ≥ k # Usual Gregory integration
        inda = ntau
        indb = m
        indg = 0
        for l = m:ntau
            @. c2 = c2 + I.GIW[ntau-m,indg] * A[inda] * B[indb]
            inda = inda - 1
            indb = indb + 1
            indg = indg + 1
        end
    elseif ntau - m > 0 # Strange boundary correction
        inda = ntau
        for j = 1:k+1
            indb = ntau
            for l = 1:k+1
                @. c2 = c2 + I.BCW[ntau-m-1,j-1,l-1] * A[inda] * B[indb]
                indb = indb - 1
            end
            inda = inda - 1
        end
    end

    # Assemble the final results
    @. C[m] = c1 + sig * c2
end

"""
    conv_mat_mat_1p(
        m::I64,
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator
    )

Try to calculate the Matsubara integral 1, i.e., convolution of A(τ-τ')
and B(τ'). The integral lower and upper limits are 0 and τ, respectively.

### Arguments
* m -> Index for imaginary time points [current τ is (m-1)δτ].
* A -> Matsubara Green's function, A(τ-τ').
* B -> Matsubara Green's function, B(τ').
* I -> A numerical integrator.

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
    ntau = A.ntau
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 1 ≤ m ≤ ntau

    # We only calculate the contributions from 0 to τ
    # Please refer to Eq.(105)-(106) in [NESSi]
    c1 = similar(C[1])
    fill!(c1, zero(T))
    #
    if m ≥ k + 1 # Usual Gregory integration
        ind = m
        for l = 1:m
            @. c1 = c1 + I.GIW[m-1,l-1] * A[ind] * B[l]
            ind = ind - 1
        end
    elseif m > 1 # Strange boundary correction
        for j = 1:k+1
            for l = 1:k+1
                @. c1 = c1 + I.BCW[m-2,j-1,l-1] * A[j] * B[l]
            end
        end
    end

    # The contributions from τ to β are discarded.

    # Assemble the final results
    @. C[m] = c1
end

#=
*Remarks* : *Matsubara Integral 2*

The Matsubara integral 2 reads:

```math
\begin{equation}
C(\tau) = \int^{\beta}_0 d\tau'\ A(\tau') B(\tau' - \tau)
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
        sig::I64
    )

Try to calculate the Matsubara integral 2, i.e., convolution of A(τ') and
B(τ'-τ). The integral lower and upper limits are 0 and β, respectively.

### Arguments
* m -> Index for imaginary time points [current τ is (m-1)δτ].
* A -> Matsubara Green's function, A(τ').
* B -> Matsubara Green's function, B(τ'-τ).
* I -> A numerical integrator.
* sig -> Set `sig = -1` for fermions or `sig = +1` for bosons.

### Returns
* C -> Matsubara Green's function, C ≡ A ∗ B.

See also: [`conv_mat_mat_2p`](@ref).
"""
function conv_mat_mat_2(
    m::I64,
    C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    sig::I64
) where {T}
    # Extract parameters
    ntau = A.ntau
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 1 ≤ m ≤ ntau
    @assert sig in (FERMI, BOSE)

    # Try to calculate the contributions from 0 to τ
    c1 = similar(C[1])
    fill!(c1, zero(T))
    #
    if m == 1
        # PASS
    elseif m < k + 1 # Strange boundary correction
        inda = 1
        for j = 1:k+1
            indb = ntau
            for l = 1:k+1
                @. c1 = c1 + I.BCW[m-2,l-1,j-1] * A[inda] * B[indb]
                indb = indb - 1
            end
            inda = inda + 1
        end
    else # Usual Gregory integration
        inda = m
        indb = ntau
        for l = 1:m
            @. c1 = c1 + I.GIW[m-1,l-1] * A[inda] * B[indb]
            inda = inda - 1
            indb = indb - 1
        end
    end

    # Try to calculate the contributions from τ to β
    c2 = similar(C[2])
    fill!(c2, zero(T))
    #
    if m == ntau
        # PASS
    elseif m > ntau - k # Strange boundary correction
        inda = ntau
        for l = 1:k+1
            for j = 1:k+1
                @. c2 = c2 + I.BCW[ntau-m-1,l-1,j-1] * A[inda] * B[j]
            end
            inda = inda - 1
        end
    elseif m > ntau - 2*k - 1 # Usual Gregory integration
        inda = m
        for l = 1:ntau-m+1
            @. c2 = c2 + I.GIW[ntau-m,l-1] * A[inda] * B[l]
            inda = inda + 1
        end
    else # Usual Gregory integration
        inda = m
        indb = 1
        for l = m:ntau
            @. c2 = c2 + I.GIW[ntau-m,ntau-l] * A[inda] * B[indb]
            inda = inda + 1
            indb = indb + 1
        end
    end

    # Assemble the final results
    @. C[m] = c1 + sig * c2
end

"""
    conv_mat_mat_2p(
        m::I64,
        C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator
    )

Try to calculate the Matsubara integral 2, i.e., convolution of A(τ') and
B(τ'-τ). The integral lower and upper limits are τ and β, respectively.

### Arguments
* m -> Index for imaginary time points [current τ is (m-1)δτ].
* A -> Matsubara Green's function, A(τ').
* B -> Matsubara Green's function, B(τ'-τ).
* I -> A numerical integrator.

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
    ntau = A.ntau
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 1 ≤ m ≤ ntau

    # The contributions from 0 to τ are discarded.

    # Try to calculate the contributions from τ to β
    c2 = similar(C[2])
    fill!(c2, zero(T))
    #
    if m == ntau
        # PASS
    elseif m > ntau - k # Strange boundary correction
        inda = ntau
        for l = 1:k+1
            for j = 1:k+1
                @. c2 = c2 + I.BCW[ntau-m-1,l-1,j-1] * A[inda] * B[j]
            end
            inda = inda - 1
        end
    elseif m > ntau - 2*k - 1 # Usual Gregory integration
        inda = m
        for l = 1:ntau-m+1
            @. c2 = c2 + I.GIW[ntau-m,l-1] * A[inda] * B[l]
            inda = inda + 1
        end
    else # Usual Gregory integration
        inda = m
        indb = 1
        for l = m:ntau
            @. c2 = c2 + I.GIW[ntau-m,ntau-l] * A[inda] * B[indb]
            inda = inda + 1
            indb = indb + 1
        end
    end

    # Assemble the final results
    @. C[m] = c2
end

#=
### *Convolution* : ``G^{R}`` *Component*

*Remarks* :

The evaluation of ``C^{R}`` at given time slice ``n`` is implemented
as follows:

```math
\begin{equation}
C^{R}(nh,mh) = C^{R}_1[A,f,B](n,m),
\end{equation}
```

where ``m = 0,\ \cdots,\ n``.

```math
\begin{equation}
C^{R}_1[A,f,B](n,m) = \int^{nh}_{mh} d\bar{t}\
    A^{R}(nh,\bar{t}) f(\bar{t}) B^{R}(\bar{t},mh).
\end{equation}
```

Actually, we implement the following equations:

```math
\begin{equation}
C^{R}_1[A,f,B](n,m) = h \sum^{n}_{j = m}
    w^{(k)}_{n-m,j-m} A^{R}_{n,j} f_j B^{R}_{j,m}, \quad n > k,\ n - m > k.
\end{equation}
```

```math
\begin{equation}
C^{R}_1[A,f,B](n,m) = h \sum^{k}_{j = 0}
    w^{(k)}_{n-m,j} A^{R}_{n,n-j} f_{n-j} \tilde{B}^{R}_{n-j,m}, \quad n > k,\ n - m \le k.
\end{equation}
```

```math
\begin{equation}
C^{R}_1[A,f,B](n,m) = h \sum^{k}_{j = 0}
    I^{(k)}_{m,n;j} \tilde{A}^{R}_{n,j} f_j \tilde{B}^{R}_{j,m}, \quad n \le k.
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
=#

"""
    conv_tstp_ret(
        n::I64,
        C::Gʳᵉᵗ{T},
        A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
        B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
        I::Integrator,
        h::F64
    )

Try to calculate.

### Arguments

### Returns

See also: [`convolution_time_step`](@ref).
"""
function conv_tstp_ret(
    n::I64,
    C::Gʳᵉᵗ{T},
    A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T},
    B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
    I::Integrator,
    h::F64
) where {T}
    # Extract parameters
    k = I.k

    # Sanity check
    @assert getdims(A) == getdims(Acc)
    @assert getdims(B) == getdims(Bcc)
    @assert getntime(A) ≥ n
    @assert getntime(B) ≥ n
    @assert getntime(C) ≥ n
    @assert n ≥ 1
    @assert h ≥ 0

    # Create Element{T}
    elem = Element{T}(undef, getdims(C))
    fill!(elem, zero(T))

    # Create VecArray{T}, whose size is indeed (n,).
    result = VecArray{T}(undef, n)
    for i = 1:n
        result[i] = copy(elem)
    end

    if n - 1 ≥ k

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
                btmp = -conj(B[m,n-j])
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
            for j = 0:k
                weight = I.XIW[m-1,n-1,j] * h
                #
                # Treat \tilde{A}^{R} term
                if n - 1 ≥ j
                    atmp = A[n,j+1]
                else
                    atmp = -conj(A[j+1,n])
                end
                #
                # Treat \tilde{B}^{R} term
                if j ≥ m-1
                    btmp = B[j+1,m]
                else
                    btmp = -conj(B[m,j+1])
                end
                #
                @. result[m] = result[m] + weight * atmp * btmp
            end
        end

    end

    @show n, result[1:n]
end

#=
### *Convolution* : ``G^{⌉}`` *Component*

*Remarks* :

The evaluation of ``C^{\rceil}`` at given time slice ``n`` is implemented
as follows:

```math
\begin{equation}
C^{\rceil}(nh,mh_{\tau}) =
    C^{\rceil}_1[A,f,B](n,m) +
    C^{\rceil}_2[A,f,B](n,m) +
    C^{\rceil}_3[A,f,B](n,m),
\end{equation}
```

where ``m = 0,\ \cdots,\ N_{\tau}``.

```math
\begin{equation}
C^{\rceil}_1[A,f,B](n,m) = \int^{nh}_{0} d\bar{t}~
    A^{R}(nh,\bar{t}) f(\bar{t}) B^{\rceil}(\bar{t},mh_{\tau}).
\end{equation}
```

Actually, we implement the following equations:

```math
\begin{equation}
C^{\rceil}_1[A,f,B](n,m) = h \sum^{n}_{j = 0}
    w^{(k)}_{n,j} A^{R}_{n,j} f_j B^{\rceil}_{j,m}, \quad n > k.
\end{equation}
```

```math
\begin{equation}
C^{\rceil}_1[A,f,B](n,m) = h \sum^{k}_{j = 0}
    w^{(k)}_{n,j} \tilde{A}^{R}_{n,j} f_j B^{\rceil}_{j,m}, \quad n \le k.
\end{equation}
```
=#

function conv_tstp_lmix()
end

function conv_ret_lmix()
end

function conv_lmix_mat()
end

#=
### *Convolution* : ``G^{<}`` *Component*
=#

function conv_tstp_less()
end

function conv_ret_less()
end

function conv_less_adv()
end

function conv_lmix_rmix()
end
