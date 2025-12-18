#
# Project : Lavender
# Source  : langreth.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/12/18
#

"""
    Integrator
"""
struct Integrator
    k   :: I64
    PIW :: PolynomialInterpolationWeights
    PDW :: PolynomialDifferentiationWeights
    XIW :: PolynomialIntegrationWeights
    BDW :: BackwardDifferentiationWeights
    GIW :: GregoryIntegrationWeights
    BCW :: BoundaryConvolutionWeights
end

"""
    Integrator(k::I64)
"""
function Integrator(k::I64)
    PIW = PolynomialInterpolationWeights(k)
    PDW = PolynomialDifferentiationWeights(k)
    XIW = PolynomialIntegrationWeights(k)
    BDW = BackwardDifferentiationWeights(k)
    GIW = GregoryIntegrationWeights(k)
    BCW = BoundaryConvolutionWeights(k)

    Integrator(k, PIW, PDW, XIW, BDW, GIW, BCW)
end

#=
*Remarks* : *Convolution*

**Convolution Type 1 : C = A ∗ B**

The convolution of two correlators ``A(t,t')`` and ``B(t,t')`` reads

```math
\begin{equation}
C(t,t') = [A \ast B](t,t')
        = \int_{\mathcal{C}} d\bar{t}\
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
        = \int_{\mathcal{C}} d\bar{t}\
          A(t,\bar{t}) f(\bar{t}) B(\bar{t},t').
\end{equation}
```

**Assumption**

In the evaluation of the above integrals we make in general no assunption
on the hermitian properties of `A` and `B`. The integrals constitute
different contributions to the convolution, which we separate into the
Matsubara, retarded, left-mixing, and lesser components of a contour
function `C`. All the equations are obtained in a straightforward way from
the Gregory integration if the integration interval includes more than
``k + 1`` function values, and from the polynomial integration or the
boundary convolution otherwise.

**Langreth Rules 1**

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

**Langreth Rules 2**

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
=#

#=
### *Driver Functions*
=#

"""
"""
function Convolution()
end

"""
"""
function ConvolutionTimeStep()
end

#=
### *Convolution* : ``G^{M}`` *Component*

*Remarks* :

The evaluation of ``C^{M}(\tau)`` is implemented as follows:

```math
\begin{equation}
C^{M}(mh_\tau) = C^{M}_1[A,f,B](m) + C^{M}_2[A,f,B](m),
\end{equation}
```

where ``m = 0,\ \cdots,\ N_{\tau}`` (It means that the number of imaginary
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
    c_mat()

Try to calculate.
"""
function c_mat(C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T}, I::Integrator, beta::F64, sig::I64) where {T}
    ntau = getntau(C)
    δτ = C64(beta / (ntau - 1))
    for m = 1:ntau
        c_mat_mat_1(m, C, A, B, I, sig)
    end
    smul!(C, δτ)
end

"""
    c_mat_mat_1()

Try to calculate.
"""
function c_mat_mat_1(m::I64, C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T}, I::Integrator, sig::I64) where {T}
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
    if m ≥ k + 1
        for j = 1:m
            @. c1 = c1 + I.GIW[m-1,j-1] * A[m-j+1] * B[j]
        end
    elseif m > 1
        for l = 1:k+1
            for j = 1:k+1
                @. c1 = c1 + I.BCW[m-2,l-1,j-1] * A[l] * B[j]
            end
        end
    end

    # Try to calculate the contributions from τ to β
    c2 = similar(C[2])
    fill!(c2, zero(T))
    #
    if ntau - m ≥ k
        for j = m:ntau
            @. c2 = c2 + I.GIW[ntau-m,ntau-j] * A[ntau-(j-m)] * B[j]
        end
    elseif ntau - m > 0
        for l = 1:k+1
            for j = 1:k+1
                @. c2 = c2 + I.BCW[ntau-m-1,l-1,j-1] * A[ntau-l+1] * B[ntau-j+1]
            end
        end
    end

    # Assemble the final results
    @. C[m] = c1 + sig * c2
end

"""
    c_mat_mat_2()

Try to calculate.
"""
function c_mat_mat_2(m::I64, C::Gᵐᵃᵗ{T}, A::Gᵐᵃᵗ{T}, B::Gᵐᵃᵗ{T}, I::Integrator, sig::I64) where {T}
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
    elseif m < k + 1
        for j = 1:k+1
            for l = 1:k+1
                @. c1 = c1 + I.BCW[m-2,l-1,j-1] * A[j] * B[ntau-l+1]
            end
        end
    else
        for l = 1:m
            @. c1 = c1 + I.GIW[m-1,l-1] * A[m-l+1] * B[ntau-l+1]
        end
    end

    # Try to calculate the contributions from τ to β
    c2 = similar(C[2])
    fill!(c2, zero(T))
    #
    if m == ntau
        # PASS
    elseif m > ntau - k
        for l = 1:k+1
            for j = 1:k+1
                @. c2 = c2 + I.BCW[ntau-m-1,l-1,j-1] * A[ntau-l+1] * B[j]
            end
        end
    elseif m > ntau - 2*k + 1
        for l = 1:ntau-m+1
            @. c2 = c2 + I.GIW[ntau-m,l-1] * A[m+l-1] * B[l]
        end
    else
        for l = m:ntau
            @. c2 = c2 + I.GIW[ntau-m,ntau-l] * A[l] * B[l-m+1]
        end
    end

    # Assemble the final results
    @. C[m] = c1 + sig * c2
end

#=
### *Convolution* : ``G^{R}`` *Component*

*Remarks* :

The evaluation of ``C^{R}`` at given time slice ``n`` is implemented
as follows:

```math
\begin{equation}
C^{R}(nh,mh) = C^{R}_1[A,f,B](n,m).
\end{equation}
```

where ``m = 0,\ \cdots,\ n``.

```math
\begin{equation}
C^{R}_1[A,f,B](n,m) = \int^{nh}_{mh} d\bar{t}\
    A^{R}(nh,\bar{t}) f(\bar{t}) B^{R}(\bar{t},mh).
\end{equation}
```

Actually, we implemented the following equations:

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
=#

"""
    c_tstp_ret()

Try to calculate.
"""
function c_tstp_ret(n::I64, C::Gʳᵉᵗ{T}, A::Gʳᵉᵗ{T}, Acc::Gʳᵉᵗ{T}, B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T}, I::Integrator, h::F64) where {T}
    # Extract parameters
    k = I.k

    # Sanity check
    @assert getdims(A) == getdims(Acc)
    @assert getdims(B) == getdims(Bcc)
    @assert getntime(A) ≥ n
    @assert getntime(B) ≥ n
    @assert getntime(C) ≥ n

    # Create Element{T}
    element = fill(zero(T), getdims(C))

    # Create VecArray{T}, whose size is indeed (n,).
    result = VecArray{T}(undef, n)
    for i = 1:n
        result[i] = copy(element)
    end

    if n - 1 ≥ k
        for m = 1:n
            ind = 0
            atmp = A[n,m] * h

            #for j = 1:m-k-1
            #    btmp = B[m,j]
            #    @show n, m, j, btmp
            #    @. result[j] = result[j] + atmp * btmp
            #end

            j1 = m - k
            if j1 < 1
                j1 = 1
            end
            #@show m, k, j1
            for j = j1:m
                ind = ind + 1
                btmp = B[m,ind]
                #@show n, m, j, I.GIW[n-j,n-m], atmp, btmp
                @. result[ind] = result[ind] + I.GIW[n-j,n-m] * atmp * btmp
            end
        end
        #
        #=
        for m = n-k:n-1
            atmp = A[n,m] * h
            for j = m+1:n
                weight = I.GIW[n-j, n-m]
                btmp = conj(B[j,m])
                @. result[j] = result[j] - weight * atmp * btmp
            end
        end
        =#
    else
        for j = 1:n
            for m = 0:k
                weight = I.XIW[j-1,n-1,m] * h
                if m ≥ j-1
                    btmp = B[m+1,j]
                else
                    btmp = conj(B[j,m+1])
                    weight = weight * (-1.0)
                end
                #
                if n - 1 ≥ m
                    atmp = A[n,m+1]
                else
                    atmp = conj(A[m+1,n])
                    weight = weight * (-1.0)
                end
                #
                @. result[j] = result[j] + weight * atmp * btmp
            end
        end
    end

    @show n, result[1:n]
end

#=
### *Convolution* : ``G^{⌉}`` *Component*
=#

function c_tstp_lmix()
end

function c_lmix_mat()
end

function c_ret_lmix()
end

#=
### *Convolution* : ``G^{<}`` *Component*
=#

function c_tstp_less()
end

function c_lmix_rmix()
end

function c_less_adv()
end

function c_ret_less()
end
