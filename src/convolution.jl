#
# File: convolution.jl
#
# Implement convolution for two contour Green's functions.
#

struct Integrator
    k   :: I64
    PIW :: PolynomialInterpolationWeights
    PDW :: PolynomialDifferentiationWeights
    XIW :: PolynomialIntegrationWeights
    BDW :: BackwardDifferentiationWeights
    GIW :: GregoryIntegrationWeights
    BCW :: BoundaryConvolutionWeights
end

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
### *Convolution*

The most general convolution of two contour Green's functions `A` and
`B` and a time-dependent function `f` is given by the integral:

```math
\begin{equation}
C(t,t') = \int_{\mathcal{C}} d \bar{t}\
    A(t,\bar{t}) f(\bar{t}) B(\bar{t},t').
\end{equation}
```

In the evaluation of this integral we make in general no assunption on
the hermitian properties of `A` and `B`.

The integrals constitute different contributions to the convolution,
which we separate into the Matsubara, retarded, left-mixing, and lesser
components of a contour function `C`. All the equations are obtained in
a straightforward way from the Gregory integration if the integration
interval includes more than ``k + 1`` function values, and from the
polynomial integration or the boundary convolution otherwise.

### *Langreth Rules*

Using the Langreth rules, the convolution integral is split into
contributions from the Matsubara, retarded, left-mixing, and lesser
components:

```math
\begin{equation}
C^{M}(\tau) = \int^{\beta}_0 d\tau'\
    A^{M} (\tau - \tau') f(0^-) B^{M}(\tau').
\end{equation}
```

```math
\begin{equation}
C^{R}(t,t') = \int^{t}_{t'} d\bar{t} \
    A^{R}(t, \bar{t}) f(\bar{t}) B^{B}(\bar{t}, t').
\end{equation}
```

```math
\begin{equation}
C^{\rceil}(t,\tau) = \int^t_0 d\bar{t}\
    A^{R}(t, \bar{t}) f(\bar{t}) B^{\rceil} (\bar{t}, \tau)
                   + \int^{\beta}_0 d\tau\
    A^{\rceil}(t,\tau') f(0^-) B^{M}(\tau' - \tau).
\end{equation}
```

```math
\begin{equation}
C^{<}(t,t') = \int^t_0 d\bar{t}\
    A^{R}(t, \bar{t}) f(\bar{t}) B^{<}(\bar{t}, t')
            + \int^{t'}_0 d\bar{t}
    A^{<}(t, \bar{t}) f(\bar{t}) B^{A}(\bar{t}, t')
            -i \int^{\beta}_0 d\tau\
    A^{\rceil}(t,\tau') f(0^-) B^{\lceil}(\tau, t').

\end{equation}
```
=#

#=
### *Driver Functions*
=#

function Convolution()
end

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

where ``m = 0,\ \cdots,\ N_{\tau}``.

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
=#

function c_mat()
end

"""
    c_mat_mat_1()

Try to calculate.
"""
function c_mat_mat_1(m::I64, C::CnMatM{T}, A::CnMatM{T}, B::CnMatM{T}, I::Integrator, sig::I64) where {T}
    # Extract parameters
    ntau = A.ntau
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 1 ≤ m ≤ ntau
    @assert sig in (-1, 1)

    # Try to calculate the contributions from 0 to τ
    c1 = similar(C[1])
    fill!(c1, zero(T))
    #
    if m ≥ k + 1
        for j = 1:m
            #@show j, A[m-j+1], B[j], I.GIW[m-1,j-1]
            @. c1 = c1 + I.GIW[m-1,j-1] * A[m-j+1] * B[j]
        end
    elseif m > 1
        for l = 1:k+1
            for j = 1:k+1
                #@show l, j, I.BCW[m-2,l-1,j-1]
                @. c1 = c1 + I.BCW[m-2,l-1,j-1] * A[l] * B[j]
            end
        end
    end
    #@show c1

    # Try to calculate the contributions from τ to β
    c2 = similar(C[2])
    fill!(c2, zero(T))
    #
    if ntau - m ≥ k
        for j = m:ntau
            #@show j, I.GIW[ntau-m,ntau-j], A[ntau-(j-m)], B[j]
            @. c2 = c2 + I.GIW[ntau-m,ntau-j] * A[ntau-(j-m)] * B[j]
        end
    elseif ntau - m > 0
        for l = 1:k+1
            for j = 1:k+1
                #@show l, j, I.BCW[ntau-m-1,l-1,j-1], A[ntau-l+1], B[ntau-j+1]
                @. c2 = c2 + I.BCW[ntau-m-1,l-1,j-1] * A[ntau-l+1] * B[ntau-j+1]
            end
        end
    end
    #@show c2

    # Assemble the final results
    @. C[m] = c1 + sig * c2
end

"""
    c_mat_mat_2()

Try to calculate.
"""
function c_mat_mat_2(m::I64, C::CnMatM{T}, A::CnMatM{T}, B::CnMatM{T}, I::Integrator, sig::I64) where {T}
    # Extract parameters
    ntau = A.ntau
    k = I.k

    # Sanity check
    @assert iscompatible(A, B)
    @assert iscompatible(B, C)
    @assert 1 ≤ m ≤ ntau
    @assert sig in (-1, 1)

    # Try to calculate the contributions from 0 to τ
    c1 = similar(C[1])
    fill!(c1, zero(T))
    #
    if m == 1
        # PASS
    elseif m < k + 1
        for j = 1:k+1
            for l = 1:k+1
                #@show l, j, I.BCW[m-2,l-1,j-1], A[j], B[ntau-l+1]
                @. c1 = c1 + I.BCW[m-2,l-1,j-1] * A[j] * B[ntau-l+1]
            end
        end
    else
        for l = 1:m
            #@show l, I.GIW[m-1,l-1], A[m-l+1], B[ntau-l+1]
            @. c1 = c1 + I.GIW[m-1,l-1] * A[m-l+1] * B[ntau-l+1]
        end
    end
    #@show c1

    # Try to calculate the contributions from τ to β
    c2 = similar(C[2])
    fill!(c2, zero(T))
    #
    if m == ntau
        # PASS
    elseif m > ntau - k
        for l = 1:k+1
            for j = 1:k+1
                #@show l, j, I.BCW[ntau-m-1,l-1,j-1], A[ntau-l+1], B[j]
                @. c2 = c2 + I.BCW[ntau-m-1,l-1,j-1] * A[ntau-l+1] * B[j]
            end
        end
    elseif m > ntau - 2*k + 1
        for l = 1:ntau-m+1
            #@show l, I.GIW[ntau-m,l-1], A[m+l-1], B[l]
            @. c2 = c2 + I.GIW[ntau-m,l-1] * A[m+l-1] * B[l]
        end
    else
        for l = m:ntau
            #@show l, I.GIW[ntau-m,ntau-l], A[l], B[l-m+1]
            @. c2 = c2 + I.GIW[ntau-m,ntau-l] * A[l] * B[l-m+1]
        end
    end
    #@show c2

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
    c_tstp_ret

Try to calculate.
"""
function c_tstp_ret(n::I64, C::CnFunM{T}, A::CnFunM{T}, Acc::CnFunM{T}, B::CnFunM{T}, Bcc::CnFunM{T}, I::Integrator, h::T) where {T}
    # Extract parameters
    k = I.k

    # Sanity check
    @assert getdims(A) == getdims(Acc)
    @assert getdims(B) == getdims(Bcc)
    @assert getntau(A) ≥ n
    @assert getntau(B) ≥ n
    @assert getntau(C) ≥ n

    # Create Element{T}
    element = fill(zero(T), getdims(C))

    # Create VecArray{T}, whose size is indeed (n+1,).
    result = VecArray{T}(undef, n + 1)
    for i = 1:n+1
        result[i] = copy(element)
    end

    if n - 1 ≥ k
        for m = 1:n
            for j = 1:m-k
            end
        end
    else

    end

end

#=
### *Convolution* : ``G^{⌉}`` *Component*
=#

function C_tstp_lmix()
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
