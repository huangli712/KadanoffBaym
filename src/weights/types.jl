#
# Project : Lavender
# Source  : types.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Abstract Type* : *AbstractWeights*
=#

"""
    AbstractWeights

Abstract weights for numerical interpolation, differentiation, and
integration. The purpose of this abstract type is to construct
the type system.
"""
abstract type AbstractWeights end


#=
### *Struct* : *PolynomialInterpolationWeights*
=#

#=
*Remarks* : *Polynomial Interpolation*

Consider a function ``y(t)`` which takes the values ``y_j`` at the points
``t_j = jh`` of an equidistant mesh ``j = 0, 1, \cdots, k`` with time
step ``h``. We denote the 𝑘th order polynomial ``\tilde{y}(t)`` passing
through the points ``y(jh) = y_j``,

```math
\begin{equation}
\tilde{y}(jh) = y_j, \quad j = 0, \cdots, k,
\end{equation}
```

by ``\mathcal{P}^{(k)}[y_0,\cdots,y_k](t)``. The interpolation can be
cast into the matrix form,

```math
\begin{equation}
\mathcal{P}^{(k)}[y_0,\cdots,y_k](t)
    =
    \sum^{k}_{a,l=0}h^{-a}t^{a}P^{(k)}_{al}y_l,
\end{equation}
```

where

```math
\begin{equation}
P^{(k)}_{al} = (M^{-1})_{al},
\end{equation}
```

with

```math
\begin{equation}
M_{ja} = j^a.
\end{equation}
```

*References* :

Please see [`NESSi`] Section `8.1` for more details.
=#

"""
    PolynomialInterpolationWeights

Weights for polynomial interpolation.
"""
struct PolynomialInterpolationWeights <: AbstractWeights
    # Order for interpolation. It starts from 0. 0 ≤ 𝑘 ≤ 10.
    k::I64

    # Matrix of weights
    W::Matrix{F64}
end


"""
    PolynomialInterpolationWeights(k::I64)

Outer constructor for the PolynomialInterpolationWeights struct. Here 𝑘
denotes the order for polynomial interpolation.
"""
function PolynomialInterpolationWeights(k::I64)
    @assert 0 ≤ k ≤ 10
    Wi = calc_poly_interpolation(k)
    PolynomialInterpolationWeights(k, Wi)
end


#=
### *Struct* : *PolynomialDifferentiationWeights*
=#

#=
*Remarks* : *Polynomial Differentiation*

An approximation for the derivative ``dy/dt`` of a function can be
obtained by taking the exact derivative of the polynomial approximant:

```math
\begin{equation}
\frac{dy}{dt} \Big|_{t = mh}
    \approx
    \frac{d}{dt} \mathcal{P}^{(k)}[y_0,\cdots,y_k](mh).
\end{equation}
```

We thus arrive at an approximative formula for polynomial differentiation

```math
\begin{equation}
\frac{dy}{dt}\Big |_{t = mh}
    \approx
    h^{-1} \sum^{k}_{l = 0} D^{(k)}_{ml}y_l,
\end{equation}
```

with

```math
\begin{equation}
D^{(k)}_{ml} = \sum^{k}_{a=1} P^{(k)}_{al}am^{a-1}.
\end{equation}
```

*References* :

Please see [`NESSi`] Section `8.2` for more details.
=#

"""
    PolynomialDifferentiationWeights

Weights for polynomial differentiation.
"""
struct PolynomialDifferentiationWeights <: AbstractWeights
    # Order for differentiation. It starts from 0. 0 ≤ 𝑘 ≤ 10.
    k::I64

    # Matrix of weights
    W::Matrix{F64}
end


"""
    PolynomialDifferentiationWeights(k::I64)

Outer constructor for the PolynomialDifferentiationWeights struct. Here 𝑘
denotes the order for polynomial differentiation.
"""
function PolynomialDifferentiationWeights(k::I64)
    @assert 0 ≤ k ≤ 10
    Wi = calc_poly_interpolation(k)
    Wd = calc_poly_differentiation(k, Wi)
    PolynomialDifferentiationWeights(k, Wd)
end


#=
### *Struct* : *PolynomialIntegrationWeights*
=#

#=
*Remarks* : *Polynomial Integration*

In some cases, the polynomial interpolation formula is also used to get
the approximation to an integral. For 0 ≤ 𝑚 ≤ 𝑛 ≤ 𝑘,

```math
\begin{equation}
\int^{nh}_{mh} dt\ y(t)
    \approx
    \int^{nh}_{mh} dt\ \mathcal{P}^{(k)}[y_0,\cdots,y_k](t).
\end{equation}
```

We thus use the following approximate relation for polynomial integration

```math
\begin{equation}
\int^{nh}_{mh} dt\ y(t)
    \approx
    h \sum^{k}_{l=0} I^{(k)}_{mnl}y_l,
\end{equation}
```

with

```math
\begin{equation}
I^{(k)}_{mnl}
    =
    \sum^{k}_{a=0} P^{(k)}_{al} \frac{n^{a+1} - m^{a+1}}{a+1}.
\end{equation}
```

*References* :

Please see [`NESSi`] Section `8.3` for more details.
=#

"""
    PolynomialIntegrationWeights

Weights for polynomial integration.
"""
struct PolynomialIntegrationWeights <: AbstractWeights
    # Order for integration. It starts from 0. 0 ≤ 𝑘 ≤ 10.
    k::I64

    # Array of weights
    W::Array{F64,3}
end


"""
    PolynomialIntegrationWeights(k::I64)

Outer constructor for the PolynomialIntegrationWeights struct. Here 𝑘
denotes the order for polynomial integration.
"""
function PolynomialIntegrationWeights(k::I64)
    @assert 0 ≤ k ≤ 10
    Wi = calc_poly_interpolation(k)
    Wt = calc_poly_integration(k, Wi)
    PolynomialIntegrationWeights(k, Wt)
end


#=
### *Struct* : *BackwardDifferentiationWeights*
=#

#=
*Remarks* : *Backward Differentiation*

Consider a function ``y(t)`` which takes the values ``y_j`` at the points
``t_j = jh`` of an equidistant mesh ``j = 0, \cdots, n`` with 𝑛 ≥ 𝑘. The
backward differentiation formula (`BDF`) of order 𝑘 approximates the
derivative ``dy/dt`` at ``t = nh`` using the function values ``y_n,
y_{n-1}, \cdots, y_{n-k}``. It is defined via the linear relation:

```math
\begin{equation}
\frac{dy}{dt}\Big |_{nh}
    \approx
    h^{-1} \sum^{k}_{j=0} a^{(k)}_{j} y_{n-j}.
\end{equation}
```

Here the coefficients for the 𝑘-th order formula are obtained by the
derivative ``\tilde{y}'(t=0)`` of the 𝑘-th order polynomial interpolation
``\tilde{y}(t)`` defined by the values ``\tilde{y}(jh) = y_{n-j}``,

```math
\begin{equation}
\frac{dy}{dt}\Big |_{nh}
    \approx
    -\frac{d}{dt} \mathcal{P}^{(k)}[y_n, y_{n-1}, \cdots, y_{n-k}](t=0).
\end{equation}
```

Note that the minus sign is due to the reversed order of the interpolated
points. Therefore, the coefficients of the `BDF` are directly related to
the coefficients for polynomial differentiation:

```math
\begin{equation}
a^{(k)}_j = -D^{(k)}_{0j}.
\end{equation}
```

*References* :

Please see [`NESSi`] Section `8.4` for more details.
=#

"""
    BackwardDifferentiationWeights

Weights for backward differentiation.
"""
struct BackwardDifferentiationWeights <: AbstractWeights
    # Order for backward differentiation. It starts from 0. 0 ≤ 𝑘 ≤ 10.
    k::I64

    # Vector of weights. The size of `W` should be 𝑘 + 2.
    W::Vector{F64}
end


"""
    BackwardDifferentiationWeights(k::I64)

Outer constructor for the BackwardDifferentiationWeights struct.

Note that the value of `BackwardDifferentiationWeights.k` is actually
𝑘 + 1, instead of 𝑘.
"""
function BackwardDifferentiationWeights(k::I64)
    @assert 0 ≤ k ≤ 10
    k₁ = k + 1
    Wi = calc_poly_interpolation(k₁)
    Wb = calc_backward_differentiation(k₁, Wi)
    BackwardDifferentiationWeights(k₁, Wb)
end


#=
### *Struct* : *GregoryIntegrationWeights*
=#

#=
*Remarks* : *Gregory Integration*

The solution of the Volterra integral equations (`VIEs`) is based on a
combination of backward differentiation formulae with so-called Gregory
quadrature rules for the integration. The 𝑘th order Gregory quadrature
rule on a linear mesh is defined by the following equation:

```math
\begin{equation}
\mathcal{I}_n
    =
    \int^{nh}_0 dt\ y(t)
    \approx
    h \sum^{\max(n,k)}_{j=0} \Omega^{(k)}_{nj}y_j.
\end{equation}
```

The matrix form of the integration weights ``\Omega^{(k)}_{nj}`` is as
follows:

```math
\begin{equation}
\Omega^{(k)}_{nj} =
\begin{pmatrix}
\sigma^{(k)}_{00} & \cdots & \sigma^{(k)}_{0k} & 0 & 0 &&&&& \\
\vdots && \vdots &&&&&&& \\
\sigma^{(k)}_{k0} & \cdots & \sigma^{(k)}_{kk} & 0 & 0 &&&&& \\
\Sigma^{(k)}_{00} & \cdots & \Sigma^{(k)}_{0k} & \omega^{(k)}_0 & 0 &&&&& \\
\vdots && \vdots && \ddots & \ddots &&&& \\
\Sigma^{(k)}_{k0} & \cdots & \Sigma^{(k)}_{kk} & \omega^{(k)}_k & \cdots & \omega^{(k)}_0 & 0 &&& \\
\omega^{(k)}_0 & \cdots & \omega^{(k)}_k & 1 & \omega^{(k)}_k & \cdots & \omega^{(k)}_0 & 0 && \\
\omega^{(k)}_0 & \cdots & \omega^{(k)}_k & 1 & 1 &\omega^{(k)}_k & \cdots & \omega^{(k)}_0 & 0 & \\
\omega^{(k)}_0 & \cdots & \omega^{(k)}_k & 1 & 1 & 1 &\omega^{(k)}_k & \cdots & \omega^{(k)}_0 & 0 \\
\end{pmatrix}
\end{equation}
```

Clearly, there are five cases for ``\Omega^{k}_{nj}``. Later, we will
introduce how to calculate the 𝑘th order Gregory integration weights
``\sigma^{(k)}``, ``\Sigma^{(k)}``, and ``\omega^{(k)}``.

*References* :

Please see [`NESSi`] Section `8.5` for more details.
=#

"""
    GregoryIntegrationWeights

Weights for Gregory integration.
"""
struct GregoryIntegrationWeights <: AbstractWeights
    # Order for Gregory integration. It starts from 0. 0 ≤ 𝑘 ≤ 10.
    k::I64

    # Matrix of weights. Its size is (𝑘 + 1, 𝑘 + 1).
    #
    # [σᵢⱼ], 0 ≤ 𝑖 ≤ 𝑘, 0 ≤ 𝑗 ≤ 𝑘.
    #
    σ::Matrix{F64}

    # Matrix of weights. Its size is (𝑘 + 1, 𝑘 + 1).
    #
    # [Σᵢⱼ], 0 ≤ 𝑖 ≤ 𝑘, 0 ≤ 𝑗 ≤ 𝑘.
    #
    Σ::Matrix{F64}

    # Vector of weights. Its size is (𝑘 + 1).
    #
    # [ωᵢ], 0 ≤ 𝑖 ≤ 𝑘.
    #
    ω::Vector{F64}
end


"""
    GregoryIntegrationWeights(k::I64)

Outer constructor for the GregoryIntegrationWeights struct. Here 𝑘
denotes the order for Gregory integration.
"""
function GregoryIntegrationWeights(k::I64)
    @assert 0 ≤ k ≤ 10
    Wi = calc_poly_interpolation(k)
    Wt = calc_poly_integration(k, Wi)
    σ, Σ, ω = calc_gregory_integration(k, Wt)
    GregoryIntegrationWeights(k, σ, Σ, ω)
end


#=
### *Struct* : *BoundaryConvolutionWeights*
=#

#=
*Remarks* : *Boundary Convolution*

Here, we introduce a 𝑘th order accurate approximation for a special kind
of convolution integral, which appears in the context of imaginary time
convolutions. Consider the following convolution integral:

```math
\begin{equation}
c(t) = \int^{t}_0 dt'\ F(t-t')G(t')
\end{equation}
```

between two functions 𝐹 and 𝐺 which are only defined for 𝑡 > 0, and can
not be continued into a differentiable function on the domain 𝑡 < 0. On
an equidistant mesh with ``t = mh`` and ``m < k``, the integration range
includes less than 𝑘 points, and the functions must be continued outside
the given integration range in order to obtain an 𝑘th order accurate
approximation.

We use the approximation

```math
\begin{equation}
c(mh) = \int^{mh}_0 dt'\
    \mathcal{P}^{(k)}[F_0,\cdots,F_k](mh-t')
    \mathcal{P}^{(k)}[G_0,\cdots,G_k](t').
\end{equation}
```

Finally, we have

```math
\begin{equation}
\int^{mh}_0 dt'\ F(t-t')G(t') =
    h\sum^{k}_{r,s = 0}F_r G_s R^{(k)}_{mrs},
\end{equation}
```

where

```math
\begin{equation}
R^{(k)}_{mrs} = \sum^{k}_{a,b = 0} P^{(k)}_{ar} P^{(k)}_{bs}
    \int^{m}_{0} dx\ (m-x)^a x^b.
\end{equation}
```

*References* :

Please see [`NESSi`] Section `8.6` for more details.
=#

"""
    BoundaryConvolutionWeights

Weights for boundary convolution.
"""
struct BoundaryConvolutionWeights <: AbstractWeights
    # Order for boundary convolution. It starts from 1. 1 ≤ 𝑘 ≤ 10.
    # For 𝑘 = 1, `W` is an empty array (just like NESSi's rcorr).
    k::I64

    # Array of weights. The size of `W` should be [𝑘 - 1, 𝑘 + 1, 𝑘 + 1].
    W::Array{F64,3}
end


"""
    BoundaryConvolutionWeights(k::I64)

Outer constructor for the BoundaryConvolutionWeights struct. Here 𝑘
denotes the order for boundary convolution.
"""
function BoundaryConvolutionWeights(k::I64)
    @assert 1 ≤ k ≤ 10
    Wi = calc_poly_interpolation(k)
    Wc = calc_boundary_convolution(k, Wi)
    BoundaryConvolutionWeights(k, Wc)
end


