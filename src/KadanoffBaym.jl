#
# File: KadanoffBaym.jl
#

"""
    KadanoffBaym

The `KadanoffBaym` package is a state-of-the-art computational framework
for simulating the non-equilibrium strongly correlated electron systems.
It provides some useful application programming interfaces to manipulate
the non-equilibrium Green's functions defined on the 𝐿-shape Kadanoff-Baym
contour, including:

* Basic integration and differentiation rules
* Basic operations for Contour Green's functions
* Basic diagrammatic algorithms based on many-body perturbation theory
* Solve Volterra integral equations
* Solve Volterra integro-differential equations
* Convolution between two contour Green's functions

This package is inspired by the `NESSi` (The Non-Equilibrium Systems
Simulation package) code, which was developed and maintained by Martin
Eckstein *et al*. Actually, it can be regarded as a replacement of the
`NESSi` package for those peoples who don't like or aren't familiar
with `C++`.

### References:

[`NESSi`]
Martin Eckstein, *et al.*,
NESSi: The Non-Equilibrium Systems Simulation package,
*Computer Physics Communications* **257**, 107484 (2020)

[`REVIEW`]
Philipp Werner, *et al.*,
Nonequilibrium dynamical mean-field theory and its applications,
*Reviews of Modern Physics* **86**, 779 (2014)

[`MABOOK`]
Johan de Villiers,
Mathematics of Approximation,
*Atlantis Press* (2012)

[`MATABLE`]
Dan Zwillinger (editor),
CRC Standard Mathematical Tables and Formulas (33rd edition),
*CRC Press* (*Taylor & Francis Group*) (2018)

[`QUADRATURE`]
Ruben J. Espinosa-Maldonado and George D. Byrne,
On the Convergence of Quadrature Formulas,
*SIAM J. Numer. Anal.* **8**, 110 (1971)
"""
module KadanoffBaym

#=
### *Using Standard Libraries*
=#

using LinearAlgebra
using InteractiveUtils

#=
### *Includes and Exports* : *const.jl*
=#

include("const.jl")
export I64, F64, C64
export FERMI, BOSE
export CZERO, CZI, CONE
export Element
export MatArray, VecArray

#=
### *Includes and Exports* : *utils.jl*
=#

include("utils.jl")
export fermi, bose
export subtypetree

#=
### *Includes and Exports* : *weights.jl*
=#

include("weights.jl")
export AbstractWeights
export PolynomialInterpolationWeights
export PolynomialDifferentiationWeights
export PolynomialIntegrationWeights
export BackwardDifferentiationWeights
export GregoryIntegrationWeights
export BoundaryConvolutionWeights

#=
### *Includes and Exports* : *types.jl*
=#

include("types.jl")
export CnAny, CnAbsMat, CnAbsVec, CnAbsFun
export Cn
export CnFunF
export CnMatM, CnRetM, CnLmixM, CnLessM
export CnMatmM, CnAdvM, CnRmixM, CnGtrM
export CnFunM
export CnMatV, CnRetV, CnLmixV, CnLessV
export CnMatmV, CnAdvV, CnRmixV, CnGtrV
export CnFunV
export getdims, getsize, getntime, getntau, gettstp, getsign
export equaldims, iscompatible
export density, distance
export memset!, zeros!, memcpy!, incr!, smul!
export read!, write
export refresh!

#=
### *Includes and Exports* : *equilibrium.jl*
=#

include("equilibrium.jl")
export init_green!

#=
### *Includes and Exports* : *convolution.jl*
=#

include("convolution.jl")
export Integrator
export Convolution, ConvolutionTimeStep
export c_mat_mat_1, c_mat_mat_2

end # END OF MODULE

#
# File: dyson.jl
#

#
# File: diagram.jl
#

#
# File: equilibrium.jl
#
# Try to construct the contour Green's function at equilibrium state.
#

"""
    init_green!(G::CnFunM{T}, H₀::Matrix{T}, μ::F64, β::F64, h::F64)

Try to generate initial contour Green's function `G`. Here, `H₀` is the
band dispersion, `μ` is the chemical potential, `β` (≡ 1/𝑇) is the
inverse temperature, and `h` (≡ δ𝑡) is the length of time step at real
time axis.
"""
function init_green!(G::CnFunM{T}, H₀::Matrix{T}, μ::F64, β::F64, h::F64) where {T}
    # Extract key parameters
    ntime = getntime(G)
    ntau = getntau(G)
    sign = getsign(G)
    ndim1, ndim2 = getdims(G)

    # Sanity check
    @assert equaldims(G)
    @assert getdims(G) == size(H₀)

    # Construct the effective Hamiltonian
    Identity = diagm(ones(T, ndim1))
    Heff = Identity * μ - H₀

    # Diagonalize the effective Hamiltonian
    vals, vecs = eigen(Heff)

    # Calculate commutator-free matrix exponentials
    Udt = exp(CZI * h * Heff)
    Ut = CnFunF(ntime, ndim1)
    Ut[0] = Identity # At Matsubara axis
    Ut[1] = Identity
    for i = 2:ntime
        Un = Ut[i-1] * Udt
        Ut[i] = Un
    end

    # Build Matsubara component of contour Green's function
    dτ = β / (ntau - 1)
    for i = 1:ntau
        τ = (i - 1) * dτ
        if sign == FERMI
            x = FERMI * vecs * diagm(fermi(β, τ, vals)) * (vecs')
        else
            x = BOSE  * vecs * diagm( bose(β, τ, vals)) * (vecs')
        end
        G.mat[i] = x
    end # END OF I LOOP

    # Build left-mixing component of contour Green's function
    for i = 1:ntau
        τ = (i - 1) * dτ
        for j = 1:ntime
            Un = Ut[j]
            if sign == FERMI
                x =  CZI * Un * vecs * diagm(fermi(β, τ, -vals)) * (vecs')
            else
                x = -CZI * Un * vecs * diagm( bose(β, τ, -vals)) * (vecs')
            end
            G.lmix[j,i] = x
        end
    end # END OF I LOOP

    # Build retarded and lesser components of contour Green's function
    if sign == FERMI
        x =  vecs * diagm(fermi(β, -vals)) * (vecs')
    else
        x = -vecs * diagm( bose(β, -vals)) * (vecs')
    end
    #
    for i = 1:ntime
        for j = 1:i
            Uni = Ut[i]
            Unj = Ut[j]
            #
            v = -CZI * Uni * (Unj')
            G.ret[i,j] = v
            #
            v =  CZI * Unj * x * (Uni')
            G.less[j,i] = v
        end
    end # END OF I LOOP

    # Debug codes
    #
    # Test G.mat
    #for i = 1:ntau
    #    @show i, G.mat[i]
    #end
    #
    # Test G.lmix
    #for i = 1:ntau
    #    for j = 1:ntime
    #        @show i, j, G.lmix[j,i]
    #    end
    #end
    #
    # Test G.ret
    #for i = 1:ntime
    #    for j = 1:i
    #        @show i, j, G.ret[i,j]
    #    end
    #end
    #
    # Test G.less
    #for i = 1:ntime
    #    for j = 1:i
    #        #@show i, j, G.less[j,i]
    #    end
    #end
end

#
# File: weights.jl
#
# Provide basic integration and differentiation rules.
#

#=
### *AbstractWeights* : *Abstract Type*
=#

"""
    AbstractWeights

Abstract weights for numerical interpolation, differentiation, and
integration. The purpose of this abstract type is to construct
the type system.
"""
abstract type AbstractWeights end

#=
### *PolynomialInterpolationWeights* : *Struct*
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
### *PolynomialDifferentiationWeights* : *Struct*
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
### *PolynomialIntegrationWeights* : *Struct*
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
### *BackwardDifferentiationWeights* : *Struct*
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
### *GregoryIntegrationWeights* : *Struct*
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
### *BoundaryConvolutionWeights* : *Struct*
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
    # Order for boundary convolution. It starts from 2. 2 ≤ 𝑘 ≤ 10.
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
    @assert 2 ≤ k ≤ 10
    Wi = calc_poly_interpolation(k)
    Wc = calc_boundary_convolution(k, Wi)
    BoundaryConvolutionWeights(k, Wc)
end

#=
### *Private Service Functions*: *Layer 1*
=#

"""
    calc_poly_interpolation(k::I64)

Calculate polynomial interpolation weights.

*References* :

See [`NESSi`] Eq.~(80).
"""
function calc_poly_interpolation(k::I64)
    Wi = zeros(F64, k + 1, k + 1)
    for j=0:k
        for i=0:k
            Wi[i+1,j+1] = i^j
        end
    end
    Wi = inv(Wi)
    return Wi
end

"""
    calc_poly_differentiation(k::I64, Wi::Matrix{F64})

Calculate polynomial differentiation weights.

*References* :

See [`NESSi`] Eq.~(85).
"""
function calc_poly_differentiation(k::I64, Wi::Matrix{F64})
    @assert size(Wi) == (k + 1, k + 1)
    Wd = zeros(F64, k + 1, k + 1)
    for j=0:k
        for i=0:k
            for a=1:k
                Wd[i+1,j+1] = Wd[i+1,j+1] + Wi[a+1,j+1] * a * (i^(a-1))
            end
        end
    end
    return Wd
end

"""
    calc_poly_integration(k::I64, Wi::Matrix{F64})

Calculate polynomial integration weights.

*References* :

See [`NESSi`] Eq.~(89).
"""
function calc_poly_integration(k::I64, Wi::Matrix{F64})
    @assert size(Wi) == (k + 1, k + 1)
    Wt = zeros(F64, k + 1, k + 1, k + 1)
    for l=0:k
        for j=0:k
            for i=0:k
                for a=0:k
                    Wt[i+1,j+1,l+1] = Wt[i+1,j+1,l+1] + Wi[a+1,l+1] * (j^(a+1.0)-i^(a+1.0)) / (a+1.0)
                end
            end
        end
    end
    return Wt
end

"""
    calc_backward_differentiation(k::I64, Wi::Matrix{F64})

Calculate backward differentiation weights.

*References* :

See [`NESSi`] Eq.~(90).
"""
function calc_backward_differentiation(k::I64, Wi::Matrix{F64})
    @assert size(Wi) == (k + 1, k + 1)
    Wb = zeros(F64, k + 1)
    for i=0:k
        Wb[i+1] = -Wi[2,i+1]
    end
    return Wb
end

#=
*Remarks* : *Gregory Quadrature Rules*

Here, we would like to introduce how to calculate the 𝑘th order Gregory
integration weights ``\Omega^{(k)}_{nj}`` step by step. It is not a piece
of cake.

(1) 𝑛 ≤ 𝑘: ``\mathcal{I}_n`` is approximated by the exact integral over
the polynomial interpolation ``\mathcal{P}^{(k)}[y_0,\cdots,y_k](t)``,

```math
\begin{equation}
\mathcal{I}_n \approx
    \int^{nh}_0 dt\ \mathcal{P}^{(k)}[y_0,\cdots,y_k](t)
    \equiv
    h \sum^{k}_{j=0}\sigma^{(k)}_{nj}y_j.
\end{equation}
```

Hence the weights for 𝑛 ≤ 𝑘, which are denoted as *starting weights*
``\Omega^{(k)}_{nj} = \sigma^{(k)}_{nj}``, are equivalent to the
polynomial integration weights,

```math
\begin{equation}
\Omega^{(k)}_{nj} = I^{(k)}_{0nj} = \sigma^{(k)}_{nj}, \quad 0 \le n \le k. 
\end{equation}
```

See [`NESSi`] Eq.~(94) and [95].

(2) 𝑛 > 𝑘: We adopt the formulism presented in [`QUADRATURE`].

The 𝑘th order Gregory quadrature formula for 𝑛 + 1 nodes reads:

```math
\begin{equation}
𝑄^{(k)}(f) = 
    \sum^{n}_{j=0} a_j f(t_j) +
    \sum^{k}_{j=0} u_j f(t_j) +
    \sum^{n}_{j=n-k} v_j f(t_j).
\end{equation}
```

Here, ``a_j`` correspond to the weights for the `trapezoidal rule`, the
``u_j`` correspond to the contribution of the forward differences, and
the ``v_j`` are the weights from the backward differences. Hence, the
final weights are a linear combination of ``a_j``, ``u_j``, and ``v_j``.

At first, ``a_j`` is easily to be computed.

Later, let us focus on ``u_j`` and ``v_j``. They fulfill the following
relation,

```math
\begin{equation}
u_j = v_{n-j} = \gamma^{(k)}_{j}, \quad j = 0, 1, 2, \cdots, k.
\end{equation}
```

For ``j = 0``,

```math
\begin{equation}
\gamma^{(k)}_0 = \sum^{k}_{i=1} \mathcal{L}_{i+1}(-1)^{i+1}.
\end{equation}
```

For ``j = 1, 2, \cdots, k``,
```math
\begin{equation}
\gamma^{(k)}_j = \sum^{k}_{i=j} 
    \mathcal{L}_{i+1} (-1)^{i+j+1} \binom{i}{j},
\end{equation}
```

where

```math
\begin{equation}
\mathcal{L}_i = \int^{1}_0 \binom{t}{i} dt,
\end{equation}
```

with

```math
\begin{equation}
\binom{m}{n} = \frac{m!}{(m-n)!n!}.
\end{equation}
```
``\binom{m}{n}`` is the so-called binomial coefficients. From ``a_j``,
``u_j``, and ``v_j``, we can easily calculate ``\Sigma^{(k)}``.

See [`QUADRATURE`] Eq.~(2.2)-(2.10).

(3) Tricks. Notice that there is a relation between ``\Sigma^{(k)}``
and ``\omega^{(k)}``,

```math
\begin{equation}
\omega^{(k)}_j = \Sigma^{(k)}_{kj}, \quad j = 0, \cdots, k.
\end{equation}
```

Actually, we always use this equation to deduce ``\omega^{(k)}`` from
``\Sigma^{(k)}``.

See [`NESSi`] Table `9`.

(4) Implementations. The weights ``a_j`` are calculated in function
`trapezoid()`. In functions `γⱼ()` and `Λ()`, the calculations of
``\gamma^{(k)}_j`` and ``\mathcal{L}_i`` are implemented, respectively.
In function `calc_gregory_weights()`, the weights ``a_j``, ``u_j``, and
``v_j`` are computed and combined. 

Finally, in function `calc_gregory_integration()`, the Gregory quadrature
weights ``\sigma^{(k)}``, ``\Sigma^{(k)}``, and ``\omega^{(k)}`` are ready.
=#

"""
    calc_gregory_integration(k::I64, Wt::Array{F64,3})

Calculate Gregory integration weights.

*References* :

See *Remarks*.
"""
function calc_gregory_integration(k::I64, Wt::Array{F64,3})
    # Allocate memory
    σ = zeros(F64, k + 1, k + 1)
    Σ = zeros(F64, k + 1, k + 1)
    ω = zeros(F64, k + 1)

    # Get starting weights σ
    for j=0:k
        for i=0:k
            σ[i+1,j+1] = Wt[1,i+1,j+1]
        end
    end

    # Get Σ for 𝑛 > 𝑘
    for n = k+1:2*k+1
        W = calc_gregory_weights(k,n)
        Σ[n-k,:] = W[1:k+1]
    end
    
    # Get ω from Σ. The ω weights can be read off from the last row of
    # the Σ weights.
    ω[:] = Σ[end,:]

    # Return the desired arrays
    return σ, Σ, ω
end

"""
    calc_gregory_weights(k::I64, n::I64)

Try to calculate the integration weights based on Gregory quadrature rule
for 𝑘-order and 𝑛 + 1 nodes. This function only works for 𝑛 > 𝑘. It just
returns a vector with 𝑛 + 1 elements (They are all Rational numbers).

*References* :

See *Remarks*.
"""
function calc_gregory_weights(k::I64, n::I64)
    @assert n > k

    # Get basic weights from trapezoid rules 
    W = trapezoid(n)

    # Get the boundary corrections
    γ = γⱼ(k)

    # Combine the basic weights and boundary corrections
    for j = 0:k
        # Contribution from forward differences
        W[j+1] = W[j+1] + γ[j+1]

        # Contribution from backward differences
        W[end-j] = W[end-j] + γ[j+1]
    end

    # Return the desired array
    return W
end

"""
    calc_boundary_convolution(k::I64, Wi::Matrix{F64})

Calculate integration weights for boundary convolution.

*References* :
    
See [`NESSi`] Eq.~(104).
"""
function calc_boundary_convolution(k::I64, Wi::Matrix{F64})
    @assert size(Wi) == (k + 1, k + 1)
    Wc = zeros(F64, k - 1, k + 1, k + 1)
    for m=1:k-1
        for i=0:k
            for j=0:k
                val = 0.0
                for a=0:k
                    for b=0:k
                        val = val + Wi[a+1,i+1] * Wi[b+1,j+1] * 𝐑(m,a,b)
                    end
                end
                Wc[m,i+1,j+1] = val
            end
        end
    end
    return Wc
end

#=
### *Private Service Functions* : *Layer 2*
=#

#=
*Remarks* : *Trapezoid Rule*

The trapezoid rule reads:

```math
\begin{equation}
\omega_{ni} = 
\begin{cases}
\frac{1}{2},\quad &\text{if}\ i = 0\ \text{or}\ n. \\
1,\quad &\text{if}\ 1 \le i \le n - 1.
\end{cases}
\end{equation}
```
=# 

"""
    trapezoid(n::I64)

Return integration weights based on the trapezoid rule. Note that the
return value of this function is a Rational number.

*References* :

See [`REVIEW`] Eq.~(A9).
"""
function trapezoid(n::I64)
    ω = fill(1//1, n + 1)
    ω[1] = 1//2
    ω[n+1] = 1//2
    return ω
end

#=
*Remarks* : *Laplace Coefficients*

In previous *Remarks*, we introduce a variable ``\mathcal{L}_k``. Let us 
recall it at first.

```math
\begin{equation}
\mathcal{L}_k = \int^{1}_0 \binom{t}{k} dt,
\end{equation}
```

It would be better to calculate the Laplace coefficients ``\Lambda_k``,
instead of calculating ``\mathcal{L}_k`` directly. The relation between
``\mathcal{L}_k`` and ``\Lambda_k`` is as follows,

```math
\begin{equation}
\Lambda_k = \mathcal{L}_k (-1)^{k-1}.
\end{equation}
```

Thus, the definition of the Laplace coefficients ``\Lambda_k`` is:

```math
\begin{equation}
\Lambda_k = (-1)^{k-1} \int^{1}_0 dt \binom{t}{k},
\end{equation}
```
 
where 𝑘 ≥ 0, and the binomial function is defined as follows:

```math
\begin{equation}
\binom{t}{k} = \frac{t!}{(t-k)!k!}.
\end{equation}
```

Specially, for arbitary 𝑡, we have

```math
\begin{equation}
\binom{t}{0} \equiv 1.
\end{equation}
```

Here we apply the following recursive formulas to evaluate ``\Lambda_k``:

```math
\begin{equation}
\Lambda_0 = -1,
\end{equation}
```
and

```math
\begin{equation}
\Lambda_k = \sum^{k-1}_{l=0} \frac{\Lambda_l}{l - k - 1}, \quad k \ge 1.
\end{equation}
```

Then we get:
```math
\begin{equation}
\Lambda_1 = \frac{1}{2},\
\Lambda_2 = \frac{1}{12},\ 
\Lambda_3 = \frac{1}{24},\
\Lambda_4 = \frac{19}{720},\
\Lambda_5 = \frac{3}{160},\
\Lambda_6 = \frac{863}{60480}.
\end{equation}
```

The above values can be used as benchmark. This algorithm is implemented
in function `Λ()`.
=#

"""
    Λ(k::I64)

Try to calculate the Laplace coefficients Λ. This a recursive function.
Note that the return value is a Rational number.

*References* :

See [`MABOOK`] Eqs.~(8.4.37), (8.4.50), and (8.4.54).
"""
function Λ(k::I64)
    # Λ₀ = -1
    if k == 0
        return -1 // 1
    # Apply [`MABOOK`] Eq.~(8.4.50) to calculate Λ_{k} for arbitary 𝑘
    else
        λ = 0 // 1
        for l = 0:k-1
            λ = λ + Λ(l) / (l - k - 1)
        end
        return λ
    end
end

#=
*Remarks* : ``\gamma^{(k)}_j`` *Coefficients*

The 𝑘th order coefficients ``\gamma^{(k)}_j`` are used to calculate the
integration weights by Gregory quadrature rules. They are defined in the
previous *Remarks*. Here, we introduce two slightly different formulas.

```math
\begin{equation}
\gamma_0 = -\sum^{k}_{i=1} \Lambda_{i+1},
\end{equation}
```

and

```math
\begin{equation}
\gamma_j = \sum^{k}_{i=j} \Lambda_{i+1} (-1)^{j+1}\binom{i}{j},
\end{equation}
```

where ``j \in [1,k]``, and ``\Lambda_i`` are the Laplace coefficients.
=#

"""
    γⱼ(k::I64)

Try to calculate the 𝑘th order coefficients γⱼ, where 𝑗 ∈ [0,𝑘].

*References* :

See [`QUADRATURE`] Eqs.~(2.5) and (2.6).
"""
function γⱼ(k::I64)
    γ = zeros(Rational{I64}, k + 1)

    # Special treatment for the first element
    for i = 1:k
        γ[1] = γ[1] - Λ(i + 1)
    end

    # Using general formula for γⱼ
    for j = 2:k+1
        for i = j-1:k
            γ[j] = γ[j] + Λ(i + 1) * (-1)^j * binomial(i, j-1) 
        end
    end

    return γ
end

#=
*Remarks* : ``\mathbf{R}`` *Integral*

In order to evaluate the following integration analytically

```math
\begin{equation}
\mathbf{R} = \int^{m}_0 dx\ (m-x)^a x^b,
\end{equation}
```

we execute the following python codes:

```python
import sympy
x, a, b, m = sympy.symbols('x a b m')
c = sympy.integrate((m-x)**a*x**b,(x,0,m))
c = sympy.simplify(c)
sympy.pprint(c)
```

The output is as follows:

```math
\begin{equation}
\mathbf{R} = \frac{m^{a+b+1}\Gamma(a+1)\Gamma(b+1)}{\Gamma(a+b+2)},
\end{equation}
```

where ``\Gamma`` is a special function.
=#

"""
    𝐑(m::I64, a::I64, b::I64)

Try to calculate integration ∫^{m}_{0} dx (m-x)ᵃxᵇ. We note that this
integration appears in [`NESSi`] Eq.~(104). This function is called by
`calc_boundary_convolution()` internally. It should not be exported.

*References* :

See [`NESSi`] Eq.~(104).
"""
function 𝐑(m::I64, a::I64, b::I64)
    return m^(a+b+1) * Γ(a+1) * Γ(b+1) / Γ(a+b+2)
end

#=
*Remarks* : ``\Gamma`` *Function*

The definition for the ``\Gamma`` function is as follows:

```math
\begin{equation}
\Gamma(z) = \int^{\infty}_0 e^{-t} t^{z - 1} dt,
\end{equation}
```

where ``z = x + iy`` and 𝑥 > 0.
=#

"""
    Γ(n::I64)

Try to calculate Γ function. This function should not be exported. Note
that the function name for the calculation of Γ function is `:tgamma`,
instead of `:gamma`. This is quite strange.

*References* :

See [`MATABLE`] Section `6.21`.
"""
function Γ(n::I64)
    xn = convert(F64, n)
    return ccall((:tgamma, Base.Math.libm), F64, (F64,), xn)
end

#=
### *Public Service Functions*
=#

"""
    Base.getindex(AW::AbstractWeights, inds::Vararg{I64})

Provide a fast access to the weights. `AW` denotes the type of weights,
and `inds` is for the indices. This function works for the following
structs only:

* `PolynomialInterpolationWeights`
* `PolynomialDifferentiationWeights`
* `PolynomialIntegrationWeights`
* `BackwardDifferentiationWeights`
* `BoundaryConvolutionWeights`
"""
function Base.getindex(AW::AbstractWeights, inds::Vararg{I64})
    @assert length(inds) == ndims(AW.W)
    indsp1 = inds .+ 1
    return AW.W[indsp1...]
end

"""
    Base.getindex(GW::GregoryIntegrationWeights, inds::Vararg{I64})

Provide a fast access to the weights. `GW` denotes the type of weights,
and `inds` is for the indices. This function works for the following
structs only:

* `GregoryIntegrationWeights`

*References* :

See *Remarks* : *Gregory Integration*.
"""
function Base.getindex(GW::GregoryIntegrationWeights, inds::Vararg{I64})
    # Check the indices
    # 𝑛 and 𝑗 are row indices and column indices, respectively.
    @assert length(inds) == 2
    n, j = inds

    # Access the weights
    #
    # Defalut weights, zero.
    val = 0.0
    #
    # 0 ≤ 𝑛, 𝑗 ≤ k
    if n ≤ GW.k
        if j ≤ GW.k
            val = GW.σ[n + 1, j + 1]
        end
    end
    #
    # 𝑘 + 1 ≤ 𝑛 ≤ 2𝑘 + 1
    if ( GW.k + 1 ) ≤ n ≤ ( 2 * GW.k + 1 )
        # 0 ≤ 𝑗 ≤ 𝑘
        if j ≤ GW.k
            val = GW.Σ[n - GW.k, j + 1]
        # 𝑘 + 1 ≤ 𝑗 ≤ 𝑛
        else
            if j ≤ n
                val = GW.ω[n + 1 - j]
            end
        end
    end
    #
    # 𝑛 ≥ 2𝑘 + 2
    if n ≥ 2 * GW.k + 2
        # 0 ≤ 𝑗 ≤ 𝑘
        if j ≤ GW.k
            val = GW.ω[j + 1]
        # 𝑘 + 1 ≤ 𝑗 ≤ 𝑛 - 𝑘 - 1
        elseif  GW.k + 1 ≤ j ≤ n - GW.k - 1
            val = 1.0
        # 𝑛 - 𝑘 ≤ 𝑗 ≤ 𝑛
        elseif n - GW.k ≤ j ≤ n
            val = GW.ω[n + 1 - j]
        end
    end

    # Return the desired weights
    return val
end

#
# File: vie2.jl
#

#
# File: utils.jl
#

#=
### *Auxiliary Functions*
=#

function fermi(β::T, ω::T) where {T}
    arg = ω * β
    if abs(arg) > 100
        arg > 0 ? zero(T) : one(T)
    else
        one(T) / ( one(T) + exp(arg) )
    end
end

function fermi(β::T, τ::T, ω::T) where {T}
    if ω < 0
        exp(ω*τ) * fermi(β, ω)
    else
        exp((τ - β) * ω) * fermi(β, -ω)
    end
end

function fermi(β::T, ω::Vector{N}) where {T,N}
    if T == N
        [fermi(β, x) for x in ω]
    else
        [fermi(β, convert(T, x)) for x in ω]
    end
end

function fermi(β::T, τ::T, ω::Vector{N}) where {T,N}
    if T == N
        [fermi(β, τ, x) for x in ω]
    else
        [fermi(β, τ, convert(T, x)) for x in ω]
    end
end

function bose(β::T, ω::T) where {T}
    arg = ω * β
    if arg < 0
        return -one(T) - bose(β, -ω)
    end

    if abs(arg) > 100
        return zero(T)
    elseif arg < 1.0e-10
        return one(T) / arg
    else
        return one(T) / ( exp(arg) - one(T) )
    end
end

function bose(β::T, τ::T, ω::T) where {T}
    if ω < 0
        exp(τ * ω) * bose(β, ω)
    else
        -exp((τ - β) * ω) * bose(β, -ω)
    end
end

function bose(β::T, ω::Vector{N}) where {T,N}
    if T == N
        [bose(β, x) for x in ω]
    else
        [bose(β, convert(T, x)) for x in ω]
    end
end

function bose(β::T, τ::T, ω::Vector{N}) where {T,N}
    if T == N
        [bose(β, τ, x) for x in ω]
    else
        [bose(β, τ, convert(T, x)) for x in ω]
    end
end

"""
    subtypetree(roottype, level::I64 = 1, indent::I64 = 4)

Display the entire type hierarchy starting from the specified `roottype`
"""
function subtypetree(roottype, level::I64 = 1, indent::I64 = 4)
    level == 1 && println(roottype)
    for s in subtypes(roottype)
        println(join(fill(" ", level * indent)) * string(s))
        subtypetree(s, level + 1, indent)
    end
end