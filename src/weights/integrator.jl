#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

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
### *Public API*
=#

"""
    Integrator

TO_BE_DONE
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

TO_BE_DONE
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

