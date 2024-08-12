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

include("global.jl")
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
export CnAbstractType
export CnAbstractMatrix
export CnAbstractVector
export CnAbstractFunction
export Cn
export Cf
export 𝔾ᵐᵃᵗ, 𝔾ʳᵉᵗ, 𝔾ˡᵐⁱˣ, 𝔾ˡᵉˢˢ
export 𝔾ᵐᵃᵗᵐ, 𝔾ᵃᵈᵛ, 𝔾ʳᵐⁱˣ, 𝔾ᵍᵗʳ    
export ℱ
export 𝕘ᵐᵃᵗ, 𝕘ʳᵉᵗ, 𝕘ˡᵐⁱˣ, 𝕘ˡᵉˢˢ
export 𝕘ᵐᵃᵗᵐ, 𝕘ᵃᵈᵛ, 𝕘ʳᵐⁱˣ, 𝕘ᵍᵗʳ
export 𝒻
export getdims, getsize, getntime, getntau, gettstp, getsign
export equaldims, iscompatible
export density, distance
export memset!, zeros!, memcpy!, incr!, smul!
export read!, write
export refresh!

#=
### *Includes and Exports* : *equilibrium.jl*
=#

include("start.jl")
export init_green!

#=
### *Includes and Exports* : *convolution.jl*
=#

include("langreth.jl")
export Integrator
export Convolution, ConvolutionTimeStep
export c_mat_mat_1, c_mat_mat_2

end # END OF MODULE
