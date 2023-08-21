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