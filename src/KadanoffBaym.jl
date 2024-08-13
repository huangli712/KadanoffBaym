#
# Project : Lavender
# Source  : KadanoffBaym.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2024/08/13
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

### References

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
using Dates
using Printf
using DelimitedFiles
using InteractiveUtils
using TOML

#=
### *Using Third-Party Libraries*
=#

#=
### *Includes And Exports* : *global.jl*
=#

#=
*Summary* :

Define some type aliases and string constants for the KadanoffBaym toolkit.

*Members* :

```text
I32, I64, API -> Numerical types (Integer).
F32, F64, APF -> Numerical types (Float).
C32, C64, APC -> Numerical types (Complex).
R32, R64, APR -> Numerical types (Union of Integer and Float).
N32, N64, APN -> Numerical types (Union of Integer, Float, and Complex).
#
__LIBNAME__   -> Name of this julia toolkit.
__VERSION__   -> Version of this julia toolkit.
__RELEASE__   -> Released date of this julia toolkit.
__AUTHORS__   -> Authors of this julia toolkit.
#
authors       -> Print the authors of KadanoffBaym to screen.
```
=#

#
include("global.jl")
#
export I32, I64, API
export F32, F64, APF
export C32, C64, APC
export R32, R64, APR
export N32, N64, APN
#
export __LIBNAME__
export __VERSION__
export __RELEASE__
export __AUTHORS__
#
export authors

#=
### *Includes And Exports* : *types.jl*
=#

#=
*Summary* :

Define some dicts and structs, which are used to store the config
parameters or represent some essential data structures.

*Members* :

```text
Element         -> Customized type.
MatArray        -> Customized type.
VecArray        -> Customized type.
#
CnAbstractType  ->
CnAbstractMatrix ->
CnAbstractVector ->
CnAbstractFunction ->
```
=#

#
include("types.jl")
#
export Element
export MatArray
export VecArray
#
export CnAbstractType
export CnAbstractMatrix
export CnAbstractVector
export CnAbstractFunction
#
export Cn
export Cf
#
export Gᵐᵃᵗ
export Gʳᵉᵗ
export Gˡᵐⁱˣ
export Gˡᵉˢˢ
export Gᵐᵃᵗᵐ
export Gᵃᵈᵛ
export Gʳᵐⁱˣ
export Gᵍᵗʳ
#
export gᵐᵃᵗ
export gʳᵉᵗ
export gˡᵐⁱˣ
export gˡᵉˢˢ
export gᵐᵃᵗᵐ
export gᵃᵈᵛ
export gʳᵐⁱˣ
export gᵍᵗʳ
#
export ℱ
export 𝒻
#
export refresh!
export getdims
export getntime
export getntau
export getsign
export getsize
export gettstp
export equaldims
export iscompatible
export density
export distance
export getindex
export setindex!
export memset!
export zeros!
export memcpy!
export incr!
export smul!
export read!
export write

#=
### *Includes And Exports* : *util.jl*
=#

#=
*Summary* :

To provide some useful utility macros and functions. They can be used
to colorize the output strings, query the environments, and parse the
input strings, etc.

*Members* :

```text
@cswitch      -> C-style switch.
@time_call    -> Evaluate a function call and print the elapsed time.
@pcs          -> Print colorful strings.
#
sorry         -> Say sorry.
prompt        -> Print some messages or logs to the output devices.
line_to_array -> Convert a line to a string array.
```
=#

#
include("util.jl")
#
export @cswitch
export @time_call
export @pcs
#
export sorry
export prompt
export line_to_array

#=
### *Includes and Exports* : *weights.jl*
=#

#
include("weights.jl")
#
export AbstractWeights
export PolynomialInterpolationWeights
export PolynomialDifferentiationWeights
export PolynomialIntegrationWeights
export BackwardDifferentiationWeights
export GregoryIntegrationWeights
export BoundaryConvolutionWeights
#
export calc_poly_interpolation
export calc_poly_differentiation
export calc_poly_integration
export calc_backward_differentiation
export calc_gregory_integration
export calc_gregory_weights
export calc_boundary_convolution
#
export trapezoid
export Λ
export γⱼ
export 𝐑
export Γ

#=
### *PreCompile*
=#

export _precompile

"""
    _precompile()

Here, we would like to precompile the whole `KadanoffBaym` toolkit to
reduce the runtime latency and speed up the successive calculations.
"""
function _precompile()
    prompt("Loading...")

    # Get an array of the names exported by the `KadanoffBaym` module
    nl = names(KadanoffBaym)

    # Go through each name
    cf = 0 # Counter
    for i in eachindex(nl)
        # Please pay attention to that nl[i] is a Symbol, we need to
        # convert it into string and function, respectively.
        str = string(nl[i])
        fun = eval(nl[i])

        # For methods only (macros must be excluded)
        if fun isa Function && !startswith(str, "@")
            # Increase the counter
            cf = cf + 1

            # Extract the signature of the function
            # Actually, `types` is a Core.SimpleVector.
            types = nothing
            try
                types = typeof(fun).name.mt.defs.sig.types
            catch
                @printf("Function %15s (#%3i) is skipped.\r", str, cf)
                continue
            end

            # Convert `types` from SimpleVector into Tuple
            # If length(types) is 1, the method is without arguments.
            T = ()
            if length(types) > 1
                T = tuple(types[2:end]...)
            end

            # Precompile them one by one
            #println(i, " -> ", str, " -> ", length(types), " -> ", T)
            precompile(fun, T)
            @printf("Function %24s (#%4i) is compiled.\r", str, cf)
        end
    end

    prompt("Well, KadanoffBaym is compiled and loaded ($cf functions).")
    prompt("We are ready to go!")
    println()
    flush(stdout)
end

"""
    __init__()

This function would be executed immediately after the module is loaded
at runtime for the first time. It works at the REPL mode only.
"""
__init__() = begin
    isinteractive() && _precompile()
end

end # END OF MODULE
