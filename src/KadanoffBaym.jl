#
# Project : Lavender
# Source  : KadanoffBaym.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/10/20
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
"""
module KadanoffBaym

#=
### *Using Standard Libraries*
=#

using Distributed
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

Define some type aliases and string constants for the KadanoffBaym library.

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
DType          -> Customized type.
ADT            -> Customized type.
#
PCONTOUR       -> Configuration dict for contour setup.
PMODEL         -> Configuration dict for model setup.
#
Element        -> Customized type for matrix.
MatArray       -> Customized type for matrix of matrix.
VecArray       -> Customized type for vector of matrix.
#
CnAbstractType -> Root abstract type.
CnAbstractContour -> Abstract type for contour.
CnAbstractMatrix -> Abstract type for matrix on contour.
CnAbstractVector -> Abstract type for vector on contour.
CnAbstractFunction -> Abstract type for contour function.
#
subtypetree    -> Display hierarchical type tree.
```
=#

#
include("types.jl")
#
export DType
export ADT
#
export PCONTOUR
export PMODE
#
export Element
export MatArray
export VecArray
#
export CnAbstractType
export CnAbstractContour
export CnAbstractMatrix
export CnAbstractVector
export CnAbstractFunction
#
export subtypetree

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
require       -> Check julia envirnoment.
setup_args    -> Setup ARGS manually.
query_args    -> Query program's arguments.
trace_error   -> Write exceptions or errors to terminal or external file.
catch_error   -> Catch the thrown exceptions or errors.
welcome       -> Print welcome message.
overview      -> Print runtime information of KadanoffBaym.
goodbye       -> Say goodbye.
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
export require
export setup_args
export query_args
export trace_error
export catch_error
export welcome
export overview
export goodbye
export sorry
export prompt
export line_to_array

#=
### *Includes And Exports* : *math.jl*
=#

#=
*Summary* :

To provide some numerical algorithms, such as Fermi and Bose functions.

*Members* :

```text
FERMI -> Basic physical constant for fermionic system.
BOSE  -> Basic physical constant for bosonic system.
#
fermi -> Fermi function.
bose  -> Bose function.
```
=#

#
include("math.jl")
#
export FERMI
export BOSE
#
export fermi
export bose

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
### *Includes And Exports* : *config.jl*
=#

#=
*Summary* :

To extract, parse, verify, and print the configuration parameters.
They are stored in external files (neq.toml) or dictionaries.

*Members* :

```text
inp_toml   -> Parse neq.toml, return raw configuration information.
fil_dict   -> Fill dicts for configuration parameters.
see_dict   -> Display all the relevant configuration parameters.
rev_dict_c -> Update dict (PCONTOUR) for configuration parameters.
rev_dict_m -> Update dict (PMODEL) for configuration parameters.
chk_dict   -> Check dicts for configuration parameters.
_v         -> Verify dict's values.
get_c      -> Extract value from dict (PCONTOUR dict), return raw value.
get_m      -> Extract value from dict (PMODEL dict), return raw value.
```
=#

#
include("config.jl")
#
export inp_toml
export fil_dict
export see_dict
export rev_dict_c
export rev_dict_m
export chk_dict
export _v
export get_c
export get_m

#
include("structs.jl")
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
include("query.jl")
#
export getdims
export getntime
export getntau
export getsign
export getsize
export gettstp
export getproperty
export equaldims
export iscompatible
export distance

#
include("indexing.jl")
#
export getindex
export setindex!

#
include("traits.jl")
#
export refresh!
export memset!
export zeros!
export memcpy!
export incr!
export smul!

#
include("operators.jl")
#

#
#include("langreth.jl")
#

#
include("vie.jl")
#

#
include("vide.jl")
#

#
include("observables.jl")
#
export density

#=
### *Includes And Exports* : *inout.jl*
=#

#=
*Summary* :

To read the input data or write the calculated results.

*Members* :

```text

```
=#

#
include("inout.jl")
#
export read!
export write

#
#include("dmft.jl")
#

#
include("base.jl")
#

#=
### *PreCompile*
=#

export _precompile

"""
    _precompile()

Here, we would like to precompile the whole `KadanoffBaym` library to
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
