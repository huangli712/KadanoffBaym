#
# Project : Lavender
# Source  : KadanoffBaym.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/01/14
#

"""
    KadanoffBaym

The `KadanoffBaym` package is a state-of-the-art computational framework
for simulating the non-equilibrium strongly correlated electron systems.
It provides some useful application programming interfaces to manipulate
the non-equilibrium Green's functions defined on the 𝐿-shape Kadanoff-Baym
contour, including:

* Basic integration and differentiation rules
* Basic operations for contour-ordered Green's functions
* Basic diagrammatic algorithms based on many-body perturbation theory
* Solve Volterra integral equations
* Solve Volterra integro-differential equations
* Convolution between two contour-ordered Green's functions

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
include("common/common.jl")
include("types/types.jl")
include("element/element.jl")
include("properties/compat.jl")
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
#
export DType
export ADT
#
export PCONTOUR
export PMODEL
export _PCONTOUR
export _PMODEL
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
export CnAbstractStationaryFunction
export FFTDomain
export FFT_TIME
export FFT_FREQ
export FFTIntegralMethod
export FFT_TRAPEZ
export FFT_CUBIC
export FFT_ADAPTIVE
export DysonBathType
export BATH_NONE
export BATH_CONST
export BATH_GAUSS
export BATH_OHMIC
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
export fermi_exp
export bose_exp

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
#
export Integrator

#=
### *Includes And Exports* : *fft.jl*
=#

#
include("fft.jl")
#
export FFTDirection
export FFT_FORWARD
export FFT_BACKWARD
export FFTPlan
export plan_fft
export allocate_fft_scratch
export clear_fft_plan_cache!
export fft!
export fft
export fft_strided!

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

#=
### *Includes And Exports* : *structs.jl*
=#

#
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
export PseudoGF
export PseudoGFStep
export Gʳᵉᵗᵐᵒᵛ
export Gˡᵉˢˢᵐᵒᵛ
export gʳᵉᵗᵐᵒᵛ
export gˡᵉˢˢᵐᵒᵛ
export ℱᵐᵒᵛ
export 𝒻ᵐᵒᵛ
export 𝒻ᵐᵒᵛView
export Cfᵐᵒᵛ

#=
### *Includes And Exports* : *properties.jl*
=#

#
include("properties.jl")
#
export getdims
export getntime
export gettmax
export getdt
export getdtau
export getbeta
export getntau
export getsign
export getsize
export gettstp
export getproperty
export equaldims
export iscompatible
export distance

#=
### *Includes And Exports* : *indexing.jl*
=#

#
include("indexing.jl")
#
export getindex
export setindex!

#=
### *Includes And Exports* : *element.jl*
=#

#
#
export CopyZone
export elemcpy!

#=
### *Includes And Exports* : *traits.jl*
=#

#
include("traits.jl")
#
export refresh!
export memset!
export zeros!
export memcpy!
export incr!
export smul!
export submatrix
export set_submatrix!

#=
### *Includes And Exports* : *moving.jl*
=#

#
include("moving.jl")
# The moving traits, bubbles, and utilities are defined in traits.jl,
# bubble.jl, and utilities.jl (the private helpers of moving.jl are
# resolved at call time).
#
export timestep_view
export get_timestep
export forward!
export set_from_G_backward!
export initialize_from_ness!
export set_from_function_backward!

#=
### *Includes And Exports* : *algebra.jl*
=#

#
include("algebra.jl")
#

#=
### *Includes And Exports* : *utilities.jl*
=#

#
include("utilities.jl")
#
export extrapolate_timestep!
export interpolation
export set_t0_from_mat!
export set_tk_from_mat!
export force_matsubara_hermitian!

#=
### *Includes And Exports* : *differentiation.jl*
=#

#
include("differentiation.jl")
#
export deriv1_matsubara!
export deriv1_element!
export deriv2_element!
export deriv1_tv!
export deriv2_tv!
export deriv1_timestep!
export deriv2_timestep!

#=
### *Includes And Exports* : *convolution.jl*
=#

#
include("convolution.jl")
#
export convolution
export convolution_time_step
#
export conv_mat
export conv_mat_mat_1
export conv_mat_mat_1p
export conv_mat_mat_2
export conv_mat_mat_2p
#
export conv_ret
#
export conv_ret_lmix
export conv_lmix_mat
#
export conv_ret_less
export conv_less_adv
export conv_lmix_rmix
#
export incr_convolution_mat!
export incr_convolution_ret!
export incr_convolution_tv!
export incr_convolution_ret_less!
export incr_convolution_less_adv!
export incr_convolution_lmix_rmix!
export incr_convolution_les!
export incr_convolution!
export convolution_les_timediag
export convolution_density_matrix

#=
### *Includes And Exports* : *response.jl*
=#

#
include("response.jl")
#
export response_convolution

#=
### *Includes And Exports* : *matsubara_fourier_core.jl*
=#

#
include("matsubara_fourier_core.jl")
#

#=
### *Includes And Exports* : *linalg.jl*
=#

#
include("linalg.jl")
#

#=
### *Includes And Exports* : *vie.jl*
=#

#
include("vie.jl")
#
export vie2_mat!
export vie2_start!
export vie2_timestep!
export vie2!
export vie2_timestep_sin!
export VIE2SinWorkspace

#=
### *Includes And Exports* : *bubble.jl*
=#

#
include("bubble.jl")
#
export bubble1!
export bubble2!

#=
### *Includes And Exports* : *vide.jl*
=#

#
include("vide.jl")
#
export dyson_mat!
export dyson_start!
export dyson_timestep!
export dyson!
export kinetic_energy_trunc

#=
### *Includes And Exports* : *equilibrium.jl*
=#

#
include("equilibrium.jl")
#
export green_equilibrium_mat!
export green_equilibrium!
export green_equilibrium_mat_bethe!
export green_equilibrium_bethe!
export green_from_H!
export green_single_pole_XX_mat!
export green_single_pole_XX_timestep!
export green_single_pole_XX!

#=
### *Includes And Exports* : *observables.jl*
=#

#
include("observables.jl")
#
export density
export density_matrix
export kinetic_energy
export correlation_energy
export current

#=
### *Includes And Exports* : *pseudo.jl*
=#

#
include("pseudo.jl")
#

#=
### *Includes And Exports* : *pseudo_convolution.jl*
=#

#
include("pseudo_convolution.jl")
#
export incr_pseudo_convolution_mat!
export incr_pseudo_convolution_tv!
export incr_pseudo_convolution!
export pseudo_convolution_timestep!
export pseudo_convolution_matsubara!
export pseudo_convolution!

#=
### *Includes And Exports* : *pseudodyson.jl*
=#

#
include("pseudodyson.jl")
#
export pseudodyson_mat!
export pseudodyson_start!
export pseudodyson_timestep!
export pseudodyson!

#=
### *Includes And Exports* : *pseudo_vie2.jl*
=#

#
include("pseudo_vie2.jl")
#
export pseudo_vie2_mat!
export pseudo_vie2_start!
export pseudo_vie2_timestep!
export pseudo_vie2!

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
#export read!
#export write

#=
### *Includes And Exports* : *fourier.jl*
=#

#
include("fourier.jl")
#
export get_dftcorr_linear
export get_dftcorr_cubic
export complex_dftcor_cubic
export dft_cplx
export adft_func
export get_tau
export get_omega
export set_first_order_tail!
export matsubara_dft!
export matsubara_ft!

#=
### *Includes And Exports* : *ness.jl*
=#

#
include("ness.jl")
#
export GridInfo
export FFTArray
export HermMatrixNESS
export time_at
export freq_at
export set_h
export set_dw
export getnft
export get_ret
export get_les
export set_ret!
export set_les!
export left_multiply!
export right_multiply!
export set_matrixelement!
export fft_to_freq!
export fft_to_time!
export integral_transform_to_freq!
export integral_transform_to_time!
export sigma_bath
export dyson_ness!
export BetheDOS
export GaussianDOS
export OhmicSymDOS
export green_equilibrium_ness!
export force_equilibrium!
export bubble1_ness!
export bubble2_ness!
export downsample
export upsample
export resample
export cntr2ness!
export ness2cntr!

#=
### *Includes And Exports* : *spectral.jl*
=#

#
include("spectral.jl")
#
export read_gf_ret_les_tavtrel
export evaluate_spectrum_windowed
export write_fftarray
export read_fftarray
export write_ness
export read_ness

#=
### *Includes And Exports* : *base.jl*
=#

#
include("base.jl")
#
export init_green!

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
