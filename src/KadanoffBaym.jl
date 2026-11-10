#
# Project : Lavender
# Source  : KadanoffBaym.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
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
### *Includes* : *Theme Barrels*

The include order below is the dependency DAG of the library: it must
go from the foundation (aliases, types) up to the applications (solvers,
observables, i/o). Dependencies only point downwards.
=#

include("common/common.jl")           # aliases, containers, statistics, meta, config, util, ...
include("weights/weights.jl")         # weight types, calc rules, Integrator
include("types/types.jl")             # contour, func, component families, green, pseudo, moving, ness
include("blocks/blocks.jl")           # blocks: CopyZone, elemcpy!, submatrix
include("indexing/indexing.jl")       # block access, getindex, setindex, ness accessors
include("properties/properties.jl")   # getters, property, equal, compat, distance, ness
include("traits/traits.jl")           # memcpy!, memset!, zeros!, incr!, smul!, ...
include("algebra/algebra.jl")         # +, -, * on contour types
include("math/math.jl")               # distributions, dos, deriv1/2, interpolation
include("fourier/fourier.jl")         # fft, dftcor, matsubara, core, transforms, convert
include("moving/moving.jl")           # views, propagation
include("convolution/convolution.jl") # kernels, incr, api, response, pseudo
include("equilibrium/equilibrium.jl") # equilibrium initial states
include("solvers/solvers.jl")         # vie/dyson families and their pseudo/ness variants
include("observables/observables.jl") # observables, bubble, spectrum
include("io/io.jl")                   # show, write, read, ness i/o

#=
### *Exports* : *common*
=#

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
export subtypetree
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
### *Exports* : *weights*
=#

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
### *Exports* : *types*
=#

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
export 𝒻ᵛⁱᵉʷᵐᵒᵛ
export Cfᵐᵒᵛ
#
export CnAbstractType
export CnAbstractContour
export CnAbstractMatrix
export CnAbstractVector
export CnAbstractFunction
export CnAbstractStationaryFunction
#
export FFTDomain
export FFT_TIME
export FFT_FREQ
export FFTIntegralMethod
export FFT_TRAPEZ
export FFT_CUBIC
export FFT_ADAPTIVE
#
export DysonBathType
export BATH_NONE
export BATH_CONST
export BATH_GAUSS
export BATH_OHMIC
#
export GridInfo
export FFTArray
export HermMatrixNESS

#=
### *Exports* : *blocks*
=#

export CopyZone
export elemcpy!

#=
### *Exports* : *indexing*
=#

export getindex
export setindex!

#=
### *Exports* : *properties*
=#

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
#
export time_at
export freq_at
export set_h
export set_dw
export getnft
export get_ret
export get_les
export set_ret!
export set_les!

#=
### *Exports* : *traits*
=#

export refresh!
export memset!
export zeros!
export memcpy!
export incr!
export smul!
export submatrix
export set_submatrix!
#
export left_multiply!
export right_multiply!
export set_matrixelement!

#=
### *Exports* : *math*
=#

export FERMI
export BOSE
#
export fermi
export bose
export fermi_exp
export bose_exp
#
export BetheDOS
export GaussianDOS
export OhmicSymDOS
#
export deriv1_matsubara!
export deriv1_element!
export deriv2_element!
export deriv1_tv!
export deriv2_tv!
export deriv1_timestep!
export deriv2_timestep!
#
export extrapolate_timestep!
export interpolation

#=
### *Exports* : *fourier*
=#

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
export force_matsubara_hermitian!
#
export fft_to_freq!
export fft_to_time!
export integral_transform_to_freq!
export integral_transform_to_time!
#
export downsample
export upsample
export resample
export cntr2ness!
export ness2cntr!

#=
### *Exports* : *moving*
=#

export timestep_view
export get_timestep
export forward!
export set_from_G_backward!
export initialize_from_ness!
export set_from_function_backward!

#=
### *Exports* : *convolution*
=#

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
#
export response_convolution
#
export incr_pseudo_convolution_mat!
export incr_pseudo_convolution_tv!
export incr_pseudo_convolution!
export pseudo_convolution_timestep!
export pseudo_convolution_matsubara!
export pseudo_convolution!

#=
### *Exports* : *equilibrium*
=#

export green_equilibrium_mat!
export green_equilibrium!
export green_equilibrium_mat_bethe!
export green_equilibrium_bethe!
export green_from_H!
export green_single_pole_XX_mat!
export green_single_pole_XX_timestep!
export green_single_pole_XX!
#
export init_green!
#
export set_t0_from_mat!
export set_tk_from_mat!
#
export green_equilibrium_ness!
export force_equilibrium!

#=
### *Exports* : *solvers*
=#

export vie2_mat!
export vie2_start!
export vie2_timestep!
export vie2!
export vie2_timestep_sin!
export VIE2SinWorkspace
#
export dyson_mat!
export dyson_start!
export dyson_timestep!
export dyson!
export kinetic_energy_trunc
#
export pseudodyson_mat!
export pseudodyson_start!
export pseudodyson_timestep!
export pseudodyson!
#
export pseudo_vie2_mat!
export pseudo_vie2_start!
export pseudo_vie2_timestep!
export pseudo_vie2!
#
export sigma_bath
export dyson_ness!

#=
### *Exports* : *observables*
=#

export density
export density_matrix
export kinetic_energy
export correlation_energy
export current
#
export bubble1!
export bubble2!
#
export bubble1_ness!
export bubble2_ness!
#
export evaluate_spectrum_windowed

#=
### *Exports* : *io*
=#

#export read!
#export write
#
export read_gf_ret_les_tavtrel
export write_fftarray
export read_fftarray
export write_ness
export read_ness

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
